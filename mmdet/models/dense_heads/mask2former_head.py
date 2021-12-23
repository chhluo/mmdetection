import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, kaiming_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList

from mmdet.core import build_assigner, build_sampler, reduce_mean
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead


@HEADS.register_module()
class Mask2FormerHead(MaskFormerHead):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_points=12544,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=...,
                 loss_mask=...,
                 loss_dice=...,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):  # res5 -> res3
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # res5 -> res3
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)
        self.transformer_decoder_norm = nn.LayerNorm(self.decoder_embed_dims)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        # TODO assigner, sampler, loss
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='MaskPseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is Mask2FormerHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official Mask2Former repo, bg_cls_weight
            # means relative classification weight of the VOID class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(self.num_classes + 1) * class_weight
            # set VOID class as the last indice
            class_weight[self.num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for i in range(self.num_transformer_feat_level):
            kaiming_init(self.decoder_input_projs[i], a=1)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape [num_queries, h, w].
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ). n is the sum of number of stuff type and
                number of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                    shape [num_queries, ].
                - label_weights (Tensor): Label weights of each image.
                    shape [num_queries, ].
                - mask_targets (Tensor): Mask targets of each image.
                    shape [num_queries, h, w].
                - mask_weights (Tensor): Mask weights of each image.
                    shape [num_queries, ].
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape [num_queries, num_points]
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape [num_gts, num_points]
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1), point_coords.repeat(num_gts, 1,
                                                       1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape [batch_size, num_queries,
                cls_out_channels].
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape [batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]:Loss components for outputs from a single decoder
                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape [batch_size, num_queries]
        labels = torch.stack(labels_list, dim=0)
        # shape [batch_size, num_queries]
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape [num_gts, h, w]
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape [batch_size, num_queries]
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape [batch_size * num_queries, ]
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_ones(self.num_classes + 1)
        class_weight[-1] = self.bg_cls_weight
        loss_cls = self.loss_bbox(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = self._get_uncertain_point_coords_with_randomness(
                mask_preds, self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            # shape [num_gts, num_points]
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1), points_coords).squeeze(1)
        # shape [num_queries, num_points]
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape [num_queries, num_points] -> [num_queries * num_points, 1]
        mask_point_preds = mask_point_preds.reshape(-1, 1)
        # shape [num_gts, num_points] -> [num_gts * num_points, ]
        mask_point_targets = mask_point_targets.reshape(-1)
        # target is (1 - mask_point_targets) !!!
        loss_mask = self.loss_mask(
            mask_point_preds,
            1 - mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def _get_uncertainty(self, logits):
        """We estimate uncerainty as L1 distance between 0.0 and the logit
        prediction in 'logits' for the foreground class in `classes`.

        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for
                class-specific or class-agnostic, where R is the
                total number of predicted masks in all images and C
                is the number of foreground classes. The values
                are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that
                contains uncertainty scores with the most uncertain
                locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))

    def _get_uncertain_point_coords_with_randomness(self, coarse_logits,
                                                    num_points,
                                                    oversample_ratio,
                                                    importance_sample_ratio):
        """Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The unceratinties are calculated for each point using
        'uncertainty_func' function that takes point's logit prediction as
        input. See PointRend paper for details.

        Args:
            coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask)
                or (N, 1, Hmask, Wmask) for class-specific or class-agnostic
                prediction.
            uncertainty_func: A function that takes a Tensor of shape
                (N, C, P) or (N, 1, P) that contains logit predictions
                for P points and returns their uncertainties as a Tensor
                of shape (N, 1, P).
            num_points (int): The number of points P to sample.
            oversample_ratio (int): Oversampling parameter.
            importance_sample_ratio (float): Ratio of points that are
                sampled via importnace sampling.

        Returns:
            point_coords (Tensor): A tensor of shape (N, P, 2) that
            contains the coordinates of P sampled points.
        """
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(
            num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = point_sample(
            coarse_logits, point_coords, align_corners=False)
        # It is crucial to calculate uncertainty based on the sampled
        # prediction value for the points. Calculating uncertainties of
        # the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits),
        # a sampled point between two coarse predictions with -1 and 1 logits
        # has 0 logits, and therefore 0 uncertainty value. However, if we
        # calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will
        # get -1 uncertainty.
        point_uncertainties = self._get_uncertainty(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(
            num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coords = torch.cat(
                [
                    point_coords,
                    torch.rand(
                        num_boxes,
                        num_random_points,
                        2,
                        device=coarse_logits.device),
                ],
                dim=1,
            )
        return point_coords

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called sequentially in every decoder
        layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            cls_pred (Tensor): Classification scores in shape (batch_size,
                num_queries, cls_out_channels).  Note `cls_out_channels`
                should includes background.
            mask_pred (Tensor): Mask scores in shape
                (batch_size, num_queries,h, w).
            attn_mask (Tensor): Attention mask in shape
                (batch_size * num_heads, num_queries, h, w).
        """
        # [nq, bs, c]
        decoder_out = self.transformer_decoder_norm(decoder_out)
        # [bs, nq, c]
        decoder_out = decoder_out.transpose(0, 1)
        # [bs, nq, c]
        cls_pred = self.cls_embed(decoder_out)
        # [bs, nq, c]
        mask_embed = self.mask_embed(decoder_out)
        # [b, q, h, w]
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='blinear',
            align_corners=False)
        # [bs, nq, h, w] -> [bs, nh, nq, h*w] -> [bs * nh, nq, h, w]
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            cls_pred_list (list[Tensor)]: Classification scores for each
                scale level. Each is a 3D-tensor with shape
                (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_list (list[Tensor]): Mask scores for each decoder
                layer. Each with shape (batch_size, num_queries, h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(
            feats, img_metas)
        # multi_scale_memorys (low to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # [b, c, h, w] -> [h*w, b, c]
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # [b, c, h, w] -> [h*w, b, c]
            decoder_positional_encoding = self.decoder_positional_encoding(
                multi_scale_memorys[i])
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # [nq, c] -> [nq, bs, c]
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,  # ! list[Tensor]
                query_key_padding_mask=None,
                # ! here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def simple_test(self, feats, img_metas, rescale=False):
        # TODO add detection result, semantic result?
        return super().simple_test(feats, img_metas, rescale=rescale)

    def semantic_inference(self, mask_cls, mask_pred):
        """Semantic segmengation inference. # TODO.

        This implementation is modified from
            https://github.com/facebookresearch/MaskFormer

        Args:
            mask_cls (Tensor): Classfication outputs for a image.
                shape = [num_queries, cls_out_channels].
            mask_pred (Tensor): Mask outputs for a image.
                shape = [num_queries, h, w].

        Returns:
            [type]: [description]
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        """Panoptic segmengation inference. # TODO.

        This implementation is modified from
            https://github.com/facebookresearch/MaskFormer

        Args:
            mask_cls (Tensor): Classfication outputs for a image.
                shape = [num_queries, cls_out_channels].
            mask_pred (Tensor): Mask outputs for a image.
                shape = [num_queries, h, w].

        Returns:
            panoptic_seg (dict[str, Tensor]):
                {'pan_results': tensor of shape = (H, W) and dtype=int32},
                each element in Tensor means:
                segment_id = _cls + instance_id * INSTANCE_OFFSET.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1
        return panoptic_seg

    def instance_inference(self, mask_cls, mask_pred):
        # TODO
        pass
