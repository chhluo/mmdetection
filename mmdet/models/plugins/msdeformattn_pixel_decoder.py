import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, kaiming_init,
                      normal_init, xavier_init)
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, ModuleList


@PLUGIN_LAYERS.register_module()
class MSDeformAttnPixelDecoder(BaseModule):
    """fpn msdeform encoder."""

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_channels=256,
                 out_channels=256,
                 num_return_feat_levels=3,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 encoder=dict(
                     type='DetrTransformerEncoder',
                     num_layers=6,
                     transformerlayers=dict(
                         type='BaseTransformerLayer',
                         attn_cfgs=dict(
                             type='MultiScaleDeformableAttention',
                             embed_dims=256,
                             num_heads=8,
                             num_levels=3,
                             num_points=4,
                             im2col_step=64,
                             dropout=0.0,
                             batch_first=False,
                             norm_cfg=None,
                             init_cfg=None),
                         feedforward_channels=1024,
                         ffn_dropout=0.0,
                         operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                     init_cfg=None),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_input_levels = len(in_channels)
        self.num_encoder_feat_levels = \
            encoder.transformerlayers.attn_cfgs.num_levels
        assert self.num_encoder_feat_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_proj_list = []
        # from top to down (low to high resolution)
        for i in range(
                self.num_input_levels - 1,
                self.num_input_levels - self.num_encoder_feat_levels - 1, -1):
            input_proj = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_proj_list.append(input_proj)
        self.input_projs = ModuleList(input_proj_list)

        self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(
            positional_encoding)
        # res_i -> res_5
        self.level_encoding = nn.Embedding(self.num_encoder_feat_levels,
                                           feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_feat_levels, 0,
                       -1):
            l_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            o_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.output_convs.append(o_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_return_feat_levels = num_return_feat_levels

    def init_weights(self):
        for i in range(0, self.num_encoder_feat_levels):
            xavier_init(
                self.input_projs[i].conv.weight,
                gain=1,
                bias=0,
                distribution='normal')

        for i in range(0,
                       self.num_input_levels - self.num_encoder_feat_levels):
            kaiming_init(self.lateral_convs[i].conv.weight, a=1)
            kaiming_init(self.output_convs[i].conv.weight, a=1)

        kaiming_init(self.mask_feature, a=1)
        normal_init(self.level_encoding, mean=0, std=1)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, feats, img_metas):
        # generate padding mask for each level, for each image
        bs = len(img_metas)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        padding_mask = feats[0].new_ones((bs, input_img_h, input_img_w),
                                         dtype=torch.float32)
        for i in range(bs):
            img_h, img_w, _ = img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0

        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        valid_radio_list = []
        spatial_shapes = []
        padding_mask = padding_mask.unsqueeze(1)
        for i in range(self.num_encoder_feat_levels):  # res5, res4, ...
            feat = feats[self.num_input_levels - i - 1]
            feat_projected = self.input_projs[i](feat)
            padding_mask_resized = F.interpolate(
                padding_mask, size=feat.shape[-2:],
                mode='nearest').to(torch.bool).squeeze(1)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[
                self.num_encoder_feat_levels - i - 1]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed

            # [batch_size, c, h_i, w_i] -> [h_i * w_i, batch_size, c]
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            valid_radio = self.get_valid_ratio(padding_mask_resized)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            valid_radio_list.append(valid_radio)
            spatial_shapes.append(feat.shape[-2:])

        # [batch_size, total_num_query], total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # [batch_size, num_encoder_feat_levels, 2]
        valid_radios = torch.stack(valid_radio_list, dim=1)
        # [total_num_query, batch_size, c]
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=0)
        device = encoder_inputs.device
        # [num_encoder_feat_levels, 2], res5, res4, ...
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(
            spatial_shapes, valid_radios, device=device)

        # [num_total_query, batch_size, c]
        memory = self.encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        # [num_total_query, batch_size, c] -> [batch_size, c, num_total_query]
        memory = memory.permute(1, 2, 0)
        c = memory.shape[1]

        # res5, res4, ...
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(bs, c, spatial_shapes[i][0], spatial_shapes[i][1])
            for i, x in enumerate(outs)
        ]

        for i in range(
                self.num_input_levels - self.num_encoder_feat_levels - 1, 0,
                -1):
            x = feats[i]
            cur_fpn = self.lateral_convs[i](x)
            y = cur_fpn + F.interpolate(
                outs[-1], size=cur_fpn.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)
            outs.append(y)

        multi_scale_features = outs[:self.num_return_feat_levels]

        return self.mask_feature(outs[-1]), multi_scale_features
