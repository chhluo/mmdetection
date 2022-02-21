# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Mask2Former(MaskFormer):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        feat = self.extract_feat(img)
        results = self.panoptic_head.simple_test(feat, img_metas, **kwargs)
        return results
