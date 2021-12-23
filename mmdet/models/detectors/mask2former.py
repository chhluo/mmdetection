from ..builder import DETECTORS
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Mask2Former(MaskFormer):

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
