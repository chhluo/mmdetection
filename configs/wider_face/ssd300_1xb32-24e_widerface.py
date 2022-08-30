_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/wider_face.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_2x.py'
]
model = dict(bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
train_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train.txt',
            data_prefix=dict(img='WIDER_train'),
            filter_cfg=dict(
                filter_empty_gt=True, bbox_min_size=17, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 20],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.002, momentum=0.9, weight_decay=5e-4))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
