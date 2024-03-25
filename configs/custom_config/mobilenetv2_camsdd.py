# _base_ = [
#     '../_base_/models/mobilenet_v2_1x.py',
#     '../_base_/datasets/imagenet_bs32_pil_resize.py',
#     '../_base_/schedules/imagenet_bs256_epochstep.py',
#     '../_base_/default_runtime.py'
# ]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 3),
    ))

# dataset settings
dataset_type = 'CamSDD'
data_preprocessor = dict(
    num_classes=30,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=r'E:\AI_dataset\CamSDD\training/',
        # split='train',
        classes=['1_Portrait', '2_Group_portrait', '3_Kids', '4_Dog', '5_Cat', '6_Macro', '7_Food',
                 '8_Beach', '9_Mountain', '10_Waterfall', '11_Snow', '12_Landscape', '13_Underwater', '14_Architecture',
                 '15_Sunset_Sunrise', '16_Blue_Sky', '17_Cloudy_Sky', '18_Greenery', '19_Autumn_leaves', '20_Flower',
                 '21_Night_shot', '22_Stage_concert', '23_Fireworks', '24_Candle_light', '25_Neon_lights',
                 '26_Indoor', '27_Backlight', '28_Text_Documents', '29_QR_images', '30_Computer_Screens'],
        with_label=True,  # or False for unsupervised tasks),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=r'E:\AI_dataset\CamSDD\validation/',
        # split='val',
        classes=['1_Portrait', '2_Group_portrait', '3_Kids', '4_Dog', '5_Cat', '6_Macro', '7_Food',
                 '8_Beach', '9_Mountain', '10_Waterfall', '11_Snow', '12_Landscape', '13_Underwater', '14_Architecture',
                 '15_Sunset_Sunrise', '16_Blue_Sky', '17_Cloudy_Sky', '18_Greenery', '19_Autumn_leaves', '20_Flower',
                 '21_Night_shot', '22_Stage_concert', '23_Fireworks', '24_Candle_light', '25_Neon_lights',
                 '26_Indoor', '27_Backlight', '28_Text_Documents', '29_QR_images', '30_Computer_Screens'],
        with_label=True,  # or False for unsupervised tasks
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 3))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule setting
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004))

# learning policy
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=1, gamma=0.98)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# runtime setting
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=200),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='auto'),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer、
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
