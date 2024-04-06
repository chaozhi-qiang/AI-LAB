auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=30,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CamSDD'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=200, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=30,
        topk=(
            1,
            3,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.045, momentum=0.9, type='SGD', weight_decay=4e-05))
param_scheduler = dict(by_epoch=True, gamma=0.98, step_size=1, type='StepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1_Portrait',
            '2_Group_portrait',
            '3_Kids',
            '4_Dog',
            '5_Cat',
            '6_Macro',
            '7_Food',
            '8_Beach',
            '9_Mountain',
            '10_Waterfall',
            '11_Snow',
            '12_Landscape',
            '13_Underwater',
            '14_Architecture',
            '15_Sunset_Sunrise',
            '16_Blue_Sky',
            '17_Cloudy_Sky',
            '18_Greenery',
            '19_Autumn_leaves',
            '20_Flower',
            '21_Night_shot',
            '22_Stage_concert',
            '23_Fireworks',
            '24_Candle_light',
            '25_Neon_lights',
            '26_Indoor',
            '27_Backlight',
            '28_Text_Documents',
            '29_QR_images',
            '30_Computer_Screens',
        ],
        data_root='E:\\AI_dataset\\CamSDD\\validation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CamSDD',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        3,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1_Portrait',
            '2_Group_portrait',
            '3_Kids',
            '4_Dog',
            '5_Cat',
            '6_Macro',
            '7_Food',
            '8_Beach',
            '9_Mountain',
            '10_Waterfall',
            '11_Snow',
            '12_Landscape',
            '13_Underwater',
            '14_Architecture',
            '15_Sunset_Sunrise',
            '16_Blue_Sky',
            '17_Cloudy_Sky',
            '18_Greenery',
            '19_Autumn_leaves',
            '20_Flower',
            '21_Night_shot',
            '22_Stage_concert',
            '23_Fireworks',
            '24_Candle_light',
            '25_Neon_lights',
            '26_Indoor',
            '27_Backlight',
            '28_Text_Documents',
            '29_QR_images',
            '30_Computer_Screens',
        ],
        data_root='E:\\AI_dataset\\CamSDD\\training/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CamSDD',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1_Portrait',
            '2_Group_portrait',
            '3_Kids',
            '4_Dog',
            '5_Cat',
            '6_Macro',
            '7_Food',
            '8_Beach',
            '9_Mountain',
            '10_Waterfall',
            '11_Snow',
            '12_Landscape',
            '13_Underwater',
            '14_Architecture',
            '15_Sunset_Sunrise',
            '16_Blue_Sky',
            '17_Cloudy_Sky',
            '18_Greenery',
            '19_Autumn_leaves',
            '20_Flower',
            '21_Night_shot',
            '22_Stage_concert',
            '23_Fireworks',
            '24_Candle_light',
            '25_Neon_lights',
            '26_Indoor',
            '27_Backlight',
            '28_Text_Documents',
            '29_QR_images',
            '30_Computer_Screens',
        ],
        data_root='E:\\AI_dataset\\CamSDD\\validation/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CamSDD',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        3,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\mobilenetv2_camsdd'
