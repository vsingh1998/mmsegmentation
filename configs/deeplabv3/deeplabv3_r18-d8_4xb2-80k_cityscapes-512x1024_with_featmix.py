_base_ = './deeplabv3_r50-d8_4xb2-80k_cityscapes-512x1024.py'
randomness = dict(seed=1)
load_from = '/home/azureuser/cloudfiles/code/Users/vikram.singh/trained-models/iter_64000_upload.pth'
log_processor = dict(by_epoch=False) 
model = dict(
    type='EncoderDecoderDeeplabFeatmix',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18,
                  strides=(1, 2, 2, 2),
                  ),
    neck=dict(
        type='MLPFPN',
        in_channels=[64,128,256,512],
        out_channels=512,
        mixer_count=1,
        start_stage=2,
        patch_dim=4,
        end_stage=3,
        feat_channels=[1,4,32,256],
        ),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=50)

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations'),
#     dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
#     dict(type='PackSegInputs')
# ]

train_dataloader = dict(
    batch_size=2,
    num_workers=4)