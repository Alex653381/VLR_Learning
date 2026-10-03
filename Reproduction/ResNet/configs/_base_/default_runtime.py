default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='accuracy/top1',#根据验证集Top-1 Accuracy保存最佳模型
        rule='greater',
        max_keep_ckpts=2,#只保留最近的普通checkpoint，避免work_dirs无限增长。最佳checkpoint会单独保留
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0,
    ),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='mmengine.TensorboardVisBackend'),#从第一次训练开始记录 loss、accuracy、learning rate 等曲线
]
#从第一次训练开始记录 loss、accuracy、learning rate 等曲线。
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends,
)

log_level = 'INFO'
load_from = None
resume = False

randomness = dict(
    seed=42,
    deterministic=False,
)