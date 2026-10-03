_base_ = ['./resnet18_catdog.py']

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5,
    val_interval=1000,
)

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    begin=0,
    end=5,
    milestones=[1000],
    gamma=0.1,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best=None,
        max_keep_ckpts=1,
    ),
)

val_cfg = None
val_dataloader = None
val_evaluator = None

test_cfg = None
test_dataloader = None
test_evaluator = None

work_dir = 'work_dirs/resnet18_catdog_smoke'