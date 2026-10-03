optim_wrapper = dict(
    optimizer=dict(
        type='SGD',#SGD优化器，随机梯度下降
        lr=0.01,#学习率
        momentum=0.9,#优化器动量配置
        weight_decay=0.0001,#权重衰减，降低过拟合风险
    )
)

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=10,
    by_epoch=True,
    milestones=[6, 9],#第6轮和第9轮结束时把学习率乘以0.1，让学习率降低，参数收敛得更精细
    gamma=0.1,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=1,#每个epoch结束后进行一次验证
)

val_cfg = dict()
test_cfg = dict()