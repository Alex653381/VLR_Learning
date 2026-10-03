#创建resnet18模型配置
pretrained_url = (
    'https://download.openmmlab.com/mmclassification/v0/resnet/'
    'resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,#resnet18
        num_stages=4,
        out_indices=(3,),#只取最后一个stage的特征，输出通道数是512
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_url,
            prefix='backbone.',#只加载backbone部分的权重，不加载分类头
        ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)