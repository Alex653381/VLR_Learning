_base_ = [
    '_base_/models/resnet18_catdog.py',
    '_base_/datasets/catdog_bs32.py',
    '_base_/schedules/catdog_finetune.py',
    '_base_/default_runtime.py',
]

work_dir = 'work_dirs/resnet18_catdog'
#不用配置load_form
#官方 ImageNet 权重已经在 ResNet backbone 的 init_cfg 中加载
#head是新的2类分类头，不需要通过load_from加载原来的1000类分类头