_base_ = [
    '_base_/models/resnet50_catdog.py',
    '_base_/datasets/catdog_bs32.py',
    '_base_/schedules/catdog_finetune.py',
    '_base_/default_runtime.py',
]

work_dir = 'work_dirs/resnet50_catdog'