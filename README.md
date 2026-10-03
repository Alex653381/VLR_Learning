# VLR_Learning

## 简介

这是一个用于VLR实验室轮转学习的代码仓库

## Reproduction

这里是经典论文、模型的学习与复现，复现过程已记录在各子目录下的 notes.md 中。

### 1.VGGNet

使用CatVSDog数据集从零训练vgg16进行二分类。

使用pytorch官方的vgg16权重（IMAGENET1K_V1）针对CatVSDog数据集进行微调训练。未参考MMPretrain框架；未实现训练可视化；未集成SENet模块。

### 2.ResNet

参考MMPretrain代码框架复现ResNet，并针对CatVSDog数据集进行微调训练。已实现 Tensorboard 训练过程可视化；未实现集成 SENet 模块；未实现 DDP 多卡训练。
