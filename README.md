# VLR_Learning

## 简介

这是一个用于VLR实验室轮转学习的代码仓库

## 论文笔记

这里是论文阅读笔记。

## Reproduction

这里是经典论文、模型的学习与复现，复现过程已记录在各子目录下的 notes.md 中。

### 数据和模型下载

仓库不直接提交数据集和模型权重。相关文件请在 GitHub Releases 中下载：

- vggnet_data.tar.gz
- vggnet_models.tar.gz
- resnet_data.tar.gz
- resnet_work_dirs.tar.gz

下载后在 `Reproduction` 目录中解压：

```bash
cd Reproduction
tar -xzf vggnet_data.tar.gz
tar -xzf vggnet_models.tar.gz
tar -xzf resnet_data.tar.gz
tar -xzf resnet_work_dirs.tar.gz
```

在 Reproduction 目录解压，会自动恢复到正确位置:
```
VGGNet/data/...
VGGNet/models/...
ResNet/data/...
ResNet/work_dirs/...
```

### 1.VGGNet

使用CatVSDog数据集从零训练vgg16进行二分类。

使用pytorch官方的vgg16权重（IMAGENET1K_V1）针对CatVSDog数据集进行微调训练。未参考MMPretrain框架；未实现训练可视化；未集成SENet模块。

### 2.ResNet

参考MMPretrain代码框架复现ResNet18，并针对CatVSDog数据集进行微调训练。已实现 Tensorboard 训练过程可视化；未实现集成 SENet 模块；未实现 DDP 多卡训练。

### 3.TurboVLA

在 Libero 上复现并评测 Turbo VLA ，使用4个 suite 评测： libero_10 、 libero_goal 、 libero_object 、libero_spatial，取平均准确率，目前最高为 93.4% 。