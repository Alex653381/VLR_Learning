# VLR_Learning

## 简介

这是一个用于VLR实验室轮转学习的代码仓库

## Reproduction

这里是经典论文、模型的学习与复现

### 1.VGGNet


#### 1.1 评估未微调的官方权重模型
下载官方预训练的VGG16权重（ImageNet V1），并用猫狗数据集进行测试与评估。在VGGNet目录下运行
`python predict_by_non_fine-tuning_pretrained_model.py`


#### 1.2 使用猫狗数据集从零训练VGG16

按照网页教程 https://cloud.tencent.com/developer/article/1980647 ，复现vgg16并使用猫狗二分类数据集从零训练参数。

制作训练集、测试集图片索引txt文件，在VGGNet目录下运行 
`python make_reference.py` 

#### 1.3 加载官方预训练权重进行微调

环境：

    System: Windows11
    
    RAM: 64GB
    
    GPU：NVIDIA GeForce RTX 5070 Laptop GPU (8 GB)
    
    CUDA Version: 12.8
   
    Python Version: Python 3.10.21
    
    Pytorch Version: torch 2.11.0+cu128  
                     
                     torchaudio 2.11.0+cu128 
                     
                     torchvision 0.26.0+cu128

模型：VGG16

预训练权重：ImageNet1K V1

训练方式：冻结卷积层，只训练分类头

数据：18000 train / 2000 val / 5000 test

epoch：5

最佳 epoch：4

best val accuracy：98.60%

test accuracy：98.34%

cat accuracy：97.56%

dog accuracy：99.12%

correct：4917/5000

测试耗时：32.40 秒

平均单张推理时间：6.48 ms
