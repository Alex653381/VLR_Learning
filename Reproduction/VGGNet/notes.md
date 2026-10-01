# Reproduction of VGGNet(VGG16)


## 1.0 目录结构

项目结构：
`````
VGGNet
│  index.md
│  list.txt
│  make_reference.py
│  predict.py
│  predict_by_non_fine-tuning_pretrained_model.py
│  test_pretrained.py
│  train.py
│  train_pretrained.py
│  
├─data
│  └─catVSdog
│      │  test.txt
│      │  train.txt
│      │  
│      ├─test_data
│      │  ├─cat
│      │  └─dog
│      └─train_data
│          ├─cat_train
│          └─dog_train
├─images
│      test_dog.jpg
│      
├─models
│      vgg16_catvsdog_head_best.pth
│      
└─process_record
        non_fine-tuning vgg16.png
`````


## 1.1 评估未微调的官方权重模型
下载官方预训练的VGG16权重（ImageNet V1），并用猫狗数据集进行测试与评估。在VGGNet目录下运行
`python predict_by_non_fine-tuning_pretrained_model.py`


## 1.2 使用猫狗数据集从零训练VGG16

按照网页教程 https://cloud.tencent.com/developer/article/1980647 ，复现vgg16并使用猫狗二分类数据集从零训练参数。

制作训练集、测试集图片索引txt文件，在VGGNet目录下运行 
`python make_reference.py` 

## 1.3 加载官方预训练权重进行微调

环境：

    System: Ubuntu-24.04
    
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

best val accuracy：98.45%

test accuracy：98.4000%

cat accuracy：98.2800%

dog accuracy：98.5200%

correct：4920/5000

Total time: 14.87 seconds

Average time per image: 2.97 ms