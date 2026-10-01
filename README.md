# VLR_Learning

## 简介

这是一个用于VLR实验室轮转学习的代码仓库

## Reproduction

这里是经典论文、模型的学习与复现

### 1.VGGNet

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
