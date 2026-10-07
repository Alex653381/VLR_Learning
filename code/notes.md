# 代码学习训练

## 一个简单的CNN网络，用于识别MNIST手写数字识别——v1
大概结构：
`````
input
conv1 + ReLU      #输入1通道，输出32通道，核3*3，步长1，padding=1
max pooling
conv2 + ReLU      #输入32通道，输出64通道，核3*3，步长1，padding=1
max pooling
fc1 + ReLU        #展平后输入到全连接层
fc2               #10个类别
output
`````
优化器：多分类交叉熵损失
训练集 60000 张图片，测试集 10000 张
共5个 epoch，batch size = 64

直接在该目录下运行：`python my_cnn_v1.py`

Test Accuracy: 99.08%