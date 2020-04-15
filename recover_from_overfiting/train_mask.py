#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
2020-2-8

'''
# 首先用有噪声的数据训练模型
## 生成有噪声的数据 transition matrix
## 训练模型

# 接收训练好的模型
# 初始化所有的 m = 1
# 循环 N_e 次
## 取D_val中的数据
## 计算 mask 中的 M = step_c(m)
## 计算L((M*W)x,y)  并计算导数
## 更新 m
## m= max(0,min(1,m)) 保证 m 属于 [0,1]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets , transforms , models
from torch.optim import lr_scheduler
from torch.autograd import variable
from Net import oral_net_bias , oral_net
import torchvision
import argparse
import copy
import collections
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 超参数设置 Hyperparameter setting
EPOCH = 100  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率
# lr=0.0001   bs=16         |lr=0.0002   bs=16
# Loss: 0.245 Acc: 81.000% |Loss: 0.260  Acc: 79.000%
# lr=0.0001   bs=32
# Loss: 0.260 | Acc: 79.000%
# lr=0.0001   bs=64         | lr=0.0002   bs=64
# Loss: 0.288 | Acc: 79.000% | Loss: 0.373 | Acc: 78.000%
# lr=0.0002   bs=128
# Loss: 0.325 | Acc: 79.000%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf' , default='./model' , help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net' , default='./model/VGG16bn.pth' , help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# class Net(nn.Module) :
#     def __init__(self , features , num_classes=2 , init_weights=False , is_bias=True) :
#         super(Net , self).__init__()
#         self.features = net.features
#         # self.pooling=nn.AvgPool2d() #kernal size ???
#         # self.classifier=nn.Linear(128,num_classes,bias=is_bias)
#         self.classifier = net.classifier
#
#         if init_weights :
#             self._initialize_weights()
#
#     def forward(self , x) :
#         x = self.features(x)
#         x = x.view(x.size(0) , -1)
#         # x=self.pooling(x)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self) :
#         for m in self.modules() :
#             print('m' , m)
#             if isinstance(m , nn.Conv2d) :
#                 # m.weight = nn.Parameter(torch.Tensor(np.ones([m.out_channels , m.in_channels , m.kernel_size[0] , m.kernel_size[1]])))
#                 m.weight.data = torch.Tensor(
#                     np.ones([m.out_channels , m.in_channels , m.kernel_size[0] , m.kernel_size[1]]))
#                 print(m)
#             elif isinstance(m , nn.BatchNorm2d) :
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m , nn.Linear) :
#                 m.weight = nn.Parameter(torch.Tensor(np.ones([m.out_features , m.in_features])))


# 准备数据集并预处理

d_t = transforms.Compose([
    transforms.RandomResizedCrop(224) ,
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])
d_t_val = transforms.Compose([
    transforms.Resize(256) ,
    transforms.CenterCrop(224) ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])
data_dir = '/media/liang/新加卷/disambigular/data/mask/mask/'
image_datasets = torchvision.datasets.ImageFolder(data_dir , d_t)

train_loader = torch.utils.data.DataLoader(image_datasets , batch_size=BATCH_SIZE , shuffle=True , num_workers=2)
# 定义损失函数和优化方式
net = torch.load("mnet.pkl")  # mask
net = net.to(device)
# criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
criterion = nn.KLDivLoss()
optimizer = torch.optim.Adam(net.parameters() , lr=LR)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = lr_scheduler.StepLR(optimizer , step_size=10 , gamma=0.1)


# 载入模型
def restore_net(model_root) :
    # restore entire network
    net = torch.load(model_root)
    return net


net_original = torch.load('alexnet_KL_sun.pkl')
net_original_temp = net_original.state_dict()
net_original_dict = copy.deepcopy(net_original.state_dict())  # 保持每次 M*W 的时候 Weight 是原始训练好的
net_dict_keys = net_original_dict.keys()


# activate parameters
def set_param_requires_grad(model) :
    for param in model.parameters() :
        param.requires_grad = True


def save_param(model , name) :
    torch.save(model , name + '.pkl')


# 训练
if __name__ == "__main__" :
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training,Mask!")  # 定义遍历数据集的次数
    set_param_requires_grad(net)
    distribution = variable(
        torch.FloatTensor([[0 , 1] , [0.2 , 0.8] , [0.4 , 0.6] , [0.6 , 0.4] , [0.8 , 0.2] , [1 , 0]]))
    for epoch in range(pre_epoch , EPOCH) :
        print('\nEpoch: %d' % (epoch + 1))
        scheduler.step()
        net.train()

        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i , data in enumerate(train_loader) :
            # 准备数据
            # print(i,data)
            length = len(train_loader)
            # print(length)
            inputs , labels = data
            inputs , labels = inputs.to(device) , labels.to(device)
            # print(inputs)
            ldl_labels = variable(torch.FloatTensor(len(labels) , 2))
            for labels_index in range(len(labels)) :
                if labels[labels_index] == 0 :
                    ldl_labels[labels_index] = distribution[5]
                elif labels[labels_index] == 1 :
                    ldl_labels[labels_index] = distribution[0]
            ldl_labels = ldl_labels.to(device)
            optimizer.zero_grad()
            # print(net)
            net_dict = net.state_dict()  # 取出 mask 的值
            # print(net_dict)
            # 对Mask 中的值 带入step_c()
            for key in net_dict :
                # print('key',key)
                # print('net',net_dict[key])
                # print('step',net_dict[key].ge(0.5))
                if key in net_dict :
                    net_original_temp[key] = torch.mul(net_dict[key].ge(0.5).float().to(device) ,
                                                       net_original_dict[key])
                    # print(net_original_dict[key])
                if re.search("mean" , key) or re.search("_var" , key) :
                    # print(net_original_dict[key])
                    net_original_temp[key] = net_original_dict[key]
                # print(net_orginal_temp[key])
            # print(net_dict)
            new = copy.deepcopy(net_dict)
            # print('*******************************************************pirnt mask_dict after backward ***************************************************************')

            # 计算 W * M calculation
            net.load_state_dict(net_original_temp)

            # for m in net.parameters() :
            # print(m)
            # print(net.state_dict())
            # forward + backward
            outputs = net(inputs)  #
            # print(outputs)
            last_layer = torch.nn.LogSoftmax()
            outputs = last_layer(outputs)
            # print(outputs,ldl_labels)
            loss = criterion(outputs , ldl_labels)
            # print(loss)
            net.load_state_dict(new)
            # for name, param in net_original.named_parameters():
            # print(param.requires_grad)
            # print(net_dict)
            # print(new)
            # print(loss)
            loss.backward()
            optimizer.step()

            # 对Mask 中的值 进行 m=max(0,min(1,m))
            for m in net.modules() :
                if isinstance(m , nn.Conv2d) :
                    # print(np.nonzero(m.weight.data))
                    # print(m,m.weight.data)
                    m.weight.data = torch.clamp(m.weight.data , min=0 , max=1).to(device)
                    # print(np.nonzero(m.weight.data))
                    m.bias.data.fill_(1).to(device)
                    # print(m,m.weight.data)
                elif isinstance(m , nn.BatchNorm2d) :  # running mean & running variance???
                    m.weight.data.fill_(1).to(device)
                    m.bias.data.fill_(1).to(device)
                elif isinstance(m , nn.Linear) :
                    m.weight.data = torch.clamp(m.weight.data , min=0 , max=1).to(device)
                    m.bias.data.fill_(1).to(device)
                    # print(m,m.weight.data)
            # print(net.state_dict())

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _ , predicted = torch.max(outputs.data , 1)
            # print(predicted,labels.data)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
            epoch + 1 , (i + 1 + epoch * length) , sum_loss / (i + 1) , 100. * correct / total))
        print(
            '*******************************************************pirnt mask_dict after backward ***************************************************************')
        print(new)
