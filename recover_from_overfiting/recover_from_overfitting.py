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
from torchvision import datasets,transforms,models
from Net import oral_net_bias,oral_net
import torchvision
import argparse
import re
import collections
# 超参数设置 Hyperparameter setting
EPOCH = 100   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1        #学习率


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/VGG16bn.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()


# 准备数据集并预处理


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''

class myDataset(datasets.ImageFolder):
    #custom dataset for loading data iamges
    def __getitem__(self, index):

        path, target = self.imgs[index]
        #print(path,target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)

data_dir = 'D:/code/Python/test_pytorch/mnist'# 此地址为没有解压的数据集，顾不可用

train_set=myDataset(data_dir , transform=transform_train)

'''
train_set = torchvision.datasets.MNIST(root='D:/code/Python/test_pytorch/mnist', train=True, download=False, transform=transform_train) #训练数据集
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

test_set = torchvision.datasets.MNIST(root='D:/code/Python/test_pytorch/mnist', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

# 模型定义 mask
net=oral_net_bias()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer=torch.optim.Adam(net.parameters(),lr=LR)

# 载入模型
def restore_net(model_root):
    # restore entire network
    net = torch.load(model_root)
    return net
net_orginal=restore_net('model_root')
net_orginal_temp=net_orginal.state_dict()
net_orginal_dict=net_orginal_temp.copy() # 保持每次 M*W 的时候 Weight 是原始训练好的
net_dict_keys=net_orginal_dict.keys()

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training,Mask!")  # 定义遍历数据集的次数

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 对Mask 中的值 进行 m=max(0,min(1,m))
            for m in net.modules() :
                if isinstance(m , nn.Conv2d) :
                    m.weight = torch.round(torch.clamp(m.weight.data))
                elif isinstance(m , nn.BatchNorm2d) :# running mean & running variance???
                    m.weight.data.fill_(1)#running mean
                    m.bias.data.fill_(1) #running variance
                elif isinstance(m , nn.Linear) :
                    m.weight.data = torch.round(torch.clamp(m.weight.data,min=0,max=1))
            # 对Mask 中的值 带入step_c()
            # 上步 round 的操作可实现 c=0.49999999999... 差一点完美
            net_dict = net.state_dict()# 取出 mask 的值

            for key in net_dict_keys:
                if key in net_dict:
                    net_orginal_temp['key']=torch.mul(net_dict['key'],net_orginal_dict['key'])
                if re.search("mean",key) or re.search("_var",key):# 保持bn层相同
                    net_dict['key']=net_orginal_dict['key']
            # 计算 W * M calculation
            net.load_state_dict(net_dict)
            net_orginal.load_state_dict(net_orginal_temp)
            # forward + backward
            outputs = net_orginal(inputs)   #
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
