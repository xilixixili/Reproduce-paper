# encoding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F


# define model
class TheModelClass(nn.Module) :
    def __init__(self) :
        super(TheModelClass , self).__init__()
        self.conv1 = nn.Conv2d(3 , 6 , 5)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6 , 16 , 5)
        self.fc1 = nn.Linear(16 * 5 * 5 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)

    def forward(self , x) :
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1 , 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main() :
    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters() , lr=0.001 , momentum=0.9)

    # print model's state_dict
    print('Model.state_dict:')
    for param_tensor in model.state_dict() :
        # 打印 key value字典
        print(param_tensor , '\t' , model.state_dict()[param_tensor].data)

    print('Model.parameters:')
    for param_tensor in model.parameters() :
        # 打印 key value字典
        print(type(param_tensor) , '\t' , param_tensor.size())

    # print optimizer's state_dict
    print('Optimizer,s state_dict:')
    for var_name in optimizer.state_dict() :
        print(var_name , '\t' , optimizer.state_dict()[var_name])


if __name__ == '__main__' :
    main()



