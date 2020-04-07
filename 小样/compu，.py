import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
'''
# 第一一个卷积层，我们可以看到它的权值是随机初始化的
w = torch.nn.Conv2d(2 , 2 , 3 , padding=1)
print(w.weight)
print("+++++++++++++++++++")

print(w.weight.data)

print("the type of w.weight：",type(w.weight))
# 第一种方法
print("1.使用另一个Conv层的权值")
q = torch.nn.Conv2d(2 , 2 , 3 , padding=1)  # 假设q代表一个训练好的卷积层
print(q.weight)  # 可以看到q的权重和w是不同的
#w.weight=q.weight
# 第二种方法

print("2.使用来自Tensor的权值")
ones=torch.Tensor(2*np.ones([2,2,3,3])) # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
# w.weight=torch.nn.Parameter(ones) # 把Tensor的值作为权值赋值给Conv层，这里需要先转为torch.nn.Parameter类型，否则将报错
# print(w.weight)


w.weight.data =torch.mul(q.weight.data,ones)  # 把一个Conv层的权重赋值给另一个Conv层
print(w.weight)
'''
# class Net(nn.Module):
#     def __init__(self):
#         super(Net , self).__init__()
#         #self.layer1=torch.nn.Linear(1, 10)
#         #self.layer2=torch.nn.Linear(10, 1)
#     def forward(self,x):
#         x=self.layer1(x)
#         x=F.relu(x)
#         x=self.layer2(x)
#         return x

net = torch.load('net.pkl')
print(net)

for param in net.parameters():
    print('===========',param,'\n', param.size())
































