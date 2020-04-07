import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
# net1 = Net(1, 10, 1)
#
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1)
# )
#
# print(net1)
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
# print(net2)
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""



torch.manual_seed(1)    # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.layer1=torch.nn.Linear(1, 10)
        self.layerF=nn.ReLU()
        self.layer2=torch.nn.Linear(10, 1)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layerF(x)
        x=self.layer2(x)
        return x

def save():
    net = Net()
    # 建网络
    print("in++++++++++++++++")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 训练
    net = net.train()  # model.train()
    for t in range(100):
        print('in loop+++++++++')
        prediction = net(x)

        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()
    torch.save(net , 'net2.pkl')  # 保存整个网络
    torch.save(net.state_dict() , 'net_params2.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


save()


# def restore_net():
#     # restore entire net1 to net2
#     net2 = torch.load('net.pkl')
#     prediction = net2(x)
#
# def restore_params():
#     # 新建 net3
#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
#
#     # 将保存的参数复制到 net3
#     net3.load_state_dict(torch.load('net_params.pkl'))
#     prediction = net3(x)
#
# # 保存 net1 (1. 整个网络, 2. 只有参数)
# save()
#
# # 提取整个网络
# restore_net()
#
# # 提取网络参数, 复制到新网络
# restore_params()