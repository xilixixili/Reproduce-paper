
import os
import time
from collections import Counter
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision import transforms

torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyperparameters
keep_prob = 0.7
random_seed = 1
learning_rate = 0.001
num_epochs = 40
batch_size = 64
pool_nb = 5  #####
test_bs = 32
n1 = 8
n2 = 8

# Architecture
num_classes = 5


def compute_accuracy(model , batch_s , num_classes , data_loader) :  # probas：sofamax输出向量
    correct_pred , num_examples = 0 , 0
    for features , targets in data_loader :
        # print(type(targets))
        features = features.to(device)
        targets = targets.to(device)
        logits , probas = model(features)
        _ , predicted = torch.max(probas , 1)  # 返回最大值及索引
        num_examples += targets.size(0)
        correct_pred += (predicted == targets).sum()
    return correct_pred.float() / num_examples * 100

def compute_mis(model,data_loader):
    mis=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])# [target][predicted]
    # mis={'one2one':0,'one2two':0,'one2three'}

    correct_pred , num_examples = 0 , 0
    for features , targets in data_loader :
        print(targets)
        features = features.to(device)
        targets = targets.to(device)
        logits , probas = model(features)
        _ , predicted = torch.max(probas , 1)  # 返回最大值及索引

        predicted=predicted.int()
        targets=targets.int()
        #print('=====',targets[0].item())
        lens=len(targets)
        for i in range(lens):
            # print('i:',i,'targets[i].item()',targets[i].item(),'predicted[i].item()',predicted[i].item())
            mis[targets[i].item()][predicted[i].item()]+=1
        num_examples += targets.size(0)
        correct_pred += (predicted == targets).sum()

    print('Total average acc: %.2f%%' % (correct_pred.float() / num_examples * 100))
    row,col = mis.shape
    sum_row=np.sum(mis,axis=1)
    print(sum_row)
    print('===============================')
    print(mis)
    for ro in range(row):
        for co in range(col):
            print(ro,'To',co,':','%5.2f%%'%(mis[ro][co]/sum_row[ro]*100),end='  ')
        print()

def save_param(model , name) :
    torch.save(model , name + '.pkl')


def get_count_by_counter(l) :
    # t1 = time.time()
    count = Counter(l)  # 类型： <class 'collections.Counter'>
    # t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)  # 类型： <type 'dict'>
    return count_dict


class Net(nn.Module) :
    def __init__(self) :
        self.bs = batch_size
        super(Net , self).__init__()

        self.conv1a = nn.Conv2d(in_channels=3 , out_channels=8 , kernel_size=(n1 , 2) , stride=1,padding=(n1 // 2 , 1))
        self.conv1b = nn.Conv2d(in_channels=3 , out_channels=8 , kernel_size=(2 , n2) , stride=1,padding=(1 , n2 // 2))
        self.cnn_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2 , 2) , stride=(2 , 2)) ,
            # conv2
            nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=(3 , 3) , stride=(1 , 1)) ,
            nn.BatchNorm2d(32) ,
            nn.ReLU() ,
            nn.MaxPool2d(kernel_size=(2 , 2) , stride=(2 , 2)) ,
            # conv3
            nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=(3 , 3) , stride=(1 , 1)) ,
            nn.BatchNorm2d(48) ,
            nn.ReLU() ,
            nn.MaxPool2d(kernel_size=(2 , 2) , stride=(2 , 2)) ,
            # conv4
            nn.Conv2d(in_channels=48 , out_channels=64 , kernel_size=(3 , 3) , stride=(1 , 1)) ,
            nn.BatchNorm2d(64) ,
            nn.ReLU() ,

            # conv5
            nn.Conv2d(in_channels=64 , out_channels=80 , kernel_size=(3 , 3) , stride=(1 , 1)) ,
            nn.BatchNorm2d(80) ,
            nn.ReLU()
        )
        self.bottom_up_att = nn.Conv2d(80 , 1 , kernel_size=1 , stride=1 , padding=0 , bias=False)  ###
        self.top_down_att = nn.Conv2d(80 , num_classes , kernel_size=1 , stride=1 , padding=0 ,bias=False)  ###每个通道对应一个类别，再与bottom做attetnion
        # self.fc = nn.Linear(in_features = 3465,out_features = 5)
        self.avg_pooling=nn.AdaptiveAvgPool2d(1)
    def forward(self , x) :
        out1 = self.conv1a(x)
        out2 = self.conv1b(x)
        x = torch.cat((out1 , out2) , 1)  # 10，16，136，549
        x = self.cnn_layer(x)
        x_1 = self.bottom_up_att(x)  # 10,1,11,63
        x_2 = self.top_down_att(x)  # 10,5,11,63
        x = x_1 * x_2  # 10,5,11,63
        x=self.avg_pooling(x)
        x=x.squeeze()
        probas = F.softmax(x , dim=1)
        return x , probas


# prepare the data set

custom_transform = transforms.Compose([
    # transforms.Resize((384, 384),
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.415 , 0.429 , 0.444] , [0.282 , 0.272 , 0.272])
])


class myDataset(datasets.ImageFolder) :
    '''custom dataset for loading data iamges'''

    def __getitem__(self , index) :

        path , target = self.imgs[index]
        # print(path,target)
        img = self.loader(path)
        if self.transform is not None :
            img = self.transform(img)
        if self.target_transform is not None :
            target = self.target_transform(target)

        return (img , target)

data_dir_test = '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/test2/test'
valid_dataset = myDataset(data_dir_test , transform=custom_transform)
valid_loader = DataLoader(valid_dataset , batch_size=batch_size , shuffle=False , num_workers=0)

with torch.no_grad() :
    model = torch.load("att_avg_nosoft.pkl")

for m in model.modules() :
    if isinstance(m , nn.Conv1d) :
        nn.init.normal(m.weight.data)
        nn.init.xavier_normal(m.weight.data)
        nn.init.kaiming_normal(m.weight.data)  # 卷积层参数初始化
        m.bias.data.fill_(0)
    elif isinstance(m , nn.Linear) :
        m.weight.data.normal_()  # 全连接层参数初始化

Cross_entropy = nn.CrossEntropyLoss()
ite=0
model = model.to(device)
print(model.parameters())
optimizer = torch.optim.SGD(model.parameters() , lr=0.01 , momentum=0.9)

start_time = time.time()

model.eval()

with torch.set_grad_enabled(False) :  # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model , num_classes=5 , batch_s=5 , data_loader=valid_loader)))
    print('======================================================')
    compute_mis(model,data_loader=valid_loader)
print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))





