""""
LEARNING DISCRIMINATIVE FEATURES FROM SPECTROGRAMS USING CENTER
LOSS FOR SPEECH EMOTION RECOGNITION
"""
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

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cudnn.enabled = False

os.environ['CUDA_VISIBLE_DEVICES']='0'
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 3e-4
num_epochs = 10
batch_size = 32
pool_nb=5#####
test_bs=32

# Architecture
num_classes = 5


def compute_accuracy(model,batch_s,num_classes, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas, fc = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def save_param(model , name) :
    torch.save(model , name + '2019.pkl')

def get_count_by_counter(l) :
    # t1 = time.time()
    count = Counter(l)  # 类型： <class 'collections.Counter'>
    # t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)  # 类型： <type 'dict'>
    return count_dict


class CenterLoss(nn.Module) :
    def __init__(self , cls_num , featur_num) :
        super().__init__()

        self.cls_num = cls_num
        self.featur_num = featur_num
        self.center = nn.Parameter(torch.rand(cls_num , featur_num))

    def forward(self , xs , ys) :  # xs=feature,ys=target
        # xs= F.normalize(xs)
        self.center_exp = self.center.index_select(dim=0 , index=ys.long())
        count = torch.histc(ys , bins=self.cls_num , min=0 , max=self.cls_num - 1)
        self.count_dis = count.index_select(dim=0 , index=ys.long()) + 1
        loss = torch.sum(torch.sum((xs - self.center_exp) ** 2 , dim=1) / 2.0 / self.count_dis.float())

        return loss


class Net(nn.Module):
    def __init__(self,num_classes,batch_size):
        self.bs=batch_size
        super(Net,self).__init__()

        # self.conv1=nn.Conv2d(in_channels=3,out_channels=48,kernel_size=(7,7),stride=(2,2))
        # self.conv2=nn.Conv2d(in_channels=48,out_channels=64,kernel_size=(3,3),stride=(1,1))
        # self.pool3=nn.MaxPool2d(kernel_size=(2,2),stride=(2.2))
        # self.conv5=nn.Conv2d(in_channels=64,out_channels=80,kernel_size=(3,3),stride=(1,1))
        # self.pool6=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        # self.conv7=nn.Conv2d(in_channels=80,out_channels=96,kernel_size=(3,3),stride=(1,1))
        # self.pool8=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.cnn_layer=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=(7,7),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.Conv2d(in_channels=64,out_channels=80,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU() ,
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.Conv2d(in_channels=80,out_channels=96,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        #size of output is Lt * Lf * 96
        # self.rnn_layer=nn.Sequential(
        #     nn.RNN(input_size=98304,hidden_size=128,num_layers=1,bidirectional=True),
        #
        #     nn.PReLU()
        # )

        self.rnn=nn.RNN(input_size=98304 , hidden_size=128 , num_layers=1 , bidirectional=True,batch_first=True)

        self.fc1=nn.Linear(in_features=256,out_features=64)
        self.fc2 = nn.Linear(in_features=64 , out_features=num_classes)


    def forward(self,x):
        out=self.cnn_layer(x)
        #print("+++++++++++++ shape of cnn output:",out.shape)
        #out=self.rnn_layer(out.view(bs,1,))#bsxuyao
        out,h=self.rnn(out.view(out.size(0),1,98304))
        #print('after rnn :=====out===',out.shape)
        out=F.relu(out.view(out.size(0),256)) # 32*256
        out_fc=self.fc1(out)
        logits=self.fc2(out_fc)
        probas=F.softmax(logits,dim=1)
        return logits,probas,out_fc


# prepare the data set
custom_transform=transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.415 , 0.429 , 0.444] , [0.282 , 0.272 , 0.272])
])


class myDataset(datasets.ImageFolder):
    '''custom dataset for loading data iamges'''
    def __getitem__(self, index):

        path, target = self.imgs[index]
        #print(path,target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)


data_dir =  '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_280_2048+550_308_1024/mix_train_transpose_and_initial'
data_dir_test='/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/test2'



#image_datasets = myDataset(data_dir , transform=custom_transform)

train_dataset=myDataset(data_dir , transform=custom_transform)

valid_dataset=myDataset(data_dir_test,transform=custom_transform)

#test_dataset=myDataset(data_dir_test,transform=custom_transform)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

#test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
#print('======================================')
# print(len(train_dataset))
# print('======================================')
# print(len(train_loader))
# print('type train_loader:',type(train_loader))
#torch.manual_seed(random_seed)
model = Net(num_classes=5,batch_size=batch_size)

for m in model.modules():
    if isinstance(m,nn.Conv1d):
        nn.init.normal(m.weight.data)
        nn.init.xavier_normal(m.weight.data)
        nn.init.kaiming_normal(m.weight.data)#卷积层参数初始化
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        m.weight.data.normal_()#全连接层参数初始化


centerloss=CenterLoss(5,64).to(device)


model = model.to(device)
print(model.parameters())
optmizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optmizer, 20, gamma=0.8)
optmizercenter = torch.optim.SGD(centerloss.parameters(), lr=0.5)


start_time = time.time()
for epoch in range(num_epochs) :
    #print("++++++++++++bs:",batch_size)
    model = model.train()  #model.train()
    batch_idx=0
    #print('type of train_loader',type(train_loader))
    for inputs , labels in train_loader :
        batch_idx+=1
        #print('======================type: ',type(inputs))
        #print(inputs)

        inputs = Variable(inputs.cuda().to(device))
        #print('====shape :',inputs.shape)

        labels = Variable(labels.cuda().to(device))

        ### FORWARD AND BACK PROP
        logits , probas,out_fc = model(inputs)
        costentry = F.cross_entropy(logits , labels)

        cost=costentry+centerloss(out_fc,labels)

        optmizer.zero_grad()
        optmizercenter.zero_grad()
        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optmizer.step()
        optmizercenter.step()

        ### LOGGING
        if not batch_idx % 50 :
            print('\n\nEpoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1 , num_epochs , batch_idx ,
                     len(train_loader) , cost))

    #model = model.eval()
    model.eval()
    with torch.set_grad_enabled(False) :  # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch + 1 , num_epochs ,
            compute_accuracy(model , num_classes=5,batch_s=32,data_loader=train_loader) ,
            compute_accuracy(model , num_classes=5,batch_s=test_bs,data_loader=valid_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

save_param(model , name='2019')
# Evaluation
model.eval()

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, num_classes=5,batch_s=test_bs,data_loader=valid_loader)))
