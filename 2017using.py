"""
USING REGIONAL SALIENCY FOR SPEECH EMOTION RECOGNITION
"""
import os
import time
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES']='1'

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 1e-4
num_epochs = 15
batch_size = 128
pool_nb=500#####


# Architecture
num_classes = 7


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    print('size of set to compute accuracy:\nlen::',len(data_loader))
    for inputs, labels in data_loader:
        inputs = Variable(inputs.cuda().to(device))
        labels = Variable(labels.cuda().to(device))
        logits, probas = model(inputs)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += labels.size(0)
        correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float()/num_examples * 100

def save_param(model , name) :
    torch.save(model , name + '2017.pkl')

def get_count_by_counter(l) :
    # t1 = time.time()
    count = Counter(l)  # 类型： <class 'collections.Counter'>
    # t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)  # 类型： <type 'dict'>
    return count_dict


class Net(nn.Module):
    def __init__(self,filter_nb):
        self.filter_nb=filter_nb
        super(Net,self).__init__()

        self.conv_8=nn.Conv2d(in_channels=3,out_channels=self.filter_nb,kernel_size=(8,550),stride=1,padding=(4,0))
        self.conv_16=nn.Conv2d(in_channels=3,out_channels=self.filter_nb,kernel_size=(16,550),stride=1,padding=(8,0))
        self.conv_32=nn.Conv2d(in_channels=3,out_channels=self.filter_nb,kernel_size=(32,550),stride=1,padding=(16,0))
        self.conv_64=nn.Conv2d(in_channels=3,out_channels=self.filter_nb,kernel_size=(64,550),stride=1,padding=(32,0))
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier=nn.Sequential(
            nn.Linear(in_features=self.filter_nb*4 , out_features=1024),
            nn.Linear(in_features=1024 , out_features=1024),
            nn.Linear(in_features=1024 , out_features=5)
        )

    def _forward(self,x):
        conv8=self.conv_8(x)
        conv16=self.conv_16(x)
        conv32=self.conv_32(x)
        conv64=self.conv_64(x)
        outputs=[conv8,conv16,conv32,conv64]
        return torch.cat(outputs, 1)
    def forward(self,x):
        out=self._forward(x)
        out=F.relu(out)
        out=self.pool(out)
        print('==========output after conv and pool:',out.shape)
        logits=self.classifier(out.view(out.size(0),-1))
        probas=F.softmax(logits,dim=1)
        return logits,probas

# prepare the data set
custom_transform=transforms.Compose([
    # transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.415 , 0.429 , 0.444] , [0.282 , 0.272 , 0.272])
])


# class Net(nn.Module):
#     def __init__(self,filter_nb):
#         self.filter_nb=filter_nb
#         super(Net,self).__init__()
#
#         self.conv1=nn.Conv2d(in_channels=3,out_channels=self.filter_nb,kernel_size=(5,550),stride=3,padding=(2,0))
#         self.pool=nn.AdaptiveAvgPool2d(output_size=1)
#         self.classifier=nn.Sequential(
#             nn.Linear(in_features=self.filter_nb , out_features=1024),
#             nn.Linear(in_features=1024 , out_features=1024),
#             nn.Linear(in_features=1024 , out_features=5)
#         )
#
#     def forward(self,x):
#         out=self.conv1(x)
#         out=F.relu(out)
#         out=self.pool(out)
#         #print('==========output after conv and pool:',out.shape)
#         logits=self.classifier(out.view(-1,self.filter_nb))
#         probas=F.softmax(logits,dim=1)
#         return logits,probas
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


data_dir =  '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/train_transpose+initial'
data_dir_test='/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/test2'



#image_datasets = myDataset(data_dir , transform=custom_transform)

train_dataset=myDataset(data_dir , transform=custom_transform)

valid_dataset=myDataset(data_dir_test,transform=custom_transform)

#test_dataset=myDataset(data_dir_test,transform=custom_transform)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

#test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
# print('======================================')
# print(len(train_dataset))
# print('======================================')
# print(len(train_loader))
# print('type train_loader:',type(train_loader))
#torch.manual_seed(random_seed)
model = Net(filter_nb=384)

for m in model.modules():
    if isinstance(m,nn.Conv1d):
        #nn.init.normal(m.weight.data)
        #nn.init.xavier_normal(m.weight.data)
        nn.init.kaiming_normal(m.weight.data)#卷积层参数初始化
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        m.weight.data.normal_()#全连接层参数初始化


model = model.to(device)
print(model.parameters())
optimizer=torch.optim.RMSprop(model.parameters(),lr=learning_rate)


start_time = time.time()
for epoch in range(num_epochs) :
    model = model.train()  #model.train()
    batch_idx=0
    print('type of train_loader',type(train_loader))
    for inputs , labels in train_loader :
        batch_idx+=1
        #print('======================type: ',type(inputs))
        #print(inputs)

        inputs = Variable(inputs.cuda().to(device))
        #print('====shape :',inputs.shape)

        labels = Variable(labels.cuda().to(device))

        ### FORWARD AND BACK PROP
        logits , probas = model(inputs)
        cost = F.cross_entropy(logits , labels)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50 :
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1 , num_epochs , batch_idx ,
                     len(train_loader) , cost))

    #model = model.eval()
    model.eval()
    #with torch.set_grad_enabled(False) :  # save memory during inference
        #print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            #epoch + 1 , num_epochs ,
            #compute_accuracy(model , train_loader) ,
            #compute_accuracy(model , valid_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

save_param(model , name='2017new_')
# Evaluation
model.eval()

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, valid_loader)))