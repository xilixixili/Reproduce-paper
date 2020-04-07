import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 1e-4
num_epochs = 10
batch_size = 128
pool_nb=5#####


# Architecture
num_classes = 5


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

class Net(nn.module):
    def __init__(self,num_class):
        super(Net,self).__init__()

        self.conv1=nn.Conv1d(in_channels=1,out_channels=1,kernel_size=filter_length,stride=filter_stride,padding=1)
        self.pool=nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier=nn.Sequential(
            nn.Linear(in_features=pool_nb , out_features=1024),
            nn.Linear(in_features=1024 , out_features=1024),
            nn.Linear(in_features=1024 , out_features=num_class)
        )

    def forward(self,x):
        out=self.conv1(x)
        out=F.relu(out)
        out=self.pool(out)
        logits=self.classifier(out.view(-1,pool_nb))
        probas=F.softmax(logits,dim=1)
        return logits,probas

class myDataset(Dataset):
    '''custom dataset for loading data iamges'''
    def __init__(self,path):
        self.img_dir=path

    def __gititem__(self,index):

        return img,label

    def __len__(self):
        return self




custom_transform=transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])
train_dataset=myDataset(transform=custom_transform)

valid_dataset=myDataset(transform=custom_transform)

test_dataset=myDataset(transform=custom_transform)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

valid_loader=DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=4)




#torch.manual_seed(random_seed)
model = Net(num_classes=num_classes)

model = model.to(device)
optimizer=torch.optim.RSMprop(model.parameters(),lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs) :
    model = model.train()  #model.train()
    for batch_idx , (features , targets) in enumerate(train_loader) :

        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits , probas = model(features)
        cost = F.cross_entropy(logits , targets)
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
    with torch.set_grad_enabled(False) :  # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch + 1 , num_epochs ,
            compute_accuracy(model , train_loader) ,
            compute_accuracy(model , valid_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))


# Evaluation
model.eval()

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))