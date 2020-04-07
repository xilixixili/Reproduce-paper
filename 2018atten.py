"""
An Attention Pooling based Representation Learning Method for
Speech Emotion Recognition
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

# import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        features = features.to(device)
        targets = targets.to(device)
        logits , probas = model(features)
        _ , predicted_labels = torch.max(probas , 1)  # 返回最大值及索引
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


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
        #out2 = out2.permute(0 , 1 , 3 , 2)
        # print('out1.size:::',out1.size() , out2.size())
        x = torch.cat((out1 , out2) , 1)  # 10，16，136，549
        # print(x.size())
        x = self.cnn_layer(x)
        # print("x after cnn: " , x.size())  # 10，80，11，63

        x_1 = self.bottom_up_att(x)  # 10,1,11,63
        x_2 = self.top_down_att(x)  # 10,5,11,63
        # print('x1:',x_1.size(),'x2:',x_2.size())
        #x_1 = F.softmax(x_1 , dim=1)
        x = x_1 * x_2  # 10,5,11,63
        x=self.avg_pooling(x)
        x=x.squeeze()
        # print("logits:" , x.size())
        # x=torch.sum(x.sum(2) , 1)
        probas = F.softmax(x , dim=1)
        # print(probas.size())
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


data_dir = '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/train_transpose+initial'
data_dir_test = '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/test2/test'

# image_datasets = myDataset(data_dir , transform=custom_transform)

train_dataset = myDataset(data_dir , transform=custom_transform)

valid_dataset = myDataset(data_dir_test , transform=custom_transform)

# test_dataset=myDataset(data_dir_test,transform=custom_transform)

train_loader = DataLoader(train_dataset , batch_size=batch_size , shuffle=True , num_workers=0)

valid_loader = DataLoader(valid_dataset , batch_size=batch_size , shuffle=False , num_workers=0)
# print(len(params))
with torch.no_grad() :
    model = Net()

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

start_time = time.time()
valid_acc=0.0
valid_acc_t=0.0
for epoch in range(num_epochs) :
    # print("++++++++++++bs:",batch_size)
    model = model.train()  # model.train()
    batch_idx = 0
    # print('type of train_loader',type(train_loader))
    for inputs , labels in train_loader :
        batch_idx += 1
        # print('======================type: ',type(inputs))
        # print(inputs)

        inputs = Variable(inputs.to(device))
        # print('====shape :',inputs.shape)

        labels = Variable(labels.to(device))
        # print("labels",labels)
        ### FORWARD AND BACK PROP
        logits , probas = model(inputs)
        # print("logits and probas",logits.size(),probas.size())
        cost = F.cross_entropy(logits , labels)
        # print("labels and cost",labels.size(),cost)

        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        scheduler.step()


        ### LOGGING
        if not batch_idx % 20 :
            print('\n\nEpoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1 , num_epochs , batch_idx ,
                     len(train_loader) , cost))

    # model = model.eval()
    model.eval()
    with torch.set_grad_enabled(False) :  # save memory during inference
        valid_acc_t=compute_accuracy(model , num_classes=5 , batch_s=batch_size , data_loader=valid_loader)
        if valid_acc<valid_acc_t:
            save_param(model , name='att++')
            valid_acc=valid_acc_t
            ite+=1
            print('model have been saved %2d times !',(ite))

        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch + 1 , num_epochs ,
            compute_accuracy(model , num_classes=5 , batch_s=batch_size , data_loader=train_loader) ,
            valid_acc))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

#save_param(model , name='1018attention_pooling1')
# Evaluation
model.eval()

with torch.set_grad_enabled(False) :  # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model , num_classes=5 , batch_s=5 , data_loader=valid_loader)))







