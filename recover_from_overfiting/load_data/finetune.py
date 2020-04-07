import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets , models , transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()

data_transforms = transforms.Compose([
    transforms.Resize((224 , 224)) ,
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.415 , 0.429 , 0.444] , [0.282 , 0.272 , 0.272])
])

data_dir = '/home/liang/disambigular/traindata'
image_datasets = datasets.ImageFolder(data_dir , data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets ,
                                          batch_size=32 ,
                                          shuffle=True ,
                                          num_workers=4)
dataset_sizes = len(image_datasets)
print(dataset_sizes)
use_gpu = torch.cuda.is_available()
print(use_gpu)


def train_model(name , model , criterion , optimizer , num_epochs=300) :
    print(name)
    lr_epoch = np.logspace(-4 , -6 , num_epochs)

    for epoch in range(num_epochs) :
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch + 1 , num_epochs))
        since = time.time()
        print('training')
        model.train(True)
        # print lr
        for p_ft in optimizer.param_groups :
            p_ft['lr'] = lr_epoch[epoch]
            print('LR: {}'.format(p_ft['lr']))

        running_loss = 0.0
        for data in dataloaders :
            inputs , labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs , labels)
            print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / dataset_sizes
        with open(name + '_loss_finetune.txt' , 'a') as f :
            f.write(str(epoch_loss) + '\n')

        if epoch == 199 :
            torch.save(model , 'finetune_model_' + name + '_' +
                       str(epoch + 1) + '.pkl')
            break

        time_elapsed = time.time() - since
        print('This epoch training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 , time_elapsed % 60))


name = 'squeezenet'
squeezenet = torch.load('squeezenet_init.pkl')
for param in squeezenet.parameters() :
    param.requires_grad = False
squeezenet.classifier = nn.Sequential(nn.Dropout(p=0.5) ,
                                      nn.Conv2d(512 , 2 , kernel_size=(1 , 1) , stride=(1 , 1)) ,
                                      nn.ReLU(True) ,
                                      nn.AvgPool2d(13 , stride=1 , padding=0 , ceil_mode=False , count_include_pad=True)
                                      )
squeezenet = squeezenet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(squeezenet.classifier.parameters() , lr=0.0001 , momentum=0.9 , weight_decay=0.0005)
train_model(name , squeezenet , criterion , optimizer , num_epochs=300)

name = 'alexnet'
alexnet = torch.load('alexnet_init.pkl')
for param in alexnet.parameters() :
    param.requires_grad = False
alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5) ,
                                   nn.Linear(9216 , 4096) ,
                                   nn.ReLU(True) ,
                                   nn.Dropout(p=0.5) ,
                                   nn.Linear(4096 , 4096) ,
                                   nn.ReLU(True) ,
                                   nn.Linear(4096 , 2) ,
                                   nn.Softmax())
alexnet = alexnet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.classifier.parameters() , lr=0.0001 , momentum=0.9 , weight_decay=0.0005)
train_model(name , alexnet , criterion , optimizer , num_epochs=300)

name = 'resnet'
resnet = torch.load('resnet_init.pkl')
for param in resnet.parameters() :
    param.requires_grad = False
resnet.fc = nn.Sequential(nn.Linear(512 , 2) ,
                          nn.Softmax())
resnet = resnet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.fc.parameters() , lr=0.0001 , momentum=0.9 , weight_decay=0.0005)
train_model(name , resnet , criterion , optimizer , num_epochs=300)
'''
name = 'cfcnn'
cfcnn = torch.load('cfcnn.pkl')
for param in cfcnn.parameters():
    param.requires_grad = False
cfcnn.fc8 = nn.Linear(1024, 2)
cfcnn = cfcnn.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cfcnn.fc8.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
train_model(name, cfcnn, criterion, optimizer, num_epochs=300)
'''

name = 'vgg'
vgg = torch.load('vgg_init.pkl')
for param in vgg.parameters() :
    param.requires_grad = False
vgg.classifier = nn.Sequential(nn.Linear(4096 , 1024) ,
                               nn.ReLU(True) ,
                               nn.Dropout(p=0.5) ,
                               nn.Linear(1024 , 2) ,
                               nn.Softmax())
vgg = vgg.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.classifier.parameters() , lr=0.0001 , momentum=0.9 , weight_decay=0.0005)
train_model(name , vgg , criterion , optimizer , num_epochs=300)
