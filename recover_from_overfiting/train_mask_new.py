"""
Recover from Overfitting to Label Noise : A Weight Pruning Perspective
train_mask
"""
import os
import time
from collections import Counter
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from ge_nosie_data import load_train_images , load_train_labels
from Net import oral_net
import re
from show_network import show_network
from Compute_acc import compute_accuracy,compute_mis
torch.backends.cudnn.enabled = False

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 1e-3
num_epochs = 200
bs = 128  # batch_size
pool_nb = 5  #####
test_bs = 32

# Architecture
num_classes = 10
data_dir='./data/mnist/'
dict_dir = 'flip045.pth' #sys05.pth
model_name='flip045_M_lr02.pth'



def save_model(model , name) :
    torch.save(model , name + '.pkl')


def save_model_dict(model , name) :
    torch.save(model.dict() , name + '.pth')


class Dataset_mnist(Dataset) :
    def __init__(self , images , labels) :
        self.images = torch.FloatTensor(images[: , np.newaxis , : , :])
        # print("============================================")
        # print("iamge size:",self.images.shape)
        self.labels = torch.LongTensor(labels)

    def __getitem__(self , idx) :
        image = self.images[idx]
        label = self.labels[idx]
        return image , label

    def __len__(self) :
        return len(self.labels)


# ==============================
train_images = load_train_images(data_dir+'train-images-idx3-ubyte')
train_labels = load_train_labels(data_dir+'train-labels-idx1-ubyte')
test_images = load_train_images(data_dir+'t10k-images-idx3-ubyte')
test_labels = load_train_labels(data_dir+'t10k-labels-idx1-ubyte')

# train_labels_noise = generate_noise_label(train_labels[-1000:] , 0.5 , False)
train_mnist = Dataset_mnist(train_images[-1000 :] , train_labels[-1000 :])
valid_mnist = Dataset_mnist(test_images[-1000 :] , test_labels[-1000 :])
# =========================
train_loader = DataLoader(train_mnist , batch_size=bs , shuffle=True , num_workers=4)
valid_loader = DataLoader(valid_mnist , batch_size=bs , shuffle=True , num_workers=4)
# 准备网络
model_origin = oral_net()

model_origin.load_state_dict(torch.load(dict_dir))
model_origin=model_origin.to(device)

model = oral_net(init_weights=True)
model=model.to(device)
model_origin_dict = copy.deepcopy(model_origin.state_dict())  # 保持每次 M*W 的时候 Weight 是原始训练好的
for m in model_origin_dict:
    print(m)
# model = model.to(device)
print("=============================================")

print(model)
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

start_time = time.time()
valid_acc = 0.0
valid_acc_t = 0.0
ite = 0

for epoch in range(num_epochs) :
    # print("++++++++++++bs:",batch_size)
    model = model.train()  # model.train()
    batch_idx = 0
    total = 0.0
    correct =0.0
    # print('type of train_loader',type(train_loader))
    for inputs , labels in train_loader :
        batch_idx += 1
        # print('======================type: ',type(inputs))
        # print(inputs)

        inputs = Variable(inputs.to(device))
        # print('====shape :',inputs.shape)
        # print('\tdtype of input:',inputs.dtype)
        labels = Variable(labels.to(device))
        # print("labels",labels)
        # 执行 step 函数的功能
        model_dict =copy.deepcopy( model.state_dict())
        model_dict_MW=copy.deepcopy(model.state_dict())
        for key in model_dict_MW :
            # model_dict_MW[key]=model_dict_MW[key].type(torch.DoubleTensor)
            # model_origin_dict[key] = model_origin_dict[key].type(torch.DoubleTensor)
            # print('model_dict_MW:',model_dict_MW[key].cuda())
            # print(model_dict_MW[key].dtype)
            # print('origin dict:',model_origin_dict[key].cuda())
            # print(model_origin_dict[key].dtype)

            #print(key,model_dict_MW[key].ge(0.5))
            #print('.ge :',model_dict_MW[key].ge(0.5).dtype,'\t',model_dict_MW[key].ge(0.5).to(device).device)
            temp=torch.mul(model_dict_MW[key].ge(0.5).to(device).type(torch.DoubleTensor), model_origin_dict[key].type(torch.DoubleTensor))
            # print('temp:',temp.dtype,'\t',temp.device)
            model_dict_MW[key] = temp
            if re.search('classifer',key):
                model_dict_MW[key]=model_origin_dict[key]
            if re.search('mean',key) or re.search('_var',key):

                model_dict_MW[key]=model_origin_dict[key]
                #print(key,model_origin_dict[key],model_dict_MW[key])
        model.load_state_dict(model_dict_MW)

        ### FORWARD AND BACK PROP
        logits = model(inputs)
        # print("logits and probas",logits.size(),probas.size())
        #print(logits)
        cost = F.cross_entropy(logits , labels)
#        print("labels and cost",cost)

        optimizer.zero_grad()
        model.load_state_dict(model_dict)
        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        # scheduler.step()

        # 对Mask 中的值 进行 m=max(0,min(1,m))
        for m in model.modules() :
            if isinstance(m , nn.Conv2d) :
                # print(np.nonzero(m.weight.data))
                #print(m,m.bias.data)
                m.weight.data = torch.clamp(m.weight.data , min=0 , max=1).to(device)
                # print(np.nonzero(m.weight.data))
                m.bias.data.fill_(1).to(device)
                #print(m,m.weight.data)
            elif isinstance(m , nn.BatchNorm2d) :  # running mean & running variance???
                continue
                # m.weight.data.fill_(1).to(device)
                # m.bias.data.fill_(1).to(device)
            elif isinstance(m , nn.Linear) :
                print(m.weight.data)
                m.weight.data = torch.clamp(m.weight.data , min=0 , max=1).to(device)
                m.bias.data.fill_(1).to(device)
                print(m.weight.data)
        model_dict=copy.deepcopy(model.state_dict())
        ### LOGGING
        if not batch_idx % 20 :
            print('\n\nEpoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1 , num_epochs , batch_idx ,
                     len(train_loader) , cost))
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        #print(predicted,labels.data)
#    print('c',correct,total)

    # model = model.eval()
    model.eval()
    model_origin.eval()
    model.load_state_dict(model_dict_MW)
    #for m in model.modules() :
           # if isinstance(m , nn.Conv2d) :
                # print(np.nonzero(m.weight.data))
              #  print(m,m.weight.data)
    with torch.set_grad_enabled(False) :  # save memory during inference
        valid_acc_t = compute_accuracy(model , num_classes=5 , batch_s=bs , data_loader=valid_loader)
        if valid_acc < valid_acc_t :
            save_model(model , name=model_name)
            valid_acc = valid_acc_t
            ite += 1
            print('model have been saved %03d times !' % (ite))

        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch + 1 , num_epochs ,
            compute_accuracy(model , num_classes=10 , batch_s=bs , data_loader=train_loader) ,
            valid_acc),end='\t')
        print(' | Origin: %.3f%% ' % (compute_accuracy(model_origin , num_classes=5 , batch_s=bs , data_loader=valid_loader) ))
        compute_mis(model,data_loader=train_loader)
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    model.load_state_dict(model_dict)

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
