

from __future__ import print_function
from __future__ import division
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import RepeatedKFold # 进行p次k折交叉验证
from sklearn import svm
from sklearn.externals import joblib
from torch.optim import lr_scheduler
from torch.autograd import Variable
import copy
import time
import numpy as np
from collections import Counter
from PIL import ImageFile
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='1'
ImageFile.LOAD_TRUNCATED_IMAGES = True
batch_size =256
num_epoch = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将来要用的全局图片地址的数据

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        #print(model.features)
        self.alex_layer = nn.Sequential(*list(model.features.children())[:13])
        self.Linear_layer = nn.Sequential(*list(model.classifier.children())[:6])
    def forward(self, x):
        x = self.alex_layer(x)
        #print("==========in forward x shape", x.shape)
        x = self.Linear_layer(x.view(x.size(0), -1))
        print('x:',x.size())
        return x

# 这里规定了图片的转换方式
custom_transform=transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip() ,
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

data_dir = '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/train_transpose+initial'

train_dataset=myDataset(data_dir , transform=custom_transform)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

# 参数保存
def save_param(model,name):
    torch.save(model,name+'.pkl')



def init_model(model_name,num_class,feature_extract,pretrain=True):
    #vgg_layer = None
    input_size = 0

    model  = models.alexnet(pretrained=pretrain)
    model = Net(model)
    print(model)
    set_param_requires_grad(model, feature_extract)

    input_size = 224

    return model, input_size

def set_param_requires_grad(model,feature_extract):
    if feature_extract:
        i=0
        for name,param in model.named_parameters():
            i=i+1
            #print(i,name, param.requires_grad)
            if i<3:
                param.requires_grad = False


def text_save(content, filename, mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def save_misclass(preds, labels, lens, mis0, mis1, mis2, mis3, mis4):
    for i in range(lens):

        # if preds[i] != labels[i]:
        if labels[i].item() == 0:
            mis0.append(preds[i].item())
            # print(mis0,preds[i].item(),labels[i].item())
        elif labels[i].item() == 1:
            mis1.append(preds[i].item())
        elif labels[i].item() == 2:
            mis2.append(preds[i].item())
        elif labels[i].item() == 3:
            mis3.append(preds[i].item())
        elif labels[i].item() == 4:
            mis4.append(preds[i].item())
            # ('mis'+labels[i].item()).append(preds[i].item())
            # print(name)


def get_count_by_counter(l):
    # t1 = time.time()
    count = Counter(l)  # 类型： <class 'collections.Counter'>
    # t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)  # 类型： <type 'dict'>
    return count_dict

# 主函数
if __name__ == "__main__":
    if torch.cuda.is_available():
        print('true')
    else:
        print('false')

    model_name = "alexnet"

    feature_extract = True
    model_ft, input_size = init_model(model_name, 5, feature_extract, True)
    model_ft = model_ft.to(device)
    print(model_ft)
    print(len(train_dataset))
    model_svm=svm.SVC(C=1.1)
#

    model_ft.eval()
    la=torch.randn(1)
    labels=torch.tensor(la).to(torch.int64)
    outputs=torch.rand(1,4096)
    for inputs , label in train_loader :
        inputs = Variable(inputs).to(device)
        #print("\ndtype of label:",label.dtype)
        #print("dtype of labels:" , labels.dtype)
        print("\nshape of labels:" , labels.shape)
        #print("\nshape of label:" , label.shape)
        labels = torch.cat([labels , label.detach()],dim=0)
        output=model_ft(inputs)
        #print("\ntype of outputs:",type(output))
        outputs=torch.cat([outputs,output.cpu().detach()],dim=0)
        #outputs.append(output.cpu().detach().numpy().tolist())
        #print("shape of outputs:",np.array(outputs).shape,'\n')
    outputs=np.array(outputs.cpu().numpy())
    row=outputs.shape[0]
    rw=np.arange(row-1)+1
    outputs=outputs[rw,:]

    labels = np.array(labels.cpu().detach().numpy())
    labels=labels[rw]
#    outputs=np.reshape(np.fromfile("outputs_no.bin"),(-1,4096))
#    labels=np.fromfile("labels_no.bin")
    print("\n before svm the  shape of outputs: ",outputs.shape)
    print(" before svm the  shape of labels: ",labels.shape,'\n')
    outputs.tofile("outputs_no.bin")
    labels.tofile("labels_no.bin")

    model_svm.fit(outputs,labels)
    joblib.dump(model_svm , "model_Svm_C1-1.m")