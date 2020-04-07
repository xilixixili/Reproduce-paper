from __future__ import print_function , division
import torch
from torch.optim import lr_scheduler
import numpy as np
from collections import Counter
from torchvision import datasets , models , transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
import copy
import torch.nn as nn
from torch.autograd import Variable
from PIL import ImageFile
torch.backends.cudnn.enabled = False
# torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.ion()
data_transforms = {
    'test' : transforms.Compose([
        # transforms.Resize((512,512)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor() ,
        transforms.Normalize([0.415 , 0.429 , 0.444] , [0.282 , 0.272 , 0.272])
    ]) ,
}

n1=8
n2=8
num_classes=5

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




# data_dir = 'H:\\TrainingData\\ICIP2019data\\traindata'

data_dir = '/media/liang/新加卷/yinyifei_new_spectgram/yinyifei_mix/550_size/550_280_2048/test2'


def init_model(num_class) :
    model_ft = torch.load("att_avg_nosoft.pkl")
    return model_ft


image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir , x) ,
                                           data_transforms[x])
                  for x in ['test']}
dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x] , batch_size=16 ,
                                               shuffle=True , num_workers=4)
               for x in ['test']}
dataset_sizes = {x : len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes  # 这里面的class_name就是train目录下的数据集种类


# device是用来控制是否使用GPU的选项
def get_count_by_counter(l) :
    # t1 = time.time()
    count = Counter(l)  # 类型： <class 'collections.Counter'>
    # t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)  # 类型： <type 'dict'>
    return count_dict


def save_misclass(preds , labels , lens , mis0 , mis1 , mis2 , mis3 , mis4) :
    for i in range(lens) :
        # print(labels[i])

        # if preds[i] != labels[i]:
        if labels[i] == 0 :
            mis0.append(preds[i].item())
            # print(mis0,preds[i].item(),labels[i].item())
        elif labels[i] == 1 :
            mis1.append(preds[i].item())
        elif labels[i] == 2 :
            mis2.append(preds[i].item())
        elif labels[i] == 3 :
            mis3.append(preds[i].item())
        elif labels[i] == 4 :
            mis4.append(preds[i].item())
            # ('mis'+labels[i].item()).append(preds[i].item())
            # print(name)


def test_model(model) :
    since = time.time()
    ta = 0
    th = 0
    tn = 0
    tp = 0
    ts = 0
    mis0 = []
    mis1 = []
    mis2 = []
    mis3 = []
    mis4 = []
    for phase in ['test'] :
        model.eval()
        running_corrects = 0

        for inputs , labels in dataloaders[phase] :
            inputs = Variable(inputs).to(device)
            labels = Variable(labels.cuda())

            with torch.set_grad_enabled(phase == 'test') :
                logits ,probs = model(inputs)
                #print('logits:',logits.size(),'outputs:',outputs.size())

                _ , preds = torch.max(probs , 1)
                print(preds)
                running_corrects += torch.sum(preds == labels.data)
                # _, labs = torch.max(labels,1)
                # print(preds)
                '''
                if preds == torch.tensor([0]).to(device):
                  ta = ta + 1
                  list.append(0)
       #shutil.move(img[num][0],path0)
                if preds == torch.tensor([1]).to(device):
                  th = th + 1
                  list.append(1)
       #shutil.move(img[num][0],path1)
                if preds == torch.tensor([2]).to(device):
                  tn = tn + 1
                  list.append(2)
       #shutil.move(img[num][0],path2)
                if preds == torch.tensor([3]).to(device):
                  tp = tp + 1
                  list.append(3)
       #shutil.move(img[num][0],path3)
                if preds == torch.tensor([4]).to(device):
                  ts = ts + 1
                  list.append(4)
                '''
                save_misclass(preds , labels , 1 , mis0 , mis1 , mis2 , mis3 , mis4)
        print('mis0' , get_count_by_counter(mis0))
        print('mis1' , get_count_by_counter(mis1))
        print('mis2' , get_count_by_counter(mis2))
        print('mis3' , get_count_by_counter(mis3))
        print('mis4' , get_count_by_counter(mis4))

        # print(preds,labels.data)
        acc = running_corrects.double() / dataset_sizes[phase]
        print(dataset_sizes[phase])
        print('Acc: {:.4f}'.format(acc))

        print()
    time_elapsed = time.time() - since

    # print(list)
    return model


if __name__ == "__main__" :
    feature_extract = True
    model_ft = init_model(5)
    model_ft = model_ft.to(device)
    # print(model_ft)

    model_ft = test_model(model_ft)
#    save_param(model_ft,model_name)