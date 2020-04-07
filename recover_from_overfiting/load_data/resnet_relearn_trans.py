from __future__ import print_function
from __future__ import division
from torchvision import datasets , transforms , models
import torch
import os
from PIL import Image
from sklearn.model_selection import RepeatedKFold  # 进行p次k折交叉验证
from torch.optim import lr_scheduler
from torch.autograd import Variable
import copy
import time
import numpy as np
import math

batch_size = 8
num_epoches = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将来要用的全局图片地址的数据
img_data_path = []  # 这个列表里面有全部数据集的路径索引，加上每张图片所对应的标签
# data_dir 指的是机器分好的类文件夹的路径

data_dir0 = '/home/liang/disambigular/net_vote_unlabel/0'
data_dir1 = '/home/liang/disambigular/net_vote_unlabel/1'
data_dir2 = '/home/liang/disambigular/net_vote_unlabel/2'
data_dir3 = '/home/liang/disambigular/net_vote_unlabel/3'
data_dir4 = '/home/liang/disambigular/net_vote_unlabel/4'
data_dir5 = '/home/liang/disambigular/net_vote_unlabel/5'
'''
data_dir0 = '/home/zhouying/testdata/0'
data_dir1 = '/home/zhouying/testdata/1'
data_dir2 = '/home/zhouying/testdata/2'
data_dir3 = '/home/zhouying/testdata/3'
data_dir4 = '/home/zhouying/testdata/4'
data_dir5 = '/home/zhouying/testdata/5'
'''
# 这里规定了图片的转换方式
d_t = transforms.Compose([
    transforms.RandomResizedCrop(224) ,
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])
d_t_val = transforms.Compose([
    transforms.Resize(256) ,
    transforms.CenterCrop(224) ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])
# 读图片的文件索引，建立索引列表
img_set0 = [os.path.join(data_dir0 , img) for img in os.listdir(data_dir0)]
img_set1 = [os.path.join(data_dir1 , img) for img in os.listdir(data_dir1)]
img_set2 = [os.path.join(data_dir2 , img) for img in os.listdir(data_dir2)]
img_set3 = [os.path.join(data_dir3 , img) for img in os.listdir(data_dir3)]
img_set4 = [os.path.join(data_dir4 , img) for img in os.listdir(data_dir4)]
img_set5 = [os.path.join(data_dir5 , img) for img in os.listdir(data_dir5)]


# 图像转换函数
# 这里是图像转换函数，上面的是图像转换方式
def datatransform(path) :
    img = Image.open(path)
    img = d_t(img)
    return img


def datatransform_val(path) :
    img = Image.open(path)
    img = d_t_val(img)
    return img


# 建立训练数据集
'''
def build_img_set():
    for path in img_set0:
        img_data_path.append((path,torch.Tensor([1]))) # 在这里直接转成tensor格式，下面就不用转换格式了
    for path in img_set1:
        img_data_path.append((path,torch.Tensor([0.8])))
    for path in img_set2:
        img_data_path.append((path,torch.Tensor([0.6])))
    for path in img_set3:
        img_data_path.append((path,torch.Tensor([0.4])))
    for path in img_set4:
        img_data_path.append((path,torch.Tensor([0.2])))
    for path in img_set5:
        img_data_path.append((path,torch.Tensor([0])))
'''


def build_img_set() :
    for path in img_set0 :
        img_data_path.append((path , torch.Tensor([1 , 0 , 0 , 0 , 0 , 0])))  # 在这里直接转成tensor格式，下面就不用转换格式了
    for path in img_set1 :
        img_data_path.append((path , torch.Tensor([0 , 1 , 0 , 0 , 0 , 0])))
    for path in img_set2 :
        img_data_path.append((path , torch.Tensor([0 , 0 , 1 , 0 , 0 , 0])))
    for path in img_set3 :
        img_data_path.append((path , torch.Tensor([0 , 0 , 0 , 1 , 0 , 0])))
    for path in img_set4 :
        img_data_path.append((path , torch.Tensor([0 , 0 , 0 , 0 , 1 , 0])))
    for path in img_set5 :
        img_data_path.append((path , torch.Tensor([0 , 0 , 0 , 0 , 0 , 1])))


# 数据加载器,返回的是转换好的traindata和valdata
def dataloader(img_data) :
    dataloader = torch.utils.data.DataLoader(
        img_data , batch_size=batch_size , num_workers=4 , shuffle=True)
    return dataloader


def set_param_requires_grad(model) :
    for param in model.parameters() :
        param.requires_grad = True


def load_net(model_name) :
    if (model_name == 'resnet') :
        net = torch.load("resnet2.pkl")
        # print(net)
        set_param_requires_grad(net)
        net.relu = torch.nn.LeakyReLU()
        net.layer1[0].relu = torch.nn.LeakyReLU()
        net.layer1[1].relu = torch.nn.LeakyReLU()
        net.layer2[0].relu = torch.nn.LeakyReLU()
        net.layer2[1].relu = torch.nn.LeakyReLU()
        net.layer3[0].relu = torch.nn.LeakyReLU()
        net.layer3[1].relu = torch.nn.LeakyReLU()
        net.layer4[0].relu = torch.nn.LeakyReLU()
        net.layer4[1].relu = torch.nn.LeakyReLU()
        net.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5) ,
            torch.nn.Linear(512 , 6 , bias=True)
        )
        # net.fc[1].weight.requires_grad = True
        # net.fc[1].bias.requires_grad = True
    return net


def trans_label(preds) :
    predict = torch.empty(0)

    i = 0
    for pred in preds :
        if pred == 0 :
            predict = torch.cat((predict , torch.tensor([[1.]])) , dim=0)
        elif pred == 1 :
            predict = torch.cat((predict , torch.tensor([[0.8]])) , dim=0)
        elif pred == 2 :
            predict = torch.cat((predict , torch.tensor([[0.6]])) , dim=0)
        elif pred == 3 :
            predict = torch.cat((predict , torch.tensor([[0.4]])) , dim=0)
        elif pred == 4 :
            predict = torch.cat((predict , torch.tensor([[0.2]])) , dim=0)
        elif pred == 5 :
            predict = torch.cat((predict , torch.tensor([[0.]])) , dim=0)
        i = i + 1
    return predict


def compare(predict) :
    pre = torch.empty(0)
    for pred in predict :
        if pred == torch.tensor([[1.]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[1.]])) , dim=0)
        elif pred == torch.tensor([[0.8]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[1.]])) , dim=0)
        elif pred == torch.tensor([[0.6]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[1.]])) , dim=0)
        elif pred == torch.tensor([[0.4]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[0.]])) , dim=0)
        elif pred == torch.tensor([[0.2]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[0.]])) , dim=0)
        elif pred == torch.tensor([[0.]]).cuda() :
            pre = torch.cat((pre , torch.tensor([[0.]])) , dim=0)
    return pre


def compare_label(label) :
    l = torch.empty(0)
    for lab in label :
        lab = torch.nonzero(lab , out=None)
        # print(lab)
        if lab == 0 :
            l = torch.cat((l , torch.tensor([[1.]])) , dim=0)
        elif lab == 1 :
            l = torch.cat((l , torch.tensor([[1.]])) , dim=0)
        elif lab == 2 :
            l = torch.cat((l , torch.tensor([[1.]])) , dim=0)
        elif lab == 3 :
            l = torch.cat((l , torch.tensor([[0.]])) , dim=0)
        elif lab == 4 :
            l = torch.cat((l , torch.tensor([[0.]])) , dim=0)
        elif lab == 5 :
            l = torch.cat((l , torch.tensor([[0.]])) , dim=0)
    # print('l',l)
    return l


# 参数保存
def save_param(model , name) :
    torch.save(model , name + '_huber_re3.pkl')


def train_model(model , criterion , optimizer , scheduler , dataloader_train , dataloader_val , size ,
                num_epoch=num_epoches) :
    since = time.time()
    # least_loss = 100
    dataloader = {"train" : dataloader_train , "val" : dataloader_val}
    for epoch in range(num_epoch) :
        print('Epoch {}/{}'.format(epoch , num_epoch - 1))
        print('-' * 10)
        for phase in ['train' , 'val'] :
            if phase == 'train' :
                scheduler.step()
                model.train()
            elif phase == 'val' :
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for input , label in dataloader[phase] :
                # print('input',input.size())
                # print('label',label)
                input = Variable(input.cuda())
                label = Variable(label.cuda())
                # print('label',label.size())
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train') :
                    output = model(input)
                    output = torch.nn.functional.softmax(output)
                    _ , preds = torch.max(output , 1)
                    predict = trans_label(preds)
                    # print('presize',predict.size())
                    # print('pre',predict)
                    predict = Variable(predict.cuda())
                    pre = compare(predict)

                    lab = compare_label(label)
                    # print('output',output)
                    # print('predict',predict)
                    loss = criterion(output , label)
                    # loss.requires_grad=True
                    # print('loss',loss)
                    if (phase == 'train') :
                        loss.backward()
                        optimizer.step()
                running_corrects += torch.sum(pre == lab).tolist()
                running_loss += loss.item() * input.size(0)
            print(running_corrects)
            epoch_loss = running_loss / size[phase]
            epoch_loss = np.abs(epoch_loss)
            epoch_acc = running_corrects / size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase , epoch_loss , epoch_acc))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    # print('least val loss: {:4f}'.format(least_loss))
    return model


# 主函数
if __name__ == "__main__" :
    model_name = 'resnet'
    # save_name='resnet_L1'

    build_img_set()
    net = load_net(model_name)  # 在这里加载这次运行需要的网络
    print(net)

    net.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5) ,
        torch.nn.Linear(512 , 6)
    )
    net.fc[1].weight.requires_grad = True
    net = net.to(device)
    print('new' , net)
    best_model = copy.deepcopy(net)
    # 这里写一个更新数据的参数列表的过程
    params_to_update = []
    for name , param in net.named_parameters() :
        if param.requires_grad == True :
            params_to_update.append(param)
    # 这里是网络加载必须的三个参数
    criterion = torch.nn.SmoothL1Loss(size_average=True)
    optimizer = torch.optim.Adam(params_to_update , lr=1e-3 , weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer , step_size=20 , gamma=0.1)

    print(len(img_data_path))
    kf = RepeatedKFold(n_splits=4 , n_repeats=1 , random_state=0)  # 这里是4折，循环一次的交叉验证
    for train_index , val_index in kf.split(img_data_path) :
        train_set = [];
        val_set = []
        for i in train_index :
            path , label = img_data_path[i]
            img = datatransform(path)
            train_set.append((img , label))
        # 到这里为止，就分出了本轮循环下的训练数据集以及每张图片所对应的标签
        for i in val_index :
            path , label = img_data_path[i]
            img = datatransform_val(path)
            val_set.append((img , label))
        len_train = len(train_set);
        len_val = len(val_set)
        size = {'train' : len_train , 'val' : len_val}
        print(len_train , len_val)  # 查看训练集和验证集的数量
        # 到这里为止，就分出了本轮循环下的验证数据集以及每张图片所对应的标签
        dataloader_train = dataloader(train_set)  # 这个是读训练数据集的数据加载器
        dataloader_val = dataloader(val_set)  # 这个是读验证数据集的数据加载器
        best_model = train_model(net , criterion , optimizer , exp_lr_scheduler , dataloader_train , dataloader_val ,
                                 size)
    save_param(best_model , model_name)
