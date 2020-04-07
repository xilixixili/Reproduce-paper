from __future__ import print_function
from __future__ import division
from torchvision import datasets , transforms , models
import torch
import os
from PIL import Image
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 原图片加载路径
# 把数据集拆成6份，分着跑，要不然会内存爆栈
data_dir = '/home/liang/disambigular/unlabeledD2/unlabeledD2'
# data_dir = '/home/liang/disambigular/unlabeled/2'
# data_dir = '/home/liang/disambigular/unlabeled/3'
# data_dir = '/home/liang/disambigular/unlabeled/4'
# data_dir = '/home/liang/disambigular/unlabeled/5'
# data_dir = '/home/liang/disambigular/unlabeled/6'
# 处理后图片保存路径

data_dir_save0 = '/home/liang/disambigular/net_vote_unlabel/0'
data_dir_save1 = '/home/liang/disambigular/net_vote_unlabel/1'
data_dir_save2 = '/home/liang/disambigular/net_vote_unlabel/2'
data_dir_save3 = '/home/liang/disambigular/net_vote_unlabel/3'
data_dir_save4 = '/home/liang/disambigular/net_vote_unlabel/4'
data_dir_save5 = '/home/liang/disambigular/net_vote_unlabel/5'
'''
data_dir_save0 = '/home/liang/disambigular/net_vote_testdata/0'
data_dir_save1 = '/home/liang/disambigular/net_vote_testdata/1'
data_dir_save2 = '/home/liang/disambigular/net_vote_testdata/2'
data_dir_save3 = '/home/liang/disambigular/net_vote_testdata/3'
data_dir_save4 = '/home/liang/disambigular/net_vote_testdata/4'
data_dir_save5 = '/home/liang/disambigular/net_vote_testdata/5'
'''
# 用列表存储分好类的图片的地址
img_list0 = []
img_list1 = []
img_list2 = []
img_list3 = []
img_list4 = []
img_list5 = []
img_dataset = []
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224) ,
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])


def init_img_set() :
    img_set = [os.path.join(data_dir , img) for img in os.listdir(data_dir)]
    return img_set


# 加载网络模型
def init_model() :
    resnet = torch.load("resnet2.pkl")
    squeezenet = torch.load("squeezenet.pkl")
    alexnet = torch.load("alexnet.pkl")
    vgg = torch.load("vgg.pkl")
    densenet = torch.load("densenet2.pkl")
    return resnet , squeezenet , alexnet , vgg , densenet


def datatransform(path) :
    img = Image.open(path)
    img = data_transforms(img)
    return img


def net_result() :
    img_set = init_img_set()
    resnet , squeezenet , alexnet , vgg , densenet = init_model()
    i = 0
    for path in img_set :
        image = datatransform(path)
        img_dataset.append((image , -1))
    print("img_finish")
    dataloader = torch.utils.data.DataLoader(img_dataset , batch_size=1 , num_workers=1 , shuffle=False , )
    print("yes")
    for image in dataloader :
        with torch.no_grad() :
            sum = -1
            img , _ = image
            img = img.to(device)
            print(img.size())
            if img.size() == torch.Size([1 , 3 , 224 , 224]) :
                output_res = resnet(img)
                print(output_res)
                output_vgg = vgg(img)
                output_squ = squeezenet(img)
                output_alexnet = alexnet(img)
                output_densenet = densenet(img)

                _ , pred_res = torch.max(output_res , 1)
                print("res:" , pred_res.item())
                _ , pred_vgg = torch.max(output_vgg , 1)
                print("vgg:" , pred_vgg.item())
                _ , pred_squ = torch.max(output_squ , 1)
                print("squ:" , pred_squ.item())
                _ , pred_alexnet = torch.max(output_alexnet , 1)
                print("alexnet:" , pred_alexnet.item())
                _ , pred_densenet = torch.max(output_densenet , 1)
                print("densenet:" , pred_densenet.item())
                sum = pred_res.item() + pred_vgg.item() + pred_squ.item() + pred_alexnet.item() + pred_densenet.item()
                sum = 5 - sum
                print("sum:" , sum)
                print()
                if sum == 0 :
                    img_list0.append(img_set[i])
                    i = i + 1
                elif sum == 1 :
                    img_list1.append(img_set[i])
                    i = i + 1
                elif sum == 2 :
                    img_list2.append(img_set[i])
                    i = i + 1
                elif sum == 3 :
                    img_list3.append(img_set[i])
                    i = i + 1
                elif sum == 4 :
                    img_list4.append(img_set[i])
                    i = i + 1
                elif sum == 5 :
                    img_list5.append(img_set[i])
                    i = i + 1
        print(i)


if __name__ == '__main__' :
    net_result()
    for path in img_list5 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save5 + '/' + name)
    print('img_list5')
    for path in img_list1 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save1 + '/' + name)
    print('img_list1')
    for path in img_list2 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save2 + '/' + name)
    print('img_list2')
    for path in img_list3 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save3 + '/' + name)
    print('img_list3')
    for path in img_list4 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save4 + '/' + name)
    print('img_list4')
    for path in img_list0 :
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save0 + '/' + name)
    print('img_list0')