from __future__ import print_function
from __future__ import division
from torchvision import datasets , transforms , models
import torch
import os
from PIL import Image
from torch.autograd import Variable

# 原图片加载路径
# data_dir = "H:\\TrainingData\\ICIP2019data\\testdata\\0"
data_dir = '/home/liang/disambigular/testdata_human/3'
# data_dir = '/home/liang/disambigular/5KappaExp/14/clo'
# 处理后图片保存路径
# data_dir_save = "/home/zhouying/下载/ICIP2019data/net"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_list0 = []
img_list1 = []
img_dataset = []
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224) ,
    transforms.RandomHorizontalFlip() ,
    transforms.ToTensor() ,
    transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
])
img_set = [os.path.join(data_dir , img) for img in os.listdir(data_dir)]
net = torch.load("vgg.pkl")


def datatransform(path) :
    img = Image.open(path)
    img = data_transforms(img)
    return img


def net_result() :
    net.eval()
    i = 0
    for path in img_set :
        image = datatransform(path)
        l = len(image)
        img_dataset.append((image , -1))
    dataloader = torch.utils.data.DataLoader(img_dataset , batch_size=1 , num_workers=1 , shuffle=False , )
    for image in dataloader :
        print(image)
        with torch.no_grad() :
            img , _ = image
            input = img.to(device)
            # print(img.size())

            output = net(input)
            # print(output)
            _ , pred = torch.max(output , 1)
            print(pred.item())
            if pred.item() == 0 or pred.item() == 1 or pred.item() == 2 :
                img_list0.append(img_set[i]);
                i = i + 1
            elif pred.item() == 3 or pred.item() == 4 or pred.item() == 5 :
                img_list1.append(img_set[i]);
                i = i + 1
    l1 = len(img_list0)
    l2 = len(img_list1)
    print(l1 , l2)
    print(l1 + l2)
    print('acc' , (l1 / (l2 + l1)))


if __name__ == '__main__' :
    net_result()

    '''
    for path in img_list0:
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save+'/'+'dis/'+name)
    for path in img_list1:
        name = path.split('/')[-1]
        single_img = Image.open(path)
        single_img.save(data_dir_save+'/'+'clo/'+name)
'''