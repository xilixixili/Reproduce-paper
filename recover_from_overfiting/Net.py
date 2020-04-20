import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module) :
    def __init__(self , features , num_classes=10 , init_weights=False , is_bias=True) :
        super(Net , self).__init__()
        self.features = features
        self.pooling = nn.AvgPool2d(kernel_size=7)  # kernal size ???
        self.classifier = nn.Linear(128 , num_classes , bias=is_bias)

        if init_weights :
            self._initialize_weights()

    def forward(self , x) :
        x = self.features(x)
        x = self.pooling(x)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++")
        # print("x before view:",x.shape)
        x = x.view(x.size(0) , -1)
        # print("x after view:" , x.shape)
        # x=self.pooling(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m , nn.Conv2d) :
                # m.weight = nn.Parameter(torch.Tensor(np.ones([m.out_channels , m.in_channels , m.kernel_size[0] , m.kernel_size[1]])))
                # m.weight.data = torch.Tensor(np.ones([m.out_channels , m.in_channels , m.kernel_size[0] , m.kernel_size[1]]))
                m.weight.data.fill_(1.0)
            elif isinstance(m , nn.BatchNorm2d) :
                m.weight.data.fill_(1.0)
                # m.bias.data.zero_()
                m.bias.data.fill_(1.0)
            elif isinstance(m , nn.Linear) :
                m.weight = nn.Parameter(torch.Tensor(np.ones([m.out_features , m.in_features])))


def make_layers(cfg , batch_norm=True , is_bias=True) :
    layers = []
    in_channels = 1  # mnist: 1   other :3
    for v in cfg :
        if v == 'M' :
            layers += [nn.MaxPool2d(kernel_size=2 , stride=2)]
        else :
            if v == 'D' :
                layers += [nn.Dropout2d(p=0.25)]
            else :
                conv2d = nn.Conv2d(in_channels , v , kernel_size=3 , padding=1 , bias=is_bias)
                if batch_norm :
                    layers += [conv2d , nn.BatchNorm2d(v) , nn.LeakyReLU(inplace=True)]
                else :
                    layers += [conv2d , nn.LeakyReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'Original' : [128 , 128 , 256 , 'M' , 'D' , 256 , 256 , 512 , 'M' , 'D' , 256 , 256 , 128]
}


def oral_net(**kwargs) :
    model = Net(make_layers(cfg['Original']) , **kwargs)
    return model


def oral_net_bias(**kwargs) :
    model = Net(make_layers(cfg['Original'] , is_bias=False) , init_weights=True , is_bias=False , **kwargs)
    return model


if __name__ == '__main__' :
    net = oral_net_bias()
    print(net)
    for param in net.parameters() :
        print('===========' , param , '\n' , param.size())
