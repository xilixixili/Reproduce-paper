import torch
import torch.nn as nn
import math
import torch.optim as optim

############# 模型参数初始化 ###############
# ————————————————— 利用model.apply(weights_init)实现初始化
def weights_init(m) :
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0 , math.sqrt(2. / n))
        if m.bias is not None :
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1 :
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1 :
        n = m.weight.size(1)
        m.weight.data.normal_(0 , 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


# ————————————————— 直接放在__init__构造函数中实现初始化
for m in self.modules() :
    if isinstance(m , nn.Conv2d) :
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0 , math.sqrt(2. / n))
        if m.bias is not None :
            m.bias.data.zero_()
    elif isinstance(m , nn.BatchNorm2d) :
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m , nn.BatchNorm1d) :
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m , nn.Linear) :
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None :
            m.bias.data.zero_()

# —————————————————
self.weight = Parameter(torch.Tensor(out_features , in_features))
self.bias = Parameter(torch.FloatTensor(out_features))
nn.init.xavier_uniform_(self.weight)
nn.init.zero_(self.bias)
nn.init.constant_(m , initm)
# nn.init.kaiming_uniform_()
# self.weight.data.normal_(std=0.001)


##### 模型参数分组weight_decay ######
def separate_bn_prelu_params(model, ignored_params=[]):
    bn_prelu_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            ignored_params += list(map(id, m.parameters()))
            bn_prelu_params += m.parameters()
        if isinstance(m, nn.BatchNorm1d):
            ignored_params += list(map(id, m.parameters()))
            bn_prelu_params += m.parameters()
        elif isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            bn_prelu_params += m.parameters()
    base_params = list(filter(lambda p: id(p) not in ignored_params, model.parameters()))

    return base_params, bn_prelu_params, ignored_params

OPTIMIZER = optim.SGD([
        {'params': base_params, 'weight_decay': WEIGHT_DECAY},
        {'params': fc_head_param, 'weight_decay': WEIGHT_DECAY * 10},
        {'params': bn_prelu_params, 'weight_decay': 0.0}
        ], lr=LR, momentum=MOMENTUM )  # , nesterov=True



###########################################
def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.
    print(optimizer)


# Module.children()与Module.modules()都是返回网络模型里的组成元素，但children()返回最外层元素，modules()返回所有级别的元素
# 下面的关键词if 'model'，其实源于模型定义文件如model_resnet,py中的‘model’，该文件中自定义的所有Model子类，都会前缀'model_resnet'，所有可通过这种方式一次性筛选出自定义的模块
def separate_irse_bn_paras(model):
    paras_only_bn = []
    paras_no_bn = []
    for layer in model.modules():
        if 'model' in str(layer.__class__):		            # eg. a=[1,2] type(a): <class 'list'>  a.__class__: <class 'list'>
            continue
        if 'container' in str(layer.__class__):             # 去掉Sequential型的模块
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_no_bn.extend([*layer.parameters()])   # extend()用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）

    return paras_only_bn, paras_no_bn


def separate_resnet_bn_paras(model) :
    all_parameters = model.parameters()
    paras_only_bn = []

    for pname , p in model.named_parameters() :
        if pname.find('bn') >= 0 :
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id , paras_only_bn))
    paras_no_bn = list(filter(lambda p : id(p) not in paras_only_bn_id , all_parameters))

    return paras_only_bn , paras_no_bn
