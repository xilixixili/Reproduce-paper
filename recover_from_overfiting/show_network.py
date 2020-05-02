"""
Recover from Overfitting to Label Noise : A Weight Pruning Perspective
train_mask
"""
import os

import torch
import torch.nn as nn


# Device

def show_network(file_name,log_name,model):
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    log_name=file_name+'/'+str(log_name)+'.txt'
    with open(log_name,"w") as f:

        f.write(str(model))
        for m in model.modules():
            if isinstance(m , nn.Conv2d) :
                # print(np.nonzero(m.weight.data))
                f.write(str(m))
                f.write('\n')
                f.write('weight:')
                f.write('\n')
                f.write(str(m.weight.data))
                f.write('\n')
                f.write('bias:')
                f.write('\n')
                f.write(str(m.bias.data))
                f.write('\n')
                # f.write('==============weight:',m,'\n',m.weight.data)
                # f.write('\n')
            elif isinstance(m , nn.BatchNorm2d) :  # running mean & running variance??

                f.write(str(m))
                f.write('\n')
                f.write('BatchNorm_weight:')
                f.write('\n')
                f.write(str(m.weight.data))
                f.write('\n')
                f.write('BatchNorm_bias:')
                f.write('\n')
                f.write(str(m.bias.data))
                f.write('\n')
            elif isinstance(m , nn.Linear) :
                f.write(str(m))
                f.write('\n')
                f.write('Linear weight:')
                f.write('\n')
                f.write(str(m.weight.data))
                f.write('\n')
                f.write('Linear bias:')
                f.write('\n')
                f.write(str(m.bias.data))
                f.write('\n')
            f.flush()
        f.close()