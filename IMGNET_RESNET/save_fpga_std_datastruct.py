import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
import os
# For training
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision
import torchvision.models as models
from torchvision import utils

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from sklearn.cluster import KMeans
import sys
import pickle

import faiss 
       
from resnet_18 import *
import os
import numpy
import random
import time
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 2
def fuse_conv_and_bn(conv, bn):
    #
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
    #
    # we're done
    return fusedconv

def save_index(kmeans, filename):
    pickle.dump(kmeans, open(filename, 'wb')) 
def save_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.float32), delimiter =', ', fmt='%f,')


def conv_filter_to_flat(layer_filter):
    n, w, h, c = layer_filter.shape
    #print(n, c, w, h)  
    kern_buff = []
    for num_filters in range(n):
        for width in range(w):
            for height in range(h):
                for num_channels in range(c):
                    kern_buff.append(layer_filter[num_filters,width,height,num_channels])
    kern = np.asarray(kern_buff, dtype=np.float32)
    return kern

def fc_filter_to_flat(layer_filter):
    w, h = layer_filter.shape
    kern_buff = []
    for num_filters in range(w):
        for num_neurons in range(h):
            kern_buff.append(layer_filter[num_filters,num_neurons])
    kern = np.asarray(kern_buff, dtype=np.float32)
    return kern

if __name__ == '__main__':

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    net = resnet18()
    pretrained_model = "./tinyimg_resnet_std.pt"
    sd = torch.load(pretrained_model, map_location=torch.device('cpu'))
    net.load_state_dict(sd['net'])
    net.eval()

    torch.set_grad_enabled(False)

    # First task is the fusing of convulution and the batch normalization layers
    c_filter = []
    # Layer 1
    fusedconv = fuse_conv_and_bn(net.conv1, net.bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[0].conv1, net.layer1[0].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[0].conv2, net.layer1[0].bn2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[1].conv1, net.layer1[1].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[1].conv2, net.layer1[1].bn2)
    c_filter.append(fusedconv.weight.data.clone())


    fusedconv = fuse_conv_and_bn(net.layer2[0].conv1, net.layer2[0].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[0].conv2, net.layer2[0].bn2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[1].conv1, net.layer2[1].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[1].conv2, net.layer2[1].bn2)
    c_filter.append(fusedconv.weight.data.clone())


    fusedconv = fuse_conv_and_bn(net.layer3[0].conv1, net.layer3[0].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[0].conv2, net.layer3[0].bn2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[1].conv1, net.layer3[1].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[1].conv2, net.layer3[1].bn2)
    c_filter.append(fusedconv.weight.data.clone())

    fusedconv = fuse_conv_and_bn(net.layer4[0].conv1, net.layer4[0].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[0].conv2, net.layer4[0].bn2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[1].conv1, net.layer4[1].bn1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[1].conv2, net.layer4[1].bn2)
    c_filter.append(fusedconv.weight.data.clone())


    
    c1_filter_saver = np.asarray(c_filter[0].permute(0,2,3,1))
    c2_filter_saver = np.asarray(c_filter[1].permute(0,2,3,1))
    c3_filter_saver = np.asarray(c_filter[2].permute(0,2,3,1))
    c4_filter_saver = np.asarray(c_filter[3].permute(0,2,3,1))
    c5_filter_saver = np.asarray(c_filter[4].permute(0,2,3,1))
    c6_filter_saver = np.asarray(c_filter[5].permute(0,2,3,1))
    c7_filter_saver = np.asarray(c_filter[6].permute(0,2,3,1))
    c8_filter_saver = np.asarray(c_filter[7].permute(0,2,3,1))
    c9_filter_saver = np.asarray(c_filter[8].permute(0,2,3,1))
    c10_filter_saver = np.asarray(c_filter[9].permute(0,2,3,1))
    c11_filter_saver = np.asarray(c_filter[10].permute(0,2,3,1))
    c12_filter_saver = np.asarray(c_filter[11].permute(0,2,3,1))
    c13_filter_saver = np.asarray(c_filter[12].permute(0,2,3,1))
    c14_filter_saver = np.asarray(c_filter[13].permute(0,2,3,1))
    c15_filter_saver = np.asarray(c_filter[14].permute(0,2,3,1))
    c16_filter_saver = np.asarray(c_filter[15].permute(0,2,3,1))
    c17_filter_saver = np.asarray(c_filter[16].permute(0,2,3,1))
    
    
    
    c1f = conv_filter_to_flat(c1_filter_saver)
    c2f = conv_filter_to_flat(c2_filter_saver)
    c3f = conv_filter_to_flat(c3_filter_saver)
    c4f = conv_filter_to_flat(c4_filter_saver)
    c5f = conv_filter_to_flat(c5_filter_saver)
    c6f = conv_filter_to_flat(c6_filter_saver)
    c7f = conv_filter_to_flat(c7_filter_saver)
    c8f = conv_filter_to_flat(c8_filter_saver)

    c9f = conv_filter_to_flat(c9_filter_saver)
    c10f = conv_filter_to_flat(c10_filter_saver)
    c11f = conv_filter_to_flat(c11_filter_saver)
    c12f = conv_filter_to_flat(c12_filter_saver)
    c13f = conv_filter_to_flat(c13_filter_saver)
    c14f = conv_filter_to_flat(c14_filter_saver)
    c15f = conv_filter_to_flat(c15_filter_saver)
    c16f = conv_filter_to_flat(c16_filter_saver)
    c17f = conv_filter_to_flat(c17_filter_saver)
    ff = fc_filter_to_flat(net.fc.weight.data.clone())


    save_to_txt(c1f, 'c1f.txt')
    save_to_txt(c2f, 'c2f.txt')
    save_to_txt(c3f, 'c3f.txt')
    save_to_txt(c4f, 'c4f.txt')
    save_to_txt(c5f, 'c5f.txt')
    save_to_txt(c6f, 'c6f.txt')
    save_to_txt(c7f, 'c7f.txt')
    save_to_txt(c8f, 'c8f.txt')
    save_to_txt(c9f, 'c9f.txt')
    save_to_txt(c10f, 'c10f.txt')
    save_to_txt(c11f, 'c11f.txt')
    save_to_txt(c12f, 'c12f.txt')
    save_to_txt(c13f, 'c13f.txt')
    save_to_txt(c14f, 'c14f.txt')
    save_to_txt(c15f, 'c15f.txt')
    save_to_txt(c16f, 'c16f.txt')
    save_to_txt(c17f, 'c17f.txt')
    save_to_txt(ff, 'ff.txt')
    print("Everything completed successfully")
