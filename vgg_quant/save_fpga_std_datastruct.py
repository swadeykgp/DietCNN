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
       
from vgg_sym import *
import os
import numpy
import random
import time
#from kneed import DataGenerator, KneeLocator
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 2

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
    
    net = VGG('VGG11')
    pretrained_model = "./cifar_vgg_sym_v3.pt"
    sd = torch.load(pretrained_model, map_location=torch.device('cpu'))
    #print(sd['net']) 
    net.load_state_dict(sd['net'])
    net.eval()

    c_filter = []
    c_filter.append(net.features[0].weight.data.clone())
    c_filter.append(net.features[2].weight.data.clone())
    c_filter.append(net.features[4].weight.data.clone())
    c_filter.append(net.features[6].weight.data.clone())
    c_filter.append(net.features[8].weight.data.clone())
    c_filter.append(net.features[10].weight.data.clone())
    c_filter.append(net.features[12].weight.data.clone())
    c_filter.append(net.features[14].weight.data.clone())
    f1_filter = net.classifier[0].weight.data.clone()

    with torch.no_grad():
        c1_bias = net.features[0].bias.clone()
        c1_bias = np.asarray(c1_bias)
        c2_bias = net.features[2].bias.clone()
        c2_bias = np.asarray(c2_bias)
        c3_bias = net.features[4].bias.clone()
        c3_bias = np.asarray(c3_bias)
        c4_bias = net.features[6].bias.clone()
        c4_bias = np.asarray(c4_bias)
        c5_bias = net.features[8].bias.clone()
        c5_bias = np.asarray(c5_bias)
        c6_bias = net.features[10].bias.clone()
        c6_bias = np.asarray(c6_bias)
        c7_bias = net.features[12].bias.clone()
        c7_bias = np.asarray(c7_bias)
        c8_bias = net.features[14].bias.clone()
        c8_bias = np.asarray(c8_bias)
        f1_bias = net.classifier[0].bias.clone()
        f1_bias = np.asarray(f1_bias)
    
    c1_filter_saver = np.asarray(c_filter[0].permute(0,2,3,1))
    c2_filter_saver = np.asarray(c_filter[1].permute(0,2,3,1))
    c3_filter_saver = np.asarray(c_filter[2].permute(0,2,3,1))
    c4_filter_saver = np.asarray(c_filter[3].permute(0,2,3,1))
    c5_filter_saver = np.asarray(c_filter[4].permute(0,2,3,1))
    c6_filter_saver = np.asarray(c_filter[5].permute(0,2,3,1))
    c7_filter_saver = np.asarray(c_filter[6].permute(0,2,3,1))
    c8_filter_saver = np.asarray(c_filter[7].permute(0,2,3,1))
    c1f = conv_filter_to_flat(c1_filter_saver)
    c2f = conv_filter_to_flat(c2_filter_saver)
    c3f = conv_filter_to_flat(c3_filter_saver)
    c4f = conv_filter_to_flat(c4_filter_saver)
    c5f = conv_filter_to_flat(c5_filter_saver)
    c6f = conv_filter_to_flat(c6_filter_saver)
    c7f = conv_filter_to_flat(c7_filter_saver)
    c8f = conv_filter_to_flat(c8_filter_saver)

    f1_filter = net.classifier[0].weight.data.clone() 

    f1f = fc_filter_to_flat(f1_filter)

    save_to_txt(c1f, 'c1f.txt')
    save_to_txt(c2f, 'c2f.txt')
    save_to_txt(c3f, 'c3f.txt')
    save_to_txt(c4f, 'c4f.txt')
    save_to_txt(c5f, 'c5f.txt')
    save_to_txt(c6f, 'c6f.txt')
    save_to_txt(c7f, 'c7f.txt')
    save_to_txt(c8f, 'c8f.txt')
    save_to_txt(f1f, 'f1f.txt')
    save_to_txt(c1_bias, 'c1b.txt')
    save_to_txt(c2_bias, 'c2b.txt')
    save_to_txt(c3_bias, 'c3b.txt')
    save_to_txt(c4_bias, 'c4b.txt')
    save_to_txt(c5_bias, 'c5b.txt')
    save_to_txt(c6_bias, 'c6b.txt')
    save_to_txt(c7_bias, 'c7b.txt')
    save_to_txt(c8_bias, 'c8b.txt')
    save_to_txt(f1_bias, 'f1b.txt')
    print("Everything completed successfully")
