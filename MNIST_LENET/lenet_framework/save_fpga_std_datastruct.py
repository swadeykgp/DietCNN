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
       
import os
import numpy
import random
import time
from kneed import DataGenerator, KneeLocator
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 2

class CNN_LeNet(nn.Module):
    def __init__(self):
        super(CNN_LeNet, self).__init__()
        # Define the net structure
        # This is the input layer first Convolution
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
    
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x

def save_index(kmeans, filename):
    pickle.dump(kmeans, open(filename, 'wb')) 
def save_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.float32), delimiter =', ', fmt='%f')


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
    
    pretrained_model = "./best_lenet_full.pt"
    
    net = CNN_LeNet() 
    net.load_state_dict(torch.load(pretrained_model))
    net.eval()
    
    c1_filter = net.conv1.weight.data.clone() 
    n1,c1,w1,h1 = c1_filter.shape 
    c2_filter = net.conv2.weight.data.clone() 
    n2,c2,w2,h2 = c2_filter.shape 
    f1_filter = net.fc1.weight.data.clone() 
    w3,h3 = f1_filter.shape 
    f2_filter = net.fc2.weight.data.clone() 
    w4,h4 = f2_filter.shape 
   
    # deal the biases 
    
    with torch.no_grad():
        c1_bias = net.conv1.bias.clone()
        c1_bias = np.asarray(c1_bias)
        c2_bias = net.conv2.bias.clone()
        c2_bias = np.asarray(c2_bias)
        f1_bias = net.fc1.bias.clone()
        f1_bias = np.asarray(f1_bias)
        f2_bias = net.fc2.bias.clone()
        f2_bias = np.asarray(f2_bias)

    c1_filter_saver = np.asarray(c1_filter.permute(0,2,3,1))
    c1f = conv_filter_to_flat(c1_filter_saver)

    c2_filter_saver = np.asarray(c2_filter.permute(0,2,3,1))
    c2f = conv_filter_to_flat(c2_filter_saver)

    f1f = fc_filter_to_flat(f1_filter)
    f2f = fc_filter_to_flat(f2_filter)


    save_to_txt(c1f, 'c1f.txt')
    save_to_txt(c2f, 'c2f.txt')
    save_to_txt(f1f, 'f1f.txt')
    save_to_txt(f2f, 'f2f.txt')
    save_to_txt(c1_bias, 'c1b.txt')
    save_to_txt(c2_bias, 'c2b.txt')
    save_to_txt(f1_bias, 'f1b.txt')
    save_to_txt(f2_bias, 'f2b.txt')
    print("Everything completed successfully")
