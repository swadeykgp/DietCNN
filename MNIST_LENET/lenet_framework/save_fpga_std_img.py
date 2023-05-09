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
from torchvision import utils
import sys
sys.path.insert(1, '../core')
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from  patchlib import *
import faiss 
       
import os
import numpy
import random
import time

HOWDY = 2
# Symbolic convolution
def discretize(x, in_channels, index,  patch_size, patch_stride, padding): 
   #x = F.pad(input=x, pad=(padding, padding, padding, padding), mode='constant', value=0) 

   if in_channels == 1:
       #img  = np.asarray(x.clone().detach()) 
       img  = np.asarray(x.clone()) 
       im_w,im_h= img.shape      
   else: 
       #img  = np.asarray(x.clone().detach().permute(1,2,0)) 
       img  = np.asarray(x.clone().permute(1,2,0)) 
       im_w,im_h,_= img.shape       
   img_sym =  image_to_symbolic(img, index, patch_size, patch_stride)  
   simg_out_w = (im_w - patch_size[0])//patch_stride + 1
   simg_out_h = (im_h - patch_size[0])//patch_stride + 1
   if in_channels == 1:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h)
   else:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h, in_channels)
   
   return img_sym


# save std image 
def generate_image_std(model, atk,  data_iter, batch_size, clamp, top_5, std):
    for data in data_iter:
        X, y = data
        img = X.permute(0,2,3,1)
        n,w,h,c = img.shape
        img_buff = []
        for num_filters in range(n):
            for width in range(w):
                for height in range(h):
                    for num_channels in range(c):
                        img_buff.append(img[num_filters,width,height,num_channels])
        img_std = np.asarray(img_buff, dtype=np.float32)
        return img_std,y.item()

# save symbolic image
def generate_image_sym(index, atk,  data_iter, patch_size, patch_stride, padding, std):
    for data in data_iter:
        X, y = data
        X = X.squeeze()
        X = X.squeeze()
        img = discretize(X, 1, index,  patch_size, patch_stride, padding)
        w,h = img.shape
        img_buff = []
        for width in range(w):
            for height in range(h):
                img_buff.append(img[width,height])
        img_sym = np.asarray(img_buff, dtype=np.float32)
        return img_sym,y.item()


def save_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.float32), delimiter =', ', fmt='%f')

def save_sym_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.int16), delimiter =', ', fmt='%d')


if __name__ == '__main__':
    if (len(sys.argv) - 1) < 1:
        print ("Call with first argument 0: Standard image, 1: DietCNN image")
        sys.exit()
    
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    bs= 1 
    transform_test = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])
    
    testset = datasets.MNIST(root='../../dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    
    transform_test_sym = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])
    
    testset_sym = datasets.MNIST(root='../../dataset', train=False, download=True, transform=transform_test_sym)
    testloader_sym = torch.utils.data.DataLoader(testset_sym, batch_size=bs, shuffle=False)
    
    index = faiss.read_index("./kmeans_mnist_fullnet_k1_s1_c128_faiss_v10.index")
    n_clusters=128
    patch_size = (1, 1)
    patch_stride = 1
    centroid_lut = index.reconstruct_n(0, n_clusters)
    
    if sys.argv[1] == "0":
        start = time.process_time()  
        start_t = time.time()  
        img, label = generate_image_std(None, None, testloader, None, False, False,0)
        save_to_txt(img, 'img_std_label_'+str(label)+'.txt')
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for standard inference",elapsed)  
        elapsed = time.process_time() - start
        end = time.time()
        print("elapsed time for standard inference:", end - start_t) 
    else:
        start = time.process_time()  
        start_t = time.time()  
        img, label = generate_image_sym(index, None, testloader, patch_size, patch_stride, 0, 1)
        save_sym_to_txt(img, 'img_sym_label_'+str(label)+'.txt')
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for symbolic inference",elapsed)
        end = time.time()
        print("elapsed time for symbolic inference:", end - start_t)
