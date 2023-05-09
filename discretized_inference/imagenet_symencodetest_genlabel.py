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

import sys
sys.path.insert(1, './core')


from  patchlib import dietcnn_purify,dietcnn_encode




import faiss 
       
import os
import numpy
import random
import time

HOWMANY = 30000 
torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)
def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  
def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  
def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def data_loader(data_dir, batch_size=1, workers=2, pin_memory=True):
#    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
#    train_loader = torch.utils.data.DataLoader(
#        train_ds,
#        batch_size=batch_size,
#        shuffle=True,
#        num_workers=workers,
#        pin_memory=pin_memory,
#        sampler=None
#    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    #return train_loader, val_loader
    return val_loader



net= models.resnet152(pretrained=True)
net.eval()
#for name, layer in net.named_modules():
#    print(name, layer)
#CHANGEME
# ********** Change the following - point to the local imagenet folder ********* #
#trainset, testset = data_loader('~/dataset/imagenet', batch_size=1)
#trainset_64, testset_64 = data_loader('~/dataset/imagenet', batch_size=64)
#trainset_8, testset_8 = data_loader('~/dataset/imagenet', batch_size=8)
testset = data_loader('~/dataset/imagenet', batch_size=1)
testset_64 = data_loader('~/dataset/imagenet', batch_size=64)
testset_8 = data_loader('~/dataset/imagenet', batch_size=8)
# ********** Change the following - point to the local imagenet folder ********* #

dir_name = "./imagenet_encoded/"
# symbolic inference test 
def encode_testset():
    total = 0

    label_buffer = []
    with torch.no_grad():
        for data in testset:
            X, y = data
            if total > HOWMANY:
                break
            y = y.squeeze()
            label_buffer.append(y)
            total += 1
            if total % 50 == 0:  
                print("label generation ongoing  ...",total)
        lb = np.array(label_buffer, dtype=np.int)        
        np.savetxt('imagenet_label_list30k.txt', lb, delimiter =', ', fmt='%d')
        print("label Encoding done ...")

def decode_testset():
    total = 0 
    with torch.no_grad():
        for data in testset:
            X, y = data
            sym_image = dietcnn_encode(X, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count) 
            imname = "image_"+str(total)
            np.savetxt(dir_name+imname+'.gz', sym_image, delimiter =', ', fmt='%d')
            total += 1
            if total % 50 == 0:  
                print("Symbolic Encoding ongoing ...",total)
        print("Symbolic Encoding done ...")



#index = faiss.read_index("./imagenet/kmeans_img_imgnet_k2_s0_c2048_v0.index")
#n_clusters=2048
patch_size = (2, 2)
channel_count = 3
repeat = 2
patch_stride = 2
index = faiss.read_index("./imagenet/kmeans_imagenet_k2_s0_512_v3_nosc_noflt.index")
n_clusters=512
centroid_lut = index.reconstruct_n(0, n_clusters)
encode_testset()
