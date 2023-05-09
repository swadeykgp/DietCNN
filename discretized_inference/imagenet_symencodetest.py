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



from  patchlib import dietcnn_purify, dietcnn_encode, dietcnn_decode




import faiss 
       
import os
import numpy
import random
import time

HOWMANY = 39999999999
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

dir_name = "./sym_imagenet_encoded_2x2/"
# symbolic inference test 
def encode_testset():
    total = 0 
    with torch.no_grad():
        for data in testset:
            X, y = data
            if total > HOWMANY:
                break
            sym_image = dietcnn_encode(X, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
            #print(sym_image.dtype)
            sym_image = sym_image.astype('u2') # 0-511 symbols
            #print(sym_image.dtype)
            imname = "image_"+str(total)
            #np.savetxt(dir_name+imname, sym_image, delimiter =', ', fmt='%d')
            np.save(dir_name+imname, sym_image)
            total += 1
            if total % 50 == 0:  
                print("Symbolic Encoding ongoing ...",total)
        print("Symbolic Encoding done ...")
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


dir_decode = "./sym_imagenet_encoded/"
def decode_testset():
    correct = 0 
    total = 0 
    net.eval()
    hm = 0 
    with torch.no_grad():
        list_of_files = sorted(filter( lambda x: os.path.isfile(os.path.join(dir_decode, x)), os.listdir(dir_decode)), key=numericalSort)
        labels = np.genfromtxt("label_name_4000.txt", dtype=np.int)
        #print(labels.shape)
        for file_name in list_of_files:
            #print(file_name)
            symbolic = np.load(dir_decode+file_name)
            Xsym = dietcnn_decode(symbolic, centroid_lut, patch_size, patch_stride, 224,224,3)
            Xsym = Xsym.unsqueeze(0)
            output = net.forward(Xsym)
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[total]:
                    correct += 1
                #else:
                #    # Whenever there is an error, print the image
                #    print("Test Image #: {}".format(total+1))
                #    print("Mispredicted label: {}".format(torch.argmax(i)))
                total += 1
            if total % 50 == 0:  
                print("Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
    print("Final Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))


## Uncomment 2 lines below for 2048 , and comment out 2 lines for 512

#index = faiss.read_index("./imagenet/kmeans_img_imgnet_k2_s0_c2048_v0.index")
#n_clusters=2048
index = faiss.read_index("./imagenet/kmeans_imagenet_k2_s0_512_v3_nosc_noflt.index")
n_clusters=512




patch_size = (2, 2)
channel_count = 3
repeat = 2
patch_stride = 2

centroid_lut = index.reconstruct_n(0, n_clusters)
encode_testset()

# after encode try decode

#decode_testset()
#net= models.wide_resnet50_2(pretrained=True)
#net.eval()
#decode_testset()
#net=models.resnext101_32x8d(pretrained=True)
#net.eval()
#decode_testset()
#net=models.alexnet(pretrained=True)
#net.eval()
#decode_testset()
