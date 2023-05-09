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
        img = discretize(X, 3, index,  patch_size, patch_stride, padding)
        w,h,c = img.shape
        img_buff = []
        for channels in range(c):
            for width in range(w):
                for height in range(h):
                    img_buff.append(img[width,height,channels])
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
    
    # Load and display samples from Tiny ImageNet  dataset
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    directory = "../../dataset/tiny-imagenet-200/"
    num_classes = 200
    # the magic normalization parameters come from the example
    transform_mean = np.array([ 0.485, 0.456, 0.406 ])
    transform_std = np.array([ 0.229, 0.224, 0.225 ])
    
    if sys.argv[1] == "0":
        val_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(74),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])
    else:    
        val_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(74),
            transforms.CenterCrop(71),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])
    
    
    ##### Related to trainset , need only for label ids ##############
    traindir = os.path.join(directory, "train")
    bs = 1
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = transform_mean, std = transform_std),
    ])
    train = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    assert num_classes == len(train_loader.dataset.classes)
    small_labels = {}
    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()
    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader.dataset.classes):
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index
    ############# All these just to get the label ids ############################
    
    valdir = os.path.join(directory, "val")
    
    val = datasets.ImageFolder(valdir, val_transform)
    
    val_loader = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=True)
    
    small_labels = {}
    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()
    
    
    val_label_map = {}
    with open(os.path.join(directory, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()
    
    
    for i in range(len(val_loader.dataset.imgs)):
        file_path = val_loader.dataset.imgs[i][0]
    
        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]
    
        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])
        
        index = faiss.read_index("./kmeans_resnet_tinyimgnet_c1_k1_s1_512_v0.index")
        n_clusters=512
        patch_size = (1, 1)
        patch_stride = 1
        centroid_lut = index.reconstruct_n(0, n_clusters)
    
    if sys.argv[1] == "0":
        start = time.process_time()  
        start_t = time.time()  
        img, label = generate_image_std(None, None, val_loader, None, False, False,0)
        save_to_txt(img, 'imgnet_img_std_label_'+str(label)+'.txt')
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for standard inference",elapsed)  
        elapsed = time.process_time() - start
        end = time.time()
        print("elapsed time for standard inference:", end - start_t) 
    else:
        start = time.process_time()  
        start_t = time.time()  
        img, label = generate_image_sym(index, None, val_loader, patch_size, patch_stride, 0, 1)
        save_sym_to_txt(img, 'imgnet_img_sym_label_'+str(label)+'.txt')
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for symbolic inference",elapsed)
        end = time.time()
        print("elapsed time for symbolic inference:", end - start_t)
