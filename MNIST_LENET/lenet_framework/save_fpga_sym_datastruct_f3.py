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
sys.path.insert(1, '../core')
from patchutils import *
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

def get_data_conv(c1_filter, c2_filter,  patch_size, stride):
    buffer = []
    index = 0 
    # Only batch size of 1 supported till now
    #print("c1_filter.shape",c1_filter.shape) 
    n, c, w, h = c1_filter.shape
    layer_filter = c1_filter
    for num_filters in range(n):
        flt_ch = layer_filter[num_filters,:,:,:]
        flt_ch = flt_ch.squeeze()
        flt = flt_ch.squeeze()
        data = extract_patches_new(flt, patch_size, stride)   
        data = np.reshape(data, (len(data), -1))
        #print("Conv one flt;", data.shape)
        buffer.append(data)
        index += 1
    #print("c2_filter.shape",c2_filter.shape) 
    n, c, w, h = c2_filter.shape
    layer_filter = c2_filter
    for num_filters in range(n):
        flt_ch = layer_filter[num_filters,:,:,:]
        flt_ch = flt_ch.squeeze()
        flt = flt_ch.squeeze()
        data = extract_patches_new(flt, patch_size, stride)   
        data = np.reshape(data, (len(data), -1))
        buffer.append(data)
        index += 1
    data = np.concatenate(buffer, axis=0)
    #print("Conv all;", data.shape)
    #return standardized_dataset
    return data

def get_data_fc(f1_filter, f2_filter,  patch_size, stride):
    index = 0 
    # Only batch size of 1 supported till now
    #print("f1_filter.shape",f1_filter.shape) 
    flt = f1_filter
    outk, ink = flt.shape 
    buffer = []
    for i in range(outk):
        data = flt[i].reshape(-1,1)   
        buffer.append(data)
    #print("f2_filter.shape",f2_filter.shape) 
    l = f2_filter.shape
    flt = f2_filter
    outk, ink = flt.shape 
    for i in range(outk):
        data = flt[i].reshape(-1,1)   
        buffer.append(data)
    data = np.concatenate(buffer, axis=0)
    #print("FC all:", data.shape)
    return data

def create_index_conv(centers, c1_filter, c2_filter,  patch_size, stride):
    data = get_data_conv(c1_filter, c2_filter, patch_size, stride )
    kmeans = KMeans(n_clusters=centers, init='k-means++', n_init=10, random_state=0, algorithm='elkan').fit(data)
    return kmeans

def create_index_fc(centers,  f1_filter, f2_filter, patch_size, stride):
    #start_t = time.time()  
    data = get_data_fc( f1_filter, f2_filter, patch_size, stride )
    #end = time.time()
    #print("elapsed time for get data fc:", end - start_t) 
    #start_t = time.time()  
    kmeans = KMeans(n_clusters=centers, init='k-means++', n_init=10, random_state=0, algorithm='elkan').fit(data)
    #end = time.time()
    #print("elapsed time for index create fc:", end - start_t) 
    return kmeans

def platform_mult(x, filter_lut, index, n_filters ):
    result = []
    for j in range(n_filters):
        input1 = torch.from_numpy(x)
        input2 = torch.from_numpy(filter_lut[j])
        inputs = np.multiply(input1 , input2)
        _, results_sym = index.search(np.asarray(inputs.reshape(1, -1)).astype(np.float32), 1)
        result.append(results_sym.item()) 
    return result

def create_conv_luts(centroid_lut,kmeans,n_clusters,n_filters,index):
    filter_lut = kmeans.cluster_centers_
    results = Parallel(n_jobs=PARALLEL)(delayed(platform_mult)(centroid_lut[i], filter_lut, index, n_filters) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    #dist_matrix = dist_matrix.reshape(n_clusters,n_filters) 
    return dist_matrix

def create_fc_luts(centroid_lut,kmeans,n_clusters,n_filters,index):
    filter_lut = kmeans.cluster_centers_
    results = Parallel(n_jobs=PARALLEL)(delayed(platform_mult)(centroid_lut[i], filter_lut, index, n_filters) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    #dist_matrix = dist_matrix.reshape(n_clusters,n_filters) 
    return dist_matrix

def platform_add(x, centroid_lut, index, n_clusters):
    result = []
    for j in range(n_clusters):
        input1 = torch.from_numpy(x)
        input2 = torch.from_numpy(centroid_lut[j])
        inputs = input1 + input2
        _, results_sym = index.search(np.asarray(inputs.reshape(1, -1)).astype(np.float32), 1)
        result.append(results_sym.item()) 
    return result

def create_add_luts(centroid_lut,n_clusters, index):
    results = Parallel(n_jobs=PARALLEL)(delayed(platform_add)(centroid_lut[i], centroid_lut, index, n_clusters) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    #dist_matrix = dist_matrix.reshape(n_clusters,n_clusters) 
    return dist_matrix

def bias_add(x, bias, index, n_bias):
    result = []
    for j in range(n_bias):
        #input1 = torch.from_numpy(x)
        #print(input1) 
        #input2 = torch.from_numpy(bias[j].item())
        #print(input2) 
        inputs = x + bias[j]
        _, results_sym = index.search(np.asarray(inputs.reshape(1, -1)).astype(np.float32), 1)
        result.append(results_sym.item()) 
    return result

def create_bias_luts(centroid_lut, n_clusters, bias,index):
    n_bias = bias.shape[0]
    #print(n_bias) 
    results = Parallel(n_jobs=PARALLEL)(delayed(bias_add)(centroid_lut[i], bias, index, n_bias) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    #dist_matrix = dist_matrix.reshape(n_clusters,n_bias) 
    return dist_matrix

def create_relu_lut(centroid_lut, n_clusters, index):
    dist_matrix = np.zeros((n_clusters,),dtype=np.int16)
    for i in range(n_clusters):
            x = centroid_lut[i]
            x = x.reshape(1,1)
            inputs = torch.from_numpy(x)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.unsqueeze(0)
            tmp = F.relu(inputs)
            _, results_sym = index.search(np.asarray(tmp.reshape(1, -1)).astype(np.float32), 1)
            #print(results_sym.item())     
            dist_matrix[i] = results_sym.item()
            #if i%32 ==0:
                #print("ReLU LUT fillup (%)  going on...  ",((i)/(n_clusters))*100)
    return np.asarray(dist_matrix)

def save_index(kmeans, filename):
    pickle.dump(kmeans, open(filename, 'wb')) 
def save_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(int), delimiter =', ', fmt='%i,')
def save_float_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.float32), delimiter =', ', fmt='%f')

import pickle

def filter_to_symbolic(kmeans, flt):
    symbol_array = kmeans.predict(flt)
    return symbol_array


def filter_to_sym_conv(kmeans, ilf, patch_size, inc, outc, fstride, pad):
    layer_filter = ilf    
    n, c, w, h = layer_filter.shape
    #print(n, c, w, h)  
    outk = (w - patch_size[0])//fstride + 1
    nsyms = outk*outk*inc
    #print(outk, nsyms)  
    sym_kern_buff = []
    #sym_kern = np.zeros((nsyms, n))
    for num_filters in range(n):
        flt_ch = layer_filter[num_filters,:,:,:]
        flt_ch = flt_ch.squeeze()
        flt = flt_ch.squeeze()
        data = extract_patches_new(flt, patch_size, fstride)  
        #print(data.shape) 
        data = np.reshape(data, (len(data), -1))
        flt_sym = filter_to_symbolic(kmeans, data)
        sym_kern_buff.append(flt_sym)
    sym_kern = np.asarray(sym_kern_buff, dtype=np.int16)
    return sym_kern

def filter_to_sym_fc(kmeans, flt, patch_size, fstride, pad):
    outk, ink = flt.shape 
    sym_kern = np.zeros((outk, ink),dtype=np.int16)
    sym_kern_buff = []
    for i in range(outk):
        buffer = []
        data = flt[i].reshape(-1,1)   
        buffer.append(data)
        data = np.concatenate(buffer, axis=0)
        flt_sym = filter_to_symbolic(kmeans, data)
        sym_kern_buff.append(flt_sym)
    sym_kern = np.asarray(sym_kern_buff, dtype=np.int16)
    return sym_kern
    

if __name__ == '__main__':

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    
    pretrained_model = "./best_lenet_full.pt"
    
    net = CNN_LeNet() 
    net.load_state_dict(torch.load(pretrained_model))
    net.eval()
    
    c1_filter = net.conv1.weight.data.clone() 
    c2_filter = net.conv2.weight.data.clone() 
    f1_filter = net.fc1.weight.data.clone() 
    f2_filter = net.fc2.weight.data.clone() 
    f3_filter = net.fc3.weight.data.clone() 
   
    conv_patch_size = (1, 1)
    all_patch_size = (1, 1)
    n_cluster_conv_filters = 64
    n_cluster_fc_filters = 128
    conv_stride = 1
    index = faiss.read_index("./kmeans_mnist_fullnet_k1_s1_c128_faiss_v10.index")
    n_clusters=128
    patch_stride = 1
    centroid_lut = index.reconstruct_n(0, n_clusters)

    start_t = time.time()  
    filter_index_conv = create_index_conv(n_cluster_conv_filters, c1_filter, c2_filter, conv_patch_size, conv_stride )
    end = time.time()
    print("elapsed time for conv:", end - start_t) 
    start_t = time.time()  
    filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter, f2_filter,  all_patch_size, patch_stride )
    end = time.time()
    print("elapsed time for fc index:", end - start_t) 
    start_t = time.time()  
    fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)   
    end = time.time()
    print("elapsed time for fc lut:", end - start_t) 
    start_t = time.time()  
    conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)   
    end = time.time()
    print("elapsed time for conv lut:", end - start_t) 
    start_t = time.time()  
    add_lut = create_add_luts(centroid_lut, n_clusters, index)
    end = time.time()
    print("elapsed time for add lut:", end - start_t) 

    # deal the biases 
    
    with torch.no_grad():
        c1_bias = net.conv1.bias.clone()
        c1_bias = np.asarray(c1_bias)

    start_t = time.time()  
    c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)  
    end = time.time()
    print("elapsed time for bias lut:", end - start_t) 
    with torch.no_grad():
        c2_bias = net.conv2.bias.clone()
        c2_bias = np.asarray(c2_bias)

    start_t = time.time()  
    c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)  
    end = time.time()
    print("elapsed time for bias lut:", end - start_t) 
    #
    with torch.no_grad():
        f1_bias = net.fc1.bias.clone()
        f1_bias = np.asarray(f1_bias)
        f2_bias = net.fc2.bias.clone()
        f2_bias = np.asarray(f2_bias)
        f3_bias = net.fc3.bias.clone()
        f3_bias = np.asarray(f3_bias)
    start_t = time.time()  
    f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)  
    end = time.time()
    print("elapsed time for bias lut:", end - start_t) 
    start_t = time.time()  
    f2_bias_lut = create_bias_luts(centroid_lut, n_clusters, f2_bias, index)  
    end = time.time()
    print("elapsed time for bias lut:", end - start_t) 
    #
    start_t = time.time()  
    f3_bias_lut = create_bias_luts(centroid_lut, n_clusters, f3_bias, index)  
    end = time.time()
    print("elapsed time for bias lut:", end - start_t) 
    start_t = time.time()  
    relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
    end = time.time()
    print("elapsed time for relu lut:", end - start_t) 

    n,ic,kw,kh = c1_filter.shape 
    filter_patch_size = (1,1)
    filter_stride = 1
    c1_sym_filter = filter_to_sym_conv(filter_index_conv, c1_filter, filter_patch_size, ic, n, filter_stride, False)
    n2,ic2,kw2,kh2 = c2_filter.shape 
    c2_sym_filter = filter_to_sym_conv(filter_index_conv, c2_filter, filter_patch_size, ic2, n2, filter_stride, False)
    f1_sym_filter = filter_to_sym_fc(filter_index_fc,f1_filter, filter_patch_size, filter_stride, False)
    f2_sym_filter = filter_to_sym_fc(filter_index_fc,f2_filter, filter_patch_size, filter_stride, False)
    f3_sym_filter = filter_to_sym_fc(filter_index_fc,f3_filter, filter_patch_size, filter_stride, False)

    #save_index(filter_index_conv, 'mnist_conv_flt_full.index')
    #save_index(filter_index_fc, 'mnist_fc_flt_full.index')
    #save_to_txt(conv_lut.reshape(-1,1), 'conv_lut.txt')
    #save_to_txt(fc_lut.reshape(-1,1), 'fc_lut.txt')
    #save_to_txt(add_lut.reshape(-1,1), 'add_lut.txt')
    #save_to_txt(c1_bias_lut.reshape(-1,1), 'c1b_lut.txt')
    #save_to_txt(c2_bias_lut.reshape(-1,1), 'c2b_lut.txt')
    #save_to_txt(f1_bias_lut.reshape(-1,1), 'f1b_lut.txt')
    #save_to_txt(f2_bias_lut.reshape(-1,1), 'f2b_lut.txt')
    #save_to_txt(f3_bias_lut.reshape(-1,1), 'f3b_lut.txt')
    save_to_txt(relu_lut.reshape(-1,1), 'relu_lut.txt')
    #save_to_txt(c1_sym_filter.reshape(-1,1), 'c1fs.txt')
    #save_to_txt(c2_sym_filter.reshape(-1,1), 'c2fs.txt')
    #save_to_txt(f1_sym_filter.reshape(-1,1), 'f1fs.txt')
    #save_to_txt(f2_sym_filter.reshape(-1,1), 'f2fs.txt')
    #save_to_txt(f3_sym_filter.reshape(-1,1), 'f3fs.txt')
    save_float_to_txt(centroid_lut.reshape(-1,1), 'centroid_lut.txt')
    print("Everything completed successfully")
