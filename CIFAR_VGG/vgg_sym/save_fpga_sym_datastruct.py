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
sys.path.insert(1, '../../core')
from patchlib import *
import pickle

import faiss 
       
import os
import numpy
import random
import time
#from kneed import DataGenerator, KneeLocator
from vgg_sym import *
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 8

#class VGG(nn.Module):
#    def __init__(self, vgg_name):
#        super(VGG, self).__init__()
#        self.features = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #21
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #18
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #15
#            nn.ReLU(inplace=True),
#            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #13
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #10
#            nn.ReLU(inplace=True),
#            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #8
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #5
#            nn.ReLU(inplace=True),
#            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #3
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
#        )
#        self.classifier = nn.Sequential(
#            nn.Linear(in_features=512, out_features=10, bias=True)
#        )
#
#
#    def forward(self, x):
#        out = self.features(x)
#        out = out.view(out.size(0), -1)
#        out = self.classifier(out)
#        return out

def get_data_conv(c_filter, num_filters, patch_size, stride):
    buffer = []
    index = 0 
    # Only batch size of 1 supported till now
    #print("c1_filter.shape",c1_filter.shape) 
    for i in range(num_filters):
        x = c_filter[i]
        n, c, w, h = x.shape
        if c == 1:
            layer_filter  = np.asarray(x.clone()) 
            n,w,h = layer_filter.shape      
        else: 
            layer_filter  = np.asarray(x.clone().permute(0,2,3,1)) 
            n, w, h, c = layer_filter.shape       
        for num_filters in range(n):
            #flt_ch = layer_filter[num_filters,:,:,:]
            #flt = flt_ch.squeeze()
            flt = layer_filter[num_filters,:,:,:]
            data = extract_patches(flt, patch_size, stride)   
            data = np.reshape(data, (len(data), -1))
            #print("Conv one flt;", data.shape)
            buffer.append(data)
            index += 1
    data = np.concatenate(buffer, axis=0)
    #print("Conv all;", data.shape)
    return data

def get_data_fc(f_filter,  patch_size, stride):
    index = 0 
    # Only batch size of 1 supported till now
    #print("f1_filter.shape",f1_filter.shape) 
    flt = f_filter
    outk, ink = flt.shape 
    buffer = []
    for i in range(outk):
        data = flt[i].reshape(-1,1)   
        buffer.append(data)
    data = np.concatenate(buffer, axis=0)
    return data

def create_index_conv(centers, c_filter, num_filters,  patch_size, stride):
    data = get_data_conv(c_filter, num_filters, patch_size, stride )
    #kmeans = KMeans(n_clusters=centers, init='k-means++', n_init=10, random_state=0, algorithm='elkan').fit(data)
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=centers, niter=200, nredo=2, verbose=True)
    kmeans.train(data.astype(np.float32))
    return kmeans.index

def create_index_fc(centers,  f_filter, patch_size, stride):
    #start_t = time.time()  
    data = get_data_fc( f_filter, patch_size, stride )
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
    #filter_lut = kmeans.reconstruct_n(0, n_filters)
    filter_lut = kmeans.cluster_centers_
    results = Parallel(n_jobs=PARALLEL)(delayed(platform_mult)(centroid_lut[i], filter_lut, index, n_filters) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    dist_matrix = dist_matrix.reshape(n_clusters,n_filters) 
    return dist_matrix

def create_fc_luts(centroid_lut,kmeans,n_clusters,n_filters,index):
    filter_lut = kmeans.cluster_centers_
    results = Parallel(n_jobs=PARALLEL)(delayed(platform_mult)(centroid_lut[i], filter_lut, index, n_filters) for i in range(n_clusters))
    dist_matrix = np.asarray(results, dtype=np.int16) 
    dist_matrix = dist_matrix.reshape(n_clusters,n_filters) 
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
    dist_matrix = dist_matrix.reshape(n_clusters,n_clusters) 
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
    dist_matrix = dist_matrix.reshape(n_clusters,n_bias) 
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
def save_lut(dm, filename):
    np.savetxt(filename, dm.astype(int), delimiter =', ', fmt='%i,')

def save_float_to_txt(dm, filename):
    np.savetxt(filename, dm.astype(np.float32), delimiter =', ', fmt='%f,')
import pickle

def filter_to_symbolic_fc(kmeans, flt):
    symbol_array = kmeans.predict(flt)
    return symbol_array

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
    sym_kern = np.zeros((nsyms, n),dtype=np.int16)
    #sym_kern = np.zeros((nsyms, n))
    for num_filters in range(n):
        flt_ch = layer_filter[num_filters,:,:,:]
        flt_ch = flt_ch.squeeze()
        flt = flt_ch.squeeze()
        data = extract_patches_new(flt, patch_size, fstride)
        data = np.reshape(data, (len(data), -1))
        #flt_sym = filter_to_symbolic_conv(kmeans, data)
        flt_sym = filter_to_symbolic(kmeans, data)
        sym_kern[:,num_filters] = flt_sym
    return sym_kern
#def filter_to_sym_conv(kmeans, ilf, patch_size, inc, outc, fstride, pad):
#    n, c, w, h = ilf.shape
#    outk = (w - patch_size[0])//fstride + 1
#    nsyms = outk*outk*inc
#    #print(outk, nsyms)  
#    sym_kern_buff = []
#    #sym_kern = np.zeros((nsyms, n))
#    if c == 1:
#        layer_filter  = np.asarray(ilf.clone()) 
#        n,w,h = layer_filter.shape      
#    else: 
#        layer_filter  = np.asarray(ilf.clone().permute(0,2,3,1)) 
#        n, w, h, c = layer_filter.shape       
#    for num_filters in range(n):
#        flt = layer_filter[num_filters,:,:,:]
#        #data = extract_patches_new(flt, patch_size, fstride)  
#        #print(data.shape) 
#        #data = np.reshape(data, (len(data), -1))
#        flt_sym = image_to_symbolic(flt, kmeans, patch_size, fstride) 
#        sym_kern_buff.append(flt_sym)
#    sym_kern = np.asarray(sym_kern_buff, dtype=np.int16)
#    return sym_kern

def filter_to_sym_fc(kmeans, flt, patch_size, fstride, pad):
    outk, ink = flt.shape 
    sym_kern = np.zeros((outk, ink),dtype=np.int16)
    sym_kern_buff = []
    for i in range(outk):
        buffer = []
        data = flt[i].reshape(-1,1)   
        buffer.append(data)
        data = np.concatenate(buffer, axis=0)
        flt_sym = filter_to_symbolic_fc(kmeans, data)
        sym_kern_buff.append(flt_sym)
    sym_kern = np.asarray(sym_kern_buff, dtype=np.int16)
    return sym_kern
    


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

    conv_patch_size = (1, 1)
    all_patch_size = (1, 1)
    n_cluster_conv_filters = 256
    n_cluster_fc_filters = 128
    conv_stride = 1
    index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_512_v0.index")
    n_clusters=512
    patch_stride = 1
    centroid_lut = index.reconstruct_n(0, n_clusters)

    #print("Conv index creation stared .....") 
    #start_t = time.time()  
    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
    #end = time.time()
    #print("elapsed time for conv index:", end - start_t) 
    #start_t = time.time()  
    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
    #end = time.time()
    #print("elapsed time for fc index:", end - start_t) 
    with open('cifar10_conv_flt.index', "rb") as f:
        filter_index_conv = pickle.load(f)
    with open('cifar10_fc_flt.index', "rb") as f:
        filter_index_fc = pickle.load(f)
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
    start_t = time.time()  
    c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)  
    c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)  
    c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)  
    c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)  
    c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)  
    c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)  
    c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)  
    c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)  
    end = time.time()
    print("elapsed time for convolution bias lut:", end - start_t) 
    
    start_t = time.time()  
    f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)  
    end = time.time()
    print("elapsed time for FC1  bias lut:", end - start_t) 
    start_t = time.time()  
    relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
    end = time.time()
    print("elapsed time for relu lut:", end - start_t) 


    # Finally the filters
    n,ic,kw,kh = c_filter[0].shape 
    c1_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[0], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[1].shape 
    c2_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[1], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[2].shape 
    c3_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[2], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[3].shape 
    c4_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[3], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[4].shape 
    c5_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[4], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[5].shape 
    c6_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[5], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[6].shape 
    c7_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[6], conv_patch_size, ic, n, conv_stride, False)
    n,ic,kw,kh = c_filter[7].shape 
    c8_sym_filter = filter_to_sym_conv(filter_index_conv, c_filter[7], conv_patch_size, ic, n, conv_stride, False)
    f1_sym_filter = filter_to_sym_fc(filter_index_fc,f1_filter, conv_patch_size, conv_stride, False)
    #faiss.write_index(filter_index_conv,"cifar10_conv_flt.index")
    #save_index(filter_index_conv, 'mnist_conv_flt_full.index')
    #save_index(filter_index_fc, 'cifar10_fc_flt.index')
    save_lut(conv_lut.reshape(-1,1), 'conv_lut.txt')
    save_lut(fc_lut.reshape(-1,1), 'fc_lut.txt')
    save_lut(add_lut.reshape(-1,1), 'add_lut.txt')
    save_lut(relu_lut.reshape(-1,1), 'relu_lut.txt')
    save_lut(c1_bias_lut.reshape(-1,1), 'c1b_lut.txt')
    save_lut(c2_bias_lut.reshape(-1,1), 'c2b_lut.txt')
    save_lut(c3_bias_lut.reshape(-1,1), 'c3b_lut.txt')
    save_lut(c4_bias_lut.reshape(-1,1), 'c4b_lut.txt')
    save_lut(c5_bias_lut.reshape(-1,1), 'c5b_lut.txt')
    save_lut(c6_bias_lut.reshape(-1,1), 'c6b_lut.txt')
    save_lut(c7_bias_lut.reshape(-1,1), 'c7b_lut.txt')
    save_lut(c8_bias_lut.reshape(-1,1), 'c8b_lut.txt')
    save_lut(f1_bias_lut.reshape(-1,1), 'f1b_lut.txt')
    save_lut(c1_sym_filter.reshape(-1,1), 'c1_sym_filter.txt')
    save_lut(c2_sym_filter.reshape(-1,1), 'c2_sym_filter.txt')
    save_lut(c3_sym_filter.reshape(-1,1), 'c3_sym_filter.txt')
    save_lut(c4_sym_filter.reshape(-1,1), 'c4_sym_filter.txt')
    save_lut(c5_sym_filter.reshape(-1,1), 'c5_sym_filter.txt')
    save_lut(c6_sym_filter.reshape(-1,1), 'c6_sym_filter.txt')
    save_lut(c7_sym_filter.reshape(-1,1), 'c7_sym_filter.txt')
    save_lut(c8_sym_filter.reshape(-1,1), 'c8_sym_filter.txt')
    save_lut(f1_sym_filter.reshape(-1,1), 'f1_sym_filter.txt')
    save_float_to_txt(centroid_lut.reshape(-1,1), 'centroid_lut.txt')
    print("Everything completed successfully")
