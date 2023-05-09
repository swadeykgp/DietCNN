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
from patchlib import *
import pickle

import faiss 
       
import os
import numpy
import random
import time
from resnet_18_sym import *

import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 8
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
    filter_lut = kmeans.reconstruct_n(0, n_filters)
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

def filter_to_sym_fc(kmeans, flt, patch_size, fstride, pad):
    outk, ink = flt.shape
    sym_kern = np.zeros((outk, ink),dtype=np.int16)
    #sym_kern = np.zeros((outk, ink))
    for i in range(outk):
        buffer = []
        data = flt[i].reshape(-1,1)
        buffer.append(data)
        #print(data.shape)
        data = np.concatenate(buffer, axis=0)
        flt_sym = filter_to_symbolic(kmeans, data)
        sym_kern[i,:] = flt_sym
    return sym_kern

   
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


if __name__ == '__main__':

    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    net = resnet18()
    pretrained_model = "./tinyimg_resnet_sym.pt"
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


    conv_patch_size = (1, 1)
    all_patch_size = (1, 1)
    patch_size = (1, 1)
    n_cluster_conv_filters = 256
    n_cluster_fc_filters = 128
    conv_stride = 1
    index = faiss.read_index("./kmeans_resnet_tinyimgnet_c1_k1_s1_512_v0.index")
    n_clusters=512
    patch_stride = 1
    centroid_lut = index.reconstruct_n(0, n_clusters)
    with open('imgnet_conv_flt.index', "rb") as f:
        filter_index_conv = pickle.load(f)
    with open('imgnet_fc_flt.index', "rb") as f:
        filter_index_fc = pickle.load(f)
    fc_lut = np.genfromtxt('./imgnet_fc_lut.txt', delimiter=',',dtype=np.int16)
    conv_lut = np.genfromtxt('./imgnet_conv_lut.txt', delimiter=',',dtype=np.int16)
    add_lut = np.genfromtxt('./imgnet_add_lut.txt', delimiter=',',dtype=np.int16)
    relu_lut = np.genfromtxt('./imgnet_relu_lut.txt', delimiter=',',dtype=np.int16)

    

    # Finally the filters
    n,ic,kw,kh = c_filter[0].shape
    print(n,ic,kw,kh)
    c1f = filter_to_sym_conv(filter_index_conv, c_filter[0], patch_size, ic, n, patch_stride, False)
    print(c1f.shape)
    sys.exit()
    n,ic,kw,kh = c_filter[1].shape
    c2f = filter_to_sym_conv(filter_index_conv, c_filter[1], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[2].shape
    c3f = filter_to_sym_conv(filter_index_conv, c_filter[2], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[3].shape
    c4f = filter_to_sym_conv(filter_index_conv, c_filter[3], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[4].shape
    c5f = filter_to_sym_conv(filter_index_conv, c_filter[4], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[5].shape
    c6f = filter_to_sym_conv(filter_index_conv, c_filter[5], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[6].shape
    c7f = filter_to_sym_conv(filter_index_conv, c_filter[6], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[7].shape
    c8f = filter_to_sym_conv(filter_index_conv, c_filter[7], patch_size, ic, n, patch_stride, False)

    n,ic,kw,kh = c_filter[8].shape
    c9f = filter_to_sym_conv(filter_index_conv, c_filter[8], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[9].shape
    c10f = filter_to_sym_conv(filter_index_conv, c_filter[9], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[10].shape
    c11f = filter_to_sym_conv(filter_index_conv, c_filter[10], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[11].shape
    c12f = filter_to_sym_conv(filter_index_conv, c_filter[11], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[12].shape
    c13f = filter_to_sym_conv(filter_index_conv, c_filter[12], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[13].shape
    c14f = filter_to_sym_conv(filter_index_conv, c_filter[13], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[14].shape
    c15f = filter_to_sym_conv(filter_index_conv, c_filter[14], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[15].shape
    c16f = filter_to_sym_conv(filter_index_conv, c_filter[15], patch_size, ic, n, patch_stride, False)
    n,ic,kw,kh = c_filter[16].shape
    c17f = filter_to_sym_conv(filter_index_conv, c_filter[16], patch_size, ic, n, patch_stride, False)
   
    ff = filter_to_sym_fc(filter_index_fc, net.fc.weight.data.clone(), patch_size, patch_stride,0)

    save_lut(conv_lut.reshape(-1,1), 'conv_lut.txt')
    save_lut(fc_lut.reshape(-1,1), 'fc_lut.txt')
    save_lut(add_lut.reshape(-1,1), 'add_lut.txt')
    save_lut(relu_lut.reshape(-1,1), 'relu_lut.txt')
    save_lut(c1f.reshape(-1,1), 'c1f.txt')
    save_lut(c2f.reshape(-1,1), 'c2f.txt')
    save_lut(c3f.reshape(-1,1), 'c3f.txt')
    save_lut(c4f.reshape(-1,1), 'c4f.txt')
    save_lut(c5f.reshape(-1,1), 'c5f.txt')
    save_lut(c6f.reshape(-1,1), 'c6f.txt')
    save_lut(c7f.reshape(-1,1), 'c7f.txt')
    save_lut(c8f.reshape(-1,1), 'c8f.txt')
    save_lut(c9f.reshape(-1,1), 'c9f.txt')
    save_lut(c10f.reshape(-1,1), 'c10f.txt')
    save_lut(c11f.reshape(-1,1), 'c11f.txt')
    save_lut(c12f.reshape(-1,1), 'c12f.txt')
    save_lut(c13f.reshape(-1,1), 'c13f.txt')
    save_lut(c14f.reshape(-1,1), 'c14f.txt')
    save_lut(c15f.reshape(-1,1), 'c15f.txt')
    save_lut(c16f.reshape(-1,1), 'c16f.txt')
    save_lut(c17f.reshape(-1,1), 'c17f.txt')
    save_lut(ff.reshape(-1,1), 'ff.txt')
    save_float_to_txt(centroid_lut.reshape(-1,1), 'centroid_lut.txt')
    print("Everything completed successfully")
