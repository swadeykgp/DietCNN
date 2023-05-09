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
#from mnist_modeldef import *
from mnist_modeldef_conv import *
from  patchlib import *
import faiss 
       
import os
import numpy
import random
import time

if (len(sys.argv) - 1) < 2:
    print ("Call with first argument 1: Standard inference, 2: DietCNN inference. Also specify a second argument as 1 for instrumentation ")
    sys.exit()

torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)

transform_test = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),
                                      transforms.Normalize((0.1309,), (0.2893,))])

testset = datasets.MNIST(root='../../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

transform_test_sym = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),
                                      transforms.Normalize((0.1309,), (0.2893,))])

testset_sym = datasets.MNIST(root='../../dataset', train=False, download=True, transform=transform_test_sym)
testloader_sym = torch.utils.data.DataLoader(testset_sym, batch_size=64, shuffle=False)

#index = faiss.read_index("./kmeans_mnist_fullnet_k2_s2_c512_faiss.index")
index = faiss.read_index("./kmeans_mnist_fullnet_k1_s1_c128_faiss_v10.index")
#n_clusters=512
n_clusters=128
patch_size = (1, 1)
patch_stride = 1
centroid_lut = index.reconstruct_n(0, n_clusters)

pretrained_model = "./mnist_bl.pt"

net = CNN_LeNet()
net.load_state_dict(torch.load(pretrained_model))
state_dict = torch.load(pretrained_model)

c1_filter = net.conv1.weight.data.clone()
c1_bias = None
c2_filter = net.conv2.weight.data.clone()
c2_bias = None
#pad =  nn.ConstantPad2d((0, 1, 0, 1), 0)
#c1_filter = pad(c1_filter)
#c2_filter = pad(c2_filter)
# First pool each kernel, this is experimental
#pooling = nn.MaxPool2d(kernel_size=2,stride=2)
#pooling = nn.AvgPool2d(kernel_size=2,stride=2)
#c1_filter = pooling(c1_filter)
#c2_filter = pooling(c2_filter)

f1_filter = net.fc1.weight.data.clone()
f2_filter = net.fc2.weight.data.clone()
f3_filter = net.fc3.weight.data.clone()

import pickle
conv_patch_size = (1, 1)
n_cluster_conv_filters = 64
n_cluster_fc_filters = 128
#n_cluster_fc_filters = 256
conv_stride = 1
with open("mnist_conv_flt.index", "rb") as f:
    filter_index_conv = pickle.load(f)
with open("mnist_fc_flt.index", "rb") as f:
    filter_index_fc = pickle.load(f)
fc_lut = np.genfromtxt('./mnist_fc_lut.txt', delimiter=',',dtype=np.int16) 
conv_lut = np.genfromtxt('./mnist_conv_lut.txt', delimiter=',',dtype=np.int16) 
add_lut = np.genfromtxt('./mnist_add_lut.txt', delimiter=',',dtype=np.int16) 
relu_lut = np.genfromtxt('./mnist_relu_lut.txt', delimiter=',',dtype=np.int16)

print(" Symbolic model loading started...")
t = time.process_time()
if sys.argv[2] == "1":
    netsym = lenet_sym(net,state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, None,None, None,None,relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride, True)
else:
    netsym = lenet_sym(net,state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, None,None, None,None,relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride, False)

elapsed_time3 = time.process_time() - t
print("Symbolic model loading completed in:",elapsed_time3)
netsym.eval()

print(" Standard model loading started...")
t = time.process_time()
if sys.argv[2] == "1":
    netstd= lenet_std(True)
else:
    netstd= lenet_std(False)
elapsed_time3 = time.process_time() - t
print("Standard std model loading completed in:",elapsed_time3)
netstd.eval()

HOWDY = 20000000 

# Test accuracy of symbolic inference
def test_fullsym_acc(model, atk,  data_iter, clamp, top_5, std):
    correct = 0 
    total = 0 
    counter = 0
    model.eval()
    for data in data_iter:
        X, y = data
        if counter > HOWDY:
            break
        counter +=1 
        if(clamp):
            X = softclamp01(X)
        if atk:
            X_atk = atk(X, y)
            X_sym = X_atk.data.cpu().numpy().copy()
        else:
            X_atk = X
            X_sym = X  

        output = model.forward(X)
        if not top_5:
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                #if True:
                    correct += 1
        else:
           top5op = output.detach().numpy()
           for idx, i in enumerate(top5op):
               ind = np.argpartition(i, -5)[-5:]
               for j in ind:
                   flag = 0
                   if j == y[idx]:
                       correct += 1
            #else:
            #    # Whenever there is an error, print the image
            #    print("Test Image #: {}".format(total+1))
            #    print("Mispredicted label: {}".format(torch.argmax(i)))
        #total += 1
        total += 64
        if std: 
            if not top_5:
                if(counter % 2 == 0):
                    print("Full standard model test accuracy :{}% ".format(100*round(correct/total, 4)))
            else:
                if(counter % 2 == 0):
                    print("Full standard  model test accuracy Top-5 :{}% ".format(100*round(correct/total, 4)))
        else:
            if not top_5:
                if(counter % 2 == 0):
                    print("Full symbolic model test accuracy DietCNN :{}% ".format(100*round(correct/total, 4)))
            else:
                if(counter % 2 == 0):
                    print("Full symbolic model test accuracy Top-5 DietCNN :{}% ".format(100*round(correct/total, 4)))
        #break 
    return round(correct/total, 4)


import torchvision.utils
from torchvision import models
#import torchattacks
#from torchattacks import *
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
#print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

#atks = [ 
#    #TIFGSM(net, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5),
#    CW(net, c=1, lr=0.01, steps=100, kappa=0),
#    #AutoAttack(net, norm='Linf', eps=8/255, version='plus'),
#    AutoAttack(net, eps=2/255, n_classes=10, version='standard'), # take this at last if time permits
#    AutoAttack(net, eps=4/255, n_classes=10, version='standard'), # take this at last if time permits
#    AutoAttack(net, eps=8/255, n_classes=10, version='standard') # take this at last if time permits
#    #DIFGSM(net, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
#    #MIFGSM(net, eps=8/255, alpha=2/255, steps=100, decay=0.1),
#    #RFGSM(net, eps=8/255, alpha=2/255, steps=100),
#    #EOTPGD(net, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
#    #APGD(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
#    #APGD(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
#    #APGDT(net, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
#    #Jitter(net, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
#    #FAB(net, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
#    #FAB(net, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
#    #Square(net, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
#    #DeepFool(net, steps=100)
#]


# This is how we call the test routine :
#
# test_fullsym_acc(model name, ignore this,  the test data loader , ignore this , whether we want top_5 or top_1 , NA for MNIST, standard or symbolic inference, instrument the counts):

import time
if sys.argv[1] == "1":
    start = time.process_time()  
    start_t = time.time()  
    acc = test_fullsym_acc(netstd, None, testloader, False, False,True)
    elapsed = time.process_time() - start
    elapsed = elapsed*1000
    #print("elapsed process time for standard inference",elapsed)  
    elapsed = time.process_time() - start
    end = time.time()
    #print("elapsed time for standard inference:", end - start_t) 
else:
    start = time.process_time()  
    start_t = time.time()  
    acc = test_fullsym_acc(netsym, None, testloader_sym, False, False, False)
    elapsed = time.process_time() - start
    elapsed = elapsed*1000
    #print("elapsed process time for symbolic inference",elapsed)  
    end = time.time()
    #print("elapsed time for symbolic inference:", end - start_t) 






