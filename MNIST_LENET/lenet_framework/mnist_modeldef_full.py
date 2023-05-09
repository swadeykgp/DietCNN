# Copied from https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18
import resource
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
import sys
sys.path.insert(1, '../core')
import random
import time
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import warnings
warnings.filterwarnings('ignore')
from patchlib import *
import faiss
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 32
np.set_printoptions(threshold=sys.maxsize)

__all__ = ['CNN_LeNet', 'CNN_LeNetStd', 'CNN_LeNetSym', 'lenet_sym', 'lenet_std', 'lenet']

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
        
# Standard Convolution

def simple_mult(A, B):
    counter = 0
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print("Cannot multiply the two matrices. Incorrect dimensions.")
      return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
                counter +=1
    return C,counter

class CNN_LeNetStd(nn.Module):
    def __init__(self, weights1, biases1, weights2, biases2, weights3,weights4, instr):
        super(CNN_LeNetStd, self).__init__()
        # Define the net structure
        self.weights1 = weights1 
        self.biases1 = biases1
        self.weights2 = weights2 
        self.biases2 = biases2
        self.f1_weights = weights3 
        self.f2_weights = weights4 
        # This is the input layer first Convolution
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
            
        self.mult_count_c1 = 0
        self.mult_overhead_c1 = 0
        self.mult_count_c2 = 0
        self.mult_overhead_c2 = 0
        self.mult_count_f1 = 0
        self.mult_overhead_f1 = 0
        self.mult_count_f2 = 0
        self.mult_overhead_f2 = 0
        self.instr = instr
     
    def update_c1(self, mc, mo):
        self.mult_overhead_c1 += mo
        self.mult_overhead_c1 /=2  
        self.mult_count_c1 = mc

    def update_c2(self, mc, mo):
        self.mult_overhead_c2 += mo
        self.mult_overhead_c2 /=2  
        self.mult_count_c2 = mc
    
    def update_f1(self, mc, mo):
        self.mult_overhead_f1 += mo
        self.mult_overhead_f1 /=2  
        self.mult_count_f1 = mc

    def update_f2(self, mc, mo):
        self.mult_overhead_f2 += mo
        self.mult_overhead_f2 /=2  
        self.mult_count_f2 = mc

    def StdConv2D(self, x,in_channels, out_channels, weights, biases, kernel_size, stride, padding):
       #x = F.pad(input=x, pad=(1, 1, 1, 1), mode='constant', value=0) 
    
       x = x.squeeze()
       if in_channels == 1:
           img  = np.asarray(x.detach()) 
           im_w,im_h= img.shape      
       else: 
           img  = np.asarray(x.detach().permute(1,2,0)) 
           im_w,im_h,_= img.shape       
    
       symbol_array =  extract_patches(img, (kernel_size,kernel_size), stride)
       w,h = symbol_array.shape
       symbol_array = symbol_array.reshape(w//in_channels,h*in_channels)
    
       #elapsed = (time.process_time() - start)
       #print('Patch extraction time:',elapsed) 
       #start = time.process_time()
       weights  = np.asarray(weights.permute(0,2,3,1)) 
    
       # Only the convolution loop will be measured 
       overhead = 0
       counts = 0 
       for k in range(out_channels): 
           kt = weights[k].squeeze()
           kt = kt.reshape(-1,1)
           #print(symbol_array)
           #print(kt)
           #print(conv_lut) 
           # Convolve symbolic input and symbolic kernel - only partial precalculated conv2d
           #print(" kernel shape:" , kt.shape)
           #print(" symbol shape:" , symbol_array.shape)
           #cpart1_array = symbol_array.dot(kt)
           start = time.process_time()
           cpart1_array,tmpcounts = simple_mult(symbol_array, kt)
           counts += tmpcounts
           cpart1_array1 = np.asarray(cpart1_array)  
           #cpart1_array = np.multiply(symbol_array, np.transpose(kt))
             
           #print("shape of conv ", cpart1_array.shape) 
           #print(" conv ", cpart1_array) 
    
           # The ADD part of convolution
           #all_center_patches = centroid_lut[cpart1_array]
           #all_sums = all_center_patches.sum(axis=1)
           cs_wins, cs_syms = cpart1_array1.shape
           #print(cpart1_array.shape) 
           all_sums = np.zeros((cs_wins,),dtype=np.int16)
           #all_sums = cpart1_array1.sum(axis=1)
           #for i in range(cpart1_array1.shape[0]):
           #    all_sums+=cpart1_array1
           for i in range(cs_wins):
               tmp_sym = cpart1_array1[i][0]
               for j in range(1,cs_syms):
                   tmp_sym += cpart1_array1[i][j]
               all_sums[i] = tmp_sym
           #all_sums = cpart1_array.sum(axis=1)
           #print("shape of sums ", all_sums.shape) 
           #print("bias  ", biases[k]) 
           #all_sums = all_sums + biases[k]
           #print(" sums ", all_sums) 
    
           elapsed = time.process_time() - start
           overhead += elapsed
           # Now convert this to symbols
           #_, all_syms_sym = index.search(all_sums.astype(np.float32), 1)
           if k ==0:
               #temp_img = np.expand_dims(all_syms_sym, axis=1)
               temp_img = np.expand_dims(all_sums, axis=1)
               output_fm = temp_img
           else:
               #temp_img = np.expand_dims(all_syms_sym, axis=1)
               temp_img = np.expand_dims(all_sums, axis=1)
               output_fm = np.concatenate((output_fm, temp_img), axis=1)
               #print("Symbolic FM shape:" , output_fm.shape)  
       output_fm = output_fm.squeeze()
       out_w = (im_w - kernel_size)//stride + 1
       out_h = (im_h - kernel_size)//stride + 1
       output_fm = output_fm.reshape(out_w, out_h, out_channels)
      
       #print(f'elapsed cycles standard convolution: {elapsed}')
    
       overhead *= 1000
       if in_channels == 1:
           self.update_c1(counts, overhead)
       else:
           self.update_c2(counts, overhead)
    
       #elapsed = (time.process_time() - start)
       #print('Convolution operation time:',elapsed)
       # Convert to torch and reshape as filter
       output_fm_t = torch.from_numpy(output_fm)
       output_fm_t = output_fm_t.permute(2,0,1)
       output_fm_t = output_fm_t.unsqueeze(0) 
       #print("output fm shape:", output_fm_t.shape)
       return output_fm_t

    def StdFC(self,x, weights):
       weights = torch.transpose(weights, 0, 1) 
       start = time.process_time()
       cpart1_array,counts = simple_mult(x, weights)
       cpart1_array = np.asarray(cpart1_array)  
       overhead = time.process_time() - start
       overhead *= 1000
       if x.shape[1] == 256:
           self.update_f1(counts, overhead)
       else:
           self.update_f2(counts, overhead)
       output_fm_t = torch.from_numpy(cpart1_array)
       return output_fm_t
    def forward_std_helper(self, x):
        # First handle the batch dimension
        x = self.StdConv2D(x,1, 6, self.weights1, self.biases1, 5, 1, 0)
        x = F.relu(x)
        x = x.float()
        x = self.pool2(x)
        x = self.StdConv2D(x,6, 16, self.weights2, self.biases2, 5, 1, 0)
        x = F.relu(x)
        x = x.float()
        x = self.pool2(x)
        x = x.reshape(1,400) 
        x = x.view(-1, 400)
        x = F.relu(self.StdFC(x, self.f1_weights))
        x = F.relu(self.StdFC(x, self.f2_weights))
        if (self.instr):
                #print('Maximum memory usage: %s Mb' % ((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024))
           print("Multiplication count C1,", self.mult_count_c1,"Multiplication count C2,", self.mult_count_c2,
                 #"Multiplication count F1,", self.mult_count_f1,"Multiplication count F2,", self.mult_count_f2,
                 "Multiplication overhead c1,", self.mult_overhead_c1,"Multiplication overhead c2,", self.mult_overhead_c2) 
                 #"Multiplication overhead F1,", self.mult_overhead_f1,"Multiplication overhead f2,", self.mult_overhead_f2)
           ##      "Pool overhead1,",po1,"Pool overhead2,",po2)  
           #print(" Total MC,", self.mult_count_c1+self.mult_count_c2+self.mult_count_f1+self.mult_count_f2,
           #      "Total overhead,",self.mult_overhead_c1+self.mult_overhead_c2+self.mult_overhead_f1+self.mult_overhead_f2)
           print(" Total MC,", self.mult_count_c1+self.mult_count_c2,
                 "Total overhead,",self.mult_overhead_c1+self.mult_overhead_c2)
        x = np.asarray(x, dtype=np.float32) 
        #x = x.squeeze()
        return x

    def forward(self, x_bat): 
        # First handle the batch dimension
        bat, _, _, _  = x_bat.shape
        results = Parallel(n_jobs=PARALLEL)(delayed(self.forward_std_helper)(x_bat[i,:,:,:].squeeze()) for i in range(bat))
        #results = []
        #for i in range(bat):
        #    results.append(forward_helper(i, x_bat[i,:,:,:].squeeze(),self.index, self.patch_size, self.patch_stride, self.weights, self.n_clusters, self.centroid_lut, self.conv_lut, self.add_lut, self.bias_lut, self.fc_lut, self.relu_lut ))
        #print(results)        
        x = np.asarray(results, dtype=np.float32) 
        #print(x.shape)
        x = torch.from_numpy(x)   
        #x = x.unsqueeze(0) 
        #x = x.reshape(bat,-1)
        x = x.float() 
        #x = self.fc1(x) #conv only
        #x = F.relu(x) #conv only
        #x = self.fc2(x) #conv only
        #x = F.relu(x) #conv only
        x = self.fc3(x)
        #print(x.shape)
        x = F.softmax(x,dim=1)
        #print(x.shape)
        return x



# Symbolic convolution
def discretize(x,in_channels, index,  patch_size, patch_stride,padding): 
   #x = F.pad(input=x, pad=(padding, padding, padding, padding), mode='constant', value=0) 

   #start = time.process_time()
   #print(x)
   x = x.squeeze()
   if in_channels == 1:
       #img  = np.asarray(x.clone().detach()) 
       img  = np.asarray(x.clone()) 
       im_w,im_h= img.shape      
   else: 
       #img  = np.asarray(x.clone().detach().permute(1,2,0)) 
       img  = np.asarray(x.clone().permute(1,2,0)) 
       im_w,im_h,_= img.shape       
   #print(im_w,im_h)
   #start = time.process_time()
   img_sym =  image_to_symbolic(img, index, patch_size, patch_stride)  
   #overhead = time.process_time() - start
   #overhead *= 1000
   simg_out_w = (im_w - patch_size[0])//patch_stride + 1
   simg_out_h = (im_h - patch_size[0])//patch_stride + 1
   if in_channels == 1:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h)
   else:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h, in_channels)
   
   return img_sym

def SymConv2D(x,in_channels, out_channels, weights, biases, kernel_size, stride, 
                 padding, # convolution parameters
                 n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params
                 conv_lut, add_lut, bias_lut, sym_kernel,sym_stride):
   if in_channels == 1:
       img_sym  = np.asarray(x) 
       im_w,im_h= img_sym.shape      
   else: 
       img_sym = x.squeeze()
       #img_sym  = np.asarray(img_sym.detach().permute(1,2,0)) 
       im_w,im_h,dc= img_sym.shape       
   #print("Symbolic image shapes:" , img_sym.shape, sym_kernel, sym_stride)
   symbol_array =  extract_patches(img_sym, (sym_kernel,sym_kernel), sym_stride)
   w,h = symbol_array.shape
   symbol_array = symbol_array.reshape(w//in_channels,h*in_channels)
   #print("Symbolic shapes:" , img_sym.shape, sym_kernel, sym_stride,w,h)
   
   #elapsed = (time.process_time() - start)
   #print('Patch extraction time:',elapsed) 
   #start = time.process_time()
   # Only the convolution loop will be measured 
   for k in range(out_channels): 
       kt = weights[:,k].squeeze()
       kt = kt.astype(int)
       #print(conv_lut) 
       #print("Symbolic kernel shape:" , kt.shape)
       # Convolve symbolic input and symbolic kernel - only partial precalculated conv2d
       start = time.process_time()
       cpart1_array1 = conv_lut[symbol_array,kt]
       ############ SYM ADD #########################
       cs_wins, cs_syms = cpart1_array1.shape
       #print(cpart1_array.shape) 
       c_out_buffer = np.zeros((cs_wins,),dtype=np.int16)
       for i in range(cs_wins):
           # sort this array by the values of the symbols, just to enforce an associativity order
           cpart1_array = cpart1_array1[i]
           #print(cpart1_array)
           cpart1_array = np.sort(cpart1_array) 
           #print(cpart1_array)
           tmp_sym = cpart1_array[0]
           for j in range(1,cs_syms):
               tmp_sym = add_lut[cpart1_array[j], tmp_sym]
               if j < (cs_syms -2):
                   if tmp_sym > cpart1_array[j+1]:
                       # problem! order is broken
                       tmp = tmp_sym
                       tmp_sym = cpart1_array[j+1]
                       cpart1_array[j+1] = tmp
                       cpart1_array = np.sort(cpart1_array)
                       j +=1
           c_out_buffer[i] = bias_lut[tmp_sym,k]
           #c_out_buffer[i] = tmp_sym
       ############ SYM ADD #########################
       if k ==0:
           temp_img = np.expand_dims(c_out_buffer, axis=1)
           output_fm = temp_img
       else:
           temp_img = np.expand_dims(c_out_buffer, axis=1)
           output_fm = np.concatenate((output_fm, temp_img), axis=1)
           #print("output fm shape:", output_fm.shape)
   output_fm = output_fm.squeeze()
   out_w = (im_w - sym_kernel)//sym_stride + 1
   out_h = (im_h - sym_kernel)//sym_stride + 1
   
   output_fm = output_fm.reshape(out_w, out_h, out_channels)
   return output_fm

def SymFC(x, weights, fc_lut, add_lut, bias_lut):
    cpart1_array1 = fc_lut[x,weights]
    # replace this by multiplication
    #outk, ink = weights.shape
    #cpart1_array1 = np.zeros((outk,ink),dtype=np.int16) 
    #for i in range(outk):
    #    cpart1_array1[i]=fc_lut[x,weights[i]]  

    #np.assert_equal(cpart1_array1,cpart1_array)

    ############ SYM ADD #########################
 
    cs_wins, cs_syms = cpart1_array1.shape
    #print(cpart1_array.shape) 
    c_out_buffer = np.zeros((1,cs_wins),dtype=np.int16)
    for i in range(cs_wins):
        # sort this array by the values of the symbols, just to enforce an associativity order
        cpart1_array = cpart1_array1[i]
        cpart1_array = np.sort(cpart1_array) 
        tmp_sym = cpart1_array[0]
        for j in range(1,cs_syms):
            tmp_sym = add_lut[cpart1_array[j], tmp_sym]
            #if j < (cs_syms -2):
            #    if tmp_sym > cpart1_array[j+1]:
            #        # problem! order is broken
            #        tmp = tmp_sym
            #        tmp_sym = cpart1_array[j+1]
            #        cpart1_array[j+1] = tmp
            #        cpart1_array = np.sort(cpart1_array)
            #        j +=1
        #c_out_buffer[i] = bias_lut[tmp_sym,k]
        #c_out_buffer[0][i] = tmp_sym
        c_out_buffer[0][i] = bias_lut[tmp_sym,i]
        ############ SYM ADD #########################
    return c_out_buffer
    
def SymReLU(x, relu_lut):
    ts = x.shape
    if len(ts) > 2:
        im_w, im_h, c = x.shape
    elif len(ts) > 1:
        im_w, im_h = x.shape
    else:
        pass

    x = np.reshape(x, (len(x), -1))
    if len(ts) > 2:
        return(relu_lut[x].reshape(im_w, im_h, c))
    elif len(ts) > 1:
        return(relu_lut[x].reshape(im_w, im_h))
    else:
        return relu_lut[x].reshape(1,-1)

def forward_helper(item, x, index, patch_size, patch_stride, weights, n_clusters, centroid_lut, conv_lut, add_lut, bias_lut, fc_lut, relu_lut): 
    try:
        # First handle the batch dimension
        #print(x.shape)
        x = discretize(x,1, index,  patch_size, patch_stride,0)
        #print(x.shape)
        x = SymConv2D(x,1, 6, weights[0], None, 5, 1, 1, # convolution parameters
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, bias_lut[0], 5, 2) 
        #print(x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,6, 16, weights[1], None, 5, 1, 1, # convolution parameters
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, bias_lut[1], 5, 2) 
        #print(x.shape)
        x = SymReLU(x, relu_lut)
        #print(x.shape)
        #x = symbolic_to_image_win(x, 5, 5, 16, centroid_lut,patch_size) # conv only
        x = symbolic_to_symbolic_win(x, 5, 5, 16, centroid_lut,patch_size)
        #print(x)
        x = torch.from_numpy(x)
        x = x.permute(2,0,1)
        x = x.unsqueeze(0)
        x = x.reshape(1, 5*5*16) 
        x = np.asarray(x, dtype=np.int16) 
        #print(x)
        #x = x.reshape(400,) # conv only
        #x = np.asarray(x, dtype=np.float32) #conv only
        x = SymFC(x, weights[2], fc_lut, add_lut,  bias_lut[2])
        #print(x.shape)
        x = SymReLU(x, relu_lut)
        x = SymFC(x, weights[3], fc_lut, add_lut, bias_lut[3])
        #print(x.shape)
        x = SymReLU(x, relu_lut)
        #print(x)
        #x = SymFC(x, weights[4], fc_lut, add_lut, bias_lut[3])
        x = centroid_lut[x[0]]
        #print(x)
        #x = centroid_lut[x]
        x = np.transpose(x)
        x = x.squeeze()
        #print(x.shape)
        #x = centroid_lut[x]
        #x = x.reshape(1,-1) 
        #x = torch.from_numpy(x)
        return x
    except:
        print('error with item:', item)



class CNN_LeNetSym(nn.Module):
    def __init__(self, symp , n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut, f2_bias_lut, f3_bias_lut, relu_lut, instr ):
        super(CNN_LeNetSym, self).__init__()
        # Define the net structure
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.bias_lut = [] 
        self.bias_lut.append(c1_bias_lut)
        self.bias_lut.append(c2_bias_lut)
        self.bias_lut.append(f1_bias_lut)
        self.bias_lut.append(f2_bias_lut)
        self.relu_lut =  relu_lut
        self.n_clusters, self.index, self.centroid_lut = n_clusters, index, centroid_lut
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.weights = symp


        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
                
        self.lut_overhead_l1 = 0
        self.disc_overhead_l1 = 0
        self.lut_overhead_l2 = 0
        self.disc_overhead_l2 = 0
        self.instr = instr
        self.lut_overhead_c1 = 0
        self.lut_overhead_c2 = 0
        self.lut_overhead_f1 = 0
        self.lut_overhead_f2 = 0
        self.lc_c1 = 0
        self.lc_c2 = 0
        self.lc_f1 = 0
        self.lc_f2 = 0


    def update_sym(self,  bs, bs2 , bs3 , bs4, bs5, conv_lut, fc_lut, add_lut,  relu_lut):
        self.c1_weights = bs 
        self.c2_weights = bs2
        self.f1_weights = bs3
        self.f2_weights = bs4
        self.f3_weights = bs5
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.relu_lut =  relu_lut

    def update_c1(self, lc, lo):
        self.lut_overhead_c1 += lo
        self.lut_overhead_c1 /=2  
        self.lc_c1 = lc

    def update_c2(self, lc, lo):
        self.lut_overhead_c2 += lo
        self.lut_overhead_c2 /=2  
        self.lc_c2 = lc   

    def forward(self, x_bat): 
        # First handle the batch dimension
        bat, _, _, _  = x_bat.shape
        results = Parallel(n_jobs=PARALLEL)(delayed(forward_helper)(i, x_bat[i,:,:,:].squeeze(),self.index, self.patch_size, self.patch_stride, self.weights, self.n_clusters, self.centroid_lut, self.conv_lut, self.add_lut, self.bias_lut, self.fc_lut, self.relu_lut) for i in range(bat))
        #results = []
        #for i in range(bat):
        #    results.append(forward_helper(i, x_bat[i,:,:,:].squeeze(),self.index, self.patch_size, self.patch_stride, self.weights, self.n_clusters, self.centroid_lut, self.conv_lut, self.add_lut, self.bias_lut, self.fc_lut, self.relu_lut ))
        x = np.asarray(results, dtype=np.float32) 
        #print(x.shape)
        x = torch.from_numpy(x)   
        #x = x.unsqueeze(0) 
        #x = x.reshape(bat,-1)
        x = x.float() 
        #x = self.fc1(x) #conv only
        #x = F.relu(x) #conv only
        #x = self.fc2(x) #conv only
        #x = F.relu(x) #conv only
        x = self.fc3(x)
        #print(x.shape)
        x = F.softmax(x,dim=1)
        #print(x.shape)
        return x

def lenet(instr) -> CNN_LeNet:
    pretrained_model = "./mnist_v0.pt"
    net = CNN_LeNet() 
    net.load_state_dict(torch.load(pretrained_model))
    num_class = 10
    net.eval()
    return net

def lenet_std(instr) -> CNN_LeNetStd:
    pretrained_model = "./mnist_v0.pt"
    state_dict = torch.load("./mnist_v0.pt") 
    net = CNN_LeNet() 
    net.load_state_dict(state_dict)
    c1_filter = net.conv1.weight.data.clone() 
    c1_bias = None 
    
    c2_filter = net.conv2.weight.data.clone() 
    c2_bias = None

    f1_filter = net.fc1.weight.data.clone() 
    f2_filter = net.fc2.weight.data.clone() 
    model = CNN_LeNetStd(c1_filter, c1_bias,c2_filter, c2_bias, f1_filter, f2_filter, instr)
    model.load_state_dict(state_dict)
    return model

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
        #print(data.shape) 
        data = np.reshape(data, (len(data), -1))
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
    
def lenet_sym(net, state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b, f2b, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride, instr) -> CNN_LeNetSym:
    sym_params = [] # These are layer 1 conv params
    with torch.no_grad():
        c1_bias = net.conv1.bias.clone()
        c1_bias = np.asarray(c1_bias)
    c1_filter = net.conv1.weight.data.clone() 
    n,ic,kw,kh = c1_filter.shape 
    filter_patch_size = (1,1)
    filter_stride = 1
    bs = filter_to_sym_conv(filter_index_conv, c1_filter, filter_patch_size, ic, n, filter_stride, False)
    sym_params.append(bs) 
    #sym_params.append(c1_bias) 

    #print('symbols in filter',len(c1_symbolic_flt[0][0]))
      

    with torch.no_grad():
        c2_bias = net.conv2.bias.clone()
        c2_bias = np.asarray(c2_bias)
    c2_filter = net.conv2.weight.data.clone() 
    n2,ic2,kw2,kh2 = c2_filter.shape 
    bs2 = filter_to_sym_conv(filter_index_conv, c2_filter, filter_patch_size, ic2, n2, filter_stride, False)
    sym_params.append(bs2) 
    #sym_params.append(c2_bias) 
 

    with torch.no_grad():
        f1_bias = net.fc1.bias.clone()
        f1_bias = np.asarray(f1_bias)
        f2_bias = net.fc2.bias.clone()
        f2_bias = np.asarray(f2_bias)
    f1_filter = net.fc1.weight.data.clone() 
    kw3,kh3 = f1_filter.shape 
    bs3 = filter_to_sym_fc(filter_index_fc,f1_filter, filter_patch_size, filter_stride, False)
    sym_params.append(bs3) 
    #sym_params.append(f1_bias) 

    f2_filter = net.fc2.weight.data.clone() 
    bs4 = filter_to_sym_fc(filter_index_fc,f2_filter, patch_size, patch_stride, False)
    sym_params.append(bs4) 
    #sym_params.append(f2_bias) 

    f3_filter = net.fc3.weight.data.clone() 
    bs5 = filter_to_sym_fc(filter_index_fc,f3_filter, patch_size, patch_stride, False)
    sym_params.append(bs5) 


    model = CNN_LeNetSym(sym_params, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b,f2b, None, relu_lut, instr) 

    model.load_state_dict(state_dict)
    #model.load_state_dict(fms_orignet)
    return model
