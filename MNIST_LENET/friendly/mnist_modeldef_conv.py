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
np.set_printoptions(threshold=sys.maxsize)




#        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
#        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
#        self.fc1 = nn.Linear(400,120)
#        self.fc2 = nn.Linear(120,84)
#        self.fc3 = nn.Linear(84, 10) 
INSTR = True
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



from hwcounter import Timer, count, count_end

__all__ = ['CNN_LeNet', 'CNN_LeNetStd', 'CNN_LeNetSym', 'lenet_sym', 'lenet_std', 'lenet']

class CNN_LeNet(nn.Module):
    def __init__(self):
        super(CNN_LeNet, self).__init__()
        # Define the net structure
        # This is the input layer first Convolution
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, bias=False) #12x12x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, bias=False) #4x4x16
        self.fc1 = nn.Linear(256,128, bias=False)
        self.fc2 = nn.Linear(128,64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False) 
    
    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x

        
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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, bias=False) #12x12x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, bias=False) #4x4x16
        self.fc1 = nn.Linear(256,128, bias=False)
        self.fc2 = nn.Linear(128,64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False) 
    
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
       #print("Symbolic image shape:" , x.shape)
       #print("weights shape:" , weights.shape)
       weights = torch.transpose(weights, 0, 1) 
       #print("Symbolic image shape:" , x.shape)
       #print("weights shape:" , weights.shape)
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
       #output_fm_t = output_fm_t.unsqueeze(0) 
       #print("output fm shape:", output_fm_t.shape)
    def forward(self, x_bat):
        # First handle the batch dimension
        bat, _, _, _  = x_bat.shape
        for b in range(bat):
            x = x_bat[b,:,:,:]
            x = x.squeeze()


            #x = F.relu(self.conv1(x))
            #print(x) 
            x = self.StdConv2D(x,1, 8, self.weights1, self.biases1, 5, 2, 0)
            #print(x) 
            #print(x.shape) 
            x = x.float()
            x = F.relu(x)
            #print(x) 
            #print(x.shape) 
            x = self.StdConv2D(x,8, 16, self.weights2, self.biases2, 5, 2, 0)
            #print(x) 
            x = x.float()
            #x = self.conv2(x)
            x = F.relu(x)
            #print(x) 
            #x = symbolic_to_image_win(x, 4, 4, 16, self.centroid_lut,self.patch_size)
            start = time.process_time()
            #print(x.shape) 
            x = x.reshape(1,256) 
            x = x.view(-1, 256)
            x = self.fc1(x)
            #x = self.StdFC(x,self.f1_weights)
            #print(x) 
            x = F.relu(x)
            #print(x) 
            #x = self.StdFC(x,self.f2_weights)
            #print(x) 
            x = self.fc2(x)
            x = F.relu(x)
            #print(x) 
            x = self.fc3(x)
            #print(x) 
            x = F.softmax(x,dim=1)
            if b ==0: 
                x_out_bat = x
                #print(x_out_bat.shape)
            else:
                x_out_bat = torch.cat((x_out_bat,x), dim=0) 
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
                
        return x_out_bat

class CNN_LeNetSym(nn.Module):
    def __init__(self, symp , n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut, f2_bias_lut, relu_lut, instr ):
        super(CNN_LeNetSym, self).__init__()
        # Define the net structure
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.c1_bias_lut = c1_bias_lut 
        self.c2_bias_lut = c2_bias_lut 
        self.f1_bias_lut = f1_bias_lut 
        self.f2_bias_lut = f2_bias_lut 
        self.relu_lut =  relu_lut
        self.n_clusters, self.index, self.centroid_lut = n_clusters, index, centroid_lut
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.c1_weights = symp[0]
        self.c1_biases = symp[1]    
        self.c2_weights = symp[2]
        self.c2_biases = symp[3]   
        self.f1_weights = symp[4]
        self.f1_biases = symp[5]   
        self.f2_weights = symp[6]
        self.f2_biases = symp[7]   
        self.f3_weights = symp[8] 
        # This is the input layer first Convolution
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, bias=False) #12x12x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, bias=False) #4x4x16
        self.fc1 = nn.Linear(256,128, bias=False)
        self.fc2 = nn.Linear(128,64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)
        
        self.lut_overhead_l1 = 0
        self.disc_overhead_l1 = 0
        self.lut_overhead_l2 = 0
        self.disc_overhead_l2 = 0

        self.lut_overhead_c1 = 0
        self.lut_overhead_c2 = 0
        self.lut_overhead_f1 = 0
        self.lut_overhead_f2 = 0
        self.lc_c1 = 0
        self.lc_c2 = 0
        self.lc_f1 = 0
        self.lc_f2 = 0

     
        self.instr = instr

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
    
    def discretize(self,x,in_channels, index,  patch_size, patch_stride,padding): 
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
       start = time.process_time()
       img_sym =  image_to_symbolic(img, index, patch_size, patch_stride)  
       overhead = time.process_time() - start
       overhead *= 1000
       simg_out_w = (im_w - patch_size[0])//patch_stride + 1
       simg_out_h = (im_h - patch_size[0])//patch_stride + 1
       if in_channels == 1:
           img_sym = img_sym.reshape(simg_out_w, simg_out_h)
       else:
           img_sym = img_sym.reshape(simg_out_w, simg_out_h, in_channels)
       
       return img_sym

    def SymConv2D(self,x,in_channels, out_channels, weights, biases, kernel_size, stride, 
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
       # Only the convolution loop will be measured 
       overhead = 0
       counts = 0 
       for k in range(out_channels): 
           kt = weights[:,k].squeeze()
           kt = kt.astype(int)
           #print(conv_lut) 
           #print("Symbolic kernel shape:" , kt.shape)
           # Convolve symbolic input and symbolic kernel - only partial precalculated conv2d
           start = time.process_time()
           cpart1_array1 = conv_lut[symbol_array,kt]
           # handling the add part
           #all_sums = all_sums + biases[k]
           # immediately discretize
           #print(cpart1_array)  
           #print(c part1_array.shape)  
           #c_out_buffer = self.centroid_lut[cpart1_array]

           ############# Cont ADD #########################
           ##print(c_out_buffer)  
           ##print(c_out_buffer.shape)  
           #all_sums = c_out_buffer.sum(axis=1)
           ##print(all_sums.shape)
           #all_sums = np.reshape(all_sums, (len(all_sums), -1)) 
           #_, c_out_buffer = index.search(all_sums.astype(np.float32), 1)
           ##print(c_out_buffer.shape)
           ############# Cont ADD #########################
           

           ############ SYM ADD #########################
           cs_wins, cs_syms = cpart1_array1.shape
           counts += cs_wins*cs_syms
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
                   #print(tmp_sym)
                   #print(cpart1_array[j])
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
               c_out_buffer[i] = tmp_sym

           ############ SYM ADD #########################
           elapsed = time.process_time() - start
           overhead += elapsed           
               
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
       overhead *= 1000
       if in_channels == 1:
           self.update_c1(counts, overhead)
       else:
           self.update_c2(counts, overhead) 
       output_fm = output_fm.reshape(out_w, out_h, out_channels)
       #output_fm = output_fm.reshape((im_w)//sym_stride, (im_h)//sym_stride, out_channels)
       #output_fm = output_fm.reshape((im_w-1)//sym_stride, (im_h-1)//sym_stride, out_channels)
       #output_fm = output_fm.reshape((im_w-2)//sym_stride, (im_h-2)//sym_stride, out_channels)
       return output_fm
       # Convert to torch and reshape as filter
       #output_fm_t = torch.from_numpy(output_fm)
       #output_fm_t = output_fm_t.permute(2,0,1)
       #output_fm_t = output_fm_t.unsqueeze(0) 
       #return output_fm_t
    def SymFC(self,x, weights, fc_lut, add_lut, bias_lut):
        #print("Symbolic image shape:" , x.shape)
        #print("weights shape:" , weights.shape)
        #print("Symbolic image shape:" , x.shape)
        #print("weights shape:" , weights.shape)
        start = time.process_time()
        cpart1_array1 = fc_lut[x,weights]
        #print(cpart1_array) 
        overhead = time.process_time() - start
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
                if j < (cs_syms -2):
                    if tmp_sym > cpart1_array[j+1]:
                        # problem! order is broken
                        tmp = tmp_sym
                        tmp_sym = cpart1_array[j+1]
                        cpart1_array[j+1] = tmp
                        cpart1_array = np.sort(cpart1_array)
                        j +=1
            #c_out_buffer[i] = bias_lut[tmp_sym,k]
            c_out_buffer[0][i] = tmp_sym



        ############ SYM ADD #########################

        return c_out_buffer

    def SymReLU(self,x):
        ts = x.shape
        if len(ts) > 2:
            im_w, im_h, c = x.shape
        elif len(ts) > 1:
            im_w, im_h = x.shape
        else:
            pass

        x = np.reshape(x, (len(x), -1))
        if len(ts) > 2:
            return(self.relu_lut[x].reshape(im_w, im_h, c))
        elif len(ts) > 1:
            return(self.relu_lut[x].reshape(im_w, im_h))
        else:
            return self.relu_lut[x].reshape(1,-1)

          
    def forward(self, x_bat):
        # First handle the batch dimension
        bat, _, _, _  = x_bat.shape
        for b in range(bat):
            x = x_bat[b,:,:,:]
            x = x.squeeze()
            x = self.discretize(x,1, self.index,  self.patch_size, self.patch_stride,0)
            #x = symbolic_to_image_win(x, 28, 28, 1, self.centroid_lut,self.patch_size)
            #x = self.conv1(x)
            x = self.SymConv2D(x,1, 8, self.c1_weights, self.c1_biases, 5, 2, 1, # convolution parameters
                self.n_clusters, self.index, self.centroid_lut, self.patch_size, self.patch_stride, # patch params 
                self.conv_lut, self.add_lut, self.c1_bias_lut, 5, 2)
            x = self.SymReLU(x)
            #print(x.shape)
            x = self.SymConv2D(x,8, 16, self.c2_weights, self.c2_biases, 5, 2, 1, # convolution parameters
                self.n_clusters, self.index, self.centroid_lut, self.patch_size, self.patch_stride, # patch params 
                self.conv_lut, self.add_lut, self.c2_bias_lut, 5, 2)
            x = self.SymReLU(x)
            x = symbolic_to_image_win(x, 4, 4, 16, self.centroid_lut,self.patch_size)
            #print("x")   
            #x = self.SymFC(x, self.f1_weights, self.fc_lut, self.add_lut,  self.f1_bias_lut)
            #x = self.SymReLU(x)
            #x = self.SymFC(x, self.f2_weights, self.fc_lut, self.add_lut, self.f2_bias_lut)
            #x = self.SymReLU(x)
            #x = self.SymFC(x, self.f3_weights, self.fc_lut, self.add_lut, self.f2_bias_lut)
            #print(x)
            #x = self.centroid_lut[x[0]]
            #x = x.reshape(1,-1)
            x = torch.from_numpy(x)
            x = x.permute(2,0,1)
            x = x.unsqueeze(0)
            x = x.reshape(1,256)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #print(x.shape)
            x = self.fc3(x)
            x = F.softmax(x,dim=1)
            #print(x.shape)
            if b ==0:
                x_out_bat = x
                #print(x_out_bat.shape)
            else:
                x_out_bat = torch.cat((x_out_bat,x), dim=0)
                #print(x_out_bat.shape)
        if self.instr: 
            print("LUT access count C1,", self.lc_c1, "LUT access count C2,", self.lc_c2, "LUT  overhead  C1,", self.lut_overhead_c1, "LUT overhead  C2,", self.lut_overhead_c2)  
                 
        return x_out_bat


def lenet() -> CNN_LeNet:
    pretrained_model = "./mnist_bl.pt"
    net = CNN_LeNet() 
    net.load_state_dict(torch.load(pretrained_model))
    num_class = 10
    net.eval()
    return net

def lenet_std(instr) -> CNN_LeNetStd:
    pretrained_model = "./mnist_bl.pt"
    state_dict = torch.load("./mnist_bl.pt") 
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
    #pooling = nn.MaxPool2d(kernel_size=2,stride=2) 
    #pad =  nn.ConstantPad2d((0, 1, 0, 1), 0)
    #pooling = nn.AvgPool2d(kernel_size=2,stride=2) 
    sym_params = [] # These are layer 1 conv params
    #net = CNN_LeNet() 
    #net.load_state_dict(state_dict)
    c1_filter = net.conv1.weight.data.clone() 
    #c1_filter = pad(c1_filter)    
    #c1_filter = pooling(c1_filter) 
    n,ic,kw,kh = c1_filter.shape
    print(c1_filter.dtype) 
    print(c1_filter.shape) 
    c1_bias = None 
    filter_patch_size = (1,1)
    filter_stride = 1
    #filter_patch_size = (2,2)
    #filter_stride = 2
    bs = filter_to_sym_conv(filter_index_conv, c1_filter, filter_patch_size, ic, n, filter_stride, False)
    print(" C1 filter shape :", bs.shape )  
    print(" C1 filter shape :", bs.dtype )  
    #print(" C1 filter orig shape :", c1_filter.shape )  
    #bs = bs.astype(int)
    #filter_lut = filter_index_conv.cluster_centers_
    #c1_renons = filter_lut[bs]
    #c1_renons = c1_renons.reshape(25,8) 
    #print(" C1 filter recons shape :", c1_renons.shape )  
    #
    #for i in range(8): 
    #    print("C1 recons filter  :" , c1_renons[:,i])   
    #    print("C1 filter original  :" , c1_filter[i,0,:,:].reshape(25,))   
    #print("C1 recons filter  :" , c1_renons)   
    #print("C1 filter original  :" , c1_filter)   
    #for i in range(6):
    #    print("filter:", i) 
    #    print(bs[:,i])   
    sym_params.append(bs) 
    sym_params.append(c1_bias) 

    #print('symbols in filter',len(c1_symbolic_flt[0][0]))
      

    c2_filter = net.conv2.weight.data.clone() 
    print(c2_filter.dtype) 
    print(c2_filter.shape) 
    #c2_filter = pad(c2_filter)    
    #c2_filter = pooling(c2_filter) 
    n2,ic2,kw2,kh2 = c2_filter.shape 
    c2_bias = None 
    bs2 = filter_to_sym_conv(filter_index_conv, c2_filter, filter_patch_size, ic2, n2, filter_stride, False)
    print("C2 filter shape :" , bs2.shape)   
    print("C2 filter shape :" , bs2.dtype)   
    
    sym_params.append(bs2) 
    sym_params.append(c2_bias) 
 

    f1_filter = net.fc1.weight.data.clone() 
    print("FC3 filter shape :" , f1_filter.shape)   
    print("FC3 filter shape :" , f1_filter.dtype)   
    #print(f1_filter.shape)
    #f1_filter = f1_filter.resize_(120, 288)    
    #print(f1_filter.shape)
    kw3,kh3 = f1_filter.shape 
    f1_bias = None
    bs3 = filter_to_sym_fc(filter_index_fc,f1_filter, filter_patch_size, filter_stride, False)
    #print("FC1 filter shape :" , bs3.shape)   
    #filter_lut = filter_index_fc.cluster_centers_
    #c1_renons = filter_lut[bs3]
    #c1_renons = c1_renons.reshape(128,256) 
    #print(" F1 filter recons shape :", c1_renons.shape )  
    
    #print("C1 recons filter  :" , c1_renons[0,:10])   
    #print("C1 filter original  :" , f1_filter[0,:10])   

    sym_params.append(bs3) 
    sym_params.append(f1_bias) 

    f2_filter = net.fc2.weight.data.clone() 
    print("FC3 filter shape :" , f2_filter.shape)   
    print("FC3 filter shape :" , f2_filter.dtype)   
    #print(f2_filter.shape)
    #f2_filter = f2_filter.resize_(120, 288)    
    f2_bias = None
    bs4 = filter_to_sym_fc(filter_index_fc,f2_filter, patch_size, patch_stride, False)
    #print("FC2 filter shape :" , bs4.shape)   
    sym_params.append(bs4) 
    sym_params.append(f2_bias) 

    f3_filter = net.fc3.weight.data.clone() 
    print("FC3 filter shape :" , f3_filter.shape)   
    print("FC3 filter shape :" , f3_filter.dtype)   
    bs5 = filter_to_sym_fc(filter_index_fc,f3_filter, patch_size, patch_stride, False)
    print("FC3 filter shape :" , bs5.shape)   
    print("FC3 filter shape :" , bs5.dtype)   
    sym_params.append(bs5) 


    model = CNN_LeNetSym(sym_params, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, None, None, None,None,relu_lut, instr) 

    model.load_state_dict(state_dict)
    #model.load_state_dict(fms_orignet)
    return model
