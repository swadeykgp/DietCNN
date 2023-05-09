import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets

# For training
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import faiss
import sys
#sys.path.insert(1, '../../core')
from lut_utils_tiny import *
import warnings
warnings.filterwarnings('ignore')
from patchlib import *
import multiprocessing
from joblib import Parallel, delayed
#from resnet_18_sym import *
from resnet_18_sym_nbn import *
from functools import reduce
#PARALLEL = 30
PARALLEL = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
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
   #start_t = time.time()
   img_sym =  image_to_symbolic(img, index, patch_size, patch_stride)  
   #overhead = time.process_time() - start
   #overhead *= 1000
   #end = time.time()
   #print("elapsed time for discretization:", end - start_t)
   simg_out_w = (im_w - patch_size[0])//patch_stride + 1
   simg_out_h = (im_h - patch_size[0])//patch_stride + 1
   if in_channels == 1:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h)
   else:
       img_sym = img_sym.reshape(simg_out_w, simg_out_h, in_channels)
   
   return img_sym

def add_syms(cpart1_array, cs_syms, add_lut, bias_lut, k):
    tmp_sym = reduce(lambda x, y: add_lut[x, y], cpart1_array)
    #tmp_sym = cpart1_array[0]
    #for j in range(1,cs_syms):
    #    tmp_sym = add_lut[cpart1_array[j], tmp_sym]
    #return bias_lut[tmp_sym,k]
    return tmp_sym

def add_sym_win(cpart1_array1, cs_wins, cs_syms, add_lut, bias_lut, k):
    if cs_wins == 1:
        output_fm = [ add_syms(cpart1_array1, cs_syms, add_lut, bias_lut, k) ]
    else:  
        output_fm = [ add_syms(cpart1_array1[i], cs_syms, add_lut, bias_lut, k) for i in range(cs_wins) ]
    return output_fm 

def SymConv2D(x,in_channels, out_channels, weights, biases, kernel_size, stride, 
                 padding, # convolution parameters
                 n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params
                 conv_lut, add_lut, bias_lut, sym_kernel,sym_stride):
    x = np.pad(x, [(padding, padding),(padding, padding), (0, 0)] , mode='constant')
    #print(" after padding: ", x.shape)
    #x = F.pad(input=x, pad=(padding, padding, padding, padding), mode='constant', value=0) 
    if in_channels == 1:
        img_sym  = np.asarray(x) 
        im_w,im_h= img_sym.shape      
    else: 
        img_sym = x.squeeze()
        #img_sym  = np.asarray(img_sym.detach().permute(1,2,0)) 
        im_w,im_h,dc= img_sym.shape       
    #print("Symbolic image shapes:" , img_sym.shape, sym_kernel, sym_stride)
    #start_t = time.time()
    symbol_array =  extract_patches(img_sym, (sym_kernel,sym_kernel), sym_stride)
    w,h = symbol_array.shape
    symbol_array = symbol_array.reshape(w//in_channels,h*in_channels)
    #print("Symbolic shapes:" , img_sym.shape, sym_kernel, sym_stride,w,h)
    
    #end = time.time()
    #print("elapsed time for conv patch extraction:", end - start_t)
    #elapsed = (time.process_time() - start)
    #print('Patch extraction time:',elapsed) 
    #start = time.process_time()
    # Only the convolution loop will be measured 
 #   for k in range(out_channels): 
 #      kt = weights[:,k].squeeze()
 #      kt = kt.astype(int)
 #      #print(conv_lut) 
 #      #print("Symbolic kernel shape:" , kt.shape)
 #      # Convolve symbolic input and symbolic kernel - only partial precalculated conv2d
 #      start = time.process_time()
 #      cpart1_array1 = conv_lut[symbol_array,kt]
 #      ############ SYM ADD #########################
 #      cs_wins, cs_syms = cpart1_array1.shape
 #      #print(cpart1_array.shape) 
 #      c_out_buffer = np.zeros((cs_wins,),dtype=np.int16)
 #      for i in range(cs_wins):
 #          # sort this array by the values of the symbols, just to enforce an associativity order
 #          cpart1_array = cpart1_array1[i]
 #          #print(cpart1_array)
 #          cpart1_array = np.sort(cpart1_array) 
 #          #print(cpart1_array)
 #          tmp_sym = cpart1_array[0]
 #          for j in range(1,cs_syms):
 #              tmp_sym = add_lut[cpart1_array[j], tmp_sym]
 #              if j < (cs_syms -2):
 #                  if tmp_sym > cpart1_array[j+1]:
 #                      # problem! order is broken
 #                      tmp = tmp_sym
 #                      tmp_sym = cpart1_array[j+1]
 #                      cpart1_array[j+1] = tmp
 #                      cpart1_array = np.sort(cpart1_array)
 #                      j +=1
 #          c_out_buffer[i] = bias_lut[tmp_sym,k]
 #          #c_out_buffer[i] = tmp_sym
 #      ############ SYM ADD #########################
 #      if k ==0:
 #          temp_img = np.expand_dims(c_out_buffer, axis=1)
 #          output_fm = temp_img
 #      else:
 #          temp_img = np.expand_dims(c_out_buffer, axis=1)
 #          output_fm = np.concatenate((output_fm, temp_img), axis=1)
 #          #print("output fm shape:", output_fm.shape)
 #   output_fm = output_fm.squeeze()
 #   out_w = (im_w - sym_kernel)//sym_stride + 1	
 #   out_h = (im_h - sym_kernel)//sym_stride + 1
 #   
 #   output_fm = output_fm.reshape(out_w, out_h, out_channels)
 
 # First perform symbol lookup based multiplication
    #print("Weight shape:", weights.shape)
    multiplied_array = conv_lut[symbol_array[:,:,np.newaxis], weights]
    wins, syms, ch = multiplied_array.shape
    #output_fm = np.zeros((cs_wins, ch),dtype=np.int16)
    # Perform the list comprehension  
    #output_fm_list = []   
    output_fm_list = Parallel(n_jobs=20)(delayed(add_sym_win)(multiplied_array[:,:,i].squeeze(), wins, syms, add_lut, bias_lut, i)  for i in range(out_channels))
    #for i in range(out_channels):
    #    output_fm_list.append(add_sym_win( multiplied_array[:,:,i].squeeze(), wins, syms, add_lut, bias_lut, i  )) 
    output_fm = np.asarray(output_fm_list, dtype=np.int16)
    output_fm = np.swapaxes(output_fm, 0, 1) 
    out_w = (im_w - sym_kernel)//sym_stride + 1
    out_h = (im_h - sym_kernel)//sym_stride + 1
    output_fm = output_fm.reshape(out_w, out_h, out_channels)
    #print(output_fm.shape)
    #print(output_fm)
     
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
    #c_out_buffer = np.zeros((1,cs_wins),dtype=np.int16)
    #for i in range(cs_wins):
    #    # sort this array by the values of the symbols, just to enforce an associativity order
    #    cpart1_array = cpart1_array1[i]
    #    cpart1_array = np.sort(cpart1_array) 
    #    tmp_sym = cpart1_array[0]
    #    for j in range(1,cs_syms):
    #        tmp_sym = add_lut[cpart1_array[j], tmp_sym]
    #        #if j < (cs_syms -2):
    #        #    if tmp_sym > cpart1_array[j+1]:
    #        #        # problem! order is broken
    #        #        tmp = tmp_sym
    #        #        tmp_sym = cpart1_array[j+1]
    #        #        cpart1_array[j+1] = tmp
    #        #        cpart1_array = np.sort(cpart1_array)
    #        #        j +=1
    #    #c_out_buffer[i] = bias_lut[tmp_sym,k]
    #    #c_out_buffer[0][i] = tmp_sym
    #    c_out_buffer[0][i] = bias_lut[tmp_sym,i]
    #    ############ SYM ADD #########################
    
    c_out_buffer = Parallel(n_jobs=20)(delayed(add_syms)(cpart1_array1[i], cs_syms, add_lut, bias_lut, i) for i in range(cs_wins))
    c_out = np.asarray(output_fm_list, dtype=np.int16)
    c_out = np.expand_dims(c_out, axis = 0)
    return c_out
    
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

##################################
##################### Stable Working
######################################
def forward_helper(item, x, index, patch_size, patch_stride, weights, n_clusters, centroid_lut, conv_lut, add_lut, bias_lut, fc_lut, relu_lut): 
    try:
        # First handle the batch dimension
        #print("input: ",x.shape)
        x = discretize(x,3, index,  patch_size, patch_stride,0) #71
        x = SymConv2D(x,3, 64, weights[0], None, 7, 4, 0, # convolution parameters 33  #1
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 7, 4) 
        #print("c1 out: ", x.shape ) #16
        x = SymReLU(x, relu_lut)

        # Layer 1

        x = SymConv2D(x,64,  64, weights[1], None, 3, 1, 1, # convolution parameters 15  
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        #print("c2 out: ",x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,64,  64, weights[2], None, 3, 1, 1, # convolution parameters 13   #3
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        #print("c3 out: ",x.shape)
        x = SymConv2D(x,64,  64, weights[3], None, 3, 1, 1, # convolution parameters 11   #4
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        #print("c4 out: ",x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,64,  64, weights[4], None, 3, 1, 1, # convolution parameters 9   #5
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        #print("c5 out: ",x.shape)

        # Layer 2

        x = SymConv2D(x,64, 128, weights[5], None, 3, 2, 1, # convolution parameters 4  #6 12
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 2) 
        #print("c6 out: ",x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,128,  128, weights[6], None, 3, 1, 1, # convolution parameters 4 #7  
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 

        #print("c7 out: ",x.shape)
        x = SymConv2D(x,128,  128, weights[7], None, 3, 1, 1, # convolution parameters 4 #8 
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        x = SymReLU(x, relu_lut)
        #print("c8 out: ",x.shape)
        x = SymConv2D(x,128,  128, weights[8], None, 3, 1, 1, # convolution parameters 4
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 

        # Layer 3

        #print("c9 out: ",x.shape)
        x = SymConv2D(x, 128, 256, weights[9], None, 3, 2, 1, # convolution parameters 2
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 2) 
        #print("c10 out: ",x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,256,  256, weights[10], None, 3, 1, 1, # convolution parameters 2
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        x = SymConv2D(x,256,  256, weights[11], None, 3, 1, 1, # convolution parameters 2
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,256,  256, weights[12], None, 3, 1, 1, # convolution parameters 2
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 


        # Layer 4

        x = SymConv2D(x, 256, 512,  weights[13], None, 3, 4, 1, # convolution parameters 1
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 2) 
        #print("c14 out: ",x.shape)
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,512 ,  512 , weights[14], None, 3, 1, 1, # convolution parameters 1
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        x = SymConv2D(x,512 ,  512 , weights[15], None, 3, 1, 1, # convolution parameters 1
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params  
            conv_lut, add_lut, None, 3, 1) 
        x = SymReLU(x, relu_lut)
        x = SymConv2D(x,512 ,  512 , weights[16], None, 3, 1, 1, # convolution parameters 1
            n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
            conv_lut, add_lut, None, 3, 1) 
        #print("c17out: ",x.shape)
        #print(x)


        #### Only conv
        x = symbolic_to_image_win(x, 2, 2, 512, centroid_lut,patch_size) # conv only
        x = torch.from_numpy(x)
        #print(x.shape)
        x = x.permute(2,0,1)
        x = x.unsqueeze(0)
        # do the average pooling here
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        #print(x.shape)
        x = x.reshape(512,) 
        x = np.asarray(x, dtype=np.float32)
        #### End Only conv
        #### Only FC
        #x = symbolic_to_symbolic_win(x, 1, 1, 512, centroid_lut,patch_size) # FC only
        #x = torch.from_numpy(x)
        #x = x.permute(2,0,1)
        #x = x.unsqueeze(0)
        #x = x.reshape(1, 512) 
        #x = np.asarray(x, dtype=np.int16) 
        #x = SymFC(x, weights[8], fc_lut, add_lut,  bias_lut[8])
        #print(x.shape)
        #x = centroid_lut[x[0]]
        #print(x)
        #x = np.transpose(x)
        #x = x.squeeze()
        #print(x.shape)
        #### End FC 
        return x
    except:
        print('error with item:', item)



############ Experimental ######################
#def forward_helper(item, x, index, patch_size, patch_stride, weights, n_clusters, centroid_lut, conv_lut, add_lut, bias_lut, fc_lut, relu_lut): 
#    # First handle the batch dimension
#    print(x.shape)
#    x = discretize(x,3, index,  patch_size, patch_stride,0) #71
#    print(x.shape)
#    x = SymConv2D(x,3, 64, weights[0], None, 7, 4, 3, # convolution parameters 33
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 7, 2) 
#    print(x.shape)
#    x = SymReLU(x, relu_lut)
#
#    # Layer 1
#
#    x = SymConv2D(x,64,  64, weights[1], None, 3, 1, 1, # convolution parameters 31
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,64,  64, weights[2], None, 3, 1, 1, # convolution parameters 29
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymConv2D(x,64,  64, weights[3], None, 3, 1, 1, # convolution parameters 27
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,64,  64, weights[4], None, 3, 1, 1, # convolution parameters 25
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    print(x.shape)
#
#    # Layer 2
#
#    x = SymConv2D(x,64, 128, weights[5], None, 3, 1, 1, # convolution parameters 23
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    print(x.shape)
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,128,  128, weights[6], None, 3, 1, 1, # convolution parameters 21
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymConv2D(x,128,  128, weights[7], None, 3, 1, 1, # convolution parameters 19
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,128,  128, weights[8], None, 3, 1, 1, # convolution parameters 17
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#
#    # Layer 3
#
#    x = SymConv2D(x, 128, 256, weights[9], None, 3, 1, 1, # convolution parameters 15
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    #print(x.shape)
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,256,  256, weights[10], None, 3, 1, 1, # convolution parameters 13
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymConv2D(x,256,  256, weights[11], None, 3, 1, 1, # convolution parameters 11
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,256,  256, weights[12], None, 3, 1, 1, # convolution parameters 9
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#
#
#    # Layer 4
#
#    x = SymConv2D(x, 256, 512,  weights[13], None, 3, 1, 1, # convolution parameters 7
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    #print(x.shape)
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,512 ,  512 , weights[14], None, 3, 1, 1, # convolution parameters 5
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymConv2D(x,512 ,  512 , weights[15], None, 3, 1, 1, # convolution parameters 3
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params  
#        conv_lut, add_lut, None, 3, 1) 
#    x = SymReLU(x, relu_lut)
#    x = SymConv2D(x,512 ,  512 , weights[16], None, 3, 1, 1, # convolution parameters 1
#        n_clusters, index, centroid_lut, patch_size, patch_stride, # patch params 
#        conv_lut, add_lut, None, 3, 1) 
#
#
#    #### Only conv
#    x = symbolic_to_image_win(x, 1, 1, 512, centroid_lut,patch_size) # conv only
#    x = torch.from_numpy(x)
#    x = x.permute(2,0,1)
#    x = x.unsqueeze(0)
#    x = x.reshape(512,) 
#    x = np.asarray(x, dtype=np.float32)
#    #### End Only conv
#    #### Only FC
#    #x = symbolic_to_symbolic_win(x, 1, 1, 512, centroid_lut,patch_size) # FC only
#    #x = torch.from_numpy(x)
#    #x = x.permute(2,0,1)
#    #x = x.unsqueeze(0)
#    #x = x.reshape(1, 512) 
#    #x = np.asarray(x, dtype=np.int16) 
#    #x = SymFC(x, weights[8], fc_lut, add_lut,  bias_lut[8])
#    #print(x.shape)
#    #x = centroid_lut[x[0]]
#    #print(x)
#    #x = np.transpose(x)
#    #x = x.squeeze()
#    #print(x.shape)
#    #### End FC 
#    return x


class VGG_Sym(nn.Module):
    def __init__(self, sym_weights , sym_biases, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, relu_lut, fclayer):
        super(VGG_Sym, self).__init__()
        # Define the net structure
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.bias_lut = sym_biases 
        self.relu_lut =  relu_lut
        self.n_clusters, self.index, self.centroid_lut = n_clusters, index, centroid_lut
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.weights = sym_weights
        # This is original network
        self.classifier = fclayer 
                            
        self.lut_overhead_l1 = 0
        self.disc_overhead_l1 = 0
        self.lut_overhead_l2 = 0
        self.disc_overhead_l2 = 0
    def update_sym(self, c1f, c2f , c3f, c4f, c5f, c6f, c7f, c8f, f1f,  conv_lut, fc_lut, add_lut, relu_lut, c1b, c2b, c3b, c4b, c5b, c6b, c7b, c8b, f1b):
        self.weights.append(c1f)
        self.weights.append(c2f)
        self.weights.append(c3f)
        self.weights.append(c4f)
        self.weights.append(c5f)
        self.weights.append(c6f)
        self.weights.append(c7f)
        self.weights.append(c8f)
        self.weights.append(f1f)
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.relu_lut =  relu_lut
        self.bias_lut = [] 
        self.bias_lut.append(c1b)
        self.bias_lut.append(c2b)
        self.bias_lut.append(c3b)
        self.bias_lut.append(c4b)
        self.bias_lut.append(c5b)
        self.bias_lut.append(c6b)
        self.bias_lut.append(c7b)
        self.bias_lut.append(c8b)
        self.bias_lut.append(f1b)

    def update_l1(self, lo, do):
        self.lut_overhead_l1 += lo
        self.lut_overhead_l1 /=2  
        self.disc_overhead_l1 += do
        self.disc_overhead_l1 /=2  

    def update_l2(self, lo, do):
        self.lut_overhead_l2 += lo
        self.lut_overhead_l2 /=2  
        self.disc_overhead_l2 += do
        self.disc_overhead_l2 /=2  
# Stable working #################################################    
    def forward(self, x_bat): 
        # First handle the batch dimension
        bat, _, _, _  = x_bat.shape
        print("Inferencing...")
        #results = Parallel(n_jobs=PARALLEL)(delayed(forward_helper)(i, x_bat[i,:,:,:].squeeze(),self.index, self.patch_size, self.patch_stride, self.weights, self.n_clusters, self.centroid_lut, self.conv_lut, self.add_lut, self.bias_lut, self.fc_lut, self.relu_lut) for i in range(bat))
        results = []
        for i in range(bat):
            results.append(forward_helper(i, x_bat[i,:,:,:].squeeze(),self.index, self.patch_size, self.patch_stride, self.weights, self.n_clusters, self.centroid_lut, self.conv_lut, self.add_lut, self.bias_lut, self.fc_lut, self.relu_lut ))
        x = np.asarray(results, dtype=np.float32) 
        #print(x.shape)

        #### FC only 
        #x = torch.from_numpy(x)   
        #x = x.unsqueeze(0) 
        #x = x.reshape(bat,-1)
        #x = x.float()
        ##### End FC
 
        #### CONV Only
        x = torch.from_numpy(x)   
        x = x.float()
        #x = self.classifier(x) #conv only
        x = self.classifier(x) #conv only
        #print(x)
        #print(x.shape)
        ##### End CONV
        return x
####################################################################

def filter_to_symbolic_conv(kmeans, flt):
    _, symbol_array = kmeans.search(flt.astype(np.float32), 1)
    symbol_array =symbol_array.squeeze()
    return symbol_array.astype(np.int)
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


def vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
                  relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride):
    torch.set_grad_enabled(False)

    # only needed for first time init
    #net.load_state_dict(sd['net'])
    sym_weights = [] # These are layer 1 conv params
    #net = CNN_LeNet() 
    #net.load_state_dict(state_dict)

    # First task is the fusing of convulution and the batch normalization layers
    c_filter = []
    # Layer 1
    fusedconv = fuse_conv_and_bn(net.conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[0].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[0].conv2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[1].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer1[1].conv2)
    c_filter.append(fusedconv.weight.data.clone())


    fusedconv = fuse_conv_and_bn(net.layer2[0].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[0].conv2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[1].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer2[1].conv2)
    c_filter.append(fusedconv.weight.data.clone())


    fusedconv = fuse_conv_and_bn(net.layer3[0].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[0].conv2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[1].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer3[1].conv2)
    c_filter.append(fusedconv.weight.data.clone())


    fusedconv = fuse_conv_and_bn(net.layer4[0].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[0].conv2)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[1].conv1)
    c_filter.append(fusedconv.weight.data.clone())
    fusedconv = fuse_conv_and_bn(net.layer4[1].conv2)
    c_filter.append(fusedconv.weight.data.clone())
    f1_filter = net.fc.weight.data.clone()

    
    fclayer = net.fc

    n,ic,kw,kh = c_filter[0].shape
    c1f = filter_to_sym_conv(filter_index_conv, c_filter[0], patch_size, ic, n, patch_stride, False)
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
    #print(c11f)
    n,ic,kw,kh = c_filter[11].shape
    c12f = filter_to_sym_conv(filter_index_conv, c_filter[11], patch_size, ic, n, patch_stride, False)
    #print(c12f)
    n,ic,kw,kh = c_filter[12].shape
    c13f = filter_to_sym_conv(filter_index_conv, c_filter[12], patch_size, ic, n, patch_stride, False)
    #print(c13f)
    n,ic,kw,kh = c_filter[13].shape
    c14f = filter_to_sym_conv(filter_index_conv, c_filter[13], patch_size, ic, n, patch_stride, False)
    #print(c14f)
    n,ic,kw,kh = c_filter[14].shape
    c15f = filter_to_sym_conv(filter_index_conv, c_filter[14], patch_size, ic, n, patch_stride, False)
    #print(c15f)
    n,ic,kw,kh = c_filter[15].shape
    c16f = filter_to_sym_conv(filter_index_conv, c_filter[15], patch_size, ic, n, patch_stride, False)
    #print(c16f)
    n,ic,kw,kh = c_filter[16].shape
    c17f = filter_to_sym_conv(filter_index_conv, c_filter[16], patch_size, ic, n, patch_stride, False)
    #print(c17f)




    sym_weights.append(c1f) 
    sym_weights.append(c2f) 
    sym_weights.append(c3f) 
    sym_weights.append(c4f) 
    sym_weights.append(c5f) 
    sym_weights.append(c6f) 
    sym_weights.append(c7f) 
    sym_weights.append(c8f) 
    sym_weights.append(c9f) 
    sym_weights.append(c10f) 
    sym_weights.append(c11f) 
    sym_weights.append(c12f) 
    sym_weights.append(c13f) 
    sym_weights.append(c14f) 
    sym_weights.append(c15f) 
    sym_weights.append(c16f) 
    sym_weights.append(c17f) 
  
    sym_biases = None # These are layer 1 conv params
    model = VGG_Sym(sym_weights , sym_biases, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, relu_lut, fclayer) 

    #model.load_state_dict(sd['net'])
    return model

#Training code - handcrafted

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), 'cifar_vgg_nobn_v2_sym.pt')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(model.state_dict(), 'cifar_vgg_nobn_v2_sym.pt')



def model_training(net, trainloader, optimizer, criterion,   centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index):
    net.train()
    print('Train')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        #old_f1f = net.weights[2] 
        #old_f2f = net.weights[3] 
        #old_flut = net.fc_lut 
        #old_clut = net.conv_lut 
        counter += 1
        # Split data into input vector and labels
        inputs, targets = data
        image, labels= data
        #image, labels = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # forward pass
        #print(image.shape)
        outputs = net(image)
        #print(outputs.shape)
        #print(outputs.dtype)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()           

 
        #backpropagate the loss
        loss.backward()
        optimizer.step()
        print('overhead started')
        n_clusters=512 
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

        filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
        filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
        fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
        conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
        add_lut = create_add_luts(centroid_lut, n_clusters, index)
        c1fl = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
        c2fl = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
        c3fl = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
        c4fl = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
        c5fl = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
        c6fl = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
        c7fl = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
        c8fl = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
        f1fl = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
        relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
    
        filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, patch_size, patch_stride )
        filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  patch_size, patch_stride )

        # Finally the filters
        n,ic,kw,kh = c_filter[0].shape
        c1f = filter_to_sym_conv(filter_index_conv, c_filter[0], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[1].shape
        c2f = filter_to_sym_conv(filter_index_conv, c_filter[1], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[2].shape
        c3f = filter_to_sym_conv(filter_index_conv, c_filter[2], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[3].shape
        c4f = filter_to_sym_conv(filter_index_conv, c_filter[3], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[4].shape
        c5f = filter_to_sym_conv(filter_index_conv, c_filter[4], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[5].shape
        c6f = filter_to_sym_conv(filter_index_conv, c_filter[5], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[6].shape
        c7f = filter_to_sym_conv(filter_index_conv, c_filter[6], conv_patch_size, ic, n, conv_stride, False)
        n,ic,kw,kh = c_filter[7].shape
        c8f = filter_to_sym_conv(filter_index_conv, c_filter[7], conv_patch_size, ic, n, conv_stride, False)
        f1f = filter_to_sym_fc(filter_index_fc,f1_filter, conv_patch_size, conv_stride, False)
  
        net.update_sym(c1f, c2f , c3f, c4f, c5f, c6f, c7f, c8f, f1f, 
                       conv_lut, fc_lut, add_lut, relu_lut, 
                       c1fl, c2fl, c3fl, c4fl, c5fl, c6fl, c7fl, c8fl, f1fl)
        print('overhead ended')

                          
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    #save_index(filter_index_tmp, 'mnist_auto_flt_stride1_128.index')
    #save_lut(lut_tmp, 'mnist_auto_lut_512_128.txt')
    return epoch_loss, epoch_acc, net 

# validation
def validate(net, testloader, criterion):
    net.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            # forward pass
            outputs = net(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            if i % 2 == 0:
                epacc = 100. * (valid_running_correct / i)
                print("running accuracy: ",epacc)
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

 

#if __name__ == '__main__':
#
#    torch.manual_seed(0)
#    np.random.seed(0)
#    print(" Data loading started...")
#    bs = 10 
#    
#    # dataset  
#   
#    directory = "/home/edgeacceleration/projects/dataset/tiny-imagenet-200/"
#    num_classes = 200
#    # the magic normalization parameters come from the example
#    transform_mean = np.array([ 0.485, 0.456, 0.406 ])
#    transform_std = np.array([ 0.229, 0.224, 0.225 ])
#    
#    train_transform = transforms.Compose([
#        transforms.RandomResizedCrop(64),
#        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize(mean = transform_mean, std = transform_std),
#    ])
#    
#    val_transform = transforms.Compose([
#        #transforms.Resize(256),
#        #transforms.CenterCrop(224),
#        transforms.Resize(74),
#        transforms.CenterCrop(64),
#        transforms.ToTensor(),
#        transforms.Normalize(mean = transform_mean, std = transform_std),
#    ])
#
#    traindir = os.path.join(directory, "train")
#    # be careful with this set, the labels are not defined using the directory structure
#    valdir = os.path.join(directory, "val")
#    
#    train = datasets.ImageFolder(traindir, train_transform)
#    val = datasets.ImageFolder(valdir, val_transform)
#    
#    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
#    val_loader = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=True)
#    
#    assert num_classes == len(train_loader.dataset.classes)
#    
#    small_labels = {}
#    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
#        line = dictionary_file.readline()
#        while line:
#            label_id, label = line.strip().split("\t")
#            small_labels[label_id] = label
#            line = dictionary_file.readline()
#    
#    #print(small_labels.items()[0])
#    
#    #The train subdirectory of Tiny ImageNet-200 has a collection of subdirectories, named using to the WordNet ids to label the images that they contain. The torchvision data loader uses the names of the subdirectories as labels, but replaces them with numeric indices when iterating the batches.
#    
#    os.listdir(traindir)[:5]
#    
#    labels = {}
#    label_ids = {}
#    for label_index, label_id in enumerate(train_loader.dataset.classes):
#        label = small_labels[label_id]
#        labels[label_index] = label
#        label_ids[label_id] = label_index
#    
#    #print(labels.items()[0])
#    
#    #print(label_ids.items()[0])
#    
#    #Another problem is that the validation directory only has one subdirectory called images. The labels for every image inside this subdirectory are defined in a file called val_annotations.txt.
#    
#    val_label_map = {}
#    with open(os.path.join(directory, "val/val_annotations.txt"), "r") as val_label_file:
#        line = val_label_file.readline()
#        while line:
#            file_name, label_id, _, _, _, _ = line.strip().split("\t")
#            val_label_map[file_name] = label_id
#            line = val_label_file.readline()
#    
#    #print(val_label_map.items()[0])
#    
#    #Finally we update the Tiny ImageNet-200 validation set labels:
#    
#    #print(val_loader.dataset.imgs[:5])
#    
#    for i in range(len(val_loader.dataset.imgs)):
#        file_path = val_loader.dataset.imgs[i][0]
#        
#        file_name = os.path.basename(file_path)
#        label_id = val_label_map[file_name]
#        
#        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])
#    
#    #val_loader.dataset.imgs[:5]
#
#
#    random_indices = list(range(0, len(val_loader), 100))
#    print(len(random_indices))
#    testset_subset = torch.utils.data.Subset(val_loader, random_indices)
#    testloader = torch.utils.data.DataLoader(testset_subset, batch_size=bs, shuffle=False)
#
#    # net
#
#
#    net = VGG('VGG11')
#    pretrained_model = "./cifar_vgg_sym_v3.pt"
#    sd = torch.load(pretrained_model)
#    #print(sd['net']) 
#    net.load_state_dict(sd['net'])
#    #net.eval()
#    #if torch.cuda.is_available():
#    #    net.cuda()
#    c_filter = []
#    c_filter.append(net.features[0].weight.data.clone())
#    c_filter.append(net.features[2].weight.data.clone())
#    c_filter.append(net.features[4].weight.data.clone())
#    c_filter.append(net.features[6].weight.data.clone())
#    c_filter.append(net.features[8].weight.data.clone())
#    c_filter.append(net.features[10].weight.data.clone())
#    c_filter.append(net.features[12].weight.data.clone())
#    c_filter.append(net.features[14].weight.data.clone())
#    f1_filter = net.classifier[0].weight.data.clone()
#
#    with torch.no_grad():
#        c1_bias = net.features[0].bias.clone()
#        c1_bias = np.asarray(c1_bias)
#        c2_bias = net.features[2].bias.clone()
#        c2_bias = np.asarray(c2_bias)
#        c3_bias = net.features[4].bias.clone()
#        c3_bias = np.asarray(c3_bias)
#        c4_bias = net.features[6].bias.clone()
#        c4_bias = np.asarray(c4_bias)
#        c5_bias = net.features[8].bias.clone()
#        c5_bias = np.asarray(c5_bias)
#        c6_bias = net.features[10].bias.clone()
#        c6_bias = np.asarray(c6_bias)
#        c7_bias = net.features[12].bias.clone()
#        c7_bias = np.asarray(c7_bias)
#        c8_bias = net.features[14].bias.clone()
#        c8_bias = np.asarray(c8_bias)
#        f1_bias = net.classifier[0].bias.clone()
#        f1_bias = np.asarray(f1_bias)
#    conv_patch_size = (1, 1)
#    patch_size = (1, 1)
#    all_patch_size = (1, 1)
#    n_cluster_conv_filters = 256
#    n_cluster_fc_filters = 32
#    conv_stride = 1
#    index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_512_v0.index")
#    #index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_256_v0.index")
#    #index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_64_v0.index")
#    n_clusters=512
#    #n_clusters=256
#    #n_clusters=64
#    patch_stride = 1
#    centroid_lut = index.reconstruct_n(0, n_clusters)
#
#    print("Conv index creation stared .....")
#    start_t = time.time()
#    filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    end = time.time()
#    print("elapsed time for conv index:", end - start_t)
#    start_t = time.time()
#    filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    end = time.time()
#    print("elapsed time for fc index:", end - start_t)
#    start_t = time.time()
#    fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    end = time.time()
#    print("elapsed time for fc lut:", end - start_t)
#    start_t = time.time()
#    conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    end = time.time()
#    print("elapsed time for conv lut:", end - start_t)
#    start_t = time.time()
#    add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    end = time.time()
#    print("elapsed time for add lut:", end - start_t)
#
#    # deal the biases 
#    start_t = time.time()
#    c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    end = time.time()
#    print("elapsed time for convolution bias lut:", end - start_t)
#    start_t = time.time()
#    f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    end = time.time()
#    print("elapsed time for FC1  bias lut:", end - start_t)
#    start_t = time.time()
#    relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    end = time.time()
#    print("elapsed time for relu lut:", end - start_t)
#
#   
#    print(" Symbolic model loading started...")
#    t = time.process_time()
#    net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#                  c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#                  f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    elapsed_time3 = time.process_time() - t
#    print("Symbolic model loading completed in:",elapsed_time3)
# 
#    for param in net.parameters():
#        param.requires_grad = True
#    learning_rate = 0.005 
#    print("Training Started !!!!!!")
#    loss_criterion = nn.CrossEntropyLoss()
#    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    epochs = 1 
#    # initialize SaveBestModel class
#    save_best_model = SaveBestModel()  
#
#    # lists to keep track of losses and accuracies
#    train_loss, valid_loss = [], []
#    train_acc, valid_acc = [], []
#    # start the training
#    for epoch in range(epochs):
#        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#        #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#        valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#        #scheduler.step(valid_epoch_loss)
#        #train_loss.append(train_epoch_loss)
#        valid_loss.append(valid_epoch_loss)
#        #train_acc.append(train_epoch_acc)
#        valid_acc.append(valid_epoch_acc)
#        #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#        # save the best model till now if we have the least loss in the current epoch
#        #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#        #with open('vggcifarsym_train_loss.txt', 'w') as f:
#        #    for item in train_loss:
#        #        f.write("%s\n" % item)
#        #with open('vggcifarsym_train_acc.txt', 'w') as f:
#        #    for item in train_acc:
#        #        f.write("%s\n" % item)
#        print('-'*50)
#    
#    # save the trained model weights for a final time
#    #save_model(epochs, net, optimizer, loss_criterion)
#    # save the loss and accuracy plots
#    print("Testing Done 64!!!!!!")
#
#    n_cluster_conv_filters = 16 
#    n_cluster_fc_filters = 16
#
#    print("Conv index creation stared .....")
#    start_t = time.time()
#    filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    end = time.time()
#    print("elapsed time for conv index:", end - start_t)
#    start_t = time.time()
#    filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    end = time.time()
#    print("elapsed time for fc index:", end - start_t)
#    start_t = time.time()
#    fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    end = time.time()
#    print("elapsed time for fc lut:", end - start_t)
#    start_t = time.time()
#    conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    end = time.time()
#    print("elapsed time for conv lut:", end - start_t)
#    start_t = time.time()
#    add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    end = time.time()
#    print("elapsed time for add lut:", end - start_t)
#
#    # deal the biases 
#    start_t = time.time()
#    c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    end = time.time()
#    print("elapsed time for convolution bias lut:", end - start_t)
#    start_t = time.time()
#    f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    end = time.time()
#    print("elapsed time for FC1  bias lut:", end - start_t)
#    start_t = time.time()
#    relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    end = time.time()
#    print("elapsed time for relu lut:", end - start_t)
#
#   
#    print(" Symbolic model loading started...")
#    t = time.process_time()
#    net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#                  c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#                  f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    elapsed_time3 = time.process_time() - t
#    print("Symbolic model loading completed in:",elapsed_time3)
# 
#    for param in net.parameters():
#        param.requires_grad = True
#    learning_rate = 0.005 
#    print("Training Started !!!!!!")
#    loss_criterion = nn.CrossEntropyLoss()
#    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    epochs = 1 
#    # initialize SaveBestModel class
#    save_best_model = SaveBestModel()  
#
#    # lists to keep track of losses and accuracies
#    train_loss, valid_loss = [], []
#    train_acc, valid_acc = [], []
#    # start the training
#    for epoch in range(epochs):
#        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#        #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#        valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#        #scheduler.step(valid_epoch_loss)
#        #train_loss.append(train_epoch_loss)
#        valid_loss.append(valid_epoch_loss)
#        #train_acc.append(train_epoch_acc)
#        valid_acc.append(valid_epoch_acc)
#        #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#        # save the best model till now if we have the least loss in the current epoch
#        #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#        #with open('vggcifarsym_train_loss.txt', 'w') as f:
#        #    for item in train_loss:
#        #        f.write("%s\n" % item)
#        #with open('vggcifarsym_train_acc.txt', 'w') as f:
#        #    for item in train_acc:
#        #        f.write("%s\n" % item)
#        print('-'*50)
#    
#    # save the trained model weights for a final time
#    #save_model(epochs, net, optimizer, loss_criterion)
#    # save the loss and accuracy plots
#    print("Testing Done 512 - 256 - 32!!!!!!")
#
#    #n_cluster_conv_filters = 16 
#    #n_cluster_fc_filters = 32
#    #
#    #print("Conv index creation stared .....")
#    #start_t = time.time()
#    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    #end = time.time()
#    #print("elapsed time for conv index:", end - start_t)
#    #start_t = time.time()
#    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    #end = time.time()
#    #print("elapsed time for fc index:", end - start_t)
#    #start_t = time.time()
#    #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    #end = time.time()
#    #print("elapsed time for fc lut:", end - start_t)
#    #start_t = time.time()
#    #conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    #end = time.time()
#    #print("elapsed time for conv lut:", end - start_t)
#    #start_t = time.time()
#    #add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for add lut:", end - start_t)
#
#    ## deal the biases 
#    #start_t = time.time()
#    #c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    #c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    #c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    #c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    #c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    #c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    #c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    #c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    #end = time.time()
#    #print("elapsed time for convolution bias lut:", end - start_t)
#    #start_t = time.time()
#    #f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    #end = time.time()
#    #print("elapsed time for FC1  bias lut:", end - start_t)
#    #start_t = time.time()
#    #relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for relu lut:", end - start_t)
#
#   
#    #print(" Symbolic model loading started...")
#    #t = time.process_time()
#    #net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#    #              c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#    #              f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    #elapsed_time3 = time.process_time() - t
#    #print("Symbolic model loading completed in:",elapsed_time3)
# 
#    #for param in net.parameters():
#    #    param.requires_grad = True
#    #learning_rate = 0.005 
#    #print("Training Started !!!!!!")
#    #loss_criterion = nn.CrossEntropyLoss()
#    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    #epochs = 1 
#    ## initialize SaveBestModel class
#    #save_best_model = SaveBestModel()  
#
#    ## lists to keep track of losses and accuracies
#    #train_loss, valid_loss = [], []
#    #train_acc, valid_acc = [], []
#    ## start the training
#    #for epoch in range(epochs):
#    #    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#    #    #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#    #    valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#    #    #scheduler.step(valid_epoch_loss)
#    #    #train_loss.append(train_epoch_loss)
#    #    valid_loss.append(valid_epoch_loss)
#    #    #train_acc.append(train_epoch_acc)
#    #    valid_acc.append(valid_epoch_acc)
#    #    #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#    #    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#    #    # save the best model till now if we have the least loss in the current epoch
#    #    #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#    #    #with open('vggcifarsym_train_loss.txt', 'w') as f:
#    #    #    for item in train_loss:
#    #    #        f.write("%s\n" % item)
#    #    #with open('vggcifarsym_train_acc.txt', 'w') as f:
#    #    #    for item in train_acc:
#    #    #        f.write("%s\n" % item)
#    #    print('-'*50)
#    #
#    ## save the trained model weights for a final time
#    ##save_model(epochs, net, optimizer, loss_criterion)
#    ## save the loss and accuracy plots
#    #print("Testing Done 16, 32!!!!!!")
#
#
#
#    #n_cluster_conv_filters = 512 
#    #n_cluster_fc_filters = 512
#
#    #print("Conv index creation stared .....")
#    #start_t = time.time()
#    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    #end = time.time()
#    #print("elapsed time for conv index:", end - start_t)
#    #start_t = time.time()
#    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    #end = time.time()
#    #print("elapsed time for fc index:", end - start_t)
#    #start_t = time.time()
#    #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    #end = time.time()
#    #print("elapsed time for fc lut:", end - start_t)
#    #start_t = time.time()
#    #conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    #end = time.time()
#    #print("elapsed time for conv lut:", end - start_t)
#    #start_t = time.time()
#    #add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for add lut:", end - start_t)
#
#    ## deal the biases 
#    #start_t = time.time()
#    #c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    #c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    #c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    #c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    #c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    #c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    #c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    #c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    #end = time.time()
#    #print("elapsed time for convolution bias lut:", end - start_t)
#    #start_t = time.time()
#    #f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    #end = time.time()
#    #print("elapsed time for FC1  bias lut:", end - start_t)
#    #start_t = time.time()
#    #relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for relu lut:", end - start_t)
#
#   
#    #print(" Symbolic model loading started...")
#    #t = time.process_time()
#    #net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#    #              c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#    #              f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    #elapsed_time3 = time.process_time() - t
#    #print("Symbolic model loading completed in:",elapsed_time3)
# 
#    #for param in net.parameters():
#    #    param.requires_grad = True
#    #learning_rate = 0.005 
#    #print("Training Started !!!!!!")
#    #loss_criterion = nn.CrossEntropyLoss()
#    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    #epochs = 1 
#    ## initialize SaveBestModel class
#    #save_best_model = SaveBestModel()  
#
#    ## lists to keep track of losses and accuracies
#    #train_loss, valid_loss = [], []
#    #train_acc, valid_acc = [], []
#    ## start the training
#    #for epoch in range(epochs):
#    #    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#    #    #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#    #    valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#    #    #scheduler.step(valid_epoch_loss)
#    #    #train_loss.append(train_epoch_loss)
#    #    valid_loss.append(valid_epoch_loss)
#    #    #train_acc.append(train_epoch_acc)
#    #    valid_acc.append(valid_epoch_acc)
#    #    #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#    #    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#    #    # save the best model till now if we have the least loss in the current epoch
#    #    #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#    #    #with open('vggcifarsym_train_loss.txt', 'w') as f:
#    #    #    for item in train_loss:
#    #    #        f.write("%s\n" % item)
#    #    #with open('vggcifarsym_train_acc.txt', 'w') as f:
#    #    #    for item in train_acc:
#    #    #        f.write("%s\n" % item)
#    #    print('-'*50)
#    #
#    ## save the trained model weights for a final time
#    ##save_model(epochs, net, optimizer, loss_criterion)
#    ## save the loss and accuracy plots
#    #print("Testing Done 512, 512!!!!!!")
#
#
#    #n_cluster_conv_filters = 512 
#    #n_cluster_fc_filters = 32
#    #
#    #print("Conv index creation stared .....")
#    #start_t = time.time()
#    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    #end = time.time()
#    #print("elapsed time for conv index:", end - start_t)
#    #start_t = time.time()
#    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    #end = time.time()
#    #print("elapsed time for fc index:", end - start_t)
#    #start_t = time.time()
#    #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    #end = time.time()
#    #print("elapsed time for fc lut:", end - start_t)
#    #start_t = time.time()
#    #conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    #end = time.time()
#    #print("elapsed time for conv lut:", end - start_t)
#    #start_t = time.time()
#    #add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for add lut:", end - start_t)
#
#    ## deal the biases 
#    #start_t = time.time()
#    #c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    #c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    #c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    #c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    #c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    #c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    #c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    #c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    #end = time.time()
#    #print("elapsed time for convolution bias lut:", end - start_t)
#    #start_t = time.time()
#    #f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    #end = time.time()
#    #print("elapsed time for FC1  bias lut:", end - start_t)
#    #start_t = time.time()
#    #relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for relu lut:", end - start_t)
#
#   
#    #print(" Symbolic model loading started...")
#    #t = time.process_time()
#    #net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#    #              c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#    #              f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    #elapsed_time3 = time.process_time() - t
#    #print("Symbolic model loading completed in:",elapsed_time3)
# 
#    #for param in net.parameters():
#    #    param.requires_grad = True
#    #learning_rate = 0.005 
#    #print("Training Started !!!!!!")
#    #loss_criterion = nn.CrossEntropyLoss()
#    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    #epochs = 1 
#    ## initialize SaveBestModel class
#    #save_best_model = SaveBestModel()  
#
#    ## lists to keep track of losses and accuracies
#    #train_loss, valid_loss = [], []
#    #train_acc, valid_acc = [], []
#    ## start the training
#    #for epoch in range(epochs):
#    #    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#    #    #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#    #    valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#    #    #scheduler.step(valid_epoch_loss)
#    #    #train_loss.append(train_epoch_loss)
#    #    valid_loss.append(valid_epoch_loss)
#    #    #train_acc.append(train_epoch_acc)
#    #    valid_acc.append(valid_epoch_acc)
#    #    #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#    #    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#    #    # save the best model till now if we have the least loss in the current epoch
#    #    #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#    #    #with open('vggcifarsym_train_loss.txt', 'w') as f:
#    #    #    for item in train_loss:
#    #    #        f.write("%s\n" % item)
#    #    #with open('vggcifarsym_train_acc.txt', 'w') as f:
#    #    #    for item in train_acc:
#    #    #        f.write("%s\n" % item)
#    #    print('-'*50)
#    #
#    ## save the trained model weights for a final time
#    ##save_model(epochs, net, optimizer, loss_criterion)
#    ## save the loss and accuracy plots
#    #print("Testing Done 512, 32!!!!!!")
#
#    #n_cluster_conv_filters = 8 
#    #n_cluster_fc_filters = 8
#    #
#    #print("Conv index creation stared .....")
#    #start_t = time.time()
#    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    #end = time.time()
#    #print("elapsed time for conv index:", end - start_t)
#    #start_t = time.time()
#    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    #end = time.time()
#    #print("elapsed time for fc index:", end - start_t)
#    #start_t = time.time()
#    #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    #end = time.time()
#    #print("elapsed time for fc lut:", end - start_t)
#    #start_t = time.time()
#    #conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    #end = time.time()
#    #print("elapsed time for conv lut:", end - start_t)
#    #start_t = time.time()
#    #add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for add lut:", end - start_t)
#
#    ## deal the biases 
#    #start_t = time.time()
#    #c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    #c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    #c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    #c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    #c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    #c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    #c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    #c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    #end = time.time()
#    #print("elapsed time for convolution bias lut:", end - start_t)
#    #start_t = time.time()
#    #f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    #end = time.time()
#    #print("elapsed time for FC1  bias lut:", end - start_t)
#    #start_t = time.time()
#    #relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for relu lut:", end - start_t)
#
#   
#    #print(" Symbolic model loading started...")
#    #t = time.process_time()
#    #net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#    #              c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#    #              f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    #elapsed_time3 = time.process_time() - t
#    #print("Symbolic model loading completed in:",elapsed_time3)
# 
#    #for param in net.parameters():
#    #    param.requires_grad = True
#    #learning_rate = 0.005 
#    #print("Training Started !!!!!!")
#    #loss_criterion = nn.CrossEntropyLoss()
#    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    #epochs = 1 
#    ## initialize SaveBestModel class
#    #save_best_model = SaveBestModel()  
#
#    ## lists to keep track of losses and accuracies
#    #train_loss, valid_loss = [], []
#    #train_acc, valid_acc = [], []
#    ## start the training
#    #for epoch in range(epochs):
#    #    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#    #    #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#    #    valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#    #    #scheduler.step(valid_epoch_loss)
#    #    #train_loss.append(train_epoch_loss)
#    #    valid_loss.append(valid_epoch_loss)
#    #    #train_acc.append(train_epoch_acc)
#    #    valid_acc.append(valid_epoch_acc)
#    #    #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#    #    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#    #    # save the best model till now if we have the least loss in the current epoch
#    #    #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#    #    #with open('vggcifarsym_train_loss.txt', 'w') as f:
#    #    #    for item in train_loss:
#    #    #        f.write("%s\n" % item)
#    #    #with open('vggcifarsym_train_acc.txt', 'w') as f:
#    #    #    for item in train_acc:
#    #    #        f.write("%s\n" % item)
#    #    print('-'*50)
#    #
#    ## save the trained model weights for a final time
#    ##save_model(epochs, net, optimizer, loss_criterion)
#    ## save the loss and accuracy plots
#    #print("Testing Done 8, 8!!!!!!")
#
#
#
#    #n_cluster_conv_filters = 128 
#    #n_cluster_fc_filters = 64
#    #print("Conv index creation stared .....")
#    #start_t = time.time()
#    #filter_index_conv = create_index_conv(n_cluster_conv_filters, c_filter, 8, conv_patch_size, conv_stride )
#    #end = time.time()
#    #print("elapsed time for conv index:", end - start_t)
#    #start_t = time.time()
#    #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter,  all_patch_size, patch_stride )
#    #end = time.time()
#    #print("elapsed time for fc index:", end - start_t)
#    #start_t = time.time()
#    #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
#    #end = time.time()
#    #print("elapsed time for fc lut:", end - start_t)
#    #start_t = time.time()
#    #conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
#    #end = time.time()
#    #print("elapsed time for conv lut:", end - start_t)
#    #start_t = time.time()
#    #add_lut = create_add_luts(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for add lut:", end - start_t)
#
#    ## deal the biases 
#    #start_t = time.time()
#    #c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)
#    #c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
#    #c3_bias_lut = create_bias_luts(centroid_lut, n_clusters, c3_bias, index)
#    #c4_bias_lut = create_bias_luts(centroid_lut, n_clusters, c4_bias, index)
#    #c5_bias_lut = create_bias_luts(centroid_lut, n_clusters, c5_bias, index)
#    #c6_bias_lut = create_bias_luts(centroid_lut, n_clusters, c6_bias, index)
#    #c7_bias_lut = create_bias_luts(centroid_lut, n_clusters, c7_bias, index)
#    #c8_bias_lut = create_bias_luts(centroid_lut, n_clusters, c8_bias, index)
#    #end = time.time()
#    #print("elapsed time for convolution bias lut:", end - start_t)
#    #start_t = time.time()
#    #f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)
#    #end = time.time()
#    #print("elapsed time for FC1  bias lut:", end - start_t)
#    #start_t = time.time()
#    #relu_lut =   create_relu_lut(centroid_lut, n_clusters, index)
#    #end = time.time()
#    #print("elapsed time for relu lut:", end - start_t)
#
#   
#    #print(" Symbolic model loading started...")
#    #t = time.process_time()
#    #net = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
#    #              c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
#    #              f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
#    #elapsed_time3 = time.process_time() - t
#    #print("Symbolic model loading completed in:",elapsed_time3)
# 
#    #for param in net.parameters():
#    #    param.requires_grad = True
#    #learning_rate = 0.005 
#    #print("Training Started !!!!!!")
#    #loss_criterion = nn.CrossEntropyLoss()
#    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#    #epochs = 1 
#    ## initialize SaveBestModel class
#    #save_best_model = SaveBestModel()  
#
#    ## lists to keep track of losses and accuracies
#    #train_loss, valid_loss = [], []
#    #train_acc, valid_acc = [], []
#    ## start the training
#    #for epoch in range(epochs):
#    #    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#    #    #train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,  centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
#    #    valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
#    #    #scheduler.step(valid_epoch_loss)
#    #    #train_loss.append(train_epoch_loss)
#    #    valid_loss.append(valid_epoch_loss)
#    #    #train_acc.append(train_epoch_acc)
#    #    valid_acc.append(valid_epoch_acc)
#    #    #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
#    #    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
#    #    # save the best model till now if we have the least loss in the current epoch
#    #    #save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
#    #    #with open('vggcifarsym_train_loss.txt', 'w') as f:
#    #    #    for item in train_loss:
#    #    #        f.write("%s\n" % item)
#    #    #with open('vggcifarsym_train_acc.txt', 'w') as f:
#    #    #    for item in train_acc:
#    #    #        f.write("%s\n" % item)
#    #    print('-'*50)
#    #
#    ## save the trained model weights for a final time
#    ##save_model(epochs, net, optimizer, loss_criterion)
#    ## save the loss and accuracy plots
#    #print("Testing Done 128, 64!!!!!!")
#
