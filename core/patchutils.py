#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

#from sklearn.cluster import MiniBatchKMeans
import time
#from scipy.spatial import distance

# for faiss
import faiss
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
np.random.seed(0)
    
# Function for patch extraction from an image
def extract_patches(fm_squeezed, kernel_size, loc, stride=0):
    fm_x, fm_y = fm_squeezed.shape
    old_x, old_y = fm_x, fm_y
    img = fm_squeezed
    #print(fm_x,fm_y)
    #print(kernel_size[0])
    # Set the stride
    if stride == 0:
        stride = kernel_size[0] 
    # handle the case where the FM is odd sized
    if fm_x % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_x = fm_x % stride
        augment_x = stride - augment_delta_x
        #print("adjusting x by",augment_x)
        fm_x = fm_x + augment_x
    else:
        augment_x = 0

    if fm_y % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_y = fm_y % stride
        augment_y = stride - augment_delta_y
        #print("adjusting y by",augment_y)
        fm_y = fm_y + augment_y
    else:
        augment_y = 0
        
    if augment_x != 0:
        img = torch.zeros(fm_x,fm_y)
        #print(img.shape)
        img[:old_x,:old_y] = fm_squeezed 
        #print("New dimensions:", fm_x,fm_y)
    elif augment_y!= 0:
        img = torch.zeros(fm_x,fm_y)
        #print(img.shape)
        img[:old_x,:old_y] = fm_squeezed 
        #print("New dimensions:", fm_x,fm_y)
    else:
        pass
     
    patches_x = ((fm_x - kernel_size[0]) // stride) + 1 
    patches_y = ((fm_y -  kernel_size[1] ) // stride) + 1
    #print("total patches possible", patches_x*patches_y)
    # All location stuff
    if loc:
        location_vectors = 4
        topleft = (0, 0)
        topright = (fm_x, 0)
        bottomleft = (0, fm_y) 
        bottomright = (fm_x, fm_y) 
        loc = np.zeros((patches_x * patches_y, location_vectors))
        patch_loc = np.zeros((patches_x * patches_y, kernel_size[0] * kernel_size[1] + location_vectors))
    
    patch = np.zeros((patches_x * patches_y, kernel_size[0],kernel_size[1]))
    patch_count = 0
    
    # Not using any padding here. last block has no slide
    for i in range(0, fm_x - kernel_size[0] + 1 ,stride):
        for j in range(0, fm_y - kernel_size[1] + 1,stride):
            #print(" for {}th row, {}th column".format(i,j))
            patch_temp = img[i:(i+kernel_size[0]), j:(j+kernel_size[1])]
            patch[patch_count,:] = patch_temp
            if loc:
                loc[patch_count][0] = distance.euclidean(topleft, (i,j))/fm_x 
                loc[patch_count][1] = distance.euclidean(topright, (i + kernel_size[0],j))/fm_x  
                loc[patch_count][2] = distance.euclidean(bottomleft, (i,  kernel_size[1]+j))/fm_x  
                loc[patch_count][3] = distance.euclidean(bottomright, (i+ kernel_size[0],  kernel_size[1]+j))/fm_x 
                patch_loc_temp = patch_temp
                #print ("patch_loc_temp shape", patch_loc_temp.shape) 
                patch_loc_temp = patch_loc_temp.flatten()
                #print ("reshaped patch_loc_temp shape", patch_loc_temp.shape) 
                patch_loc[patch_count,:] = np.concatenate((patch_loc_temp, loc[patch_count]), axis = 0)
            patch_count += 1
    #print("total patches collected", patch_count)
    if loc:
        return patch, patch_loc, loc
    else:
        return patch

# Function for patch extraction from an image
def make_patches_conv_pre(img, indices, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count):
    # get this in symbolic form
    _, sym_fm = fm_to_symbolic_fm(img, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count, ana=True)
    extracted_symbols = sym_fm[indices]
    return extracted_symbols

# Function for patch extraction from an image
def make_patches_conv(img, k_x, fm_x, stride, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count):

    # get this in symbolic form
    std_fm, sym_fm = fm_to_symbolic_fm(img, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count, ana=True)
    # fm symbol shape , this is what happened after symbolic abstraction
    #print(len(sym_fm)) 
    fm_y = fm_x
    #print(fm_x, fm_y)    
    k_y = k_x
    symbol_stride = stride//patch_size[0]
    #print('symbol_stride', symbol_stride)    
    all_indices = []
    one_indices = []
    fm_sym_buffer = []
    for i in range(0, fm_x - k_x + 1 ,symbol_stride):
        for j in range(0, fm_y - k_y + 1,symbol_stride):
            for k in range(k_x):
                for l in range(k_y):
                    index_x = i + k
                    index_y = j + l
                    print("Index x & y:",index_x,index_y) 
                    print("i,j,k, & l :",i,j,k,l) 
                    one_indices.append(index_x*fm_x + index_y)
            #print('one_indices:', one_indices) 
            all_indices.append(one_indices)
            extracted_symbols = sym_fm[one_indices]
            #print('Extracted symbols:', extracted_symbols)  
            fm_sym_buffer.append(extracted_symbols)  
            one_indices = []
    conv_indices = np.array(all_indices, dtype=np.int32)
    #np.savetxt('conv_indices_I224_K11_S4.txt', conv_indices, delimiter =', ', fmt='%d')    
    np.savetxt('conv_indices_I224_K7_S2.txt', conv_indices, delimiter =', ', fmt='%d')    
    return fm_sym_buffer

# Function for patch extraction from an image
def extract_patches_conv(fm_squeezed, kernel_size, stride, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count):
    fm_x, fm_y = fm_squeezed.shape
    old_x, old_y = fm_x, fm_y
    img = fm_squeezed
    #print(fm_x,fm_y)
    #print(kernel_size[0])
    # Set the stride
    if stride == 0:
        stride = kernel_size[0] 
    # handle the case where the FM is odd sized
    if fm_x % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_x = fm_x % stride
        augment_x = stride - augment_delta_x
        #print("adjusting x by",augment_x)
        fm_x = fm_x + augment_x
    else:
        augment_x = 0

    if fm_y % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_y = fm_y % stride
        augment_y = stride - augment_delta_y
        #print("adjusting y by",augment_y)
        fm_y = fm_y + augment_y
    else:
        augment_y = 0
        
    if augment_x != 0:
        img = torch.zeros(fm_x,fm_y)
        #print(img.shape)
        img[:old_x,:old_y] = fm_squeezed 
        #print("New dimensions:", fm_x,fm_y)
    elif augment_y!= 0:
        img = torch.zeros(fm_x,fm_y)
        #print(img.shape)
        img[:old_x,:old_y] = fm_squeezed 
        #print("New dimensions:", fm_x,fm_y)
    else:
        pass
     
    patches_x = ((fm_x - kernel_size[0]) // stride) + 1 
    patches_y = ((fm_y -  kernel_size[1] ) // stride) + 1
    #print("total patches possible", patches_x*patches_y)
    
    patch_count = 0
    
    # Not using any padding here. last block has no slide
    fm_sym_buffer = []
    fm_std_buffer = []
    for i in range(0, fm_x - kernel_size[0] + 1 ,stride):
        for j in range(0, fm_y - kernel_size[1] + 1,stride):
            #print(" for {}th row, {}th column".format(i,j))
            patch_temp = img[i:(i+kernel_size[0]), j:(j+kernel_size[1])]
            std_fm, sym_fm = fm_to_symbolic_fm(patch_temp, n_clusters, kmeans, centroid_lut,  patch_size, patch_stride, channel_count, ana=True)
            std_fm = np.squeeze(std_fm)
            fm_sym_buffer.append(list(sym_fm))
            fm_std_buffer.append(list(std_fm))
            patch_count += 1
    abstract_fm =  np.array(fm_std_buffer, dtype=np.double)
    #print("total patches collected", patch_count)
    return abstract_fm, fm_sym_buffer



# Function for patch extraction from an image to non image with location
def extract_patches_loc(fm_squeezed, kernel_size, stride=0):
    fm_x, fm_y = fm_squeezed.shape
    img = fm_squeezed
    #print(fm_x,fm_y)
    #print(kernel_size[0])
    # Set the stride
    if stride == 0:
        stride = kernel_size[0] 
        # handle the case where the FM is odd sized
        if fm_x % stride != 0:
            # repeat the last row and column  stride times
            augment = fm_x % stride
            
            fm_x = fm_x + augment
            fm_y = fm_y + augment
            img = torch.zeros(fm_x,fm_y)
            img[:fm_x-1,:fm_y-1] = fm_squeezed 
            #print("New dimensions:", fm_x,fm_y)
     
    patches_x = ((fm_x - kernel_size[0]) // stride) + 1 
    patches_y = ((fm_y -  kernel_size[1] ) // stride) + 1
    #print("total patches possible", patches_x*patches_y)
    # All location stuff
    patch = np.zeros((patches_x * patches_y, kernel_size[0], kernel_size[1]))
    patch_count = 0
    
    # Not using any padding here. last block has no slide
    for i in range(0, fm_x - kernel_size[0] + 1 ,stride):
        for j in range(0, fm_y - kernel_size[1] + 1,stride):
            #print(" for {}th row, {}th column".format(i,j))
            patch_temp = img[i:(i+kernel_size[0]), j:(j+kernel_size[1])]
            patch[patch_count,:] = patch_temp
            patch_count += 1
    #print("total patches collected", patch_count)
    return patch

# Function for clustering the patches
# Define a custom function that will clamp the images between 0 & 1 , without being too harsh as torch.clamp 
def softclamp01(image_tensor):
    image_tensor_shape = image_tensor.shape
    image_tensor = image_tensor.view(image_tensor.size(0), -1)
    image_tensor -= image_tensor.min(1, keepdim=True)[0]
    image_tensor /= image_tensor.max(1, keepdim=True)[0]
    image_tensor = image_tensor.view(image_tensor_shape)
    return image_tensor

# cluster patch varriants

# Function for clustering the patches - vanilla , useful as a basic routine
# This is incremental image building, collct patches and pass on to the faiss


def cluster_patches_inc(train_patch_loader, patch_size, n_clusters, channel_count, repeat, stride, location=False):
    # we need only a StandardGpuResources per GPU
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)

    D = 16 # This is a issue, will solve this
    res = faiss.StandardGpuResources() 

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device = 0

    kmeans.index = faiss.GpuIndexFlatL2(res, D, cfg)


    buffer = []
    t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    idx = 0
    for _ in range(repeat):
        #net.eval()
        with torch.no_grad():
             counter = 0
             for entry in train_patch_loader:
                if (counter % 500) == 0:
                    print("Images  done:{}, percentage complete:{}".format(counter*500, 100*(counter*500)/len(train_patch_loader)))
                counter = counter + 1     
                img_src, label = entry
                #cond2d_orignet = net(img_src)
                #cond2d_orignet = img_src
                cond2d_orignet = softclamp01(img_src)
                #print("Image shape:",cond2d_orignet.shape) 
                for ch in range(channel_count):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    idx += 1
                    if idx % (channel_count*200) == 0:
                        data = np.concatenate(buffer, axis=0)
                        #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                        #kmeans.partial_fit(data)
                        kmeans.train(data.astype(np.float32))
                        buffer = []
                    if idx % (channel_count*200) == 0:
                        print('Part 1: Partial fit of %4i out of %i'
                              % (idx, repeat * len(train_patch_loader) * channel_count))
                
    dt = time.time() - t0
    print('done in %.2fs.' % dt)
    faiss.write_index(kmeans.index,"kmeans_k"+str(patch_size[0])+"_s0_"+str(n_clusters)+"_v1_sc.index")
    if location:
        return kmeans, loc
    else:
        return kmeans


# Function for clustering the patches - vanilla , useful as a basic routine
# This is at once image building, collect all patches and pass on to the faiss at once
# This function can also return the extracted patch dataset, if required

def cluster_patches_singleshot(train_patch_loader, patch_size, n_clusters, channel_count, repeat, stride, getdata=False, location=False):
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)
    buffer = []
    #t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    index = 0
    for _ in range(repeat):
        #net.eval()
        with torch.no_grad():
             for entry in train_patch_loader:
                img_src, label = entry
                #cond2d_orignet = net(img_src)
                cond2d_orignet = softclamp01(img_src)
                #print("Image shape:",cond2d_orignet.shape) 
                for ch in range(channel_count):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    index += 1
                        #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                        #kmeans.partial_fit(data)
                    if index % (channel_count*2000) == 0:
                        print('Patch Clustering: Patch extraction for all images happenning  of %4i out of %i'
                              % (index, repeat * len(train_patch_loader) * channel_count))
                
             data = np.concatenate(buffer, axis=0)
             kmeans.train(data.astype(np.float32))
    #dt = time.time() - t0
    #print('done in %.2fs.' % dt)
    faiss.write_index(kmeans.index,"kmeans_img_k"+str(patch_size[0])+"_s0_c"+str(n_clusters)+"_sc_ss.index")
    if getdata:
        return kmeans, data


# Function for clustering the patches - in case the patch data is passed from the caller function 
# This is at once image building, collect all patches and pass on to the faiss at once

def cluster_patches_withdata(data, patch_size, n_clusters, channel_count, repeat, stride, location=False):
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=False)
    kmeans.train(data.astype(np.float32))
    #dt = time.time() - t0
    #print('done in %.2fs.' % dt)
    #faiss.write_index(kmeans.index,"kmeans_img_k"+str(patch_size[0])+"_s0_c"+str(n_clusters)+"_sc.index")
    faiss.write_index(kmeans.index,"kmeans_flt_imgnet_alexnet_k"+str(patch_size[0])+"_s0_c"+str(n_clusters)+"_.index")
    if location:
        return kmeans, loc
    else:
        return kmeans


#for name, layer in net.named_modules():
#    print(name, layer)
def extract_layers(model, layers_to_drop = 1):
    return nn.Sequential(*list(model.children())[:-layers_to_drop])
# lets drop the last 30 layers
def extract_lowernet(net, layers_to_drop=9):
    fms_orignet = extract_layers(model = net, layers_to_drop=layers_to_drop)
    for param in fms_orignet.parameters():
        param.requires_grad = False
    return fms_orignet    

# the two functions below are replica of the patch extraction function - at one go
# Only difference is that her we extract symbols from the whole network - now in 
# an adhoc form - this one is only for Lecun MNIST model 
# Automating this part is something we need to take up later

# Basic definitions for MNIST model
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
    
    def forward_symbolic(self, x, n_clusters, index, centroid_lut, patch_size, stride, channel_count): 
        channel_count = 1
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv1(x))
        channel_count = 6
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool1(x)
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv2(x))
        channel_count = 16
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x    




def cluster_patches_singleshot_fullnet_imgnet(train_patch_loader,conv1_net, patch_size, n_clusters, channel_count, repeat, stride, getdata=False, location=False):
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)
    buffer = []
    #t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    index = 0

    for _ in range(repeat):
        # Network architecture 
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        ch_l1 = 64
        # self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        ch_l2 = 16
        # self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.fc1 = nn.Linear(400,120)
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(84, 10) 

        with torch.no_grad():
             for entry in train_patch_loader:
                img_src, label = entry
                #cond2d_orignet = net(img_src)
                img_sc = softclamp01(img_src)
                cond2d_orignet = img_sc
                #print("Image shape:",cond2d_orignet.shape) 
                for ch in range(channel_count):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    index += 1
                        #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                        #kmeans.partial_fit(data)
                    if index % (channel_count*50) == 0:
                        print('Patch Clustering: Patch extraction for images happenning  of %4i out of %i'
                              % (index, repeat * len(train_patch_loader) * channel_count))
                
                # lets get the patches from the first conv layer 
                #for name, layer in fms_orignet.named_modules():
                #    print(name, layer)
                cond2d_orignet = conv1_net(img_sc)
                for ch in range(ch_l1):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    index += 1
                        #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                        #kmeans.partial_fit(data)
                    if index % (channel_count*200) == 0:
                        print('Patch Clustering: Patch extraction for conv1 FMs  happenning  of %4i out of %i'
                              % (index, repeat * len(train_patch_loader) * channel_count))

             data = np.concatenate(buffer, axis=0)
             kmeans.train(data.astype(np.float32))
    #dt = time.time() - t0
    #print('done in %.2fs.' % dt)
    faiss.write_index(kmeans.index,"kmeans_imagenet_fullnet_k"+str(patch_size[0])+"_s0_c"+str(n_clusters)+"_sc.index")
    if getdata:
        return kmeans, data
from patchify import patchify, unpatchify
import time

def extract_patches_new(img, patch_size, patch_stride):
    if len(img.shape) > 2:
        img = np.asarray(img.permute(1,2,0))
        w,h,c = img.shape
    else:
        w,h = img.shape
        img = np.asarray(img)
        c = 1 
    out_w = (w - patch_size[0])//patch_stride + 1 
    out_h = (h - patch_size[0])//patch_stride + 1 
    
    #print("How many patches?",out_w,out_w)
    #print("Channels?",c)
    
    if c == 1:
        #patch_array = extract_patches_strided((patch_size[0],patch_size[1]), img, patch_stride)
        #patch_array = skimage.util.shape.view_as_blocks(img, (patch_size[0],patch_size[1]))
        patch_array = patchify(img, (patch_size[0],patch_size[1]), step=patch_stride)
    else:
        #patch_array = extract_patches_strided((patch_size[0],patch_size[1],c), img, patch_stride)
        #patch_array = skimage.util.shape.view_as_blocks(img, (patch_size[0],patch_size[1],c))
        patch_array = patchify(img, (patch_size[0],patch_size[1],c), step=patch_stride)
        #print("Patch array shape", patch_array.shape)
    patch_array = patch_array.reshape(out_w*out_h*c, patch_size[0]*patch_size[1])    
    return patch_array

def cluster_patches_inc_imgnet_newextract(train_patch_loader, conv1_net, conv2_net, conv3_net, conv4_net, patch_size, n_clusters, channel_count, repeat, location=False):
    # we need only a StandardGpuResources per GPU
    kmeans512 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    kmeans2048 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=2048, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)

    #D = 16 # This is a issue, will solve this
    #res = faiss.StandardGpuResources() 

    #cfg = faiss.GpuIndexFlatConfig()
    #cfg.useFloat16 = True
    #cfg.device = 0

    #kmeans2048.index = faiss.GpuIndexFlatL2(res, D, cfg)
    #kmeans512.index = faiss.GpuIndexFlatL2(res, D, cfg)
    #kmeans2304.index = faiss.GpuIndexFlatL2(res, D, cfg)


    buffer = []
    t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    for _ in range(repeat):
        #net.eval()
        with torch.no_grad():
             counter = 1
             for entry in train_patch_loader:
                img, label = entry
                #cond2d_orignet = net(img_src)
                #cond2d_orignet = img_src
                #img_sc = softclamp01(img_src)
                img_src = img.squeeze()
                cond2d_orignet = img_src
                #print("Image shape:",cond2d_orignet.shape) 
                data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                # lets get the patches from the first conv layer 
                #for name, layer in fms_orignet.named_modules():
                #    print(name, layer)
                cond2d_orignet = conv1_net(img)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,stride)   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                cond2d_orignet = conv2_net(img)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,stride)   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                cond2d_orignet = conv3_net(img)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,stride)   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                cond2d_orignet = conv4_net(img)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,stride)   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                if counter % 50 == 0:
                    print('Imagenet Clustering: images processed so far...', counter)
                # 1 image entry generates 3 ch + 64 ch conv = 67 entries . If we take 10000 images at once, what happens?  
                if counter % 10000 == 0:
                    data = np.concatenate(buffer, axis=0)
                    #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                    #kmeans.partial_fit(data)
                    kmeans512.train(data.astype(np.float32))
                    faiss.write_index(kmeans512.index,"kmeans_full_part_imagenet_k"+str(patch_size[0])+"_s0_"+str(n_clusters)+"_v1_sc.index")
                    buffer = []
                    print('Imagenet Clustering: Partial Fit  ongoing...')
                counter +=1
                
    dt = time.time() - t0
    print('done in %.2fs.' % dt)
    faiss.write_index(kmeans512.index,"kmeans_full_imagenet_k"+str(patch_size[0])+"_s0_"+str(n_clusters)+"_v1_sc.index")
    if location:
        return kmeans512, loc
    else:
        return kmeans512



def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    sigma = np.cov(X, rowvar=False) # [M x M]
    U,S,V = np.linalg.svd(sigma)
    # Whitening constant: prevents division by zero
    epsilon = 0.1
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X.T).T
    X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
    return X_ZCA_rescaled

def cluster_patches_inc_imgnet_newextract_noflt(train_patch_loader, conv1_net, conv2_net, conv3_net, conv4_net, patch_size, n_clusters, channel_count, repeat, location=False):
    # we need only a StandardGpuResources per GPU
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    #kmeans2048 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=2048, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)

    #D = 16 # This is a issue, will solve this
    #res = faiss.StandardGpuResources() 

    #cfg = faiss.GpuIndexFlatConfig()
    #cfg.useFloat16 = True
    #cfg.device = 0

    #kmeans2048.index = faiss.GpuIndexFlatL2(res, D, cfg)
    #kmeans512.index = faiss.GpuIndexFlatL2(res, D, cfg)
    #kmeans2304.index = faiss.GpuIndexFlatL2(res, D, cfg)


    buffer = []
    t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    counter = 1
    for _ in range(repeat):
        #net.eval()
        with torch.no_grad():
             for entry in train_patch_loader:
                img, label = entry
                #cond2d_orignet = net(img_src)
                #cond2d_orignet = img_src
                #img_sc = softclamp01(img_src)
                #img_src = img.squeeze()
                #cond2d_orignet = img_src
                #print("Image shape:",cond2d_orignet.shape) 
                #data = extract_patches_new(cond2d_orignet, patch_size, patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #data = zca_whitening_matrix(data) # get ZCAMatrix
                #data = data.copy(order='C')
                #data = np.ascontiguousarray(data, dtype=np.float32)
                #buffer.append(data)
                ## lets get the patches from the first conv layer 
                ##for name, layer in fms_orignet.named_modules():
                ##    print(name, layer)
                cond2d_orignet = conv1_net(img)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                #cond2d_orignet = conv2_net(img)
                #cond2d_orignet = cond2d_orignet.squeeze()
                #data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #buffer.append(data)
                #cond2d_orignet = conv3_net(img)
                #cond2d_orignet = cond2d_orignet.squeeze()
                #data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #buffer.append(data)
                #cond2d_orignet = conv4_net(img)
                #cond2d_orignet = cond2d_orignet.squeeze()
                #data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #buffer.append(data)
                if counter % 50 == 0:
                    print('Alexnet C1 Clustering: images processed so far...', counter)
                # 1 image entry generates 3 ch + 64 ch conv = 67 entries . If we take 10000 images at once, what happens?  
                if counter % 5000 == 0:
                    data = np.concatenate(buffer, axis=0)
                    #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                    #kmeans.partial_fit(data)
                    kmeans.train(data.astype(np.float32))
                    faiss.write_index(kmeans.index,"kmeans_alexnet_c1_k"+str(patch_size[0])+"_s2_"+str(n_clusters)+"_repeat2_v7.index")
                    buffer = []
                    print('Alexnet C1 Clustering: Partial Fit  ongoing...')
                counter +=1
                
    dt = time.time() - t0
    print('done in %.2fs.' % dt)
    faiss.write_index(kmeans.index,"kmeans_alexnet_c1_k"+str(patch_size[0])+"_s2_"+str(n_clusters)+"_repeat2_v7.index")
    return kmeans



def cluster_patches_inc_imgnet(train_patch_loader, conv1_net, patch_size, n_clusters, channel_count, repeat, stride, location=False):
    # we need only a StandardGpuResources per GPU
    kmeans512 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    kmeans2048 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=2048, niter=200, nredo=repeat, verbose=True)
    kmeans2304 = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=2304, niter=200, nredo=repeat, verbose=True)
    #kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, verbose=True)

    D = 16 # This is a issue, will solve this
    res = faiss.StandardGpuResources() 

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device = 0

    kmeans2048.index = faiss.GpuIndexFlatL2(res, D, cfg)
    kmeans512.index = faiss.GpuIndexFlatL2(res, D, cfg)
    kmeans2304.index = faiss.GpuIndexFlatL2(res, D, cfg)


    buffer = []
    ch_l1 = 64
    t0 = time.time()
    # The online learning part: cycle over the whole dataset 6 times
    idx1 = 0
    idx2 = 0
    for _ in range(repeat):
        #net.eval()
        with torch.no_grad():
             counter = 0
             for entry in train_patch_loader:
                img_src, label = entry
                if counter % 250 == 0:
                    print(counter, "Images done")
                counter +=1
                #cond2d_orignet = net(img_src)
                #cond2d_orignet = img_src
                img_sc = softclamp01(img_src)
                cond2d_orignet = img_sc
                #print("Image shape:",cond2d_orignet.shape) 
                for ch in range(channel_count):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    if idx1 % (channel_count*500*2) == 0:
                        print('Imagenet Clustering:Patch extraction for input image of %4i out of %i repeated channels'
                              % (idx1, repeat * len(train_patch_loader) * channel_count))
                    idx1 += 1
                # lets get the patches from the first conv layer 
                #for name, layer in fms_orignet.named_modules():
                #    print(name, layer)
                cond2d_orignet = conv1_net(img_sc)
                for ch in range(ch_l1):
                    fm_squeezed = cond2d_orignet[:,ch,:,:]
                    img = np.squeeze(fm_squeezed)
                    if location:
                        _, data, loc = extract_patches(img, patch_size, True, stride)
                    else:
                        data = extract_patches(img, patch_size, False, stride)
                    data = np.reshape(data, (len(data), -1))
                    #print ("data shape: {}, dtype:{}".format(data.shape, data.dtype))
                    buffer.append(data)
                    #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                    #kmeans.partial_fit(data)
                    if idx2 % (channel_count*500*2) == 0:
                        print('Imagenet Clustering: Patch extraction for conv1 FMs  happenning  of %4i out of %i repeated channels'
                              % (idx2, repeat * len(train_patch_loader) * ch_l1))
                    idx2 += 1
                # 1 image entry generates 3 ch + 64 ch conv = 67 entries . If we take 10000 images at once, what happens?  
                if (idx1 + idx2) % (4000*67) == 0:
                    data = np.concatenate(buffer, axis=0)
                    #print ("data shape b4 fit: {}, dtype:{}".format(data.shape, data.dtype))
                    #kmeans.partial_fit(data)
                    kmeans512.train(data.astype(np.float32))
                    kmeans2048.train(data.astype(np.float32))
                    kmeans2304.train(data.astype(np.float32))
                    faiss.write_index(kmeans512.index,"kmeans_full_part_imagenet_k"+str(patch_size[0])+"_s0_"+str(n_clusters)+"_v1_sc.index")
                    faiss.write_index(kmeans2048.index,"kmeans_full_part_imagenet_k"+str(patch_size[0])+"_s0_"+str(2048)+"_v1_sc.index")
                    faiss.write_index(kmeans2304.index,"kmeans_full_part_imagenet_k"+str(patch_size[0])+"_s0_"+str(2304)+"_v1_sc.index")
                    buffer = []
                    print('Imagenet Clustering: Partial Fit  image and first conv layer done of %4i out of %i repeated channels'
                          % (idx1 + idx2, repeat * len(train_patch_loader) * channel_count +  repeat * len(train_patch_loader) * ch_l1 ))
                
    dt = time.time() - t0
    print('done in %.2fs.' % dt)
    faiss.write_index(kmeans512.index,"kmeans_full_imagenet_k"+str(patch_size[0])+"_s0_"+str(n_clusters)+"_v1_sc.index")
    faiss.write_index(kmeans2048.index,"kmeans_full_imagenet_k"+str(patch_size[0])+"_s0_"+str(2048)+"_v1_sc.index")
    faiss.write_index(kmeans2304.index,"kmeans_full_imagenet_k"+str(patch_size[0])+"_s0_"+str(2304)+"_v1_sc.index")
    if location:
        return kmeans512, loc
    else:
        return kmeans512


def cluster_patches_singleshot_fullnet_mnist(train_patch_loader,conv1_net, conv2_net, fc1_net, fc2_net, patch_size, n_clusters, channel_count, repeat, stride, getdata=False, location=False):
    kmeans = faiss.Kmeans(d=(patch_size[0]*patch_size[1]), k=n_clusters, niter=200, nredo=repeat, verbose=True)
    buffer = []
    index = 0
    counter = 1
    for _ in range(repeat):
        # Network architecture 
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.fc1 = nn.Linear(400,120)
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(84, 10) 
        with torch.no_grad():
             for entry in train_patch_loader:
                img, label = entry
                img = softclamp01(img)
                img_src = img.squeeze()
                data = extract_patches_new(img_src, patch_size, patch_size[0])
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                index += 1
                # lets get the patches from the first conv layer 
                cond2d_orignet = conv1_net(img)
                #cond2d_orignet = softclamp01(cond2d_orignet)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                index += 1
                cond2d_orignet = conv2_net(img)
                #cond2d_orignet = softclamp01(cond2d_orignet)
                cond2d_orignet = cond2d_orignet.squeeze()
                data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                index += 1
                #cond2d_orignet = fc1_net(img)
                #cond2d_orignet = cond2d_orignet.squeeze()
                #data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #buffer.append(data)
                #index += 1
                #cond2d_orignet = fc2_net(img)
                #cond2d_orignet = cond2d_orignet.squeeze()
                #data = extract_patches_new(cond2d_orignet, patch_size,patch_size[0])   
                #data = np.reshape(data, (len(data), -1))
                #buffer.append(data)
                #index += 1
                if counter % 500 == 0:
                    print('MNIST Clustering: images processed so far...', counter)
                if counter % 5000 == 0:
                    data = np.concatenate(buffer, axis=0)
                    kmeans.train(data.astype(np.float32))
                    faiss.write_index(kmeans.index,"kmeans_mnist_fullnet_k"+str(patch_size[0])+"_s2_c"+str(n_clusters)+"_faiss_sc.index")
                    buffer = []
                    print('MNIST Clustering: Partial Fit  ongoing...')
                counter +=1
    #dt = time.time() - t0
    #print('done in %.2fs.' % dt)
    faiss.write_index(kmeans.index,"kmeans_mnist_fullnet_k"+str(patch_size[0])+"_s2_c"+str(n_clusters)+"_faiss_sc.index")


def reconstruct_fm_from_patches(patches, fm_size, kernel_size, stride=0):
    fm_x, fm_y = fm_size[0], fm_size[1]
    old_x, old_y = fm_x, fm_y
    #print(fm_x,fm_y)
    #print(kernel_size[0])
    # Set the stride
    if stride == 0:
        stride = kernel_size[0]  

    # handle the case where the FM is odd sized
    if stride == 0:
        stride = kernel_size[0] 
    # handle the case where the FM is odd sized
    if fm_x % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_x = fm_x % stride
        augment_x = stride - augment_delta_x
        fm_x = fm_x + augment_x
    else:
        augment_x = 0

    if fm_y % stride != 0:
        # repeat the last row and column  stride times
        augment_delta_y = fm_y % stride
        augment_y = stride - augment_delta_y
        fm_y = fm_y + augment_x
    else:
        augment_y = 0
        
    patches_x = ((fm_x - kernel_size[0]) // stride) + 1 
    patches_y = ((fm_y -  kernel_size[1]) // stride) + 1
    #print("total patches possible to be used", patches_x*patches_y)
    fm = np.zeros((fm_x, fm_y), dtype=np.double)
    patch_count = 0
    for i in range(0, fm_x - kernel_size[0] + 1 ,stride):
        for j in range(0, fm_y - kernel_size[1] + 1,stride):
            if stride == kernel_size[0]:
                fm[i:(i+kernel_size[0]), j:(j+kernel_size[1])] = patches[patch_count]
            else:
                # This is the case where we implement the sliding window mechanism   
                # first stride on the first row will not need any averaging
                if i == 0: # first row
                    fm[i:(i+kernel_size[0]), j:(j+kernel_size[1])] =  patches[patch_count]
                else:  # all first columns
                    if j == 0:
                        fm[i:(i+kernel_size[0] - stride), j:(j+kernel_size[1])] += (patches[patch_count])[0:(kernel_size[0] - stride), 0:(kernel_size[1])]
                        fm[i:(i+kernel_size[0] - stride), j:(j+kernel_size[1])] /=2
                        # Now the final part that would need no averaging
                        fm[(i+kernel_size[0] - stride):(i+kernel_size[0]), j:(j+kernel_size[1])] = (patches[patch_count])[(kernel_size[0] - stride):(kernel_size[0]), 0:(kernel_size[1])]  
                    else: # general case , patch can be anywhere in the middle having overlaps in every direction
                        fm[i:(i+kernel_size[0] - stride), j:(j+kernel_size[1])] += (patches[patch_count])[0:(kernel_size[0] - stride), 0:(kernel_size[1])]
                        fm[i:(i+kernel_size[0] - stride), j:(j+kernel_size[1])] /=2
                        fm[i:(i+kernel_size[0]), j:(j+kernel_size[1] - stride)] += (patches[patch_count])[0:(kernel_size[0]), 0:(kernel_size[1] - stride)]
                        fm[i:(i+kernel_size[0]), j:(j+kernel_size[1] - stride)] /=2
                        # Now the final part that would need no averaging
                        fm[(i+kernel_size[0] - stride):(i+kernel_size[0]), (j+kernel_size[1] - stride):(j+kernel_size[1])] = (patches[patch_count])[(kernel_size[0] - stride):(kernel_size[0]), (kernel_size[1] - stride):(kernel_size[1])]  
            patch_count += 1
    #print("total patches used", patch_count)
    # handle the case where the FM is odd sized
    if old_x % stride != 0 or old_y % stride != 0:
        #img = np.zeros((fm_x-augment,fm_y-augment))
        img = np.zeros((old_x,old_y))
        img[:,:] = fm[:old_x,:old_y]
        return img
    else:
        return fm

# Function for abstracting an image to symbolic form - this is the simplest version
         

def fm_to_symbolic_fm(fm, n_clusters, kmeans, centroid_lut,  patch_size, stride, channel_count, ana=False, multi=False, instr=False, pdf=None):
    if ana:
        buffer_symset = []
    if multi:
        buffer2 = []
        buffer3 = []    
    buffer1 = []
    #print(fm.shape)
    for ch in range(channel_count):
        if channel_count == 1:
            img = fm
            #print(fm.shape)
            fm_x = fm.shape[0]  
            fm_y = fm.shape[1] 
        else:     
            img = fm[ch,:,:]
            #print(fm.shape)
            fm_x = fm.shape[1]  
            fm_y = fm.shape[2]  
            img = img.squeeze()
        #print(img.shape)
        #data, _ , _ = extract_patches(img, patch_size, stride)
        data = extract_patches(img, patch_size, False, stride)
        #print ("data shape: {}".format(data.shape))
        data = np.reshape(data, (len(data), -1))
        #buffer.append(data)
        #clusters = kmeans.predict(data)
        if multi:
            distances, clusters = kmeans.search(data.astype(np.float32), multi)
        else:        
            distances, clusters = kmeans.search(data.astype(np.float32), 1)
        #buffer = []
        #print("cluster shape: {}".format(clusters.shape))
        reconstructed_patches = []
        if multi:
            reconstructed_patches2 = []
            reconstructed_patches3 = []
        for center in clusters:
            #print("center patch shape: {}".format(center.shape))
            center_patch = centroid_lut[center]
            
            if multi:
                center_patch2 = centroid_lut[center[1]]
                center_patch3 = centroid_lut[center[2]]
            if ana:
                buffer_symset.append(center.item())
            #print("center symbol shape: {}".format(center_patch.shape))
            center_patch = center_patch.reshape(patch_size)
            if instr:
                idx = center.item()
                pdf[idx] += 1
            if multi:
                center_patch2 = center_patch2.reshape(patch_size)
                center_patch3 = center_patch3.reshape(patch_size)    
            #print(center_patch.shape)
            reconstructed_patches.append(center_patch)
            if multi:
                reconstructed_patches2.append(center_patch2)
                reconstructed_patches3.append(center_patch3)
            
        #print(len(reconstructed_patches)) 
        if multi:   
            reconstructed_patch_centers2 =  np.array(reconstructed_patches2, dtype=np.double)
            reconstructed_patch_centers3 =  np.array(reconstructed_patches3, dtype=np.double)
        reconstructed_patch_centers =  np.array(reconstructed_patches, dtype=np.double)
        if multi:
            reconstructed_fm2 = reconstruct_fm_from_patches(reconstructed_patch_centers2, (fm_x, fm_y), patch_size, stride)
            reconstructed_fm3 = reconstruct_fm_from_patches(reconstructed_patch_centers3, (fm_x, fm_y), patch_size, stride)
        #print(reconstructed_patch_centers.shape)
        reconstructed_fm = reconstruct_fm_from_patches(reconstructed_patch_centers, (fm_x, fm_y), patch_size, stride)
        #print(reconstructed_fm.shape)
        if multi:
            buffer2.append(reconstructed_fm2)
            buffer3.append(reconstructed_fm3)
        buffer1.append(reconstructed_fm)
        
    reconstructed_fm_ch =  np.array(buffer1, dtype=np.double)
     
    if multi:
        reconstructed_fm_ch2 =  np.array(buffer2, dtype=np.double)
        reconstructed_fm_ch3 =  np.array(buffer3, dtype=np.double)
    if ana:  
        reconstructed_fm_sym =  np.array(buffer_symset, dtype=np.int32)  
    #print(reconstructed_fm_ch.shape)
    if ana:  
        return reconstructed_fm_ch, reconstructed_fm_sym
    elif multi:
        return reconstructed_fm_ch , reconstructed_fm_ch2, reconstructed_fm_ch3
    elif instr:
        return reconstructed_fm_ch  , pdf 
    else:                
        return reconstructed_fm_ch  
