import numpy as np
from patchify import patchify, unpatchify
import time
import torch
torch.manual_seed(0)
np.random.seed(0)
from skimage.util import view_as_blocks


def extract_patches(img, patch_size, patch_stride):
    if len(img.shape) > 2:
        w,h,c = img.shape
    else:
        w,h = img.shape
        c = 1
    out_w = (w - patch_size[0])//patch_stride + 1
    out_h = (h - patch_size[0])//patch_stride + 1
    
    #print("How many patches?",out_w,out_w)
    
    if c == 1:
        #patch_array = extract_patches_strided((patch_size[0],patch_size[1]), img, patch_stride)
        #patch_array = view_as_blocks(img, (patch_size[0],patch_size[1]))
        patch_array = patchify(img, (patch_size[0],patch_size[1]), step=patch_stride)
    else:
        #patch_array = extract_patches_strided((patch_size[0],patch_size[1],c), img, patch_stride)
        #patch_array = view_as_blocks(img, (patch_size[0],patch_size[1],c))
        patch_array = patchify(img, (patch_size[0],patch_size[1],c), step=patch_stride)
        #print("Patch array shape", patch_array.shape)
    patch_array = patch_array.reshape(out_w*out_h*c, patch_size[0]*patch_size[1])    
    return patch_array

def image_to_symbolic(img, index, patch_size, patch_stride):
    patch_array = extract_patches(img,  patch_size, patch_stride)
    #np.savetxt('firstimage.txt', patch_array, delimiter =', ', fmt='%4.10f')    
    #start = time.process_time()
    _, symbol_array = index.search(patch_array.astype(np.float32), 1)
    #elapsed = (time.process_time() - start)
    #print('Symbol conversion time:',elapsed) 
    symbol_array =symbol_array.squeeze()
    #print("Symbolic array shape", symbol_array.shape)
    return symbol_array
    #return patch_array

def symbolic_to_image(symbolic, w,h,c , centroid_lut,patch_size, patch_stride):
    out_w = (w - patch_size[0])//patch_stride + 1
    out_h = (h - patch_size[0])//patch_stride + 1
    if c == 1:
        sym_img = centroid_lut[symbolic.reshape(out_w,out_h)]
        sym_img = sym_img.reshape(out_w,out_h,patch_size[0],patch_size[0])
        reconstructed_image = unpatchify(sym_img, (w,h))
        
    else:
        sym_img = centroid_lut[symbolic.reshape(out_w,out_h,c)] #112, 112, 1, 2, 2, 3
        sym_img = sym_img.reshape(out_w,out_h, 1, patch_size[0], patch_size[0], c)
        reconstructed_image = unpatchify(sym_img, (w,h,c))
    return reconstructed_image


# Legacy patch extraction method - to be removed, still used in filters LUT creation
# Function for patch extraction from an image
def extract_patches_legacy(fm_squeezed, kernel_size, loc, stride=0):
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

# main function for purification defense
def dietcnn_purify(images, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count):
    #adv = images.detach().clone()
    adv = images
    n, c, w, h = adv.shape
    #print(n,c,h,w)
    for i in range(n):
        X = adv[i,:,:,:]
        X = X.squeeze()
        X = X.permute(1,2,0)
        X = np.asarray(X)
        symbolic = image_to_symbolic(X, index, patch_size, patch_stride)
        Xsym_ = symbolic_to_image(symbolic, w, h, c, centroid_lut, patch_size, patch_stride)
        Xsym_ = torch.from_numpy(Xsym_)
        Xsym_ = Xsym_.permute(2,0,1)
        Xsym = torch.tensor(Xsym_, requires_grad=True)
        if i == 0:
            purified_images = Xsym.float()
            purified_images = purified_images.unsqueeze(0)
        else:
            purified_images = torch.cat([purified_images, (Xsym.float()).unsqueeze(0)], dim=0)
    adv_purified = purified_images.detach()
    #print(adv_purified.shape)
    adv_purified.requires_grad_()
    adv_purified.retain_grad()
    return adv_purified

# main function for purification defense
def dietcnn_encode(images, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count):
    X = images
    X = X.squeeze()
    X = X.permute(1,2,0)
    X = np.asarray(X)
    symbolic = image_to_symbolic(X, index, patch_size, patch_stride)
    return symbolic


# main function for purification defense
def dietcnn_decode(symbolic, centroid_lut, patch_size, patch_stride, w,h,c):
    Xsym_ = symbolic_to_image(symbolic, w, h, c, centroid_lut, patch_size, patch_stride)
    Xsym_ = torch.from_numpy(Xsym_)
    Xsym_ = Xsym_.permute(2,0,1)
    Xsym = torch.tensor(Xsym_, requires_grad=True)
    return Xsym
