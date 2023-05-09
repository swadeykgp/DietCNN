import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from tqdm.auto import tqdm

# For training
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import faiss
import sys
sys.path.insert(1, '../../core')
from lut_utils_full import *
import warnings
warnings.filterwarnings('ignore')
from patchlib import *
import multiprocessing
from joblib import Parallel, delayed
PARALLEL = 32

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
    def __init__(self, symp , n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut, f2_bias_lut, f3_bias_lut, relu_lut ):
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
        # This is the input layer first Convolution
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

    def update_sym(self, c1f, c2f , f1f, f2f, conv_lut, fc_lut, add_lut, relu_lut, c1b, c2b , f1b, f2b):
        self.weights.append(c1f)
        self.weights.append(c2f)
        self.weights.append(f1f)
        self.weights.append(f2f)
        self.conv_lut = conv_lut 
        self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.relu_lut =  relu_lut
        self.bias_lut = [] 
        self.bias_lut.append(c1b)
        self.bias_lut.append(c2b)
        self.bias_lut.append(f1b)
        self.bias_lut.append(f2b)

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


def lenet_sym(net, state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b, f2b, f3b, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride) -> CNN_LeNetSym:
    # only needed for first time init
    net.load_state_dict(state_dict)
    sym_params = [] # These are layer 1 conv params
    net = CNN_LeNet() 
    net.load_state_dict(state_dict)
    c1_filter = net.conv1.weight.data.clone() 
    n,ic,kw,kh = c1_filter.shape 
    c1_bias = None
    filter_patch_size = (1,1)
    filter_stride = 1
    bs = filter_to_sym_conv(filter_index_conv, c1_filter, filter_patch_size, ic, n, filter_stride, False)
    sym_params.append(bs) 
    #sym_params.append(c1_bias) 


    c2_filter = net.conv2.weight.data.clone() 
    n2,ic2,kw2,kh2 = c2_filter.shape 
    c2_bias = None
    bs2 = filter_to_sym_conv(filter_index_conv, c2_filter, filter_patch_size, ic2, n2, filter_stride, False)
    
    sym_params.append(bs2) 
    #sym_params.append(c2_bias) 


    f1_filter = net.fc1.weight.data.clone() 
    kw3,kh3 = f1_filter.shape 
    f1_bias = None
    bs3 = filter_to_sym_fc(filter_index_fc, f1_filter, patch_size, patch_stride, False)

    sym_params.append(bs3) 
    #sym_params.append(f1_bias) 

    f2_filter = net.fc2.weight.data.clone() 
    f2_bias = None
    bs4 = filter_to_sym_fc(filter_index_fc, f2_filter, patch_size, patch_stride, False)
    sym_params.append(bs4) 
    #sym_params.append(f2_bias) 

    f3_filter = net.fc3.weight.data.clone() 
    bs5 = filter_to_sym_fc(filter_index_fc,f3_filter, patch_size, patch_stride, False)
    sym_params.append(bs5) 


    model = CNN_LeNetSym(sym_params, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b, f2b, f3b, relu_lut)

    model.load_state_dict(state_dict)
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
            torch.save(model.state_dict(), 'best_lenet_full.pt')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(model.state_dict(), 'best_lenet_full.pt')



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
        image, labels = data
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
        
        #if counter % 50 == 0 and counter > 0:
        #if counter % 1 == 0 and counter > 0:

        c1_filter = net.conv1.weight.data.clone() 
        c2_filter = net.conv2.weight.data.clone() 
        f1_filter = net.fc1.weight.data.clone()
        f2_filter = net.fc2.weight.data.clone()
        #f3_filter = net.fc3.weight.data.clone()
        with torch.no_grad():
            c1_bias = net.conv1.bias.clone()
            c1_bias = np.asarray(c1_bias)
            c2_bias = net.conv2.bias.clone()
            c2_bias = np.asarray(c2_bias)
            f1_bias = net.fc1.bias.clone()
            f1_bias = np.asarray(f1_bias)
            f2_bias = net.fc2.bias.clone()
            f2_bias = np.asarray(f2_bias)
            #f3_bias = net.fc3.bias.clone()
            #f3_bias = np.asarray(f3_bias)
        filter_index_conv = create_index_conv(n_cluster_conv_filters, c1_filter, c2_filter, patch_size, patch_stride )
        filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter, f2_filter,  patch_size, patch_stride )
        n,ic,kw,kh = c1_filter.shape
        c1f = filter_to_sym_conv(filter_index_conv, c1_filter, patch_size, ic, n, patch_stride, False)

        n2,ic2,kw2,kh2 = c2_filter.shape
        c2f = filter_to_sym_conv(filter_index_conv, c2_filter, patch_size, ic2, n2, patch_stride, False)

        f1_filter = net.fc1.weight.data.clone() 
        f1f = filter_to_sym_fc(filter_index_fc, f1_filter, patch_size, patch_stride, False)

        f2_filter = net.fc2.weight.data.clone() 
        f2f = filter_to_sym_fc(filter_index_fc, f2_filter, patch_size, patch_stride, False)
        

        conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
        c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)  
        c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
        fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
        add_lut = create_add_luts(centroid_lut, n_clusters, index)
        relu_lut = create_relu_lut(centroid_lut, n_clusters, index)
        f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)  
        f2_bias_lut = create_bias_luts(centroid_lut, n_clusters, f2_bias, index)  
        #f3_bias_lut = create_bias_luts(centroid_lut, n_clusters, f3_bias, index)  
        net.update_sym(c1f, c2f , f1f, f2f, conv_lut, fc_lut, add_lut, relu_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut , f2_bias_lut)

                          
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
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

 

if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    apply_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])
    #bs = 128 
    bs = 256 
  
    trainset = datasets.MNIST(root='../../../dataset', train=True, download=True, transform=apply_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, pin_memory=True)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, pin_memory=True)
    apply_test_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])
    testset = datasets.MNIST(root='../../../dataset', train=False, download=True, transform=apply_test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, pin_memory=True)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=True)
    #pretrained_model = "./mnist_v0.pt"
    pretrained_model = "./best_lenet_full.pt"
    
    index = faiss.read_index("./kmeans_mnist_fullnet_k1_s1_c128_faiss_v10.index")
    n_clusters=128
    patch_size = (1, 1)
    patch_stride = 1 
    centroid_lut = index.reconstruct_n(0, n_clusters)

    net = CNN_LeNet() 
    net.load_state_dict(torch.load(pretrained_model))
    state_dict = torch.load(pretrained_model) 
    
    #to_device(net, device)
    with torch.no_grad():
        c1_bias = net.conv1.bias.clone()
        c1_bias = np.asarray(c1_bias)
        c2_bias = net.conv2.bias.clone()
        c2_bias = np.asarray(c2_bias)
        f1_bias = net.fc1.bias.clone()
        f1_bias = np.asarray(f1_bias)
        f2_bias = net.fc2.bias.clone()
        f2_bias = np.asarray(f2_bias)
        f3_bias = net.fc3.bias.clone()
        f3_bias = np.asarray(f3_bias)
        
    c1_filter = net.conv1.weight.data.clone() 
    c2_filter = net.conv2.weight.data.clone() 
    
    f1_filter = net.fc1.weight.data.clone()
    f2_filter = net.fc2.weight.data.clone()
    f3_filter = net.fc3.weight.data.clone()

    n_cluster_conv_filters = 64
    n_cluster_fc_filters = 128

    filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter, f2_filter,  patch_size, patch_stride )
    filter_index_conv = create_index_conv(n_cluster_conv_filters, c1_filter, c2_filter, patch_size, patch_stride )
    fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
    conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
    add_lut = create_add_luts(centroid_lut, n_clusters, index)
    relu_lut = create_relu_lut(centroid_lut, n_clusters, index)
    c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)  
    c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)  
    f1_bias_lut = create_bias_luts(centroid_lut, n_clusters, f1_bias, index)  
    f2_bias_lut = create_bias_luts(centroid_lut, n_clusters, f2_bias, index)  
    f3_bias_lut = create_bias_luts(centroid_lut, n_clusters, f3_bias, index)  
    
    print(" Symbolic model loading started...")
    t = time.process_time()
    net = lenet_sym(net,state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut,f2_bias_lut, f3_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
    elapsed_time3 = time.process_time() - t
    print("Symbolic model loading completed in:",elapsed_time3)
 
    for param in net.parameters():
        param.requires_grad = True
    learning_rate = 0.005 
    print("Training Started !!!!!!")
    #optimizer = optim.Adam(net.parameters(), lr=0.1)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-2)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-2)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    #loss_criterion = nn.CrossEntropyLoss
    loss_criterion = F.nll_loss
    epochs = 60
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()  

    # lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, net = model_training(net, trainloader, optimizer, loss_criterion,   centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index)
        valid_epoch_loss, valid_epoch_acc = validate(net, testloader, loss_criterion)
        scheduler.step(valid_epoch_loss)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(valid_epoch_loss, epoch, net, optimizer, loss_criterion)
        with open('lenetconv_train_loss.txt', 'w') as f:
            for item in train_loss:
                f.write("%s\n" % item)
        with open('lenetconv_valid_loss.txt', 'w') as f:
            for item in valid_loss:
                f.write("%s\n" % item)
        with open('lenetconv_valid_acc.txt', 'w') as f:
            for item in valid_acc:
                f.write("%s\n" % item)
        with open('lenetconv_train_acc.txt', 'w') as f:
            for item in train_acc:
                f.write("%s\n" % item)
        print('-'*50)
    
    # save the trained model weights for a final time
    save_model(epochs, net, optimizer, loss_criterion)
    # save the loss and accuracy plots
    print("Training Done !!!!!!")
