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
from lut_utils_sig import *
import warnings
warnings.filterwarnings('ignore')
from patchlib import *

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

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
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x

class CNN_LeNetSym(nn.Module):
    def __init__(self, symp , n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1_bias_lut, f2_bias_lut, relu_lut ):
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

    def update_sym(self, c1f, c2f ,  conv_lut, add_lut,  relu_lut, c1b, c2b ):
        self.c1_weights = c1f
        self.c2_weights = c2f 
        #self.f1_weights = None 
        #self.f2_weights = None
        #self.f3_weights = None
        self.conv_lut = conv_lut 
        #self.fc_lut = fc_lut 
        self.add_lut = add_lut 
        self.relu_lut =  relu_lut
        self.c1_bias_lut =  c1b
        self.c2_bias_lut =  c2b

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
                   #if j < (cs_syms -2):
                   #    if tmp_sym > cpart1_array[j+1]:
                   #        # problem! order is broken
                   #        tmp = tmp_sym
                   #        tmp_sym = cpart1_array[j+1]
                   #        cpart1_array[j+1] = tmp
                   #        cpart1_array = np.sort(cpart1_array)
                   #        j +=1
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
            c_out_buffer[i] = bias_lut[tmp_sym,k]
            #c_out_buffer[0][i] = tmp_sym



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
            x = self.SymConv2D(x,1, 6, self.c1_weights, self.c1_biases, 5, 1, 1, # convolution parameters
                self.n_clusters, self.index, self.centroid_lut, self.patch_size, self.patch_stride, # patch params 
                self.conv_lut, self.add_lut, self.c1_bias_lut, 5, 2) 
            x = self.SymReLU(x)
            x = self.SymConv2D(x,6, 16, self.c2_weights, self.c2_biases, 5, 1, 1, # convolution parameters
                self.n_clusters, self.index, self.centroid_lut, self.patch_size, self.patch_stride, # patch params 
                self.conv_lut, self.add_lut, self.c2_bias_lut, 5, 2) 
            x = self.SymReLU(x)
            x = symbolic_to_image_win(x, 5, 5, 16, self.centroid_lut,self.patch_size)
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
            x = x.reshape(1,400) 
            #x = x.unsqueeze(0)
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = self.fc3(x)
            x = F.softmax(x,dim=1)
            if b ==0:
                x_out_bat = x
                #print(x_out_bat.shape)
            else:
                x_out_bat = torch.cat((x_out_bat,x), dim=0)
                #print(x_out_bat.shape)
        return x_out_bat
       

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


def lenet_sym(net, state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b, f2b, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride) -> CNN_LeNetSym:
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
    sym_params.append(c1_bias) 


    c2_filter = net.conv2.weight.data.clone() 
    n2,ic2,kw2,kh2 = c2_filter.shape 
    c2_bias = None
    bs2 = filter_to_sym_conv(filter_index_conv, c2_filter, filter_patch_size, ic2, n2, filter_stride, False)
    
    sym_params.append(bs2) 
    sym_params.append(c2_bias) 


    f1_filter = net.fc1.weight.data.clone() 
    kw3,kh3 = f1_filter.shape 
    f1_bias = None
    bs3 = filter_to_sym_fc(filter_index_fc, f1_filter, patch_size, patch_stride, False)

    sym_params.append(bs3) 
    sym_params.append(f1_bias) 

    f2_filter = net.fc2.weight.data.clone() 
    f2_bias = None
    bs4 = filter_to_sym_fc(filter_index_fc, f2_filter, patch_size, patch_stride, False)
    sym_params.append(bs4) 
    sym_params.append(f2_bias) 

    f3_filter = net.fc3.weight.data.clone() 
    bs5 = filter_to_sym_fc(filter_index_fc,f3_filter, patch_size, patch_stride, False)
    sym_params.append(bs5) 


    model = CNN_LeNetSym(sym_params, n_clusters, index, centroid_lut, patch_size, patch_stride, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, f1b, f2b, relu_lut)

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
            torch.save(model.state_dict(), 'best_lenet_sig.pt')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(model.state_dict(), 'best_lenet_sig.pt')



def model_training(net, trainloader, optimizer, criterion,   centroid_lut, n_cluster_conv_filters, n_cluster_fc_filters, patch_size, patch_stride,  index):
    net.train()
    print('Train')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
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
        with torch.no_grad():
            c1_bias = net.conv1.bias.clone()
            c1_bias = np.asarray(c1_bias)
            c2_bias = net.conv2.bias.clone()
            c2_bias = np.asarray(c2_bias)
        
        filter_index_conv = create_index_conv(n_cluster_conv_filters, c1_filter, c2_filter, patch_size, patch_stride )
        conv_lut = create_conv_luts(centroid_lut, filter_index_conv , n_clusters, n_cluster_conv_filters, index)
        c1_bias_lut = create_bias_luts(centroid_lut, n_clusters, c1_bias, index)  
        c2_bias_lut = create_bias_luts(centroid_lut, n_clusters, c2_bias, index)
        #f1_filter = net.fc1.weight.data.clone()
        #f2_filter = net.fc2.weight.data.clone()
        #f3_filter = net.fc3.weight.data.clone()
        #filter_index_fc = create_index_fc(n_cluster_fc_filters, f1_filter, f2_filter,  patch_size, patch_stride )
        #fc_lut = create_fc_luts(centroid_lut, filter_index_fc , n_clusters, n_cluster_fc_filters, index)
        add_lut = create_add_luts(centroid_lut, n_clusters, index)
        relu_lut = create_relu_lut(centroid_lut, n_clusters, index)

        n,ic,kw,kh = c1_filter.shape
        c1f = filter_to_sym_conv(filter_index_conv, c1_filter, patch_size, ic, n, patch_stride, False)

        n2,ic2,kw2,kh2 = c2_filter.shape
        c2f = filter_to_sym_conv(filter_index_conv, c2_filter, patch_size, ic2, n2, patch_stride, False)

        #bs3 = filter_to_sym_fc(filter_index_fc, f1_filter, patch_size, patch_stride, False)
        #bs4 = filter_to_sym_fc(filter_index_fc, f2_filter, patch_size, patch_stride, False)
        #bs5 = filter_to_sym_fc(filter_index_fc, f3_filter, patch_size, patch_stride, False)

        net.update_sym( c1f, c2f ,  conv_lut, add_lut,  relu_lut, c1_bias_lut, c2_bias_lut)
        #print("Symbolic net updated") 
                          
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
   
    device = get_default_device()
    

    trainset = datasets.MNIST(root='../../../dataset', train=True, download=True, transform=apply_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, pin_memory=True)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False, pin_memory=True)
    apply_test_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])
    testset = datasets.MNIST(root='../../../dataset', train=False, download=True, transform=apply_test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, pin_memory=True)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=True)
    #pretrained_model = "./mnist_bl.pt"
    pretrained_model = "./best_lenet.pt"
    
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
        
    c1_filter = net.conv1.weight.data.clone() 
    c2_filter = net.conv2.weight.data.clone() 
    
    f1_filter = net.fc1.weight.data.clone()
    f2_filter = net.fc2.weight.data.clone()

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
    
    print(" Symbolic model loading started...")
    t = time.process_time()
    net = lenet_sym(net,state_dict, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, c1_bias_lut, c2_bias_lut, None,None,relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
    elapsed_time3 = time.process_time() - t
    print("Symbolic model loading completed in:",elapsed_time3)
 
    for param in net.parameters():
        param.requires_grad = True

    print("Training Started !!!!!!")
    learning_rate=0.002
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
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
