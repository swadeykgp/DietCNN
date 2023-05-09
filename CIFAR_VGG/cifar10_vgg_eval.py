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
sys.path.insert(1, '../core')
from lut_utils_cifar import *
import warnings
warnings.filterwarnings('ignore')
from patchlib import *
import multiprocessing
from joblib import Parallel, delayed
from vgg_sym import *
from eval_vgg_cifar import *
PARALLEL = 30 

HOWDY = 20000000 

# Test accuracy of symbolic inference
def test_fullsym_acc(model, atk,  data_iter, batch_size, clamp, top_5, std):
    correct = 0 
    total = 0 
    counter = 0
    model.eval()
    for data in data_iter:
        X, y = data
        if counter > HOWDY:
            break
        if(clamp):
            X = softclamp01(X)
        if atk:
            X_atk = atk(X, y)
            X_sym = X_atk.data.cpu().numpy().copy()
        else:
            X_atk = X
            X_sym = X  
        start_t = time.time()

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
        counter +=batch_size 
        total += batch_size
        if std == 0: 
            if not top_5:
                if(counter > 0 and counter % batch_size == 0):
                    print("PyTorch optimized model test accuracy :{}% ".format(100*round(correct/total, 4)))
            else:
                if(counter > 0 and counter % batch_size == 0):
                    print("PyTorch  model test accuracy Top-5 :{}% ".format(100*round(correct/total, 4)))
        elif std == 1: 
            if not top_5:
                if(counter > 0 and counter % batch_size == 0):
                    print("Full standard model test accuracy :{}% ".format(100*round(correct/total, 4)))
            else:
                if(counter > 0 and counter % batch_size == 0):
                    print("Full standard  model test accuracy Top-5 :{}% ".format(100*round(correct/total, 4)))
        else:
            if not top_5:
                if(counter > 0 and counter % batch_size == 0):
                    print("Full symbolic model test accuracy DietCNN :{}% ".format(100*round(correct/total, 4)))
            else:
                if(counter > 0 and counter % batch_size == 0):
                    print("Full symbolic model test accuracy Top-5 DietCNN :{}% ".format(100*round(correct/total, 4)))
        #break 
        end = time.time()
        print("elapsed time for one batch:", end - start_t, ", batch number: ", counter//batch_size)
    return round(correct/total, 4)


if __name__ == '__main__':
    if (len(sys.argv) - 1) < 2:
        print ("Call with first argument 0: Standard PyTorch inference, 1: Standard MAC implementation, 2: DietCNN inference . Also specify a second argument as 1 for instrumentation ")
        sys.exit()
    
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    bs = 25 
    #bs = 1 
  
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #trainset = torchvision.datasets.CIFAR10(root='../../dataset', train=True, download=False, transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='../../dataset/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #random_indices = list(range(0, len(testset), 10))
    #print(len(random_indices))
    #testset_subset = torch.utils.data.Subset(testset, random_indices)
    #testloader = torch.utils.data.DataLoader(testset_subset, batch_size=bs, shuffle=False)



    net = VGG('VGG11')
    pretrained_model = "./cifar_vgg_sym_v3.pt"
    #pretrained_model = "./cifar_vgg_nobn_sym.pt"
    sd = torch.load(pretrained_model, map_location=torch.device('cpu'))
    #print(sd['net']) 
    net.load_state_dict(sd['net'])
    #net.eval()
    #if torch.cuda.is_available():
    #    net.cuda()



    #c_filter = []
    #c_filter.append(net.features[0].weight.data.clone())
    #c_filter.append(net.features[2].weight.data.clone())
    #c_filter.append(net.features[4].weight.data.clone())
    #c_filter.append(net.features[6].weight.data.clone())
    #c_filter.append(net.features[8].weight.data.clone())
    #c_filter.append(net.features[10].weight.data.clone())
    #c_filter.append(net.features[12].weight.data.clone())
    #c_filter.append(net.features[14].weight.data.clone())
    #f1_filter = net.classifier[0].weight.data.clone()

    #with torch.no_grad():
    #    c1_bias = net.features[0].bias.clone()
    #    c1_bias = np.asarray(c1_bias)
    #    c2_bias = net.features[2].bias.clone()
    #    c2_bias = np.asarray(c2_bias)
    #    c3_bias = net.features[4].bias.clone()
    #    c3_bias = np.asarray(c3_bias)
    #    c4_bias = net.features[6].bias.clone()
    #    c4_bias = np.asarray(c4_bias)
    #    c5_bias = net.features[8].bias.clone()
    #    c5_bias = np.asarray(c5_bias)
    #    c6_bias = net.features[10].bias.clone()
    #    c6_bias = np.asarray(c6_bias)
    #    c7_bias = net.features[12].bias.clone()
    #    c7_bias = np.asarray(c7_bias)
    #    c8_bias = net.features[14].bias.clone()
    #    c8_bias = np.asarray(c8_bias)
    #    f1_bias = net.classifier[0].bias.clone()
    #    f1_bias = np.asarray(f1_bias)
    conv_patch_size = (1, 1)
    patch_size = (1, 1)
    all_patch_size = (1, 1)
    n_cluster_conv_filters = 256
    n_cluster_fc_filters = 128
    conv_stride = 1
    index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_512_v0.index")
    #index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_256_v0.index")
    #index = faiss.read_index("./kmeans_vgg11_fullnet_cifar10_k1_64_v0.index")
    n_clusters=512
    #n_clusters=256
    #n_clusters=64
    patch_stride = 1
    centroid_lut = index.reconstruct_n(0, n_clusters)
    with open('cifar10_conv_flt.index', "rb") as f:
        filter_index_conv = pickle.load(f)
    with open('cifar10_fc_flt.index', "rb") as f:
        filter_index_fc = pickle.load(f)
    fc_lut = np.genfromtxt('./cifar10_fc_lut.txt', delimiter=',',dtype=np.int16)
    conv_lut = np.genfromtxt('./cifar10_conv_lut.txt', delimiter=',',dtype=np.int16)
    add_lut = np.genfromtxt('./cifar10_add_lut.txt', delimiter=',',dtype=np.int16)
    relu_lut = np.genfromtxt('./cifar10_relu_lut.txt', delimiter=',',dtype=np.int16)
    c1_bias_lut = np.genfromtxt('./cifar10_c1_bias_lut.txt', delimiter=',',dtype=np.int16)
    c2_bias_lut = np.genfromtxt('./cifar10_c2_bias_lut.txt', delimiter=',',dtype=np.int16)
    c3_bias_lut = np.genfromtxt('./cifar10_c3_bias_lut.txt', delimiter=',',dtype=np.int16)
    c4_bias_lut = np.genfromtxt('./cifar10_c4_bias_lut.txt', delimiter=',',dtype=np.int16)
    c5_bias_lut = np.genfromtxt('./cifar10_c5_bias_lut.txt', delimiter=',',dtype=np.int16)
    c6_bias_lut = np.genfromtxt('./cifar10_c6_bias_lut.txt', delimiter=',',dtype=np.int16)
    c7_bias_lut = np.genfromtxt('./cifar10_c7_bias_lut.txt', delimiter=',',dtype=np.int16)
    c8_bias_lut = np.genfromtxt('./cifar10_c8_bias_lut.txt', delimiter=',',dtype=np.int16)
    f1_bias_lut = np.genfromtxt('./cifar10_f1_bias_lut.txt', delimiter=',',dtype=np.int16)

   
    print(" Symbolic model loading started...")
    t = time.process_time()
    netsym = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
                  c1_bias_lut, c2_bias_lut, c3_bias_lut, c4_bias_lut, c5_bias_lut, c6_bias_lut, c7_bias_lut, c8_bias_lut, 
                  f1_bias_lut, relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
    elapsed_time3 = time.process_time() - t
    print("Symbolic model loading completed in:",elapsed_time3)
 
    if sys.argv[1] == "0":
        start = time.process_time()  
        start_t = time.time()  
        acc = test_fullsym_acc(net, None, testloader, bs, False, False,0)
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for standard inference",elapsed)  
        elapsed = time.process_time() - start
        end = time.time()
        print("elapsed time for standard inference:", end - start_t) 
    elif sys.argv[1] == "1":
        start = time.process_time()  
        start_t = time.time()  
        #acc = test_fullsym_acc(netstd, None, testloader_sym, bs, False, False, 1)
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for symbolic inference",elapsed)  
        end = time.time()
        print("elapsed time for symbolic inference:", end - start_t) 
    else:
        start = time.process_time()  
        start_t = time.time()  
        acc = test_fullsym_acc(netsym, None, testloader, bs, False, False, 2)
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for symbolic inference",elapsed)  
        end = time.time()
        print("elapsed time for symbolic inference:", end - start_t) 

