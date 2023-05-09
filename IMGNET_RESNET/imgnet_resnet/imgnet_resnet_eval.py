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
from lut_utils_tiny import *
import warnings
warnings.filterwarnings('ignore')
from patchlib import *
import multiprocessing
from joblib import Parallel, delayed
from evalutils_resnet_imgnet import *
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

    #bs = 25 
    bs = 5 
  
    directory = "/home/edgeacceleration/projects/dataset/tiny-imagenet-200/"
    num_classes = 200
    # the magic normalization parameters come from the example
    transform_mean = np.array([ 0.485, 0.456, 0.406 ])
    transform_std = np.array([ 0.229, 0.224, 0.225 ])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = transform_mean, std = transform_std),
    ])
    
    if sys.argv[1] == "2":
        val_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(74),
            transforms.CenterCrop(64),
            #transforms.Resize(71),
            #transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])
    else:
        val_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.Resize(74),
            #transforms.CenterCrop(71),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform_mean, std = transform_std),
        ])
            
    traindir = os.path.join(directory, "train")
    # be careful with this set, the labels are not defined using the directory structure
    valdir = os.path.join(directory, "val")
    
    train = datasets.ImageFolder(traindir, train_transform)
    val = datasets.ImageFolder(valdir, val_transform)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=True)
    
    assert num_classes == len(train_loader.dataset.classes)
    
    small_labels = {}
    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()
    
    #print(small_labels.items()[0])
    
    #The train subdirectory of Tiny ImageNet-200 has a collection of subdirectories, named using to the WordNet ids to label the images that they contain. The torchvision data loader uses the names of the subdirectories as labels, but replaces them with numeric indices when iterating the batches.
    
    os.listdir(traindir)[:5]
    
    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader.dataset.classes):
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index
    
    #print(labels.items()[0])
    
    #print(label_ids.items()[0])
    
    #Another problem is that the validation directory only has one subdirectory called images. The labels for every image inside this subdirectory are defined in a file called val_annotations.txt.
    
    val_label_map = {}
    with open(os.path.join(directory, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()
    
    #print(val_label_map.items()[0])
    
    #Finally we update the Tiny ImageNet-200 validation set labels:
    
    #print(val_loader.dataset.imgs[:5])
    
    for i in range(len(val_loader.dataset.imgs)):
        file_path = val_loader.dataset.imgs[i][0]
        
        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]
        
        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])
    
    #val_loader.dataset.imgs[:5]



   
    net = resnet18() 
    #pretrained_model = "./tinyimg_resnet_sym_nbn.pt"
    pretrained_model = "./tinyimg_resnet_sym_nbn_v1.pt"
    sd = torch.load(pretrained_model)
    #print(sd['net']) 
    net.load_state_dict(sd['net'])

    conv_patch_size = (1, 1)
    patch_size = (1, 1)
    all_patch_size = (1, 1)
    n_cluster_conv_filters = 256
    n_cluster_fc_filters = 128
    conv_stride = 1
    index = faiss.read_index("./kmeans_resnet_tinyimgnet_c1_k1_s1_512_v1_noflt.index")
    n_clusters=512
    #n_clusters=256
    #n_clusters=64
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
   
    print(" Symbolic model loading started...")
    t = time.process_time()
    netsym = vgg_sym(net,sd, filter_index_conv, filter_index_fc, conv_lut, fc_lut, add_lut, 
                  relu_lut, n_clusters, index, centroid_lut, patch_size, patch_stride)
    elapsed_time3 = time.process_time() - t
    print("Symbolic model loading completed in:",elapsed_time3)


    # Setup for standard inference


 
    if sys.argv[1] == "0":
        start = time.process_time()  
        start_t = time.time()  
        acc = test_fullsym_acc(net, None, val_loader, bs, False, False,0)
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
        acc = test_fullsym_acc(netsym, None, val_loader, bs, False, True, 2)
        elapsed = time.process_time() - start
        elapsed = elapsed*1000
        print("elapsed process time for symbolic inference",elapsed)  
        end = time.time()
        print("elapsed time for symbolic inference:", end - start_t) 
