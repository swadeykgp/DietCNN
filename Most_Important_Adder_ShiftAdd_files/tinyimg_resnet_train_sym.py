from __future__ import print_function, division
#from modeldef_alexnet_reduced import AlexNet
from resnet_18_sym import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, datasets
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, datasets
from tqdm.auto import tqdm

# For training
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
# For training
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from utils import progress_bar
cudnn.benchmark = True
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
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

val_transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.Resize(74),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = transform_mean, std = transform_std),
])



# Training
def tinytrain(epoch, trainloader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def tinytest(epoch, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './tinyimg_resnet_adder.pt')
        best_acc = acc



if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" Data loading started...")
    bs = 16

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
    #pretrained_model = "./tinyimg_resnet.pt"
    #sd = torch.load(pretrained_model)
    #print(sd['net']) 
    #net.load_state_dict(sd['net'], strict=False)
    if torch.cuda.is_available():
        net.cuda()
    num_epochs = 500 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        tinytrain(epoch, train_loader, optimizer, criterion)
        tinytest(epoch, val_loader, criterion)
        scheduler.step()
    

    print("Finished Training") 

