from __future__ import print_function, division
#from modeldef_alexnet_reduced import AlexNet
from vgg_sym import *
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
from utils import progress_bar
cudnn.benchmark = True
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Basic model parameters.

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder.adder2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=0, bias=False)

#def top_k_error(top_k, total):
#    return 100.0 - top_k / total * 100.0
#def run_epoch(epoch, loader, model, optimizer, scheduler, train=True, log_every=100):
#    running_loss = 0.0
#    running_top_1 = 0.0
#    running_top_5 = 0.0
#    running_total = 0.0
#    
#    epoch_top_1 = 0.0
#    epoch_top_5 = 0.0
#    epoch_total = 0.0
#   
#    best_acc = 0.0
#    best_model_wts = copy.deepcopy(model.state_dict()) 
#    model.train(mode=train)
#    epoch_loss = 0.0
#    #if train:
#    #    model.train()
#    #else:
#    #    model.eval()
#    
#    for batch_number, (batch_inputs, batch_labels) in tqdm(enumerate(loader), total=len(loader)):
#        batch_inputs, batch_labels = Variable(batch_inputs.cuda()), Variable(batch_labels.cuda())
#
#        if train:
#            optimizer.zero_grad()
#
#        batch_logits = model(batch_inputs)
#        
#        if train:
#            batch_loss = criterion(batch_logits, batch_labels)
#            batch_loss.backward()
#        
#            optimizer.step()
#            
#            running_loss += batch_loss.data.cpu().item()
#        
#        batch_labels = batch_labels.data.cpu().numpy()
#        batch_predictions = batch_logits.topk(5)[1].data.cpu().numpy()
#    
#        for i in range(len(batch_labels)):
#            if batch_labels[i] == batch_predictions[i, 0]:
#                running_top_1 += 1
#                running_top_5 += 1
#                epoch_top_1 += 1
#                epoch_top_5 += 1
#            else:
#                for j in range(1, 5):
#                    if batch_labels[i] == batch_predictions[i, j]:
#                        running_top_5 += 1
#                        epoch_top_5 += 1
#                        break
#        
#        running_total += len(batch_labels)
#        epoch_total += len(batch_labels)
#
#        if batch_number % log_every == log_every - 1:
#            if train:
#                print("[Batch {:5d}] Loss: {:.3f} Top-1 Error: {:.3f} Top-5 Error: {:.3f}".format(
#                    batch_number + 1,
#                    running_loss / log_every,
#                    top_k_error(running_top_1, running_total),
#                    top_k_error(running_top_5, running_total)
#                ))
#            
#            running_loss = 0.0
#            running_top_1 = 0.0
#            running_top_5 = 0.0
#            running_total = 0.0
#        if train == False:
#            # deep copy the model
#            if epoch_top_1 > best_acc:
#                #print("Best accuracy improved: ",epoch_top_1," , at epoch: ", epoch) 
#                best_acc = epoch_top_1
#                best_model_wts = copy.deepcopy(model.state_dict())
#    #scheduler.step(epoch_loss)
#    model.load_state_dict(best_model_wts)
#    return top_k_error(epoch_top_1, epoch_total), top_k_error(epoch_top_5, epoch_total), model
#

#class SaveBestModel:
#    """
#    Class to save the best model while training. If the current epoch's 
#    validation loss is less than the previous least less, then save the
#    model state.
#    """
#    def __init__(
#        self, best_valid_loss=float('inf')
#    ):
#        self.best_valid_loss = best_valid_loss
#
#    def __call__(
#        self, current_valid_loss,
#        epoch, model, optimizer, criterion
#    ):
#        if current_valid_loss < self.best_valid_loss:
#            self.best_valid_loss = current_valid_loss
#            print(f"\nBest validation loss: {self.best_valid_loss}")
#            print(f"\nSaving best model for epoch: {epoch+1}\n")
#            torch.save(model.state_dict(), 'best_tinyimg_alex.pt')
#
#def save_model(epochs, model, optimizer, criterion):
#    """
#    Function to save the trained model to disk.
#    """
#    print(f"Saving final model...")
#    torch.save(model.state_dict(), 'best_tinyimg_alex.pt')

# Training
def train(epoch, trainloader, optimizer, criterion):
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


def test(epoch, testloader, criterion):
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
        print('Saving model with validation acc, loss: ',acc,' ,',test_loss)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './cifar_vgg_adder.pt')
        best_acc = acc




if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" Data loading started...")
    bs = 64 

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        #transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='../../dataset', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='../../dataset', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    random_indices = list(range(0, len(testset), 256))
    testset_subset = torch.utils.data.Subset(testset, random_indices)
    testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=64, shuffle=False)

    random_indices1 = list(range(0, len(trainset), 2048))
    trainset_subset = torch.utils.data.Subset(trainset, random_indices1)
    trainloader_subset = torch.utils.data.DataLoader(trainset_subset, batch_size=64, shuffle=False)
    
    #net = AlexNet()
    net = VGG('VGG11')
    #pretrained_model = "./cifar_vgg_nobn_sym.pt"
    #sd = torch.load(pretrained_model)
    #print(sd['net']) 
    #net.load_state_dict(sd['net'])
#    start_epoch = sd['epoch']
#    net.eval()
#
    #torch.nn.init.kaiming_uniform_(net.c1.weight.data, nonlinearity='relu')
    #nn.init.constant_(net.c1.bias.data, 0)  
    #torch.nn.init.kaiming_uniform_(net.c2.weight.data, nonlinearity='relu')
    #nn.init.constant_(net.c2.bias.data, 0)  
    #torch.nn.init.kaiming_uniform_(net.f1.weight.data, nonlinearity='relu')
    #nn.init.constant_(net.fc1.bias.data, 0)  
    #torch.nn.init.kaiming_uniform_(net.f2.weight.data, nonlinearity='relu')
    #nn.init.constant_(net.fc2.bias.data, 0)  
    #torch.nn.init.kaiming_uniform_(net.f3.weight.data)
    #nn.init.constant_(net.f3.bias.data, 0)  

    if torch.cuda.is_available():
        net.cuda()
    num_epochs = 500 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        train(epoch, trainloader_subset, optimizer, criterion)
        test(epoch, testloader_subset, criterion)
        scheduler.step()
    

    print("Finished Training") 

#    print("Training Started !!!!!!")
#    learning_rate=0.1
#    #optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#    loss_criterion = nn.CrossEntropyLoss()
#    #loss_criterion = F.nll_loss
#    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
#    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#    epochs = 200 
#    net = train_model(net, loss_criterion, optimizer, scheduler, dataloaders, num_epochs=epochs)
#    save_model(epochs, net, optimizer, loss_criterion)
