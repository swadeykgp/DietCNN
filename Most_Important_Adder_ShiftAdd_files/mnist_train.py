#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
import adder
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./cache/data/')
parser.add_argument('--output_dir', type=str, default='./cache/models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder.adder2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=0, bias=False)

class CNN_LeNet(nn.Module):
    def __init__(self):
        super(CNN_LeNet, self).__init__()
        # Define the net structure
        # This is the input layer first Convolution
        #self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv1 = conv3x3(1, 6, stride=1)
        self.bn1 = nn.BatchNorm2d(6) 
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        #self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2 = conv3x3(6, 16, stride=1)
        self.bn2 = nn.BatchNorm2d(16) 
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
    
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x

acc = 0
acc_best = 0

transform_train = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])

transform_test = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                          transforms.Normalize((0.1309,), (0.2893,))])

data_train = datasets.MNIST(root='../../../dataset', train=True, download=True, transform=transform_train) 
data_test = datasets.MNIST(root='../../../dataset', train=False, download=True, transform=transform_test)

data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=False, pin_memory=True)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, pin_memory=True)

net = CNN_LeNet().cuda()
print(net)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
 
def test(epoch):
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(net.state_dict(), './adder_mnist.pt')


    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc*100))
 
 
def train_and_test(epoch):
    train(epoch)
    test(epoch)
    #scheduler.step()
 
 
def main():
    epoch = 410
    for e in range(1, epoch):
        train_and_test(e)
 
 
if __name__ == '__main__':
    main()
