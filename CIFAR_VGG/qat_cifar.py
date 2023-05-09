from __future__ import print_function, division
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
from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def calcScaleZeroPointSym(min_val, max_val,num_bits=8):
  
  # Calc Scale 
  max_val = max(abs(min_val), abs(max_val))
  qmin = 0.
  qmax = 2.**(num_bits-1) - 1.

  scale = max_val / qmax

  return scale, 0

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax   

    q_x = x/scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)

def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())

def quantizeLayer(x, layer, stat, scale_x, zp_x, vis=False, axs=None, X=None, y=None, sym=False, num_bits=8):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # WEIGHTS SIMULATED QUANTISED

  # quantise weights, activations are already quantised
  if sym:
    w = quantize_tensor_sym(layer.weight.data,num_bits=num_bits) 
    b = quantize_tensor_sym(layer.bias.data,num_bits=num_bits)
  else:
    w = quantize_tensor(layer.weight.data, num_bits=num_bits) 
    b = quantize_tensor(layer.bias.data, num_bits=num_bits)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  ## END WEIGHTS QUANTISED SIMULATION


  if vis:
    axs[X,y].set_xlabel("Visualising weights of layer: ")
    visualise(layer.weight.data, axs[X,y])

  # QUANTISED OP, USES SCALE AND ZERO POINT TO DO LAYER FORWARD PASS. (How does backprop change here ?)
  # This is Quantisation Arithmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  if sym:
    scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'])
  else:
    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by saturating range to num_bits range.
  if sym:
    X = x.float()
    layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data)
  else:
    X = x.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data - zp_w)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data + zp_b)

  # All int computation
  if sym:  
    x = (layer(X)) 
  else:
    x = (layer(X)) + zero_point_next 
  
  # cast to int
  x.round_()

  # Perform relu too
  x = F.relu(x)

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return x, scale_next, zero_point_next

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
  max_val, _ = torch.max(x, dim=1)
  min_val, _ = torch.min(x, dim=1)

  # add ema calculation

  if key not in stats:
    stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
  else:
    stats[key]['max'] += max_val.sum().item()
    stats[key]['min'] += min_val.sum().item()
    stats[key]['total'] += 1
  
  weighting = 2.0 / (stats[key]['total']) + 1

  if 'ema_min' in stats[key]:
    stats[key]['ema_min'] = weighting*(min_val.mean().item()) + (1- weighting) * stats[key]['ema_min']
  else:
    stats[key]['ema_min'] = weighting*(min_val.mean().item())

  if 'ema_max' in stats[key]:
    stats[key]['ema_max'] = weighting*(max_val.mean().item()) + (1- weighting) * stats[key]['ema_max']
  else: 
    stats[key]['ema_max'] = weighting*(max_val.mean().item())

  stats[key]['min_val'] = stats[key]['min']/ stats[key]['total']
  stats[key]['max_val'] = stats[key]['max']/ stats[key]['total']
  
  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
  x = model.features[1](model.features[0](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
  x =  model.features[3](model.features[2](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
  x = model.features[5](model.features[4](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
  x = model.features[7](model.features[6](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
  x = model.features[9](model.features[8](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv6')
  x = model.features[11](model.features[10](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv7')
  x = model.features[13](model.features[12](x))
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv8')
  x = model.features[15](model.features[14](x))

  #x = x.view(x.size(0), -1)  
  x = x.view(-1, 512) 
  
  stats = updateStats(x, stats, 'fc')

  x = model.classifier(x)

  return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"], "ema_min": value["ema_min"], "ema_max": value["ema_max"] }
    return final_stats

def quantForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8):
  X = 0
  y = 0
  # Quantise before inputting into incoming layers
  if sym:
    x = quantize_tensor_sym(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)
  else:
    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)

    # Quantise before inputting into incoming layers
  if sym:
    x = quantize_tensor_sym(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=nb)
  else:
    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=nb)

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.features[0], stats['conv2'], x.scale, x.zero_point)
  #x = model.features[1](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[2], stats['conv3'], scale_next, zero_point_next)
  #x = model.features[3](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[4], stats['conv4'], scale_next, zero_point_next)
  #x = model.features[5](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[6], stats['conv5'], scale_next, zero_point_next)
  #x = model.features[7](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[8], stats['conv6'], scale_next, zero_point_next)
  #x = model.features[9](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[10], stats['conv7'], scale_next, zero_point_next)
  #x = model.features[11](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[12], stats['conv8'], scale_next, zero_point_next)
  #x = model.features[13](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.features[14], stats['fc'], scale_next, zero_point_next)
  #x = model.features[15](x)
    
  
  #x = x.view(x.size(0), -1)  
  x = x.view(-1, 512)   
  
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = model.classifier(x)

  return x

import torch

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x,num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False):

  #print(x.shape)
  #print(model.features[0].weight.data.shape) 
  #x = model.features[0](x)
  #x = model.features[1](x)
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
  #x =  model.features[3](model.features[2](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
  #x = model.features[5](model.features[4](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
  #x = model.features[7](model.features[6](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
  #x = model.features[9](model.features[8](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv6')
  #x = model.features[11](model.features[10](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv7')
  #x = model.features[13](model.features[12](x))
  #stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv8')
  #x = model.features[15](model.features[14](x))
  ##x = x.view(x.size(0), -1)  
  #x = x.view(-1, 512) 
  #stats = updateStats(x, stats, 'fc')
  #x = model.classifier(x)

  conv1weight = model.features[0].weight.data
  model.features[0].weight.data = FakeQuantOp.apply(model.features[0].weight.data, num_bits)
  x = model.features[1](model.features[0](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

  conv2weight = model.features[2].weight.data
  model.features[2].weight.data = FakeQuantOp.apply(model.features[2].weight.data, num_bits)
  x = model.features[3](model.features[2](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])

  conv3weight = model.features[4].weight.data
  model.features[4].weight.data = FakeQuantOp.apply(model.features[4].weight.data, num_bits)
  x = model.features[5](model.features[4](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max'])


  conv4weight = model.features[6].weight.data
  model.features[6].weight.data = FakeQuantOp.apply(model.features[6].weight.data, num_bits)
  x = model.features[7](model.features[6](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max'])


  conv5weight = model.features[8].weight.data
  model.features[8].weight.data = FakeQuantOp.apply(model.features[8].weight.data, num_bits)
  x = model.features[9](model.features[8](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'])



  conv6weight = model.features[10].weight.data
  model.features[10].weight.data = FakeQuantOp.apply(model.features[10].weight.data, num_bits)
  x = model.features[11](model.features[10](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv6')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv6']['ema_min'], stats['conv6']['ema_max'])


  conv7weight = model.features[12].weight.data
  model.features[12].weight.data = FakeQuantOp.apply(model.features[12].weight.data, num_bits)
  x = model.features[13](model.features[12](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv7')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv7']['ema_min'], stats['conv7']['ema_max'])


  conv8weight = model.features[14].weight.data
  model.features[14].weight.data = FakeQuantOp.apply(model.features[14].weight.data, num_bits)
  x = model.features[15](model.features[14](x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv8')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv8']['ema_min'], stats['conv8']['ema_max'])

  x = x.view(-1, 512) 
  x = model.classifier(x)

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc')


  return x, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, stats

# Training
def train(epoch, trainloader, optimizer, criterion, model, device, stats, act_quant=False, num_bits=8):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #outputs = net(inputs)
        outputs, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, stats = quantAwareTrainingForward(model, inputs, stats, num_bits=num_bits, act_quant=act_quant)
        model.features[0].weight.data   = conv1weight
        model.features[2].weight.data   = conv2weight
        model.features[4].weight.data   = conv3weight
        model.features[6].weight.data   = conv4weight
        model.features[8].weight.data   = conv5weight
        model.features[10].weight.data  = conv6weight
        model.features[12].weight.data  = conv7weight
        model.features[14].weight.data  = conv8weight
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, testloader, criterion, model, device, stats, act_quant, num_bits=8):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #outputs = net(inputs)
            outputs, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, stats = quantAwareTrainingForward(model, inputs, stats, num_bits=num_bits, act_quant=act_quant)
            model.features[0].weight.data   = conv1weight
            model.features[2].weight.data   = conv2weight
            model.features[4].weight.data   = conv3weight
            model.features[6].weight.data   = conv4weight
            model.features[8].weight.data   = conv5weight
            model.features[10].weight.data  = conv6weight
            model.features[12].weight.data  = conv7weight
            model.features[14].weight.data  = conv8weight
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
        torch.save(state, './cifar_qat.pt')
        best_acc = acc


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" Data loading started...")
    bs = 128
    num_bits=8
    stats = {} 
  
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
    
    net = VGG('VGG11')
    if torch.cuda.is_available():
        net.cuda()
    num_epochs = 500 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        if epoch > 50:
          act_quant = True 
        else:
          act_quant = False 
        train(epoch, train_loader, optimizer, criterion,  net, device, stats, act_quant=False, num_bits=8)
        test(epoch, test_loader, criterion, net, device, stats, act_quant, num_bits=8)
        scheduler.step()
    

    print("Finished Training") 
