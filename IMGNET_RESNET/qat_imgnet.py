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
  x = model.conv1(x)
  x = model.bn1(x)
  x = model.relu(x)

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l10c1')
  x = model.layer1[0].conv1(x)
  x = model.layer1[0].bn1(x)
  x = model.layer1[0].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l10c2')
  x = model.layer1[0].conv2(x)
  x = model.layer1[0].bn2(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l11c1')
  x = model.layer1[1].conv1(x)
  x = model.layer1[1].bn1(x)
  x = model.layer1[1].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l11c2')
  x = model.layer1[1].conv2(x)
  x = model.layer1[1].bn2(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l20c1')
  x = model.layer2[0].conv1(x)
  x = model.layer2[0].bn1(x)
  x = model.layer2[0].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l20c2')
  x = model.layer2[0].conv2(x)
  x = model.layer2[0].bn2(x)
  #x = model.layer2[0].downsample[0](x)
  #x = model.layer2[0].downsample[1](x)  
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l21c1')
  x = model.layer2[1].conv1(x)  
  x = model.layer2[1].bn1(x)
  x = model.layer2[1].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l21c2')
  x = model.layer2[1].conv2(x)  
  x = model.layer2[1].bn2(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l30c1')
  x = model.layer3[0].conv1(x)
  x = model.layer3[0].bn1(x)
  x = model.layer3[0].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l30c2')
  x = model.layer3[0].conv2(x)
  x = model.layer3[0].bn2(x)
  #x = model.layer3[0].downsample[0](x)
  #x = model.layer3[0].downsample[1](x) 
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l31c1')
  x = model.layer3[1].conv1(x)
  x = model.layer3[1].bn1(x)
  x = model.layer3[1].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l31c2')
  x = model.layer3[1].conv2(x)   
  x = model.layer3[1].bn2(x) 
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l40c1')
  x = model.layer4[0].conv1(x)
  x = model.layer4[0].bn1(x)
  x = model.layer4[0].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l40c2')
  x = model.layer4[0].conv2(x)
  x = model.layer4[0].bn2(x)
  #x = model.layer4[0].downsample[0](x)
  #x = model.layer4[0].downsample[1](x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l41c1')
  x = model.layer4[1].conv1(x)  
  x = model.layer4[1].bn1(x)
  x = model.layer4[1].relu(x)
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l41c2')
  x = model.layer4[1].conv2(x) 
  x = model.layer4[1].bn2(x) 
  x = x.view(-1, 512) 
  
  stats = updateStats(x, stats, 'fc')

  x = model.fc(x)

  
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

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['l10c1'], x.scale, x.zero_point)
  x = model.bn1(x)
  x = model.relu(x)

  x, scale_next, zero_point_next = quantizeLayer(x, model.layer1[0].conv1, stats['l10c2'], scale_next, zero_point_next)
  x = model.layer1[0].bn1(x)
  x = model.layer1[0].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer1[0].conv2, stats['l11c1'], scale_next, zero_point_next)
  x = model.layer1[0].bn2(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer1[1].conv1, stats['l11c2'], scale_next, zero_point_next)
  x = model.layer1[1].bn1(x)
  x = model.layer1[1].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer1[1].conv2, stats['l20c1'], scale_next, zero_point_next)
  x = model.layer1[1].bn2(x)

  x, scale_next, zero_point_next = quantizeLayer(x, model.layer2[0].conv1, stats['l20c2'], scale_next, zero_point_next)
  x = model.layer2[0].bn1(x)
  x = model.layer2[0].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer2[0].conv2, stats['l21c1'], scale_next, zero_point_next)
  x = model.layer2[0].bn2(x)
  #x = model.layer2[0].downsample[0](x)
  #x = model.layer2[0].downsample[1](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer2[1].conv1, stats['l21c2'], scale_next, zero_point_next)
  x = model.layer2[1].bn1(x)
  x = model.layer2[1].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer2[1].conv2, stats['l30c1'], scale_next, zero_point_next)
  x = model.layer2[1].bn2(x)

  x, scale_next, zero_point_next = quantizeLayer(x, model.layer3[0].conv1, stats['l30c2'], scale_next, zero_point_next)
  x = model.layer3[0].bn1(x)
  x = model.layer3[0].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer3[0].conv2, stats['l31c1'], scale_next, zero_point_next)
  x = model.layer3[0].bn2(x)
  #x = model.layer3[0].downsample[0](x)
  #x = model.layer3[0].downsample[1](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer3[1].conv1, stats['l31c2'], scale_next, zero_point_next)
  x = model.layer3[1].bn1(x)
  x = model.layer3[1].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer3[1].conv2, stats['l40c1'], scale_next, zero_point_next)
  x = model.layer3[1].bn2(x)  

  x, scale_next, zero_point_next = quantizeLayer(x, model.layer4[0].conv1, stats['l40c2'], scale_next, zero_point_next)
  x = model.layer4[0].bn1(x)
  x = model.layer4[0].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer4[0].conv2, stats['l41c1'], scale_next, zero_point_next)
  x = model.layer4[0].bn2(x)
  #x = model.layer4[0].downsample[0](x)
  #x = model.layer4[0].downsample[1](x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer4[1].conv1, stats['l41c2'], scale_next, zero_point_next)
  x = model.layer4[1].bn1(x)
  x = model.layer4[1].relu(x)
  x, scale_next, zero_point_next = quantizeLayer(x, model.layer4[1].conv2, stats['fc'], scale_next, zero_point_next)
  x = model.layer4[1].bn2(x)  

  x = x.view(-1, 512)   
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = model.fc(x)

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

  ######## Outer layer #######
  conv1weight = model.conv1.weight.data
  model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits)
  x = model.conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])
  x = model.bn1(x)
  x = model.relu(x)

  ######## layer 1 #######
  conv2weight = model.layer1[0].conv1.weight.data
  model.layer1[0].conv1.weight.data = FakeQuantOp.apply(model.layer1[0].conv1.weight.data, num_bits)
  x = model.layer1[0].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l10c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l10c1']['ema_min'], stats['l10c1']['ema_max'])
  x = model.layer1[0].bn1(x)
  x = model.layer1[0].relu(x)

  conv3weight = model.layer1[0].conv2.weight.data
  model.layer1[0].conv2.weight.data = FakeQuantOp.apply(model.layer1[0].conv2.weight.data, num_bits)
  x = model.layer1[0].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l10c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l10c2']['ema_min'], stats['l10c2']['ema_max'])
  x = model.layer1[0].bn2(x)


  conv4weight = model.layer1[1].conv1.weight.data
  model.layer1[1].conv1.weight.data = FakeQuantOp.apply(model.layer1[1].conv1.weight.data, num_bits)
  x = model.layer1[1].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l11c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l11c1']['ema_min'], stats['l11c1']['ema_max'])
  x = model.layer1[1].bn1(x)
  x = model.layer1[1].relu(x)

  conv5weight = model.layer1[1].conv2.weight.data
  model.layer1[1].conv2.weight.data = FakeQuantOp.apply(model.layer1[1].conv2.weight.data, num_bits)
  x = model.layer1[1].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l11c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l11c2']['ema_min'], stats['l11c2']['ema_max'])
  x = model.layer1[1].bn2(x)

  ######## layer 2 #######
  conv6weight = model.layer2[0].conv1.weight.data
  model.layer2[0].conv1.weight.data = FakeQuantOp.apply(model.layer2[0].conv1.weight.data, num_bits)
  x = model.layer2[0].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l20c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l20c1']['ema_min'], stats['l20c1']['ema_max'])
  x = model.layer2[0].bn1(x)
  x = model.layer2[0].relu(x)

  conv7weight = model.layer2[0].conv2.weight.data
  model.layer2[0].conv2.weight.data = FakeQuantOp.apply(model.layer2[0].conv2.weight.data, num_bits)
  x = model.layer2[0].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l20c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l20c2']['ema_min'], stats['l20c2']['ema_max'])
  x = model.layer2[0].bn2(x)

  conv8weight = model.layer2[1].conv1.weight.data
  model.layer2[1].conv1.weight.data = FakeQuantOp.apply(model.layer2[1].conv1.weight.data, num_bits)
  x = model.layer2[1].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l21c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l21c1']['ema_min'], stats['l21c1']['ema_max'])
  x = model.layer2[1].bn1(x)
  x = model.layer2[1].relu(x)

  conv9weight = model.layer2[1].conv2.weight.data
  model.layer2[1].conv2.weight.data = FakeQuantOp.apply(model.layer2[1].conv2.weight.data, num_bits)
  x = model.layer2[1].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l21c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l21c2']['ema_min'], stats['l21c2']['ema_max'])
  x = model.layer2[1].bn2(x)

  ######## layer 3 #######
  conv10weight = model.layer3[0].conv1.weight.data
  model.layer3[0].conv1.weight.data = FakeQuantOp.apply(model.layer3[0].conv1.weight.data, num_bits)
  x = model.layer3[0].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l30c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l30c1']['ema_min'], stats['l30c1']['ema_max'])
  x = model.layer3[0].bn1(x)
  x = model.layer3[0].relu(x)

  conv11weight = model.layer3[0].conv2.weight.data
  model.layer3[0].conv2.weight.data = FakeQuantOp.apply(model.layer3[0].conv2.weight.data, num_bits)
  x = model.layer3[0].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l30c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l30c2']['ema_min'], stats['l30c2']['ema_max'])
  x = model.layer3[0].bn2(x)

  conv12weight = model.layer3[1].conv1.weight.data
  model.layer3[1].conv1.weight.data = FakeQuantOp.apply(model.layer3[1].conv1.weight.data, num_bits)
  x = model.layer3[1].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l31c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l31c1']['ema_min'], stats['l31c1']['ema_max'])
  x = model.layer3[1].bn1(x)
  x = model.layer3[1].relu(x)

  conv13weight = model.layer3[1].conv2.weight.data
  model.layer3[1].conv2.weight.data = FakeQuantOp.apply(model.layer3[1].conv2.weight.data, num_bits)
  x = model.layer3[1].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l31c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l31c2']['ema_min'], stats['l31c2']['ema_max'])
  x = model.layer3[1].bn2(x)
  ######## layer 4 #######
  conv14weight = model.layer4[0].conv1.weight.data
  model.layer4[0].conv1.weight.data = FakeQuantOp.apply(model.layer4[0].conv1.weight.data, num_bits)
  x = model.layer4[0].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l40c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l40c1']['ema_min'], stats['l40c1']['ema_max'])
  x = model.layer4[0].bn1(x)
  x = model.layer4[0].relu(x)

  conv15weight = model.layer4[0].conv2.weight.data
  model.layer4[0].conv2.weight.data = FakeQuantOp.apply(model.layer4[0].conv2.weight.data, num_bits)
  x = model.layer4[0].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l40c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l40c2']['ema_min'], stats['l40c2']['ema_max'])
  x = model.layer4[0].bn2(x)
  conv16weight = model.layer4[1].conv1.weight.data
  model.layer4[1].conv1.weight.data = FakeQuantOp.apply(model.layer4[1].conv1.weight.data, num_bits)
  x = model.layer4[1].conv1(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l41c1')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l41c1']['ema_min'], stats['l41c1']['ema_max'])
  x = model.layer4[1].bn1(x)
  x = model.layer4[1].relu(x)

  conv17weight = model.layer4[1].conv2.weight.data
  model.layer4[1].conv2.weight.data = FakeQuantOp.apply(model.layer4[1].conv2.weight.data, num_bits)
  x = model.layer4[1].conv2(x)
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'l41c2')
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, stats['l41c2']['ema_min'], stats['l41c2']['ema_max'])
  x = model.layer4[1].bn2(x)
  ######## layer ends  #######

  x = x.view(-1, 512) 
  x = model.fc(x)

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc')


  return x, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight, conv14weight, conv15weight, conv16weight,  conv17weight, stats

# Training
def tinytrain(epoch, trainloader, optimizer, criterion, model, device, stats, act_quant=False, num_bits=8):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #outputs = net(inputs)
        outputs, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight, conv14weight, conv15weight, conv16weight,  conv17weight, stats = quantAwareTrainingForward(model, inputs, stats, num_bits=num_bits, act_quant=act_quant)
        model.conv1.weight.data            = conv1weight
        model.layer1[0].conv1.weight.data  = conv2weight
        model.layer1[0].conv2.weight.data  = conv3weight
        model.layer1[1].conv1.weight.data  = conv4weight
        model.layer1[1].conv2.weight.data  = conv5weight
        model.layer2[0].conv1.weight.data  = conv6weight
        model.layer2[0].conv2.weight.data  = conv7weight
        model.layer2[1].conv1.weight.data  = conv8weight
        model.layer2[1].conv2.weight.data  = conv9weight
        model.layer3[0].conv1.weight.data  = conv10weight
        model.layer3[0].conv2.weight.data  = conv11weight
        model.layer3[1].conv1.weight.data  = conv12weight
        model.layer3[1].conv2.weight.data  = conv13weight
        model.layer4[0].conv1.weight.data  = conv14weight
        model.layer4[0].conv2.weight.data  = conv15weight
        model.layer4[1].conv1.weight.data  = conv16weight
        model.layer4[1].conv2.weight.data  = conv17weight
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def tinytest(epoch, testloader, criterion, model, device, stats, act_quant, num_bits=8):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #outputs = net(inputs)
            outputs, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight, conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, conv13weight, conv14weight, conv15weight, conv16weight,  conv17weight, stats = quantAwareTrainingForward(model, inputs, stats, num_bits=num_bits, act_quant=act_quant)
            model.conv1.weight.data  = conv1weight
            model.layer1[0].conv1.weight.data  = conv2weight
            model.layer1[0].conv2.weight.data  = conv3weight
            model.layer1[1].conv1.weight.data  = conv4weight
            model.layer1[1].conv2.weight.data  = conv5weight
            model.layer2[0].conv1.weight.data  = conv6weight
            model.layer2[0].conv2.weight.data  = conv7weight
            model.layer2[1].conv1.weight.data  = conv8weight
            model.layer2[1].conv2.weight.data  = conv9weight
            model.layer3[0].conv1.weight.data  = conv10weight
            model.layer3[0].conv2.weight.data  = conv11weight
            model.layer3[1].conv1.weight.data  = conv12weight
            model.layer3[1].conv2.weight.data  = conv13weight
            model.layer4[0].conv1.weight.data  = conv14weight
            model.layer4[0].conv2.weight.data  = conv15weight
            model.layer4[1].conv1.weight.data  = conv16weight
            model.layer4[1].conv2.weight.data  = conv17weight
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
        torch.save(state, './imgnet_qat.pt')
        best_acc = acc


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(" Data loading started...")
    bs = 128
    num_bits=8
    stats = {} 
  

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
        tinytrain(epoch, train_loader, optimizer, criterion,  net, device, stats, act_quant=False, num_bits=8)
        tinytest(epoch, val_loader, criterion, net, device, stats, act_quant, num_bits=8)
        scheduler.step()
    

    print("Finished Training") 
