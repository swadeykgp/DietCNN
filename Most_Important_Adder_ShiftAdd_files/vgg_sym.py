import torch
import torch.nn as nn
import adder


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            conv3x3(3, 64, stride=2), #21 #0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), 
            #nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3(64, 128), #18 #3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), 
            #nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3(128, 256), #15 #6
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), 
            conv3x3(256, 256), #13 #8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), 
            #nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3(256, 512), #10 #11
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), 
            conv3x3(512, 512), #8 #13
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), 
            #nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3(512, 512), #5 #16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), 
            conv3x3(512, 512), #3 #18
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), 
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=10, bias=True)
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(net)
    print(y.size())

test()
