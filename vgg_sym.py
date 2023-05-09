import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)), #21 #0
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #18 #2
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #15 #4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #13 #6
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #10 #8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #8 #10
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #5 #12
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)), #3 #14
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=10, bias=True)
        )


    def forward(self, x):
        out = self.features(x)
        #out = out.view(out.size(0), -1)
        out = out.view(-1, 512)
        out = self.classifier(out)
        return out


#def test():
#    net = VGG('VGG11')
#    x = torch.randn(2,3,32,32)
#    y = net(x)
#    print(net)
#    print(y.size())
#
#test()
