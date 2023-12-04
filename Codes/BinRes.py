import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        neck = self.neck(x)
        res = neck + x
        res = self.relu(res)
        return res
    
class Conv_Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_Bottleneck, self).__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        )
        self.conv = nn.Conv2d(in_channels , out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        neck = self.neck(x)
        conv = self.conv(x)
        res = neck + conv
        res = self.relu(res)
        return res
    
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()

        self.prep0 = nn.Conv2d(in_channels, 32, kernel_size=7)
        self.prep1 = nn.MaxPool2d((2,2))

        self.conv = nn.Sequential(
            Bottleneck(32,32,3),
            Bottleneck(32,32,3),
            Conv_Bottleneck(32,64,3)
        )

        self.final0 = nn.AvgPool2d((2,2))
        self.final1 = nn.Sequential(
            nn.Linear(in_features=64*62*62, out_features=1024),
            nn.ReLU(inplace=True)
        )
        self.final2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x = self.prep0(x)
        x = self.prep1(x)
        x = self.conv(x)
        x = self.final0(x)
        print(x.size(0))
        res = x.view(128, -1)
        
        res = self.final1(res)
        res = self.final2(res)
        return res