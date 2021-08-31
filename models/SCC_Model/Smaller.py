"""
Based on code from
https://github.com/milesial/Pytorch-UNet/tree/master/unet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class Smaller(nn.Module):
    def __init__(self, test=False):
        super(Smaller, self).__init__()
        vgg = models.vgg16(pretrained=True)

        # ==============
        # Initialising
        # ==============

        """
        Loaded each layer individually because I was having enormous technical
        issues with loading state dicts when running the test set because of the 
        key names. This is an ugly solution but it's a solution that allowed me to 
        build the network how I wanted and work
        """

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if not test:
            print('Training - Loading state dicts\n')
            self.conv1.load_state_dict(vgg.features[0].state_dict())
            self.conv2.load_state_dict(vgg.features[2].state_dict())
            self.conv3.load_state_dict(vgg.features[5].state_dict())
            self.conv4.load_state_dict(vgg.features[7].state_dict())
            self.conv5.load_state_dict(vgg.features[10].state_dict())
            self.conv6.load_state_dict(vgg.features[12].state_dict())
            self.conv7.load_state_dict(vgg.features[14].state_dict())
            self.conv8.load_state_dict(vgg.features[17].state_dict())
            self.conv9.load_state_dict(vgg.features[19].state_dict())
            self.conv10.load_state_dict(vgg.features[21].state_dict())
        else:
            print('Testing - NOT loading state dicts\n')

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==============
        # Freezing first 5 layers
        # ==============

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.conv5.parameters():
            param.requires_grad = False

        # ==============
        # Modulising layers
        # ==============

        self.inc = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
        )

        self.down1 = nn.Sequential(
            self.maxpool,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
        )

        self.down2 = nn.Sequential(
            self.maxpool,
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.conv7,
            self.relu,
        )

        self.down3 = nn.Sequential(
            self.maxpool,
            self.conv8,
            self.relu,
            self.conv9,
            self.relu,
            self.conv10,
            self.relu,
        )

        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Up(nn.Module):
    """
    Up-sampling using interpolation then double convolution, space allowed for skip connections
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        """
        Forward pass with inputs from skip connections
        """
        x1 = self.up(x1)

        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
