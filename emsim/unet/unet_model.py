"""
FROM: https://github.com/milesial/Pytorch-UNet
Full assembly of the parts to form the complete network
"""

import torch.nn.functional as F

from .unet_parts import *

netdebug = False

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        if(netdebug): print("Inc:",x1.shape)
        x2 = self.down1(x1)
        if(netdebug): print("Down1:",x2.shape)
        x3 = self.down2(x2)
        if(netdebug): print("Down2:",x3.shape)
        x4 = self.down3(x3)
        if(netdebug): print("Down3:",x4.shape)
        x5 = self.down4(x4)
        if(netdebug): print("Down4:",x5.shape)
        x = self.up1(x5, x4)
        if(netdebug): print("Up1:",x.shape)
        x = self.up2(x, x3)
        if(netdebug): print("Up2:",x.shape)
        x = self.up3(x, x2)
        if(netdebug): print("Up3:",x.shape)
        x = self.up4(x, x1)
        if(netdebug): print("Up4:",x.shape)
        logits = self.outc(x)
        if(netdebug): print("Outc:",logits.shape)
        return logits
