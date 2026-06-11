import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNetEncoderDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, backbone='resnet34'):
        super().__init__()

        if backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif backbone == 'resnet18':
            resnet = models.resnet18(weights=None)
            state_dict = torch.load("resnet18-5c106cde.pth", map_location="cpu", weights_only=False)
            resnet.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("backbone must be resnet18 or resnet34")

        w = resnet.conv1.weight
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet.conv1.weight.data[:, :3, :, :] = w
        resnet.conv1.weight.data[:, 3:, :, :] = w[:, :1, :, :]

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up4 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x0 = self.layer0(x)   
        x1 = self.layer1(x0)  
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)  
        x4 = self.layer4(x3)  

        d4 = self.up4(x4)              
        d4 = d4 + x3

        d3 = self.up3(d4)              
        d3 = d3 + x2

        d2 = self.up2(d3)             
        d2 = d2 + x1

        d1 = self.up1(d2)              
        d1 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.final(d1)
        return out
