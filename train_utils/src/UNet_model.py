import torch
import torch.nn as nn
from src.BasicBlock import se_block, cbam_block, eca_block



#UNet
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DownConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature, norm_layer=None, is_attention=False):
        super(DownConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.is_attention = is_attention
        self.out_feature = out_feature
        self.conv = nn.Sequential(
            conv3x3(in_feature, out_feature),
            norm_layer(out_feature),
            nn.ReLU(inplace=True),
            conv3x3(out_feature, out_feature),
            norm_layer(out_feature),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.is_attention:
            self.attention_layer = se_block(out_feature)

    def forward(self, x):
        skip = self.conv(x)
        if self.is_attention:
            x = self.attention_layer(skip)
            x = self.pool(x)
        else:
            x = self.pool(skip)
        return x, skip

class UpConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature, norm_layer=None):
        super(UpConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.transconv = nn.ConvTranspose2d(in_feature, in_feature//2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            conv3x3(in_feature, out_feature),
            norm_layer(out_feature),
            nn.ReLU(inplace=True),
            conv3x3(out_feature, out_feature),
            norm_layer(out_feature),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.transconv(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):

    def __init__(self, in_channel=3, num_classes=1, is_attention=[False, False, False, False, False]):
        super(UNet, self).__init__()
        self.is_attention = is_attention
        self.down_layer1 = DownConvBlock(in_channel, 64, is_attention=is_attention[0])
        self.down_layer2 = DownConvBlock(64, 128, is_attention=is_attention[1])
        self.down_layer3 = DownConvBlock(128, 256, is_attention=is_attention[2])
        self.down_layer4 = DownConvBlock(256, 512, is_attention=is_attention[3])
        self.down_layer5 = nn.Sequential(
            conv3x3(512, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            conv3x3(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        if self.is_attention[4]:
            self.attention_layer = se_block(1024)

        self.transconv_layer4 = UpConvBlock(1024, 512)
        self.transconv_layer3 = UpConvBlock(512, 256)
        self.transconv_layer2 = UpConvBlock(256, 128)
        self.transconv_layer1 = UpConvBlock(128, 64)

        self.output_layer = nn.Sequential(
            conv1x1(64, num_classes),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x, skip1 = self.down_layer1(x)
        x, skip2 = self.down_layer2(x)
        x, skip3 = self.down_layer3(x)
        x, skip4 = self.down_layer4(x)
        x = self.down_layer5(x)
        if self.is_attention[4]:
            x = self.attention_layer(x)
        x = self.transconv_layer4(x, skip4)
        x = self.transconv_layer3(x, skip3)
        x = self.transconv_layer2(x, skip2)
        x = self.transconv_layer1(x, skip1)
        x = self.output_layer(x)

        return x



def unet_model(in_channel=9, num_classes=1, is_attention=[False, False, False, False, False]):

    model = UNet(in_channel=in_channel, num_classes=num_classes, is_attention=is_attention)

    return model
