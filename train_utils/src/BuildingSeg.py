import numpy as np
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from src.BasicBlock import se_block, cbam_block, eca_block, DANet, ASPP, InceptionBlock, MKBlock, TransposeConv, Upsample, MultiScale_Upconv, \
    ContextAggregation

'''
使用光学数据，实现建筑物分割任务
模型：Unet  主干：resnet50
loss=BCE+Dice
评价指标：IoU  F1_score  Recall (or Sensitivity or True Positive Rate)
'''


class Bottleneck(nn.Module):  # 输出通道加倍
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.Bottleneck_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.Bottleneck_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv1(x)
        identity = self.bn1(identity)
        identity = self.relu(identity)

        out = self.Bottleneck_conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.Bottleneck_conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class decoder_conv(nn.Module):  # 输出通道减半
    def __init__(self, in_channels, out_channels):
        super(decoder_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out


class TriUnet_encoder(nn.Module):
    def __init__(self, inchannels, ori_filter=32):
        super(TriUnet_encoder, self).__init__()

        self.encoder1 = Bottleneck(inchannels, ori_filter)
        self.encoder2 = Bottleneck(ori_filter, ori_filter*2)
        self.encoder3 = Bottleneck(ori_filter*2, ori_filter*4)
        self.encoder4 = Bottleneck(ori_filter*4, ori_filter*8)
        self.encoder5 = Bottleneck(ori_filter*8, ori_filter*16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip1 = self.encoder1(x)  # 256 256 16
        skip2 = self.encoder2(self.maxpool(skip1))  # 128 128 32
        skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256

        return skip1, skip2, skip3, skip4, bridge

class TriUnet_decoder(nn.Module):
    def __init__(self, num_feature):
        super(TriUnet_decoder, self).__init__()
         # encoder
        self.decoder4 = TransposeConv(num_feature, num_feature//2)
        self.decoder_conv4 = decoder_conv(num_feature, num_feature//2)   #(256, 128)

        self.decoder3 = TransposeConv(num_feature//2, num_feature//4)
        self.decoder_conv3 = decoder_conv(num_feature//2, num_feature//4) #(128, 64)

        self.decoder2 = TransposeConv(num_feature//4, num_feature//8)
        self.decoder_conv2 = decoder_conv(num_feature//4, num_feature//8) #(64, 32)

        self.decoder1 = TransposeConv(num_feature//8, num_feature//16)
        self.decoder_conv1 = decoder_conv(num_feature//8, num_feature//16) #(32, 16)

    def forward(self, skip1, skip2, skip3, skip4, bridge):

        out = self.decoder4(bridge)
        out = self.decoder_conv4(torch.cat([out, skip4], dim=1))

        out = self.decoder3(out)
        out = self.decoder_conv3(torch.cat([out, skip3], dim=1))

        out = self.decoder2(out)
        out = self.decoder_conv2(torch.cat([out, skip2], dim=1))

        out = self.decoder1(out)
        out = self.decoder_conv1(torch.cat([out, skip1], dim=1))

        return out

class UnetSeg(nn.Module):
    def __init__(self, channel_branchB, ori_filter=64):
        super(UnetSeg, self).__init__()

        self.part1_branchB = TriUnet_encoder(channel_branchB, ori_filter)
        self.part3_branchB = TriUnet_decoder(ori_filter*16)

        self.finalLayerB = nn.Conv2d(ori_filter, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x[:, 1:5, :, :]   #光学

        Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)

        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)

        out = self.sigmoid(self.finalLayerB(Bout))

        return out
