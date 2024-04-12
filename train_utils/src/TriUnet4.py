import numpy as np
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from src.BasicBlock import se_block, cbam_block, eca_block, DANet, ASPP, InceptionBlock, MKBlock, TransposeConv, Upsample, MultiScale_Upconv, \
    ContextAggregation

'''
multi-encoder and single-decoder
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
    def __init__(self, inchannels):
        super(TriUnet_encoder, self).__init__()

        self.encoder1 = Bottleneck(inchannels, 16)
        self.encoder2 = Bottleneck(16, 32)
        self.encoder3 = Bottleneck(32, 64)
        self.encoder4 = Bottleneck(64, 128)
        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip1 = self.encoder1(x)  # 256 256 16
        skip2 = self.encoder2(self.maxpool(skip1))  # 128 128 32
        skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256

        return skip1, skip2, skip3, skip4, bridge


class TriUnet_fuse(nn.Module):      #融合两分支
    def __init__(self, num_feature):
        super(TriUnet_fuse, self).__init__()

        self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
        self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)

    def forward(self, fuse_main, fuse_other):

        out = self.fuse(torch.cat([fuse_main, fuse_other], dim=1))
        out = self.conv(out)

        out = out + fuse_main

        return out  # 这是middle feature，之后开始上采样


class TriUnet_bridge(nn.Module):
    def __init__(self, num_feature):
        super(TriUnet_bridge, self).__init__()

        self.encoder5 = Bottleneck(num_feature, num_feature * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        bridge = self.encoder5(self.maxpool(x)) # 16 16 256
        return bridge


class TriUnet_decoder(nn.Module):
    def __init__(self, num_feature):
        super(TriUnet_decoder, self).__init__()

        # encoder
        self.decoder4 = TransposeConv(256, 128)
        self.decoder_conv4 = decoder_conv(256, 128)

        self.decoder3 = TransposeConv(128, 64)
        self.decoder_conv3 = decoder_conv(128, 64)

        self.decoder2 = TransposeConv(64, 32)
        self.decoder_conv2 = decoder_conv(64, 32)

        self.decoder1 = TransposeConv(32, 16)
        self.decoder_conv1 = decoder_conv(32, 16)

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



class TriUnet_encoder35(nn.Module):
    def __init__(self):
        super(TriUnet_encoder35, self).__init__()

        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip4):

        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return bridge



class TriUnet_fuse21(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加ASPP模块
    def __init__(self, num_feature):
        super(TriUnet_fuse21, self).__init__()
        self.multiscale = ASPP(num_feature * 2, num_feature * 2)
        self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
        self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)

    def forward(self, fuse_main, fuse_A):

        out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
        out = self.fuse(out)
        out = self.conv(out)

        out = out + fuse_main

        return out  # 这是middle feature，之后开始上采样


class TriUnet_fuse22(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加MK模块
    def __init__(self, num_feature):
        super(TriUnet_fuse22, self).__init__()
        self.multiscale = MKBlock(num_feature * 2)
        self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
        self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)

    def forward(self, fuse_main, fuse_A):

        out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
        out = self.fuse(out)
        out = self.conv(out)

        out = out + fuse_main

        return out

class TriUnet_encoder36(nn.Module):
    def __init__(self):
        super(TriUnet_encoder36, self).__init__()

        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip4):

        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return bridge

class TriUnet41(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC
    def __init__(self, channel_branchA, channel_branchB, channel_branchC):
        super(TriUnet41, self).__init__()

        self.part1_branchA = TriUnet_encoder(channel_branchA)
        self.part2_branchB1 = Bottleneck(channel_branchB, 16)
        self.part2_branchC1 = Bottleneck(channel_branchC, 16)
        self.part2_branchB2 = TriUnet_fuse22(16)
        self.part2_branchC2 = TriUnet_fuse22(16)

        self.part2_branchB3 = Bottleneck(16, 32)
        self.part2_branchC3 = Bottleneck(16, 32)
        self.part2_branchB4 = TriUnet_fuse22(32)
        self.part2_branchC4 = TriUnet_fuse22(32)

        self.part2_branchB5 = Bottleneck(32, 64)
        self.part2_branchC5 = Bottleneck(32, 64)
        self.part2_branchB6 = TriUnet_fuse22(64)
        self.part2_branchC6 = TriUnet_fuse22(64)

        self.part2_branchB7 = Bottleneck(64, 128)
        self.part2_branchC7 = Bottleneck(64, 128)
        self.part2_branchB8 = TriUnet_fuse22(128)
        self.part2_branchC8 = TriUnet_fuse22(128)

        self.part2_branchB9 = TriUnet_encoder35()
        self.part2_branchC9 = TriUnet_encoder35()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branch = TriUnet_decoder(256)

        self.finalLayer = nn.Conv2d(16, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]   #光学
        C = x[:, 0:1, :, :]

        Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
        Bskip1 = self.part2_branchB1(B)
        Cskip1 = self.part2_branchC1(C)

        RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
        RCskip1 = self.part2_branchC2(Cskip1, Bskip1)

        Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
        Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))

        RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
        RCskip2 = self.part2_branchC4(Cskip2, Bskip2)

        Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
        Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))

        RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
        RCskip3 = self.part2_branchC6(Cskip3, Bskip3)

        Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
        Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))

        out = self.part3_branch(Bskip1, Bskip2, Bskip3, Bskip4, Abridge)

        out = self.relu(self.finalLayer(out))

        return out



class TriUnet_fuse42(nn.Module):        #T4.1的B C分支第五层（bridge）融合，然后输出一个
    def __init__(self, num_feature):
        super(TriUnet_fuse42, self).__init__()
        self.multiscale = MKBlock(num_feature * 2)
        self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
        self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)

    def forward(self, fuse_main, fuse_A):

        out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
        out = self.fuse(out)
        out = self.conv(out)

        out = out

        return out



class TriUnet42(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC
    def __init__(self, channel_branchA, channel_branchB, channel_branchC):
        super(TriUnet42, self).__init__()

        self.part1_branchA = TriUnet_encoder(channel_branchA)
        self.part2_branchB1 = Bottleneck(channel_branchB, 16)
        self.part2_branchC1 = Bottleneck(channel_branchC, 16)
        self.part2_branchB2 = TriUnet_fuse22(16)
        self.part2_branchC2 = TriUnet_fuse22(16)

        self.part2_branchB3 = Bottleneck(16, 32)
        self.part2_branchC3 = Bottleneck(16, 32)
        self.part2_branchB4 = TriUnet_fuse22(32)
        self.part2_branchC4 = TriUnet_fuse22(32)

        self.part2_branchB5 = Bottleneck(32, 64)
        self.part2_branchC5 = Bottleneck(32, 64)
        self.part2_branchB6 = TriUnet_fuse22(64)
        self.part2_branchC6 = TriUnet_fuse22(64)

        self.part2_branchB7 = Bottleneck(64, 128)
        self.part2_branchC7 = Bottleneck(64, 128)
        self.part2_branchB8 = TriUnet_fuse22(128)
        self.part2_branchC8 = TriUnet_fuse22(128)

        self.part2_branchB9 = TriUnet_encoder35()
        self.part2_branchC9 = TriUnet_encoder35()

        self.part2_branchC10 = TriUnet_fuse42(256)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branch = TriUnet_decoder(256)

        self.finalLayer = nn.Conv2d(16, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]   #光学
        C = x[:, 0:1, :, :]

        Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
        Bskip1 = self.part2_branchB1(B)
        Cskip1 = self.part2_branchC1(C)

        RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
        RCskip1 = self.part2_branchC2(Cskip1, Bskip1)

        Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
        Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))

        RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
        RCskip2 = self.part2_branchC4(Cskip2, Bskip2)

        Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
        Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))

        RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
        RCskip3 = self.part2_branchC6(Cskip3, Bskip3)

        Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
        Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))

        RBskip4= self.part2_branchB8(Bskip4, Cskip4)
        RCskip4= self.part2_branchC8(Cskip4, Bskip4)

        Bbridge = self.part2_branchB9(RBskip4)
        Cbridge = self.part2_branchC9(RCskip4)

        BCbridge = self.part2_branchC10(Bbridge, Cbridge)

        out = self.part3_branch(Askip1, Askip2, Askip3, Askip4, BCbridge)

        out = self.relu(self.finalLayer(out))

        return out
