import numpy as np
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from src.BasicBlock import se_block, cbam_block, ASPP, Upsample, MKBlock, TransposeConv, UpConv, GDFM, AttentionBlock, ContextAggregation

'''
multi-(encoder-decoder)
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
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.relu(self.bn1(self.conv1(x)))
        # out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

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


class TriUnet_decoderC(nn.Module):
    def __init__(self, num_feature):
        super(TriUnet_decoderC, self).__init__()
         # encoder
        self.decoder4 = TransposeConv(num_feature*2, num_feature//2)
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

# class TriUnet3(nn.Module):      #只倒数第二层光学和夜光融合，且单独分割分支
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet3, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         # self.part2_branchA = TriUnet_fuse(256)
#         self.part2_branchB1 = TriUnet_fuse(128)
#         self.part2_branchC1 = TriUnet_fuse(128)
#
#         self.part2_branchB2 = TriUnet_bridge(128)
#         self.part2_branchC2 = TriUnet_bridge(128)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchB_seg = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB_seg = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, _ = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, _ = self.part1_branchC(C)
#
#         Bskip4 = self.part2_branchB1(Bskip4, Cskip4)
#         Cskip4 = self.part2_branchC1(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB2(Bskip4)
#         Cbridge = self.part2_branchC2(Cskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Bout_seg = self.part3_branchB_seg(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Bout_seg = self.finalLayerB_seg(Bout_seg)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Bout_seg, Cout], dim=1)))
#
#         return Aout, Bout, Bout_seg, Cout, out

# class TriUnet31(nn.Module): #只倒数第二层光学和夜光融合
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet31, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part2_branchB1 = TriUnet_fuse(128)
#         self.part2_branchC1 = TriUnet_fuse(128)
#
#         self.part2_branchB2 = TriUnet_bridge(128)
#         self.part2_branchC2 = TriUnet_bridge(128)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, _ = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, _ = self.part1_branchC(C)
#
#         Bskip4 = self.part2_branchB1(Bskip4, Cskip4)
#         Cskip4 = self.part2_branchC1(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB2(Bskip4)
#         Cbridge = self.part2_branchC2(Cskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

class TriUnet_encoder32(nn.Module):
    def __init__(self):
        super(TriUnet_encoder32, self).__init__()

        self.encoder2 = Bottleneck(16, 32)
        self.encoder3 = Bottleneck(32, 64)
        self.encoder4 = Bottleneck(64, 128)
        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip1):
        skip2 = self.encoder2(self.maxpool(skip1))  # 128 128 32
        skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return skip2, skip3, skip4, bridge

# class TriUnet32(nn.Module): #只第一层光学和夜光融合
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet32, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse(16)
#         self.part2_branchC2 = TriUnet_fuse(16)
#
#         self.part2_branchB5 = TriUnet_encoder32()
#         self.part2_branchC5 = TriUnet_encoder32()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2, Bskip3, Bskip4, Bbridge = self.part2_branchB5(RBskip1)
#         Cskip2, Cskip3, Cskip4, Cbridge = self.part2_branchC5(RCskip1)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

class TriUnet_encoder33(nn.Module):
    def __init__(self):
        super(TriUnet_encoder33, self).__init__()

        self.encoder3 = Bottleneck(32, 64)
        self.encoder4 = Bottleneck(64, 128)
        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip2):
        skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return skip3, skip4, bridge

# class TriUnet33(nn.Module):     #第一、二层光学和夜光融合
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet33, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse(16)
#         self.part2_branchC2 = TriUnet_fuse(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse(32)
#         self.part2_branchC4 = TriUnet_fuse(32)
#
#         self.part2_branchB5 = TriUnet_encoder33()
#         self.part2_branchC5 = TriUnet_encoder33()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3, Bskip4, Bbridge = self.part2_branchB5(RBskip2)
#         Cskip3, Cskip4, Cbridge = self.part2_branchC5(RCskip2)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

class TriUnet_encoder34(nn.Module):
    def __init__(self):
        super(TriUnet_encoder34, self).__init__()

        self.encoder4 = Bottleneck(64, 128)
        self.encoder5 = Bottleneck(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip3):
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return skip4, bridge

# class TriUnet34(nn.Module):     #第一、二、三层光学和夜光融合  BBC
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet34, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse(16)
#         self.part2_branchC2 = TriUnet_fuse(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse(32)
#         self.part2_branchC4 = TriUnet_fuse(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse(64)
#         self.part2_branchC6 = TriUnet_fuse(64)
#         self.part2_branchB7 = TriUnet_encoder34()
#         self.part2_branchC7 = TriUnet_encoder34()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4, Bbridge = self.part2_branchB7(RBskip3)
#         Cskip4, Cbridge = self.part2_branchC7(RCskip3)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out


class TriUnet_encoder35(nn.Module):
    def __init__(self, ori_filter=32):
        super(TriUnet_encoder35, self).__init__()

        self.encoder5 = Bottleneck(ori_filter*8, ori_filter*16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip4):

        bridge = self.encoder5(self.maxpool(skip4))  # 16 16 256
        return bridge

# class TriUnet35(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC  没有MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet35, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse(16)
#         self.part2_branchC2 = TriUnet_fuse(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse(32)
#         self.part2_branchC4 = TriUnet_fuse(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse(64)
#         self.part2_branchC6 = TriUnet_fuse(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse(128)
#         self.part2_branchC8 = TriUnet_fuse(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

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

# class TriUnet36a(nn.Module):     #T36  decoder时BBC修改为cat(AB)BC
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet36a, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = T6_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet36b(nn.Module):     #T36  decoder时BBC修改为(A+B)BC
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet36b, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge+Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out
# class sTriUnet_fuse22(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加MK模块
#     def __init__(self, num_feature):
#         super(sTriUnet_fuse22, self).__init__()
#         self.multiscale = sMKBlock(num_feature * 2)
#         self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
#         self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A):
#
#         out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
#         out = self.fuse(out)
#         out = self.conv(out)
#         out = out + fuse_main
#         return out
#
# class sTriUnet36(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(sTriUnet36, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = sTriUnet_fuse22(16)
#         self.part2_branchC2 = sTriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = sTriUnet_fuse22(32)
#         self.part2_branchC4 = sTriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = sTriUnet_fuse22(64)
#         self.part2_branchC6 = sTriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = sTriUnet_fuse22(128)
#         self.part2_branchC8 = sTriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out
#
# class lTriUnet_fuse22(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加MK模块
#     def __init__(self, num_feature):
#         super(lTriUnet_fuse22, self).__init__()
#         self.multiscale = lMKBlock(num_feature * 2)
#         self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
#         self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A):
#
#         out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
#         out = self.fuse(out)
#         out = self.conv(out)
#         out = out + fuse_main
#         return out
#
# class lTriUnet36(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(lTriUnet36, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = lTriUnet_fuse22(16)
#         self.part2_branchC2 = lTriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = lTriUnet_fuse22(32)
#         self.part2_branchC4 = lTriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = lTriUnet_fuse22(64)
#         self.part2_branchC6 = lTriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = lTriUnet_fuse22(128)
#         self.part2_branchC8 = lTriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet364(nn.Module):     #第一、二、三、四、五层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet364, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#         self.part2_branchB10 = TriUnet_fuse22(256)
#         self.part2_branchC10 = TriUnet_fuse22(256)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         RBbridge = self.part2_branchB10(Bbridge, Cbridge)
#         RCbridge = self.part2_branchC10(Cbridge, Bbridge)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, RBbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, RBbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, RCbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class double_fusion(nn.Module):        #MK+DANet
#     def __init__(self, num_feature):
#         super(double_fusion, self).__init__()
#         self.multiscale = MKBlock(num_feature * 2)
#         self.fuse = DANet(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
#         self.reduce_dim = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A):
#
#         out = self.multiscale(torch.cat([fuse_main, fuse_A], dim=1))
#         out = self.fuse(out)
#         out = self.reduce_dim(out)
#         out = out + fuse_main
#         return out

# class T362_fuse(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加MK模块
#     def __init__(self, num_feature):
#         super(T362_fuse, self).__init__()
#         self.multiscale = MKBlock(num_feature * 3)
#         self.fuse = cbam_block(num_feature * 3)     # self.fuse = eca_block(num_feature * 2)
#         self.dim_reduce = nn.Conv2d(num_feature * 3, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A, fuse_B):
#
#         out = self.multiscale(torch.cat([fuse_main, fuse_A, fuse_B], dim=1))
#         out = self.fuse(out)
#         out = self.dim_reduce(out)
#         return out + fuse_main

# class TriUnet362(nn.Module):     #三独立encoder  第四层三分支融合  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet362, self).__init__()
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         # self.part2_branchA2 = T362_fuse(16)
#         # self.part2_branchB2 = T362_fuse(16)
#         # self.part2_branchC2 = T362_fuse(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         # self.part2_branchA4 = T362_fuse(32)
#         # self.part2_branchB4 = T362_fuse(32)
#         # self.part2_branchC4 = T362_fuse(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         # self.part2_branchA6 = T362_fuse(64)
#         # self.part2_branchB6 = T362_fuse(64)
#         # self.part2_branchC6 = T362_fuse(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = T362_fuse(128)
#         self.part2_branchB8 = T362_fuse(128)
#         self.part2_branchC8 = T362_fuse(128)
#
#         self.part2_branchA9 = TriUnet_encoder35()
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         # Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Askip1 = self.part2_branchA1(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         # RAskip1 = self.part2_branchA2(Askip1, Bskip1, Cskip1)
#         # RCskip1 = self.part2_branchC2(Askip1, Bskip1, Cskip1)
#         # RBskip1 = self.part2_branchB2(Askip1, Bskip1, Cskip1)
#
#         Askip2 = self.part2_branchA3(self.maxpool(Askip1))
#         Bskip2 = self.part2_branchB3(self.maxpool(Bskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(Cskip1))
#         # Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
#         # Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         # Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         # RAskip2 = self.part2_branchA4(Askip2, Bskip2, Cskip2)
#         # RBskip2 = self.part2_branchB4(Askip2, Bskip2, Cskip2)
#         # RCskip2 = self.part2_branchC4(Askip2, Bskip2, Cskip2)
#
#         Askip3 = self.part2_branchA5(self.maxpool(Askip2))
#         Bskip3 = self.part2_branchB5(self.maxpool(Bskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(Cskip2))
#         # Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#         # Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         # Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         # RAskip3 = self.part2_branchA6(Askip3, Bskip3, Cskip3)
#         # RBskip3 = self.part2_branchB6(Askip3, Bskip3, Cskip3)
#         # RCskip3 = self.part2_branchC6(Askip3, Bskip3, Cskip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(Askip3))
#         Bskip4 = self.part2_branchB7(self.maxpool(Bskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(Cskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Bskip4, Cskip4)
#         RBskip4= self.part2_branchB8(Askip4, Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Askip4, Bskip4, Cskip4)
#
#         Abridge = self.part2_branchA9(RAskip4)
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out
#
# class TriUnet363(nn.Module):     #T362的第四层融合改为第五层三分支融合  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet363, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part2_branchA8 = T362_fuse(256)
#         self.part2_branchB8 = T362_fuse(256)
#         self.part2_branchC8 = T362_fuse(256)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#         RAbridge= self.part2_branchA8(Abridge, Bbridge, Cbridge)
#         RBbridge= self.part2_branchB8(Abridge, Bbridge, Cbridge)
#         RCbridge= self.part2_branchC8(Abridge, Bbridge, Cbridge)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, RAbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, RBbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, RCbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out
#
# class TriUnet361(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet361, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchBs = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerBs = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(5, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Bouts = self.part3_branchBs(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Bouts = self.finalLayerBs(Bouts)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Bouts, Cout], dim=1)))
#
#         return Aout, Bout, Bouts, Cout, out

# class TriUnet46(nn.Module):     #第一、二、三、四层(光学+夜光)和SAR融合 加MK+CBAM  decoder部分AB  B分支也做回归分支
#     def __init__(self, channel_branchA, channel_branchBC):
#         super(TriUnet46, self).__init__()
#
#         self.part2_branchB1 = Bottleneck(channel_branchBC, 16)
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchA2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchA4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchA6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchA8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchA9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(2, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 0:5, :, :]   #光学+夜光
#
#         Bskip1 = self.part2_branchB1(B)
#         Askip1 = self.part2_branchA1(A)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Askip1)
#         RAskip1 = self.part2_branchA2(Askip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Askip2)
#         RAskip2 = self.part2_branchA4(Askip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Askip3)
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Askip4)
#         RAskip4= self.part2_branchA8(Askip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Abridge = self.part2_branchA9(RAskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout], dim=1)))
#
#         return Aout, Bout, out

# class newTriUnet46(nn.Module):     #第一、二、三、四层(光学+夜光)和SAR融合 加MK+CBAM  decoder部分AB  B分支做分割分支
#     def __init__(self, channel_branchA, channel_branchBC):
#         super(newTriUnet46, self).__init__()
#
#         self.part2_branchB1 = Bottleneck(channel_branchBC, 16)
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchA2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchA4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchA6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchA8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchA9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 0:5, :, :]   #光学+夜光
#
#         Bskip1 = self.part2_branchB1(B)
#         Askip1 = self.part2_branchA1(A)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Askip1)
#         RAskip1 = self.part2_branchA2(Askip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Askip2)
#         RAskip2 = self.part2_branchA4(Askip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Askip3)
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Askip4)
#         RAskip4= self.part2_branchA8(Askip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Abridge = self.part2_branchA9(RAskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout], dim=1)))
#
#         return Aout, Bout, out

# class shaocat(nn.Module):
#     def __init__(self, num_feature):
#         super(shaocat, self).__init__()
#         self.fuse = cbam_block(num_feature * 2)     # self.fuse = eca_block(num_feature * 2)
#         self.conv = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A):
#         concat = torch.cat([fuse_main, fuse_A], dim=1)
#         fuse_out = self.fuse(concat)
#         out = self.conv(fuse_out)
#         return out

# class shao(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB):
#         super(shao, self).__init__()
#
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchA3 = Bottleneck(16, 32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB6 = shaocat(64) #TriUnet_fuse22(64)
#         self.part2_branchA6 = shaocat(64) #TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB8 = shaocat(128)  #TriUnet_fuse22(128)
#         self.part2_branchA8 = shaocat(128)  #TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchA9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 1+1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#
#         Bskip1 = self.part2_branchB1(B)
#         Askip1 = self.part2_branchA1(A)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(Bskip1))
#         Askip2 = self.part2_branchA3(self.maxpool(Askip1))
#
#         Bskip3 = self.part2_branchB5(self.maxpool(Bskip2))
#         Askip3 = self.part2_branchA5(self.maxpool(Askip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Askip3)
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Askip4)
#         RAskip4= self.part2_branchA8(Askip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Abridge = self.part2_branchA9(RAskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#
#         # out = self.relu(self.finalLayer(torch.cat([Aout, Bout], dim=1)))
#         out = self.finalLayer(torch.cat([Aout, Bout], dim=1))
#
#         return Bout, out

# class T5_decoder(nn.Module):
#     def __init__(self, num_feature):
#         super(T5_decoder, self).__init__()
#
#         # encoder
#         self.decoder4 = TransposeConv(num_feature*3, num_feature//2)
#         self.decoder_conv4 = decoder_conv(num_feature, num_feature//2)   #(256, 128)
#
#         self.decoder3 = TransposeConv(num_feature//2, num_feature//4)
#         self.decoder_conv3 = decoder_conv(num_feature//2, num_feature//4) #(128, 64)
#
#         self.decoder2 = TransposeConv(num_feature//4, num_feature//8)
#         self.decoder_conv2 = decoder_conv(num_feature//4, num_feature//8) #(64, 32)
#
#         self.decoder1 = TransposeConv(num_feature//8, num_feature//16)
#         self.decoder_conv1 = decoder_conv(num_feature//8, num_feature//16) #(32, 16)
#
#     def forward(self, skip1, skip2, skip3, skip4, bridgeA, bridgeB, bridgeC):
#
#         bridge = torch.cat([bridgeA, bridgeB, bridgeC], dim=1)
#
#         out = self.decoder4(bridge)
#         out = self.decoder_conv4(torch.cat([out, skip4], dim=1))
#
#         out = self.decoder3(out)
#         out = self.decoder_conv3(torch.cat([out, skip3], dim=1))
#
#         out = self.decoder2(out)
#         out = self.decoder_conv2(torch.cat([out, skip2], dim=1))
#
#         out = self.decoder1(out)
#         out = self.decoder_conv1(torch.cat([out, skip1], dim=1))
#
#         return out

# class TriUnet5(nn.Module):     #各分支不融合，最后一层交互  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet5, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part3_branchA = T5_decoder(256)
#         self.part3_branchBr = T5_decoder(256)
#         self.part3_branchBs = T5_decoder(256)
#         self.part3_branchC = T5_decoder(256)
#
#         self.regA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.segB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.regC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regF = nn.Conv2d(5, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge, Cbridge)
#         Boutr = self.part3_branchBr(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Bouts = self.part3_branchBs(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Abridge, Bbridge, Cbridge)
#
#         Aout = self.regA(Aout)
#         Boutr = self.regB(Boutr)
#         Bouts = self.segB(Bouts)
#         Cout = self.regC(Cout)
#
#         out = self.relu(self.regF(torch.cat([Aout, Boutr, Bouts, Cout], dim=1)))
#
#         return Aout, Boutr, Bouts, Cout, out

# class TriUnet5(nn.Module):     #各分支不融合，最后一层交互  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet5, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part3_branchA = T5_decoder(256)
#         self.part3_branchB = T5_decoder(256)
#         self.part3_branchC = T5_decoder(256)
#
#         self.regA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.segB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.regC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regF = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge, Cbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Abridge, Bbridge, Cbridge)
#
#         Aout = self.regA(Aout)
#         Bout = self.segB(Bout)
#         Cout = self.regC(Cout)
#
#         out = self.relu(self.regF(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class T51_decoder(nn.Module):
#     def __init__(self, num_feature):
#         super(T51_decoder, self).__init__()
#
#         self.fuse = cbam_block(num_feature * 3)
#         # encoder
#         self.decoder4 = TransposeConv(num_feature*3, num_feature//2)
#         self.decoder_conv4 = decoder_conv(num_feature, num_feature//2)   #(256, 128)
#
#         self.decoder3 = TransposeConv(num_feature//2, num_feature//4)
#         self.decoder_conv3 = decoder_conv(num_feature//2, num_feature//4) #(128, 64)
#
#         self.decoder2 = TransposeConv(num_feature//4, num_feature//8)
#         self.decoder_conv2 = decoder_conv(num_feature//4, num_feature//8) #(64, 32)
#
#         self.decoder1 = TransposeConv(num_feature//8, num_feature//16)
#         self.decoder_conv1 = decoder_conv(num_feature//8, num_feature//16) #(32, 16)
#
#     def forward(self, skip1, skip2, skip3, skip4, bridgeA, bridgeB, bridgeC):
#
#         concat = torch.cat([bridgeA, bridgeB, bridgeC], dim=1)
#         bridge = self.fuse(concat)
#
#         out = self.decoder4(bridge)
#         out = self.decoder_conv4(torch.cat([out, skip4], dim=1))
#
#         out = self.decoder3(out)
#         out = self.decoder_conv3(torch.cat([out, skip3], dim=1))
#
#         out = self.decoder2(out)
#         out = self.decoder_conv2(torch.cat([out, skip2], dim=1))
#
#         out = self.decoder1(out)
#         out = self.decoder_conv1(torch.cat([out, skip1], dim=1))
#
#         return out
#
# class TriUnet51(nn.Module):     #各分支不融合，最后一层交互  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet51, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part3_branchA = T51_decoder(256)
#         self.part3_branchBr = T51_decoder(256)
#         self.part3_branchBs = T51_decoder(256)
#         self.part3_branchC = T51_decoder(256)
#
#         self.regA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.segB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.regC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regF = nn.Conv2d(5, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge, Cbridge)
#         Boutr = self.part3_branchBr(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Bouts = self.part3_branchBs(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Abridge, Bbridge, Cbridge)
#
#         Aout = self.regA(Aout)
#         Boutr = self.regB(Boutr)
#         Bouts = self.segB(Bouts)
#         Cout = self.regC(Cout)
#
#         out = self.relu(self.regF(torch.cat([Aout, Boutr, Bouts, Cout], dim=1)))
#
#         return Aout, Boutr, Bouts, Cout, out

# class T6_encoder(nn.Module):
#     def __init__(self, inchannels):
#         super(T6_encoder, self).__init__()
#
#         self.encoder1 = Bottleneck(inchannels, 16)
#         self.encoder2 = Bottleneck(16, 32)
#         self.encoder3 = Bottleneck(32, 64)
#         self.encoder4 = Bottleneck(64, 128)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         skip1 = self.encoder1(x)  # 256 256 16
#         skip2 = self.encoder2(self.maxpool(skip1))  # 128 128 32
#         skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
#         skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128
#
#         return skip1, skip2, skip3, skip4

# class T6_decoder(nn.Module):
#     def __init__(self, num_feature):
#         super(T6_decoder, self).__init__()
#
#         # encoder
#         self.decoder4 = TransposeConv(num_feature*2, num_feature//2)
#         self.decoder_conv4 = decoder_conv(num_feature, num_feature//2)   #(256, 128)
#
#         self.decoder3 = TransposeConv(num_feature//2, num_feature//4)
#         self.decoder_conv3 = decoder_conv(num_feature//2, num_feature//4) #(128, 64)
#
#         self.decoder2 = TransposeConv(num_feature//4, num_feature//8)
#         self.decoder_conv2 = decoder_conv(num_feature//4, num_feature//8) #(64, 32)
#
#         self.decoder1 = TransposeConv(num_feature//8, num_feature//16)
#         self.decoder_conv1 = decoder_conv(num_feature//8, num_feature//16) #(32, 16)
#
#     def forward(self, skip1, skip2, skip3, skip4, bridgeA, bridgeB):
#
#         concat = torch.cat([bridgeA, bridgeB], dim=1)
#         out = self.decoder4(concat)
#         out = self.decoder_conv4(torch.cat([out, skip4], dim=1))
#
#         out = self.decoder3(out)
#         out = self.decoder_conv3(torch.cat([out, skip3], dim=1))
#
#         out = self.decoder2(out)
#         out = self.decoder_conv2(torch.cat([out, skip2], dim=1))
#
#         out = self.decoder1(out)
#         out = self.decoder_conv1(torch.cat([out, skip1], dim=1))
#
#         return out



# class TriUnet6(nn.Module):     #AC第四、五层融合 MK+DANet BBB
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet6, self).__init__()
#
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part2_branchA1 = T6_encoder(channel_branchA)
#         self.part2_branchC1 = T6_encoder(channel_branchC)
#
#         self.part2_branchA8 = double_fusion(128)
#         self.part2_branchC8 = double_fusion(128)
#
#         self.part2_branchA9 = Bottleneck(128, 256)
#         self.part2_branchC9 = Bottleneck(128, 256)
#
#         self.part2_branchA10 = double_fusion(256)
#         self.part2_branchC10 = double_fusion(256)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = T6_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = T6_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Askip1, Askip2, Askip3, Askip4 = self.part2_branchA1(A)
#         Cskip1, Cskip2, Cskip3, Cskip4 = self.part2_branchC1(C)
#
#         RAskip4= self.part2_branchA8(Askip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4)
#
#         Abridge = self.part2_branchA9(self.maxpool(RAskip4))
#         Cbridge = self.part2_branchC9(self.maxpool(RCskip4))
#
#         RAbridge = self.part2_branchA10(Abridge, Cbridge)
#         RCbridge = self.part2_branchC10(Cbridge, Abridge)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, RAbridge, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, RCbridge, Bbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet61(nn.Module):     #AC第四、五层融合 MK+DANet BBB
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet61, self).__init__()
#
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchA2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchA4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchA6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchA9 = Bottleneck(128, 256)
#         self.part2_branchC9 = Bottleneck(128, 256)
#         # self.part2_branchA10 = double_fusion(256)
#         # self.part2_branchC10 = double_fusion(256)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = T6_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = T6_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#
#         Askip1 = self.part2_branchA1(A)
#         Cskip1 = self.part2_branchC1(C)
#
#         RAskip1 = self.part2_branchA2(Askip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Askip1)
#
#         Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RAskip2 = self.part2_branchA4(Askip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Askip2)
#
#         Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RAskip3 = self.part2_branchA6(Askip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Askip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4)
#
#         Abridge = self.part2_branchA9(self.maxpool(RAskip4))
#         Cbridge = self.part2_branchC9(self.maxpool(RCskip4))
#
#         # RAbridge = self.part2_branchA10(Abridge, Cbridge)
#         # RCbridge = self.part2_branchC10(Cbridge, Abridge)
#
#         # Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, RAbridge, Bbridge)
#         # Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         # Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, RCbridge, Bbridge)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge, Bbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out


# class TriUnet365(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet365, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = double_fusion(16)
#         self.part2_branchC2 = double_fusion(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = double_fusion(32)
#         self.part2_branchC4 = double_fusion(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = double_fusion(64)
#         self.part2_branchC6 = double_fusion(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = double_fusion(128)
#         self.part2_branchC8 = double_fusion(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class GDFM_fusion(nn.Module):        #GDFM
#     def __init__(self, num_feature, H, W):
#         super(GDFM_fusion, self).__init__()
#         self.fusion = GDFM(num_feature, H, W)
#         self.reduce_dim = nn.Conv2d(num_feature * 2, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A):
#
#         out = self.fusion(fuse_main, fuse_A)
#         out = self.reduce_dim(out)
#         out = out + fuse_main
#         return out

# class GDFM(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK   内存不足
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(GDFM, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = GDFM_fusion(16, 256, 256)
#         self.part2_branchC2 = GDFM_fusion(16, 256, 256)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = GDFM_fusion(32, 128, 128)
#         self.part2_branchC4 = GDFM_fusion(32, 128, 128)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = GDFM_fusion(64, 64, 64)
#         self.part2_branchC6 = GDFM_fusion(64, 64, 64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = GDFM_fusion(128, 32, 32)
#         self.part2_branchC8 = GDFM_fusion(128, 32, 32)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet52(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet52, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.part3_branchA = T51_decoder(256)
#         self.part3_branchB = T51_decoder(256)
#         self.part3_branchC = T51_decoder(256)
#
#         self.regA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.segB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.regC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regF = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge, Bbridge, Cbridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Abridge, Bbridge, Cbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Abridge, Bbridge, Cbridge)
#
#         Aout = self.regA(Aout)
#         Bout = self.segB(Bout)
#         Cout = self.regC(Cout)
#
#         out = self.relu(self.regF(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet53(nn.Module):     #各分支不融合，最后一层交互  ABC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet53, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part1_branchB = TriUnet_encoder(channel_branchB)
#         self.part1_branchC = TriUnet_encoder(channel_branchC)
#
#         self.ABA = TriUnet_fuse22(256)
#         self.ABB = TriUnet_fuse22(256)
#         self.ACA = TriUnet_fuse22(256)
#         self.ACC = TriUnet_fuse22(256)
#         self.BCB = TriUnet_fuse22(256)
#         self.BCC = TriUnet_fuse22(256)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.regA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.segB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.regC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.regF = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)
#         Cskip1, Cskip2, Cskip3, Cskip4, Cbridge = self.part1_branchC(C)
#
#         A1 = self.ABA(Abridge, Bbridge)
#         B1 = self.ABB(Bbridge, Abridge)
#         A2 = self.ACA(Abridge, Cbridge)
#         C1 = self.ACC(Cbridge, Abridge)
#         B2 = self.BCB(Bbridge, Cbridge)
#         C2 = self.BCC(Cbridge, Bbridge)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, A1+A2)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, B1+B2)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, C1+C2)
#
#         Aout = self.regA(Aout)
#         Bout = self.segB(Bout)
#         Cout = self.regC(Cout)
#
#         out = self.relu(self.regF(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet366(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet366, self).__init__()
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchA2 = T362_fuse(16)
#         self.part2_branchB2 = T362_fuse(16)
#         self.part2_branchC2 = T362_fuse(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchA4 = T362_fuse(32)
#         self.part2_branchB4 = T362_fuse(32)
#         self.part2_branchC4 = T362_fuse(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchA6 = T362_fuse(64)
#         self.part2_branchB6 = T362_fuse(64)
#         self.part2_branchC6 = T362_fuse(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = T362_fuse(128)
#         self.part2_branchB8 = T362_fuse(128)
#         self.part2_branchC8 = T362_fuse(128)
#
#         self.part2_branchA9 = TriUnet_encoder35()
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1 = self.part2_branchA1(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RAskip1 = self.part2_branchA2(Askip1, Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Askip1, Bskip1)
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1, Askip1)
#
#         Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RAskip2 = self.part2_branchA4(Askip2, Bskip2, Cskip2)
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2, Askip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Askip2, Bskip2)
#
#         Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3, Cskip3)
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3, Askip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Askip3, Bskip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Bskip4, Cskip4)
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4, Askip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4, Bskip4)
#
#         Abridge = self.part2_branchA9(RAskip4)
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet367(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet367, self).__init__()
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         # self.part2_branchA2 = T362_fuse(16)
#         # self.part2_branchB2 = T362_fuse(16)
#         # self.part2_branchC2 = T362_fuse(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         # self.part2_branchA4 = T362_fuse(32)
#         # self.part2_branchB4 = T362_fuse(32)
#         # self.part2_branchC4 = T362_fuse(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchA6 = T362_fuse(64)
#         self.part2_branchB6 = T362_fuse(64)
#         self.part2_branchC6 = T362_fuse(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = T362_fuse(128)
#         self.part2_branchB8 = T362_fuse(128)
#         self.part2_branchC8 = T362_fuse(128)
#
#         self.part2_branchA9 = TriUnet_encoder35()
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1 = self.part2_branchA1(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         Askip2 = self.part2_branchA3(self.maxpool(Askip1))
#         Bskip2 = self.part2_branchB3(self.maxpool(Bskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(Cskip1))
#
#         Askip3 = self.part2_branchA5(self.maxpool(Askip2))
#         Bskip3 = self.part2_branchB5(self.maxpool(Bskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(Cskip2))
#
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3, Cskip3)
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3, Askip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Askip3, Bskip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Bskip4, Cskip4)
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4, Askip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4, Bskip4)
#
#         Abridge = self.part2_branchA9(RAskip4)
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class lT362_fuse(nn.Module):        #在TriUnet2.0的基础上，自注意力机制前加MK模块
#     def __init__(self, num_feature):
#         super(lT362_fuse, self).__init__()
#         self.multiscale = lMKBlock(num_feature * 3)
#         self.fuse = cbam_block(num_feature * 3)     # self.fuse = eca_block(num_feature * 2)
#         self.dim_reduce = nn.Conv2d(num_feature * 3, num_feature, kernel_size=1, stride=1)
#
#     def forward(self, fuse_main, fuse_A, fuse_B):
#
#         out = self.multiscale(torch.cat([fuse_main, fuse_A, fuse_B], dim=1))
#         out = self.fuse(out)
#         out = self.dim_reduce(out)
#         return out + fuse_main
#
# class lTriUnet367(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(lTriUnet367, self).__init__()
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         # self.part2_branchA2 = T362_fuse(16)
#         # self.part2_branchB2 = T362_fuse(16)
#         # self.part2_branchC2 = T362_fuse(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         # self.part2_branchA4 = T362_fuse(32)
#         # self.part2_branchB4 = T362_fuse(32)
#         # self.part2_branchC4 = T362_fuse(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchA6 = lT362_fuse(64)
#         self.part2_branchB6 = lT362_fuse(64)
#         self.part2_branchC6 = lT362_fuse(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = lT362_fuse(128)
#         self.part2_branchB8 = lT362_fuse(128)
#         self.part2_branchC8 = lT362_fuse(128)
#
#         self.part2_branchA9 = TriUnet_encoder35()
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1 = self.part2_branchA1(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         Askip2 = self.part2_branchA3(self.maxpool(Askip1))
#         Bskip2 = self.part2_branchB3(self.maxpool(Bskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(Cskip1))
#
#         Askip3 = self.part2_branchA5(self.maxpool(Askip2))
#         Bskip3 = self.part2_branchB5(self.maxpool(Bskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(Cskip2))
#
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3, Cskip3)
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3, Askip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Askip3, Bskip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Bskip4, Cskip4)
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4, Askip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4, Bskip4)
#
#         Abridge = self.part2_branchA9(RAskip4)
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out

# class TriUnet368(nn.Module):
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(TriUnet368, self).__init__()
#
#         self.part2_branchA1 = Bottleneck(channel_branchA, 16)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         # self.part2_branchA2 = T362_fuse(16)
#         # self.part2_branchB2 = T362_fuse(16)
#         # self.part2_branchC2 = T362_fuse(16)
#
#         self.part2_branchA3 = Bottleneck(16, 32)
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchA4 = T362_fuse(32)
#         self.part2_branchB4 = T362_fuse(32)
#         self.part2_branchC4 = T362_fuse(32)
#
#         self.part2_branchA5 = Bottleneck(32, 64)
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchA6 = T362_fuse(64)
#         self.part2_branchB6 = T362_fuse(64)
#         self.part2_branchC6 = T362_fuse(64)
#
#         self.part2_branchA7 = Bottleneck(64, 128)
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchA8 = T362_fuse(128)
#         self.part2_branchB8 = T362_fuse(128)
#         self.part2_branchC8 = T362_fuse(128)
#
#         self.part2_branchA9 = TriUnet_encoder35()
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.part3_branchA = TriUnet_decoder(256)
#         self.part3_branchB = TriUnet_decoder(256)
#         self.part3_branchC = TriUnet_decoder(256)
#
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1 = self.part2_branchA1(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         Askip2 = self.part2_branchA3(self.maxpool(Askip1))
#         Bskip2 = self.part2_branchB3(self.maxpool(Bskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(Cskip1))
#
#         RAskip2 = self.part2_branchA4(Askip2, Bskip2, Cskip2)
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2, Askip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Askip2, Bskip2)
#
#         Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RAskip3 = self.part2_branchA6(Askip3, Bskip3, Cskip3)
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3, Askip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Askip3, Bskip3)
#
#         Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RAskip4= self.part2_branchA8(Askip4, Bskip4, Cskip4)
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4, Askip4)
#         RCskip4= self.part2_branchC8(Cskip4, Askip4, Bskip4)
#
#         Abridge = self.part2_branchA9(RAskip4)
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abridge)
#         Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(Bout)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

# class AttentionUNet_decoder(nn.Module):
#     def __init__(self, num_feature):
#         super(AttentionUNet_decoder, self).__init__()
#          # encoder
#         self.decoder4 = UpConv(num_feature, num_feature//2)
#         self.decoder_conv4 = decoder_conv(num_feature, num_feature//2)   #(256, 128)
#
#         self.decoder3 = UpConv(num_feature//2, num_feature//4)
#         self.decoder_conv3 = decoder_conv(num_feature//2, num_feature//4) #(128, 64)
#
#         self.decoder2 = UpConv(num_feature//4, num_feature//8)
#         self.decoder_conv2 = decoder_conv(num_feature//4, num_feature//8) #(64, 32)
#
#         self.decoder1 = UpConv(num_feature//8, num_feature//16)
#         self.decoder_conv1 = decoder_conv(num_feature//8, num_feature//16) #(32, 16)
#
#     def forward(self, skip1, skip2, skip3, skip4, bridge):
#
#         out = self.decoder4(bridge)
#         out = self.decoder_conv4(torch.cat([out, skip4], dim=1))
#
#         out = self.decoder3(out)
#         out = self.decoder_conv3(torch.cat([out, skip3], dim=1))
#
#         out = self.decoder2(out)
#         out = self.decoder_conv2(torch.cat([out, skip2], dim=1))
#
#         out = self.decoder1(out)
#         out = self.decoder_conv1(torch.cat([out, skip1], dim=1))
#
#         return out

# class AttentionUNet(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
#     def __init__(self, channel_branchA, channel_branchB, channel_branchC):
#         super(AttentionUNet, self).__init__()
#
#         self.part1_branchA = TriUnet_encoder(channel_branchA)
#         self.part2_branchB1 = Bottleneck(channel_branchB, 16)
#         self.part2_branchC1 = Bottleneck(channel_branchC, 16)
#         self.part2_branchB2 = TriUnet_fuse22(16)
#         self.part2_branchC2 = TriUnet_fuse22(16)
#
#         self.part2_branchB3 = Bottleneck(16, 32)
#         self.part2_branchC3 = Bottleneck(16, 32)
#         self.part2_branchB4 = TriUnet_fuse22(32)
#         self.part2_branchC4 = TriUnet_fuse22(32)
#
#         self.part2_branchB5 = Bottleneck(32, 64)
#         self.part2_branchC5 = Bottleneck(32, 64)
#         self.part2_branchB6 = TriUnet_fuse22(64)
#         self.part2_branchC6 = TriUnet_fuse22(64)
#
#         self.part2_branchB7 = Bottleneck(64, 128)
#         self.part2_branchC7 = Bottleneck(64, 128)
#         self.part2_branchB8 = TriUnet_fuse22(128)
#         self.part2_branchC8 = TriUnet_fuse22(128)
#
#         self.part2_branchB9 = TriUnet_encoder35()
#         self.part2_branchC9 = TriUnet_encoder35()
#
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         #decoder stage
#         self.part3_branchA = AttentionUNet_decoder(256)
#         self.part3_branchC = AttentionUNet_decoder(256)
#
#         # branch B
#         self.Up5 = UpConv(256, 128)   #不是用TransposeConv,而是nn.Upsample+conv3*3,实现通道减半，尺寸2倍
#         self.Att5 = AttentionBlock(F_g=128, F_l=128, n_coefficients=256)
#         self.UpConv5 = ConvBlock(256, 128)
#
#         self.Up4 = UpConv(128, 64)
#         self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=128)
#         self.UpConv4 = ConvBlock(128, 64)
#
#         self.Up3 = UpConv(64, 32)
#         self.Att3 = AttentionBlock(F_g=32, F_l=32, n_coefficients=64)
#         self.UpConv3 = ConvBlock(64, 32)
#
#         self.Up2 = UpConv(32, 16)
#         self.Att2 = AttentionBlock(F_g=16, F_l=16, n_coefficients=32)
#         self.UpConv2 = ConvBlock(32, 16)
#
#         # final stage
#         self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayerB = nn.Conv2d(16, 2, kernel_size=1, stride=1)
#         self.finalLayerC = nn.Conv2d(16, 1, kernel_size=1, stride=1)
#         self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
#
#         self.relu = nn.ReLU(inplace=True)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         A = x[:, 5:7, :, :]
#         B = x[:, 1:5, :, :]   #光学
#         C = x[:, 0:1, :, :]
#
#         Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
#         Bskip1 = self.part2_branchB1(B)
#         Cskip1 = self.part2_branchC1(C)
#
#         RBskip1 = self.part2_branchB2(Bskip1, Cskip1)
#         RCskip1 = self.part2_branchC2(Cskip1, Bskip1)
#
#         Bskip2 = self.part2_branchB3(self.maxpool(RBskip1))
#         Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))
#
#         RBskip2 = self.part2_branchB4(Bskip2, Cskip2)
#         RCskip2 = self.part2_branchC4(Cskip2, Bskip2)
#
#         Bskip3 = self.part2_branchB5(self.maxpool(RBskip2))
#         Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))
#
#         RBskip3 = self.part2_branchB6(Bskip3, Cskip3)
#         RCskip3 = self.part2_branchC6(Cskip3, Bskip3)
#
#         Bskip4 = self.part2_branchB7(self.maxpool(RBskip3))
#         Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))
#
#         RBskip4= self.part2_branchB8(Bskip4, Cskip4)
#         RCskip4= self.part2_branchC8(Cskip4, Bskip4)
#
#         Bbridge = self.part2_branchB9(RBskip4)
#         Cbridge = self.part2_branchC9(RCskip4)
#
#         # decoder stage
#         Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
#         Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)
#
#
#         # branch B
#         d5 = self.Up5(Bbridge)
#
#         s4 = self.Att5(gate=d5, skip_connection=Bskip4)
#         d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
#         d5 = self.UpConv5(d5)
#
#         d4 = self.Up4(d5)
#         s3 = self.Att4(gate=d4, skip_connection=Bskip3)
#         d4 = torch.cat((s3, d4), dim=1)
#         d4 = self.UpConv4(d4)
#
#         d3 = self.Up3(d4)
#         s2 = self.Att3(gate=d3, skip_connection=Bskip2)
#         d3 = torch.cat((s2, d3), dim=1)
#         d3 = self.UpConv3(d3)
#
#         d2 = self.Up2(d3)
#         s1 = self.Att2(gate=d2, skip_connection=Bskip1)
#         d2 = torch.cat((s1, d2), dim=1)
#         d2 = self.UpConv2(d2)
#
#         # final stage
#         # Aout = self.relu(self.finalLayerA(Aout))
#         # Bout = self.sigmoid(self.finalLayerB(d2))
#         # Cout = self.relu(self.finalLayerC(Cout))
#         Aout = self.finalLayerA(Aout)
#         Bout = self.finalLayerB(d2)
#         Cout = self.finalLayerC(Cout)
#
#         out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))
#
#         return Aout, Bout, Cout, out


class TriUnet36t(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        # 对二分类分割任务，输出也可以是单通道的，通过sigmoid限制其输出在0 1之间
        # self.finalLayerB = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        # self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        # self.sigmoid = nn.Sigmoid()

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t_VIIRS(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t_VIIRS, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 4:6, :, :]
        B = x[:, 0:4, :, :]   #光学
        C = x[:, 6:7, :, :]

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t_optical(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t_optical, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        # 对二分类分割任务，输出也可以是单通道的，通过sigmoid限制其输出在0 1之间
        # self.finalLayerB = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        # self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        # self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]   #光学

        Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
        Bskip1 = self.part2_branchB1(B)
        Cskip1 = self.part2_branchC1(B)

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t_sar(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t_sar, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchA, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]   #光学
        C = x[:, 0:1, :, :]

        Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)
        Bskip1 = self.part2_branchB1(B)
        Cskip1 = self.part2_branchC1(A)

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out


class TriUnet36tC(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36tC, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoderC(self.layer[4])
        self.part3_branchC = TriUnet_decoderC(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)

        com_centerB = torch.cat([Abridge, Bbridge], dim=1)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, com_centerB)
        com_centerC = torch.cat([Abridge, Cbridge], dim=1)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, com_centerC)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t1(nn.Module):     #对比实验：夜光光学融合，夜光分割输出
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t1, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(5, 1, kernel_size=1, stride=1)

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)  # 分割任务多通道输出
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t2(nn.Module):     #夜光SAR融合，回归输出
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t2, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchB = TriUnet_encoder(channel_branchB, ori_filter=self.layer[0])
        self.part2_branchA1 = Bottleneck(channel_branchA, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchA2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchA3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchA4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchA5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchA6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchA7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchA8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchA9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]
        C = x[:, 0:1, :, :]

        Bskip1, Bskip2, Bskip3, Bskip4, Bbridge = self.part1_branchB(B)

        Askip1 = self.part2_branchA1(A)
        Cskip1 = self.part2_branchC1(C)

        RAskip1 = self.part2_branchA2(Askip1, Cskip1)
        RCskip1 = self.part2_branchC2(Cskip1, Askip1)

        Askip2 = self.part2_branchA3(self.maxpool(RAskip1))
        Cskip2 = self.part2_branchC3(self.maxpool(RCskip1))

        RAskip2 = self.part2_branchA4(Askip2, Cskip2)
        RCskip2 = self.part2_branchC4(Cskip2, Bskip2)

        Askip3 = self.part2_branchA5(self.maxpool(RAskip2))
        Cskip3 = self.part2_branchC5(self.maxpool(RCskip2))

        RAskip3 = self.part2_branchA6(Askip3, Cskip3)
        RCskip3 = self.part2_branchC6(Cskip3, Askip3)

        Askip4 = self.part2_branchA7(self.maxpool(RAskip3))
        Cskip4 = self.part2_branchC7(self.maxpool(RCskip3))

        RAskip4= self.part2_branchA8(Askip4, Cskip4)
        RCskip4= self.part2_branchC8(Cskip4, Askip4)

        Abridge = self.part2_branchA9(RAskip4)
        Cbridge = self.part2_branchC9(RCskip4)

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Bbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36t3(nn.Module):     #夜光SAR融合，夜光分割
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36t3, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=self.layer[0])
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        # 对二分类分割任务，输出也可以是单通道的，通过sigmoid限制其输出在0 1之间
        # self.finalLayerB = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        # self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        # self.sigmoid = nn.Sigmoid()

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        # Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class TriUnet36(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    """
    2023/12/23   TriUnet_decoder中使用到的decoder_conv加上了bn层（以前只有conv和relu）

    """
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(TriUnet36, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = TriUnet_encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = TriUnet_fuse22(self.layer[0])
        self.part2_branchC2 = TriUnet_fuse22(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = TriUnet_fuse22(self.layer[1])
        self.part2_branchC4 = TriUnet_fuse22(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = TriUnet_fuse22(self.layer[2])
        self.part2_branchC6 = TriUnet_fuse22(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = TriUnet_fuse22(self.layer[3])
        self.part2_branchC8 = TriUnet_fuse22(self.layer[3])

        self.part2_branchB9 = TriUnet_encoder35(ori_filter=32)
        self.part2_branchC9 = TriUnet_encoder35(ori_filter=32)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = TriUnet_decoder(self.layer[4])
        self.part3_branchB = TriUnet_decoder(self.layer[4])
        self.part3_branchC = TriUnet_decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        # self.finalLayerB = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        # self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        # 对二分类分割任务，输出也可以是单通道的，通过sigmoid限制其输出在0 1之间
        self.finalLayerB = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

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

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        # Bout = self.finalLayerB(Bout)  # 分割任务多通道输出
        Bout = self.sigmoid(self.finalLayerB(Bout))  # 分割任务单通道输出

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out
