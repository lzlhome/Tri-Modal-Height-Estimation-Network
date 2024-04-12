import numpy as np
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from src.BasicBlock import se_block, cbam_block, eca_block, TransposeConv, Upsample, MultiScale_Upconv, \
    ContextAggregation


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


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# class MSK(nn.Module):
#     def __init__(self, channel):
#         super(MSK, self).__init__()
#
#         self.attention = eca_block(channel * 2)
#         self.changechannel = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1)
#
#     def forward(self, x, x_other):
#         input = torch.cat([x, x_other], dim=1)
#         attentioned = self.attention(input)
#         V = self.changechannel(attentioned)
#         return x + V


class MSK(nn.Module):
    def __init__(self, channel, reduction_ratio=4, L=32):
        super(MSK, self).__init__()

        self.SPLIT_conv3 = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, padding=1)
        self.SPLIT_conv5 = nn.Conv2d(channel * 2, channel * 2, kernel_size=5, padding=2)
        self.SPLIT_conv1 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1)

        self.F_fc = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel * 2)
        )
        # self.activation = F.softmax()
        self.conv_V = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1)

    def forward(self, x, x_other):
        input = torch.cat([x, x_other], dim=1)
        S1 = self.SPLIT_conv3(input)
        S2 = self.SPLIT_conv5(input)
        S = self.SPLIT_conv1(S1 + S2)
        Fgp = torch.mean(S, dim=(2, 3), keepdim=True)
        attention = self.F_fc(Fgp)
        attention = F.softmax(attention, dim=0)
        Z = S1 * attention.expand_as(S1) + S2 * attention.expand_as(S2)
        V = self.conv_V(Z)
        return x + V


class BiUnet_encoder(nn.Module):
    def __init__(self, inchannels):
        super(BiUnet_encoder, self).__init__()

        self.encoder1 = Bottleneck(inchannels, 16)
        self.encoder2 = Bottleneck(16, 32)
        self.encoder3 = Bottleneck(32, 64)
        self.encoder4 = Bottleneck(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip1 = self.encoder1(x)  # 256 256 16
        skip2 = self.encoder2(self.maxpool(skip1))  # 128 128 32
        skip3 = self.encoder3(self.maxpool(skip2))  # 64 64 64
        skip4 = self.encoder4(self.maxpool(skip3))  # 32 32 128

        return skip1, skip2, skip3, skip4


class BiUnet_bridge(nn.Module):
    def __init__(self, num_feature):
        super(BiUnet_bridge, self).__init__()

        self.featurefuse = MSK(num_feature)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = Bottleneck(num_feature, 256)

    def forward(self, skip4, skip4_other):
        out = self.featurefuse(skip4, skip4_other)
        out = self.maxpool(out)
        bridge = self.encoder5(out)

        return bridge  # 这是middle feature，之后开始上采样


class BiUnet_decoder(nn.Module):
    def __init__(self, num_feature):
        super(BiUnet_decoder, self).__init__()

        self.featurefuse = MSK(num_feature)

        # encoder
        self.decoder4 = TransposeConv(256, 128)
        self.decoder_conv4 = decoder_conv(256, 128)

        self.decoder3 = TransposeConv(128, 64)
        self.decoder_conv3 = decoder_conv(128, 64)

        self.decoder2 = TransposeConv(64, 32)
        self.decoder_conv2 = decoder_conv(64, 32)

        self.decoder1 = TransposeConv(32, 16)
        self.decoder_conv1 = decoder_conv(32, 16)

    def forward(self, skip1, skip2, skip3, skip4, bridge, bridge_other):
        out = self.featurefuse(bridge, bridge_other)
        out = self.decoder4(out)
        out = self.decoder_conv4(torch.cat([out, skip4], dim=1))

        out = self.decoder3(out)
        out = self.decoder_conv3(torch.cat([out, skip3], dim=1))

        out = self.decoder2(out)
        out = self.decoder_conv2(torch.cat([out, skip2], dim=1))

        out = self.decoder1(out)
        out = self.decoder_conv1(torch.cat([out, skip1], dim=1))

        return out


class BiUnet(nn.Module):  # 包含BiUnet_encoder和BiUnet_bridge的完整的模型
    def __init__(self, channel_branchA, channel_branchB):
        super(BiUnet, self).__init__()

        self.part1_branchA = BiUnet_encoder(channel_branchA)
        self.part1_branchB = BiUnet_encoder(channel_branchB)

        self.part2_branchA = BiUnet_bridge(128)
        self.part2_branchB = BiUnet_bridge(128)

        self.part3_branchA = BiUnet_decoder(256)
        self.part3_branchB = BiUnet_decoder(256)

        self.finalLayerA = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        self.finalLayerB = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        # self.finalLayer = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 4:6, :, :]
        B = x[:, 0:4, :, :]
        Askip1, Askip2, Askip3, Askip4 = self.part1_branchA(A)
        Bskip1, Bskip2, Bskip3, Bskip4 = self.part1_branchB(B)

        Abrideg = self.part2_branchA(Askip4, Bskip4)
        Bbrideg = self.part2_branchB(Bskip4, Askip4)

        Aout = self.part3_branchA(Askip1, Askip2, Askip3, Askip4, Abrideg, Bbrideg)
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbrideg, Abrideg)

        Aout = self.finalLayerA(Aout)
        Bout = self.finalLayerB(Bout)

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout], dim=1)))

        y_reg = out
        y_seg = Bout

        return y_reg, y_seg
