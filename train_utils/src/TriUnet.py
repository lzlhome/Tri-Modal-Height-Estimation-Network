import numpy as np
import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch.nn.functional as F
from src.BasicBlock import MKBlock, cbam_block, eca_block, TransposeConv, Upsample, MultiScale_Upconv, \
    ContextAggregation


# class filternum_multiresBlock(nn.Module):  # MultiResUNet的decoder每层filter num
#     def __init__(self):
#         super(filternum_multiresBlock, self).__init__()
#         self.alpha = 1.67
#
#     def forward(self, filters):
#         a = int(filters*self.alpha*0.167)
#         b = int(filters*self.alpha*0.333)
#         c = int(filters*self.alpha* 0.5)
#         all = a + b + c
#         return [a, b, c, all]

        # self.out_channle = [int(self.filters*self.alpha*0.167), int(self.filters*self.alpha*0.333), int(self.filters*self.alpha* 0.5)]

def filternum_multiresBlock(ref_filters):
    alpha = 1.67
    a = int(ref_filters * alpha * 0.167)
    b = int(ref_filters * alpha * 0.333)
    c = int(ref_filters * alpha * 0.5)
    all = a + b + c
    return [a, b, c, all]


class multiresBottleneck(nn.Module):  # 输出通道加倍
    def __init__(self, in_channels, ref_filters):    # ref_filters为同层Unet的filter
        super(multiresBottleneck, self).__init__()
        self.out_channle = filternum_multiresBlock(ref_filters)

        self.iden_conv = nn.Conv2d(in_channels, self.out_channle[3], kernel_size=1, stride=1)
        self.iden_bn = nn.BatchNorm2d(self.out_channle[3])

        self.Bottleneck_conv1 = nn.Conv2d(in_channels, self.out_channle[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channle[0])

        self.Bottleneck_conv2 = nn.Conv2d(self.out_channle[0], self.out_channle[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channle[1])

        self.Bottleneck_conv3 = nn.Conv2d(self.out_channle[1], self.out_channle[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.out_channle[2])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.iden_bn(self.iden_conv(x))
        out1 = self.relu(self.bn1(self.Bottleneck_conv1(x)))
        out2 = self.relu(self.bn2(self.Bottleneck_conv2(out1)))
        out3 = self.bn3(self.Bottleneck_conv3(out2))

        out = torch.concat([out1, out2, out3], dim=1)
        out = self.relu(out+identity)

        return out

class ResPath(nn.Module):  # 输出通道加倍
    def __init__(self, in_channels, out_channels, respath_length):   #respath_length = 1 2 3 4
        super(ResPath, self).__init__()
        self.respath_length = respath_length

        # ResPath4
        self.iden_conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.iden_bn4 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)

        if respath_length<4:  # ResPath3
            self.iden_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
            self.iden_bn3 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(out_channels)

        if respath_length<3:  # ResPath2
            self.iden_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
            self.iden_bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        if respath_length<2:  # ResPath1
            self.iden_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
            self.iden_bn1 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity4 = self.iden_bn4(self.iden_conv4(x))
        out4 = self.bn4(self.conv4(x))
        out = self.relu(identity4+out4)

        if self.respath_length<4:  # ResPath3
            identity3 = self.iden_bn3(self.iden_conv3(out))
            out3 = self.bn3(self.conv3(out))
            out = self.relu(identity3+out3)

        if self.respath_length<3:  # ResPath2
            identity2 = self.iden_bn2(self.iden_conv2(out))
            out2 = self.bn2(self.conv2(out))
            out = self.relu(identity2+out2)

        if self.respath_length<2:  # ResPath1
            identity1 = self.iden_bn1(self.iden_conv1(out))
            out1 = self.bn1(self.conv1(out))
            out = self.relu(identity1+out1)

        return out


class MultiResUNet(nn.Module):
    def __init__(self, in_channel, ori_filter, num_classes=1):  #ori_filter是Unet的第一层的输出通道  16/32/64···
        super(MultiResUNet, self).__init__()

        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.encoder_block1 = multiresBottleneck(in_channel, self.layer[0])
        self.encoder_block2 = multiresBottleneck(filternum_multiresBlock(self.layer[0])[3], self.layer[1])
        self.encoder_block3 = multiresBottleneck(filternum_multiresBlock(self.layer[1])[3], self.layer[2])
        self.encoder_block4 = multiresBottleneck(filternum_multiresBlock(self.layer[2])[3], self.layer[3])
        self.encoder_block5 = multiresBottleneck(filternum_multiresBlock(self.layer[3])[3], self.layer[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = TransposeConv(filternum_multiresBlock(self.layer[4])[3], self.layer[3])
        self.respath4 = ResPath(filternum_multiresBlock(self.layer[3])[3], self.layer[3], 4)
        self.decoder_block6 = multiresBottleneck(self.layer[3] * 2, self.layer[3])

        self.upconv3 = TransposeConv(filternum_multiresBlock(self.layer[3])[3], self.layer[2])
        self.respath3 = ResPath(filternum_multiresBlock(self.layer[2])[3], self.layer[2], 3)
        self.decoder_block7 = multiresBottleneck(self.layer[2] * 2, self.layer[2])

        self.upconv2 = TransposeConv(filternum_multiresBlock(self.layer[2])[3], self.layer[1])
        self.respath2 = ResPath(filternum_multiresBlock(self.layer[1])[3], self.layer[1], 2)
        self.decoder_block8 = multiresBottleneck(self.layer[1] * 2, self.layer[1])

        self.upconv1 = TransposeConv(filternum_multiresBlock(self.layer[1])[3], self.layer[0])
        self.respath1 = ResPath(filternum_multiresBlock(self.layer[0])[3], self.layer[0], 1)
        self.decoder_block9 = multiresBottleneck(self.layer[0] * 2, self.layer[0])

        self.final_layer = nn.Conv2d(self.layer[0], num_classes, kernel_size=1, stride=1)
        self.final_bn = nn.BatchNorm2d(num_classes)
        if num_classes == 1:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        skip1 = self.encoder_block1(x)
        skip2 = self.encoder_block2(self.maxpool(skip1))
        skip3 = self.encoder_block3(self.maxpool(skip2))
        skip4 = self.encoder_block4(self.maxpool(skip3))
        bridge = self.encoder_block5(self.maxpool(skip4))

        up4 = self.decoder_block6(torch.concat([self.upconv4(bridge), self.respath4(skip4)], dim=1))
        up3 = self.decoder_block7(torch.concat([self.upconv3(up4), self.respath3(skip3)], dim=1))
        up2 = self.decoder_block8(torch.concat([self.upconv2(up3), self.respath2(skip2)], dim=1))
        up1 = self.decoder_block9(torch.concat([self.upconv1(up2), self.respath1(skip1)], dim=1))

        out = self.activation(self.final_bn(self.final_layer(up1)))

        return out


class MultiResUNet_encoder(nn.Module):
    def __init__(self, in_channel, ori_filter=32):  #ori_filter是Unet的第一层的输出通道  16/32/64···
        super(MultiResUNet_encoder, self).__init__()

        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.encoder_block1 = multiresBottleneck(in_channel, self.layer[0])
        self.encoder_block2 = multiresBottleneck(filternum_multiresBlock(self.layer[0])[3], self.layer[1])
        self.encoder_block3 = multiresBottleneck(filternum_multiresBlock(self.layer[1])[3], self.layer[2])
        self.encoder_block4 = multiresBottleneck(filternum_multiresBlock(self.layer[2])[3], self.layer[3])
        self.encoder_block5 = multiresBottleneck(filternum_multiresBlock(self.layer[3])[3], self.layer[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip1 = self.encoder_block1(x)
        skip2 = self.encoder_block2(self.maxpool(skip1))
        skip3 = self.encoder_block3(self.maxpool(skip2))
        skip4 = self.encoder_block4(self.maxpool(skip3))
        bridge = self.encoder_block5(self.maxpool(skip4))

        return [skip1, skip2, skip3, skip4, bridge]


class MultiResUNet_decoder(nn.Module):
    def __init__(self, ori_filter, num_classes=1):  #ori_filter是Unet的第一层的输出通道  16/32/64···
        super(MultiResUNet_decoder, self).__init__()

        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.upconv4 = TransposeConv(filternum_multiresBlock(self.layer[4])[3], self.layer[3])
        self.respath4 = ResPath(filternum_multiresBlock(self.layer[3])[3], self.layer[3], 4)
        self.decoder_block6 = multiresBottleneck(self.layer[3] * 2, self.layer[3])

        self.upconv3 = TransposeConv(filternum_multiresBlock(self.layer[3])[3], self.layer[2])
        self.respath3 = ResPath(filternum_multiresBlock(self.layer[2])[3], self.layer[2], 3)
        self.decoder_block7 = multiresBottleneck(self.layer[2] * 2, self.layer[2])

        self.upconv2 = TransposeConv(filternum_multiresBlock(self.layer[2])[3], self.layer[1])
        self.respath2 = ResPath(filternum_multiresBlock(self.layer[1])[3], self.layer[1], 2)
        self.decoder_block8 = multiresBottleneck(self.layer[1] * 2, self.layer[1])

        self.upconv1 = TransposeConv(filternum_multiresBlock(self.layer[1])[3], self.layer[0])
        self.respath1 = ResPath(filternum_multiresBlock(self.layer[0])[3], self.layer[0], 1)
        self.decoder_block9 = multiresBottleneck(self.layer[0] * 2, self.layer[0])

        self.final_layer = nn.Conv2d(filternum_multiresBlock(self.layer[0])[3], num_classes, kernel_size=1, stride=1)
        self.final_bn = nn.BatchNorm2d(num_classes)
        if num_classes == 1:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, skip1, skip2, skip3, skip4, bridge):   #skip需要respath, up需要upsampling

        up4 = self.decoder_block6(torch.concat([self.upconv4(bridge), self.respath4(skip4)], dim=1))
        up3 = self.decoder_block7(torch.concat([self.upconv3(up4), self.respath3(skip3)], dim=1))
        up2 = self.decoder_block8(torch.concat([self.upconv2(up3), self.respath2(skip2)], dim=1))
        up1 = self.decoder_block9(torch.concat([self.upconv1(up2), self.respath1(skip1)], dim=1))

        out = self.final_layer(up1)
        out = self.final_bn(out)
        out = self.activation(out)

        return out


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


class MultiResUNet_T36(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(MultiResUNet_T36, self).__init__()

        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = MultiResUNet_encoder(channel_branchA, self.layer[0])

        self.Bencoder_block1 = multiresBottleneck(channel_branchB, self.layer[0])
        self.Bencoder_block2 = multiresBottleneck(filternum_multiresBlock(self.layer[0])[3], self.layer[1])
        self.Bencoder_block3 = multiresBottleneck(filternum_multiresBlock(self.layer[1])[3], self.layer[2])
        self.Bencoder_block4 = multiresBottleneck(filternum_multiresBlock(self.layer[2])[3], self.layer[3])
        self.Bencoder_block5 = multiresBottleneck(filternum_multiresBlock(self.layer[3])[3], self.layer[4])

        self.Cencoder_block1 = multiresBottleneck(channel_branchC, self.layer[0])
        self.Cencoder_block2 = multiresBottleneck(filternum_multiresBlock(self.layer[0])[3], self.layer[1])
        self.Cencoder_block3 = multiresBottleneck(filternum_multiresBlock(self.layer[1])[3], self.layer[2])
        self.Cencoder_block4 = multiresBottleneck(filternum_multiresBlock(self.layer[2])[3], self.layer[3])
        self.Cencoder_block5 = multiresBottleneck(filternum_multiresBlock(self.layer[3])[3], self.layer[4])

        self.Bfuse1 = TriUnet_fuse22(filternum_multiresBlock(self.layer[0])[3])
        self.Cfuse1 = TriUnet_fuse22(filternum_multiresBlock(self.layer[0])[3])

        self.Bfuse2 = TriUnet_fuse22(filternum_multiresBlock(self.layer[1])[3])
        self.Cfuse2 = TriUnet_fuse22(filternum_multiresBlock(self.layer[1])[3])

        self.Bfuse3 = TriUnet_fuse22(filternum_multiresBlock(self.layer[2])[3])
        self.Cfuse3 = TriUnet_fuse22(filternum_multiresBlock(self.layer[2])[3])

        self.Bfuse4 = TriUnet_fuse22(filternum_multiresBlock(self.layer[3])[3])
        self.Cfuse4 = TriUnet_fuse22(filternum_multiresBlock(self.layer[3])[3])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Adecoder = MultiResUNet_decoder(ori_filter)
        self.Bdecoder = MultiResUNet_decoder(ori_filter, num_classes=2)
        self.Cdecoder = MultiResUNet_decoder(ori_filter)

        self.final_Layer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]   #光学
        C = x[:, 0:1, :, :]

        Askip1, Askip2, Askip3, Askip4, Abridge = self.part1_branchA(A)

        Bskip1 = self.Bencoder_block1(B)
        Cskip1 = self.Cencoder_block1(C)
        RBskip1 = self.Bfuse1(Bskip1, Cskip1)
        RCskip1 = self.Cfuse1(Cskip1, Bskip1)

        Bskip2 = self.Bencoder_block2(self.maxpool(RBskip1))
        Cskip2 = self.Cencoder_block2(self.maxpool(RCskip1))
        RBskip2 = self.Bfuse2(Bskip2, Cskip2)
        RCskip2 = self.Cfuse2(Cskip2, Bskip2)

        Bskip3 = self.Bencoder_block3(self.maxpool(RBskip2))
        Cskip3 = self.Cencoder_block3(self.maxpool(RCskip2))
        RBskip3 = self.Bfuse3(Bskip3, Cskip3)
        RCskip3 = self.Cfuse3(Cskip3, Bskip3)

        Bskip4 = self.Bencoder_block4(self.maxpool(RBskip3))
        Cskip4 = self.Cencoder_block4(self.maxpool(RCskip3))
        RBskip4= self.Bfuse4(Bskip4, Cskip4)
        RCskip4= self.Cfuse4(Cskip4, Bskip4)

        Cbridge = self.Bencoder_block5(self.maxpool(RBskip4))
        Bbridge = self.Cencoder_block5(self.maxpool(RCskip4))

        Aout = self.Adecoder(Askip1, Askip2, Askip3, Askip4, Bbridge)
        Bout = self.Bdecoder(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.Cdecoder(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        out = self.relu(self.final_Layer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out






