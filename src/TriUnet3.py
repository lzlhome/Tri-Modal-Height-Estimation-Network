import torch
import torch.nn as nn
from src.BasicBlock import ChannelAttention, SpatialAttention, DirectionalASPP, TransposeConv


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
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        return out

class encoder(nn.Module):
    def __init__(self, inchannels, ori_filter=32):
        super(encoder, self).__init__()

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

class encoder_layer(nn.Module):
    def __init__(self, ori_filter=32):
        super(encoder_layer, self).__init__()
        self.encoder = Bottleneck(ori_filter*8, ori_filter*16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, skip4):
        bridge = self.encoder(self.maxpool(skip4))  # 16 16 256
        return bridge

class decoder(nn.Module):
    def __init__(self, num_feature):
        super(decoder, self).__init__()
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


class DSEunit(nn.Module):
    def __init__(self, num_feature):
        super(DSEunit, self).__init__()
        self.RefinedD0 = DirectionalASPP(num_feature * 2, num_feature * 2)
        self.RefinedD45 = DirectionalASPP(num_feature * 2, num_feature * 2)
        self.RefinedD90 = DirectionalASPP(num_feature * 2, num_feature * 2)
        self.RefinedD135 = DirectionalASPP(num_feature * 2, num_feature * 2)

        self.conv = nn.Conv2d(num_feature * 2, num_feature * 2, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, F):
        a = self.RefinedD0(F)
        b = self.RefinedD45(F)
        c = self.RefinedD90(F)
        d = self.RefinedD135(F)
        Q = torch.cat([a, b, c, d], dim=1)

        H = self.relu(self.conv(Q))
        out = H + F

        return out

class TACunit(nn.Module):
    def __init__(self, num_feature):
        super(TACunit, self).__init__()
        self.calibratedChannelAttention = ChannelAttention(num_feature)
        self.calibratedSpatialAttention = SpatialAttention(num_feature)

    def forward(self, M, T):
        a = self.calibratedChannelAttention(M)
        b = self.calibratedSpatialAttention(M)
        out = a * T + b * T

        return out

class CAE(nn.Module):
    def __init__(self, num_feature):
        super(CAE, self).__init__()
        self.DSE = DSEunit(num_feature)

        self.squeeze = nn.Conv2d(num_feature * 2, num_feature, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        self.TAC = TACunit(num_feature)

    def forward(self, M, N):
        F = torch.cat([M, N], dim=1)
        E = self.DSE(F)
        T = self.relu(self.squeeze(E))
        out = self.TAC(T, M)

        return out


class M4(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(M4, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = CAE(self.layer[0])
        self.part2_branchC2 = CAE(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = CAE(self.layer[1])
        self.part2_branchC4 = CAE(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = CAE(self.layer[2])
        self.part2_branchC6 = CAE(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = CAE(self.layer[3])
        self.part2_branchC8 = CAE(self.layer[3])

        self.part2_branchB9 = encoder_layer(ori_filter=self.layer[0])
        self.part2_branchC9 = encoder_layer(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = decoder(self.layer[4])
        self.part3_branchB = decoder(self.layer[4])
        self.part3_branchC = decoder(self.layer[4])

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
        Bout = self.part3_branchB(Bskip1, Bskip2, Bskip3, Bskip4, Bbridge)
        Cout = self.part3_branchC(Cskip1, Cskip2, Cskip3, Cskip4, Cbridge)

        Aout = self.finalLayerA(Aout)
        Cout = self.finalLayerC(Cout)
        Bout = self.finalLayerB(Bout)

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class M1(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(M1, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchB2 = CAE(self.layer[0])
        self.part2_branchC2 = CAE(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = CAE(self.layer[1])
        self.part2_branchC4 = CAE(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = CAE(self.layer[2])
        self.part2_branchC6 = CAE(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = CAE(self.layer[3])
        self.part2_branchC8 = CAE(self.layer[3])

        self.part2_branchB9 = encoder_layer(ori_filter=self.layer[0])
        self.part2_branchC9 = encoder_layer(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = decoder(self.layer[4])
        self.part3_branchB = decoder(self.layer[4])
        self.part3_branchC = decoder(self.layer[4])

        self.finalLayerA = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)
        self.finalLayerC = nn.Conv2d(self.layer[0], 1, kernel_size=1, stride=1)

        # 对分割任务，输出通道数通常等于类别数，通过softmax进行归一化确保每个像素被分配到一个类别
        self.finalLayerB = nn.Conv2d(self.layer[0], 2, kernel_size=1, stride=1)
        self.finalLayer = nn.Conv2d(4, 1, kernel_size=1, stride=1)

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

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class M2(nn.Module):     #第一、二、三、四层光学和夜光融合  BBC 加MK
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(M2, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = CAE(self.layer[0])
        self.part2_branchC2 = CAE(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = CAE(self.layer[1])
        self.part2_branchC4 = CAE(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = CAE(self.layer[2])
        self.part2_branchC6 = CAE(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = CAE(self.layer[3])
        self.part2_branchC8 = CAE(self.layer[3])

        self.part2_branchB9 = encoder_layer(ori_filter=self.layer[0])
        self.part2_branchC9 = encoder_layer(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = decoder(self.layer[4])
        self.part3_branchB = decoder(self.layer[4])
        self.part3_branchC = decoder(self.layer[4])

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

        out = self.relu(self.finalLayer(torch.cat([Aout, Bout, Cout], dim=1)))

        return Aout, Bout, Cout, out

class M3(nn.Module):     #对比实验：夜光光学融合，夜光分割输出
    def __init__(self, channel_branchA, channel_branchB, channel_branchC, ori_filter=32):
        super(M3, self).__init__()
        self.layer = [ori_filter, ori_filter*2, ori_filter*4, ori_filter*8, ori_filter*16]

        self.part1_branchA = encoder(channel_branchA, ori_filter=self.layer[0])
        self.part2_branchB1 = Bottleneck(channel_branchB, self.layer[0])
        self.part2_branchC1 = Bottleneck(channel_branchC, self.layer[0])
        self.part2_branchB2 = CAE(self.layer[0])
        self.part2_branchC2 = CAE(self.layer[0])

        self.part2_branchB3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchC3 = Bottleneck(self.layer[0], self.layer[1])
        self.part2_branchB4 = CAE(self.layer[1])
        self.part2_branchC4 = CAE(self.layer[1])

        self.part2_branchB5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchC5 = Bottleneck(self.layer[1], self.layer[2])
        self.part2_branchB6 = CAE(self.layer[2])
        self.part2_branchC6 = CAE(self.layer[2])

        self.part2_branchB7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchC7 = Bottleneck(self.layer[2], self.layer[3])
        self.part2_branchB8 = CAE(self.layer[3])
        self.part2_branchC8 = CAE(self.layer[3])

        self.part2_branchB9 = encoder_layer(ori_filter=self.layer[0])
        self.part2_branchC9 = encoder_layer(ori_filter=self.layer[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3_branchA = decoder(self.layer[4])
        self.part3_branchB = decoder(self.layer[4])
        self.part3_branchC = decoder(self.layer[4])

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

