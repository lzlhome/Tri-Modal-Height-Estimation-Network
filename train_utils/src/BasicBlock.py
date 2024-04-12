import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class ChannelAttention(nn.Module):  # channel-wise
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


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


# class DANet(nn.Module):
#     def __init__(self, channel, ratio=8, kernel_size=7):
#         super(DANet, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio=ratio)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)
#
#
#     def forward(self, x):
#         x1 = x * self.channelattention(x)
#         x2 = x * self.spatialattention(x)
#         x = x1 * x2
#
#         return x

class DANet(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(DANet, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.conv = nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = x * self.channelattention(x)
        x2 = x * self.spatialattention(x)
        x = self.conv(torch.cat([x1, x2], dim=1))

        return x

class GDFM(nn.Module):
     def __init__(self, channel, H, W, ratio=64):
        super(GDFM, self).__init__()

        self.convaK = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convaV = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convaQ = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convbK = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convbV = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convbQ = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.af1 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.af2 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.af3 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.af4 = nn.Linear(channel*H*W//ratio, channel*H*W, bias=True)

        self.bf1 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.bf2 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.bf3 = nn.Linear(channel*H*W, channel*H*W//ratio, bias=True)
        self.bf4 = nn.Linear(channel*H*W//ratio, channel*H*W, bias=True)

     def forward(self, x1, x2):

        [B,C,H,W] = x1.size()

        aK = self.convaK(x1).view(B, -1)    #[B,C,H,W] -- [B,C,H,W] -- [B,C*H*W]
        aV = self.convaV(x1).view(B, -1)
        aQ = self.convaQ(x1).view(B, -1)
        bK = self.convbK(x2).view(B, -1)
        bV = self.convbV(x2).view(B, -1)
        bQ = self.convbQ(x2).view(B, -1)

        faK = self.af1(aK).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N] -- [B,N,C]
        faV = self.af2(aV).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N]
        fbQ = self.af3(bQ).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N] -- [B,N,C]
        fbK = self.bf1(bK).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N] -- [B,N,C]
        fbV = self.bf2(bV).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N]
        faQ = self.bf3(aQ).view(B, C, -1).permute(0, 2, 1)    #[B,C*H*W] -- [B,C*N] -- [B,C,N] -- [B,N,C]

        KV1 = F.softmax(torch.matmul(faK, faV))  # [B,N,N]
        KV2 = F.softmax(torch.matmul(fbK, fbV))  # [B,N,N]

        out1 = self.af4(torch.matmul(KV1, fbQ).view(B, -1)).view(B, C, H, W)  # [B,N,C] -- [B,N*C] -- [B, C*H*W] -- [B,C,H,W]
        out2 = self.bf4(torch.matmul(KV2, faQ).view(B, -1)).view(B, C, H, W)

        return torch.concat([x1 + out1, x2 + out2], dim=1)   #[B,2C,H,W]

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

'''
ASPP模块：多尺度特征提取，后常接差值上采样
'''

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[12, 24, 36]):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[0], padding=dilations[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[1], padding=dilations[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[2], padding=dilations[2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv1x1_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 1x1卷积分支
        branch_1x1 = self.conv1x1(x)

        # 3x3卷积分支
        branch_3x3_1 = self.conv3x3_1(x)
        branch_3x3_2 = self.conv3x3_2(x)
        branch_3x3_3 = self.conv3x3_3(x)

        # 平均池化分支
        branch_pool = F.adaptive_avg_pool2d(x, 1)
        branch_pool = self.conv1x1_pool(branch_pool)
        branch_pool = F.interpolate(branch_pool, size=x.size()[2:], mode='bilinear', align_corners=False)

        # 拼接分支结果
        out = self.project(torch.cat([branch_1x1, branch_3x3_1, branch_3x3_2, branch_3x3_3, branch_pool], dim=1))

        return out


class InceptionBlock(nn.Module):    #输入输出通道数一样
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        # 1x1 Convolution to reduce channels
        self.conv1x1_reduce = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_reduce = nn.BatchNorm2d(64)

        # 3x3 Convolution
        self.conv3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.conv3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(128)

        # 5x5 Convolution
        self.conv5_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(32)

        # Max Pooling followed by 1x1 Convolution to reduce channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convmaxpool = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.bnmaxpool = nn.BatchNorm2d(32)



    def forward(self, x):
        # 1x1 Convolution to reduce channels
        out1x1 = self.conv1x1_reduce(x)
        out1x1 = self.bn1x1_reduce(out1x1)
        out1x1 = nn.ReLU(inplace=True)(out1x1)

        # 3x3 Convolution
        out3x3 = self.conv3_1x1(x)
        out3x3 = self.conv3x3(out3x3)
        out3x3 = self.bn3x3(out3x3)
        out3x3 = nn.ReLU(inplace=True)(out3x3)

        # 5x5 Convolution
        out5x5 = self.conv5_1x1(x)
        out5x5 = self.conv5x5(out5x5)
        out5x5 = self.bn5x5(out5x5)
        out5x5 = nn.ReLU(inplace=True)(out5x5)

        # Max Pooling followed by 1x1 Convolution to reduce channels
        out_pool = self.maxpool(x)
        out_pool = self.convmaxpool(out_pool)
        out_pool = self.bnmaxpool(out_pool)
        out_pool = nn.ReLU(inplace=True)(out_pool)

        # Concatenate the outputs along the channel dimension
        out = torch.cat([out1x1, out3x3, out5x5, out_pool], dim=1)

        return out


class MKBlock(nn.Module):    #输入输出通道数一样
    def __init__(self, in_channels):
        super(MKBlock, self).__init__()
        # 1x1 Convolution to reduce channels
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1x1 = nn.BatchNorm2d(in_channels)
        # 3x3 Convolution
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(in_channels)
        # 5x5 Convolution
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        # 1x1 Convolution to reduce channels
        out1x1 = nn.ReLU(inplace=True)(self.bn1x1(self.conv1x1(x)))
        # 3x3 Convolution
        out3x3 = nn.ReLU(inplace=True)(self.bn3x3(self.conv3x3(x)))
        # 5x5 Convolution
        out5x5 = nn.ReLU(inplace=True)(self.bn5x5(self.conv5x5(x)))
        # Concatenate the outputs along the channel dimension
        out = out1x1 + out3x3 + out5x5
        return out

class sMKBlock(nn.Module):    #输入输出通道数一样
    def __init__(self, in_channels):
        super(sMKBlock, self).__init__()
        # 3x3 Convolution
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(in_channels)
        # 5x5 Convolution
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        # 3x3 Convolution
        out3x3 = nn.ReLU(inplace=True)(self.bn3x3(self.conv3x3(x)))
        # 5x5 Convolution
        out5x5 = nn.ReLU(inplace=True)(self.bn5x5(self.conv5x5(x)))
        # Concatenate the outputs along the channel dimension
        out = out3x3 + out5x5
        return out

class lMKBlock(nn.Module):    #输入输出通道数一样
    def __init__(self, in_channels):
        super(lMKBlock, self).__init__()
        # 3x3 Convolution
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(in_channels)
        # 5x5 Convolution
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(in_channels)
        # 7x7 Convolution to reduce channels
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.bn7x7 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        # 3x3 Convolution
        out3x3 = nn.ReLU(inplace=True)(self.bn3x3(self.conv3x3(x)))
        # 5x5 Convolution
        out5x5 = nn.ReLU(inplace=True)(self.bn5x5(self.conv5x5(x)))
        # 7x7 Convolution to reduce channels
        out7x7 = nn.ReLU(inplace=True)(self.bn7x7(self.conv7x7(x)))
        # Concatenate the outputs along the channel dimension
        out = out7x7 + out3x3 + out5x5
        return out


class SepBlock(nn.Module):
    def __init__(self, in_channels):
        super(SepBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        out = self.bn(out)
        out += residual
        return out

'''
上采样模块
'''

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class TransposeConv(nn.Module):
    '''
    转置卷积：默认kernel=2, stride=2, [b,input_dim,h,w] -> [b,output_dim,2h,2w]
    '''
    def __init__(self, input_dim, output_dim, kernel=2, stride=2):
        super(TransposeConv, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)
    def forward(self, x):
        out = self.upsample(x)
        return out


class Upsample(nn.Module):
    '''
    interpolate+3*3 conv代替transposed convolution——减少伪影,默认scale_factor=2，[b,input_dim,h,w] -> [b,output_dim,2h,2w]
    '''
    def __init__(self, input_dim, output_dim, scale_factor=2):
        super(Upsample, self).__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MultiScale_Upconv(nn.Module):
    '''
    ref: Height estimation from single aerial images using a deep convolutional encoder-decoder network
    多核卷积+线性内插,默认卷积核尺寸：3*3 3*2 2*3 2*2
    [b,input_dim,h,w] -> [b,output_dim,2h,2w]
    '''
    def __init__(self, input_dim, output_dim, kernel_size=[(1, 1), (3, 3), (5, 5)]):
        super(MultiScale_Upconv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size[0], padding=((kernel_size[0][0] - 1) // 2, (kernel_size[0][1] - 1) // 2)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size[1], padding=((kernel_size[1][0] - 1) // 2, (kernel_size[1][1] - 1) // 2)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size[2], padding=((kernel_size[2][0] - 1) // 2, (kernel_size[2][1] - 1) // 2)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU())

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size[3], padding=((kernel_size[3][0] - 1) // 2, (kernel_size[3][1] - 1) // 2)),
        #     nn.BatchNorm2d(output_dim),
        #     nn.ReLU())

        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        # out4 = self.conv4(x)
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=False)
        # out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=False)
        # out = self.relu(out1 + out2 + out3 + out4)
        out = self.relu(out1 + out2 + out3)

        return out


class ContextAggregation(nn.Module):
    '''
    ref: Height estimation from single aerial images using a deep convolutional encoder-decoder network
    对要进行skip connection的低层次特征，空洞卷积，提取信息
    [b,input_dim,h,w] -> [b,output_dim,h,w]
    '''
    def __init__(self, input_dim, dilation=[3, 6, 9]):
        super(ContextAggregation, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=dilation[0], dilation=dilation[0], bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=dilation[1], dilation=dilation[1], bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=dilation[2], dilation=dilation[2], bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = out1 + out2 + out3

        return out

