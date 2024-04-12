import torch
import torch.nn as nn


class ChannelAttention(nn.Module):  # channel-wise
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)


class DirectionalASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 3, 5]):
        super(DirectionalASPP, self).__init__()

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

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
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

        # 拼接分支结果
        out = self.project(torch.cat([branch_1x1, branch_3x3_1, branch_3x3_2, branch_3x3_3], dim=1))

        return out


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

