import torch
import torch.nn as nn

class EntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntryBlock, self).__init__()
        self.pointwise1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pointwise2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pointwise3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(self.pointwise1, self.bn1)
        self.layer2 = nn.Sequential(self.pointwise2, self.bn2)
        self.layer3 = nn.Sequential(self.pointwise3, self.bn3)
        self.concat_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out += self.concat_layer(residual)
        out = self.relu(out)
        return out


class SepBlock(nn.Module):
    def __init__(self, in_channels):
        super(SepBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.layer = nn.Sequential(self.relu, self.depthwise_conv, self.pointwise_conv, self.bn)

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out = self.layer(out)
        out += residual
        return out


class DepthwiseSepModel(nn.Module):
    def __init__(self, in_channel, mid_channel=512, SepBlock_num=12, num_classes=1):
        super(DepthwiseSepModel, self).__init__()
        # torch.autograd.set_detect_anomaly(True)  # 启用异常检测
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.entry_block = EntryBlock(in_channel, mid_channel)
        self.sep_blocks = nn.Sequential(
            *[SepBlock(mid_channel) for _ in range(SepBlock_num)]
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_channel, num_classes, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.downsample(x)
        out = self.entry_block(out)
        out = self.sep_blocks(out)
        out = self.upsample(out)
        return out
