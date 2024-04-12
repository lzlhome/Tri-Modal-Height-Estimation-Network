import torch
import torch.nn as nn
import torchvision.models as models

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

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.upsample(x)
        out = self.bn(out)
        out = self.relu(out)
        return out




class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=False)

        # 修改最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x



