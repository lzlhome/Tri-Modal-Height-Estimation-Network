from .resnet_backbone import resnet50, resnet101
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.pool_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        feats = [x]
        for layer in self.pool_layers:
            feats.append(F.interpolate(layer(x), size=x.shape[2:], mode='bilinear', align_corners=True))
        return torch.cat(feats, dim=1)

class PSPNet(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False):
        super(PSPNet, self).__init__()
        resnet = resnet50(in_channel=in_channel, replace_stride_with_dilation=[False, False, False], pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.pyramid_pooling = PyramidPoolingModule(2048, [6, 3, 2, 1])
        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)
        x = self.cls(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

