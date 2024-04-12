import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List
from collections import OrderedDict
from torch import nn, Tensor
from src.BasicBlock import se_block, cbam_block, eca_block, TransposeConv, Upsample, MultiScale_Upconv, ContextAggregation
from src.resnet_backbone import resnet50

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
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

class ResUNet(nn.Module):
    def __init__(self, backbone):
        super(ResUNet, self).__init__()
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = backbone

        # Decoder part
        self.decoder_layer3 = decoder_conv(2048, 1024)
        self.decoder_layer2 = decoder_conv(1024, 512)
        self.decoder_layer1 = decoder_conv(512, 256)

        # Upsampling layer
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)   #二分类  num_classes=2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[:, 1:5, :, :]   #光学

        # Encoder (use intermediate layers)
        intermediate_outputs = self.backbone(x)

        # Decoder with skip connections
        upfeature4 = self.upsample4(intermediate_outputs['layer4'])
        de_feature3 = self.decoder_layer3(torch.cat([intermediate_outputs['layer3'], upfeature4], dim=1))

        upfeature3 = self.upsample3(de_feature3)
        de_feature2 = self.decoder_layer2(torch.cat([intermediate_outputs['layer2'], upfeature3], dim=1))

        upfeature2 = self.upsample2(de_feature2)
        de_feature1 = self.decoder_layer1(torch.cat([intermediate_outputs['layer1'], upfeature2], dim=1))

        # Continue with the rest of the upsampling
        upfeature1 = self.upsample(self.upsample1(de_feature1))

        # Final output
        x = self.sigmoid(self.final_conv(upfeature1))

        return x

def UNet_resnet50(pretrain_backbone=False, replace_stride_with_dilation=[False, False, False]):
    """
    2023.12.27   利用光学图像实现分割任务  单分支   二分类  num_classes=2  输出层使用sigmoid激活，单通道，不需要num_classes参数

    """
    backbone = resnet50(in_channel=3)

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50-0676ba61.pth", map_location='cpu'))
        print('load finish!')

    return_layers = {'conv1': 'conv1', 'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = ResUNet(backbone)

    return model
