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
    """
    backbone: resnet50 or resnet101
    in_channel: revise the input channels of 'conv1' of resnet for different input feature
    """
    def __init__(self, backbone, in_channel):
        super(ResUNet, self).__init__()
        self.in_channel = in_channel

        # backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)
        if self.in_channel == 4:        #分类 building and background
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
        x = self.final_conv(upfeature1)
        if self.in_channel == 4:
            x = self.sigmoid(x)

        return x

def revise_resnet50_conv1(pretrain_backbone=False, replace_stride_with_dilation=[False, False, False], in_channel=4):
    """
    单分支的UNet_resnet50  输出单通道
    in_channel=4  光学图像  实现分割任务  输出层使用sigmoid激活
    """
    backbone = resnet50(in_channel=3)

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50-0676ba61.pth", map_location='cpu'))
        print('resnet50 weights load finish!')

    return_layers = {'conv1': 'conv1', 'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return backbone

def UNet_resnet50_single(pretrain_backbone=False, replace_stride_with_dilation=[False, False, False], in_channel=4):
    """
    单分支的UNet_resnet50  输出单通道
    pretrain_backbone：是否加载resnet50的预训练参数
    in_channel=4  光学图像  实现分割任务  输出层使用sigmoid激活
    """
    backbone = revise_resnet50_conv1(pretrain_backbone=pretrain_backbone, replace_stride_with_dilation=replace_stride_with_dilation, in_channel=in_channel)

    model = ResUNet(backbone, in_channel)

    return model

def UNet_resnet50_single_pretrained(in_channel=4):
    """
    加载UNet_resnet50_single在光学图像上训练的参数  这样会覆盖UNet_resnet50_single的resnet50的预训练参数
    """
    model = UNet_resnet50_single(in_channel=4)

    # UNet_resnet50_single在光学图像上训练的权重
    weights_dict = torch.load("UNet_resnet50_025_weights/model_144.pth", map_location='cpu')['model']
    model_dict = model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(weights_dict)
    model.load_state_dict(weights_dict)
    print('UNet_resnet50_single weights load finish!')

    if in_channel != 4:
        model.backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model

class UNet_resnet50_trible(nn.Module):
    """
    三分支各自进行，没有交互，光学实现分割任务，最后对各分支输出的单通道结果进行操作
    pretrain_resunet  是否加载分割任务的权重
    pretrain_backbone  不加载分割任务的权重的话，是否加载resnet50参数
    """
    def __init__(self, pretrain_resunet=[False, True, False], pretrain_backbone=[True, True, True], threshold=0.5):
        super(UNet_resnet50_trible, self).__init__()
        self.threshold = threshold

        if pretrain_resunet[0]:    #加载分割任务的权重
            self.branch_SAR = UNet_resnet50_single_pretrained(2)
        else:
            self.branch_SAR = UNet_resnet50_single(pretrain_backbone=pretrain_backbone[0], in_channel=2)

        if pretrain_resunet[1]:    #加载分割任务的权重
            self.branch_optical = UNet_resnet50_single_pretrained(4)
        else:
            self.branch_optical = UNet_resnet50_single(pretrain_backbone=pretrain_backbone[1], in_channel=4)

        if pretrain_resunet[2]:    #加载分割任务的权重
            self.branch_NTL = UNet_resnet50_single_pretrained(1)
        else:
            self.branch_NTL = UNet_resnet50_single(pretrain_backbone=pretrain_backbone[2], in_channel=1)

        # Final output layer
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1)   #branch_optical输出的概率分布图分别与另外两个分支相乘，再相加，再卷积输出

    def forward(self, x):
        A = x[:, 5:7, :, :]
        B = x[:, 1:5, :, :]
        C = x[:, 0:1, :, :]

        x1 = self.branch_SAR(A)
        prob = self.branch_optical(B)   # 输出是单通道的概率图
        x3 = self.branch_NTL(C)

        x = prob * x1 + prob * x3   #branch_optical 输出的概率分布图二值化后分别与另外两个分支相乘，再相加，再卷积输出
        # binarized_prob = (prob > self.threshold).float()    # 二值化（阈值）
        # x = binarized_prob * x1 + binarized_prob * x3   #branch_optical 输出的概率分布图二值化后分别与另外两个分支相乘，再相加，再卷积输出

        x = self.final_conv(x)

        return x1, prob, x3, x





class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        x = x * attn
        return x

class UNet_resnet50_trible_WithAttention(nn.Module):
    def __init__(self, backbone, pretrain_backbone=False, threshold=0.5):
        super(UNet_resnet50_trible_WithAttention, self).__init__()

        self.threshold = threshold

        self.SARbackbone = revise_resnet50_conv1(pretrain_backbone, 2)
        self.opticalbackbone = revise_resnet50_conv1(pretrain_backbone, 4)
        self.NTLbackbone = revise_resnet50_conv1(pretrain_backbone, 1)

        # Final output layer
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1)   #branch_optical输出的概率分布图分别与另外两个分支相乘，再相加，再卷积输出

        # Attention blocks in the encoder
        self.attention1 = AttentionBlock(64, 256)
        self.attention2 = AttentionBlock(256, 512)
        self.attention3 = AttentionBlock(512, 1024)
        self.attention4 = AttentionBlock(1024, 2048)

        # Decoder part with attention blocks and skip connections
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # Encoder (use intermediate layers)
        intermediate_outputs = self.backbone(x)

        # Apply attention blocks in the encoder and use attentioned features as input to the next layer
        x = self.attention1(intermediate_outputs['layer1'])
        intermediate_outputs['layer1'] = self.backbone.layer1(x)

        x = self.attention2(intermediate_outputs['layer2'])
        intermediate_outputs['layer2'] = self.backbone.layer2(x)

        x = self.attention3(intermediate_outputs['layer3'])
        intermediate_outputs['layer3'] = self.backbone.layer3(x)

        x = self.attention4(intermediate_outputs['layer4'])
        intermediate_outputs['layer4'] = self.backbone.layer4(x)

        # Decoder with skip connections
        x = self.upsample(intermediate_outputs['layer4'])
        x = torch.cat([x, intermediate_outputs['layer4']], dim=1)

        # Continue with the rest of the decoder
        x = self.decoder(x)

        # Final output
        x = self.final_conv(x)

        return x


class ResUNetWithAttention(nn.Module):
    def __init__(self, backbone, input_channels, num_classes=2):
        super(ResUNetWithAttention, self).__init__()

        # Modify the first convolutional layer in the backbone
        backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = backbone

        # Attention blocks in the encoder
        self.attention1 = AttentionBlock(64, 256)
        self.attention2 = AttentionBlock(256, 512)
        self.attention3 = AttentionBlock(512, 1024)
        self.attention4 = AttentionBlock(1024, 2048)

        # Decoder part with attention blocks and skip connections
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (use intermediate layers)
        intermediate_outputs = self.backbone(x)

        # Apply attention blocks in the encoder and use attentioned features as input to the next layer
        x = self.attention1(intermediate_outputs['layer1'])
        intermediate_outputs['layer1'] = self.backbone.layer1(x)

        x = self.attention2(intermediate_outputs['layer2'])
        intermediate_outputs['layer2'] = self.backbone.layer2(x)

        x = self.attention3(intermediate_outputs['layer3'])
        intermediate_outputs['layer3'] = self.backbone.layer3(x)

        x = self.attention4(intermediate_outputs['layer4'])
        intermediate_outputs['layer4'] = self.backbone.layer4(x)

        # Decoder with skip connections
        x = self.upsample(intermediate_outputs['layer4'])
        x = torch.cat([x, intermediate_outputs['layer4']], dim=1)

        # Continue with the rest of the decoder
        x = self.decoder(x)

        # Final output
        x = self.final_conv(x)

        return x

