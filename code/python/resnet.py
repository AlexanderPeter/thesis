# Copied from https://github.com/princeton-vl/selfstudy/blob/master/models/resnet.py

import torch
import torch.nn as nn
from torch.functional import F

# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

# NOTE: Only ResNet50 models included so far
model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride

        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.pool = nn.MaxPool2d(2)
        self.bn = norm_layer(planes)
        self.relu = nn.CELU(inplace=True, alpha=0.075)

    def forward(self, x):
        out = self.conv1(x)
        if self.stride == 2:
            out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        is_half=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        midplanes = planes
        if is_half:
            midplanes = midplanes // 2

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = norm_layer(midplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def pool_and_flatten(x):
    x = F.adaptive_avg_pool2d(x, (1, 1))
    return torch.flatten(x, 1)


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        out_feats=256,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        in_channels=3,
        is_half=False,
        skip_init_pool=True,
        prepool_fc=True,
        feat_layer=-1,
        freeze_at_k=None,
        is_resnet9=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.is_half = is_half
        self.skip_init_pool = skip_init_pool
        self.feat_layer = feat_layer
        self.prepool_fc = prepool_fc
        self.freeze_at_k = freeze_at_k
        self.out_feats = out_feats
        self.n_layers = 4

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        k, s, p = (3, 1, 1) if is_resnet9 else (7, 2, 3)
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=k, stride=s, padding=p, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if not is_resnet9:
            b = 32 if is_half else 64
            self.layer1 = self._make_layer(block, b, layers[0])
            self.layer2 = self._make_layer(
                block,
                2 * b,
                layers[1],
                stride=2,
                dilate=replace_stride_with_dilation[0],
            )
            self.layer3 = self._make_layer(
                block,
                4 * b,
                layers[2],
                stride=2,
                dilate=replace_stride_with_dilation[1],
            )
            self.layer4 = self._make_layer(
                block,
                8 * b,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
            )
            fc_in_feats = 8 * b * block.expansion

        else:
            self.n_layers = 3
            mid_f = 128 if is_half else 256
            if is_half:
                mid_f = 128
                downsample = nn.Sequential(
                    conv1x1(64, 128, 1),
                    nn.BatchNorm2d(128),
                )
            else:
                mid_f = 256
                downsample = None

            self.layer1 = nn.Sequential(
                ConvBlock(64, mid_f // 2, stride=2),
                BasicBlock(mid_f // 2, 128, downsample=downsample, is_half=is_half),
            )
            self.layer2 = nn.Sequential(ConvBlock(128, mid_f, stride=2))
            self.layer3 = nn.Sequential(
                ConvBlock(mid_f, 512, stride=2), BasicBlock(512, 512, is_half=is_half)
            )
            fc_in_feats = 512

        self.fc = nn.Linear(fc_in_feats, out_feats)
        if feat_layer != -1:
            l = self._modules[f"layer{feat_layer + 1}"][0]
            self.out_feats = l.conv1.in_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.skip_init_pool:
            x = self.maxpool(x)

        all_feats = [x]
        layer_feats = [x]

        for i in range(self.n_layers):
            layer = self._modules[f"layer{i+1}"]
            for l in layer:
                all_feats += [l(all_feats[-1])]

            if self.freeze_at_k is not None and self.freeze_at_k == i:
                all_feats[-1] = all_feats[-1].detach()

            layer_feats += [all_feats[-1]]

        if self.prepool_fc:
            x = all_feats[-1].permute(0, 2, 3, 1)
            all_feats += [self.fc(x).permute(0, 3, 1, 2)]
            all_feats += [pool_and_flatten(all_feats[-1])]
        else:
            all_feats += [self.fc(pool_and_flatten(all_feats[-1]))]

        feats = all_feats[-1]
        if self.feat_layer != -1:
            feats = pool_and_flatten(layer_feats[self.feat_layer])

        return feats, all_feats

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
      progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
