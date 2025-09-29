"""
Independent convolution modules for ODE components.

This module contains all necessary convolution components copied from streamingflow
to make ode_modules completely independent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, norm='bn', activation='relu'):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        # Normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck module with residual connection."""

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1,
                 groups=1, upsample=False, downsample=False, dropout=0.0):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=1, stride=2, output_padding=padding_size,
                padding=padding_size, groups=groups)
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=dilation, stride=2, padding=padding_size, groups=groups)
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=dilation, stride=1, padding=padding_size, groups=groups)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            bottleneck_conv,
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        if in_channels != out_channels or upsample or downsample:
            if upsample:
                self.projection = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            elif downsample:
                self.projection = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            else:
                self.projection = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.projection = None

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.layers(x)
        if self.dropout is not None:
            y = self.dropout(y)

        if self.projection is not None:
            x = self.projection(x)

        return self.activation(x + y)


class Block(nn.Module):
    """ConvNeXt Block for advanced feature processing."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.GroupNorm(1, dim, eps=1e-6)  # Simplified LayerNorm
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # Simplified drop path

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # Apply norm
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Bottleblock(nn.Module):
    """Simplified bottle block."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = in_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DeepLabHead(nn.Sequential):
    """DeepLab head for segmentation tasks."""

    def __init__(self, in_channels, out_channels, mid_channels=256):
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, 1)
        )


__all__ = ['ConvBlock', 'Bottleneck', 'Block', 'DeepLabHead', 'Bottleblock']