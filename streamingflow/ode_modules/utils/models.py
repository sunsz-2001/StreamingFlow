"""
Independent model utilities for ODE components.

This module contains model utilities and helper functions needed by the ODE modules,
making ode_modules completely independent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolutions import ConvBlock


class SmallEncoder(nn.Module):
    """Small encoder for efficient feature compression."""

    def __init__(self, in_channels, out_channels, filter_size=16):
        super().__init__()
        self.filter_size = filter_size

        self.conv1 = ConvBlock(in_channels, filter_size, 3, 1, norm='bn', activation='relu')
        self.conv2 = ConvBlock(filter_size, filter_size * 2, 3, 2, norm='bn', activation='relu')
        self.conv3 = ConvBlock(filter_size * 2, out_channels, 3, 1, norm='bn', activation='relu')

    def forward(self, x, return_skip=False):
        skip1 = self.conv1(x)
        skip2 = self.conv2(skip1)
        out = self.conv3(skip2)

        if return_skip:
            return out, [skip1, skip2]
        return out


class SmallDecoder(nn.Module):
    """Small decoder for feature decompression."""

    def __init__(self, in_channels, out_channels, filter_size=16, use_skip=True):
        super().__init__()
        self.use_skip = use_skip

        self.conv1 = ConvBlock(in_channels, filter_size, 3, 1, norm='bn', activation='relu')
        self.up1 = nn.ConvTranspose2d(filter_size, filter_size, 3, 2, 1, 1)
        self.conv2 = ConvBlock(filter_size, filter_size, 3, 1, norm='bn', activation='relu')
        self.conv3 = ConvBlock(filter_size, out_channels, 3, 1, norm='bn', activation='relu')

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ConvNet(nn.Module):
    """Simple ConvNet for parameter prediction."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(32, in_channels // 2)

        self.layers = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 3, 1, norm='bn', activation='relu'),
            ConvBlock(mid_channels, mid_channels, 3, 1, norm='bn', activation='relu'),
            nn.Conv2d(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layers(x)


def rsample_normal(params, max_log_sigma=5.0, min_log_sigma=-5.0):
    """
    Reparameterized sampling from normal distribution.

    Args:
        params: Tensor containing mean and log_sigma concatenated along channel dim
        max_log_sigma: Maximum log sigma value
        min_log_sigma: Minimum log sigma value

    Returns:
        Sampled tensor using reparameterization trick
    """
    mean, log_sigma = params.chunk(2, dim=1)

    # Clamp log_sigma to prevent numerical issues
    log_sigma = torch.clamp(log_sigma, min_log_sigma, max_log_sigma)
    sigma = torch.exp(log_sigma)

    # Reparameterization trick
    eps = torch.randn_like(mean)
    return mean + eps * sigma


__all__ = ['SmallEncoder', 'SmallDecoder', 'ConvNet', 'rsample_normal']