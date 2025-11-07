import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
except ImportError as exc:  # pragma: no cover
    spconv = None
    _SPCONV_IMPORT_ERROR = exc
else:
    _SPCONV_IMPORT_ERROR = None


class GDSCA(nn.Module):
    """Gated Dynamic Spatial-Channel Attention.

    Applies 3D spatial attention followed by channel-wise gating.
    Designed to operate on tensor with layout [B, C, T, H, W].
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels % reduction != 0:
            reduction = max(1, channels // reduction)
        self.spatial_conv = nn.Conv3d(channels, 1, kernel_size=1)
        self.spatial_gate = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.channel_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_weight = self.spatial_gate(self.spatial_conv(x))
        x = x * spatial_weight
        pooled = self.avg_pool(x).flatten(1)
        channel_weight = self.channel_gate(self.fc2(F.relu(self.fc1(pooled), inplace=True))).view(
            x.size(0), x.size(1), 1, 1, 1
        )
        return x * channel_weight


class SparseGDSCA(nn.Module):
    """Applies GDSCA on SparseConvTensor by densifying-channel gating and resparsifying."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if spconv is None:
            raise ImportError(
                "spconv is required for SparseGDSCA. Please install spconv before enabling GDSCA."
            ) from _SPCONV_IMPORT_ERROR
        self.channels = channels
        self.block = GDSCA(channels, reduction)

    def forward(self, tensor: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        features = tensor.features
        if features.numel() == 0:
            return tensor

        spatial_shape = tensor.spatial_shape
        batch_size = tensor.batch_size
        dense = tensor.dense()  # [B, C, T, H, W]
        attn = self.block(dense)
        attn = attn.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]

        coords = tensor.indices
        b = coords[:, 0]
        t = coords[:, 1]
        y = coords[:, 2]
        x = coords[:, 3]

        sampled = attn[b, t, y, x]
        return tensor.replace_feature(sampled)
