from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamingflow.models.layers.gdsca import SparseGDSCA

try:
    import spconv.pytorch as spconv
except ImportError as exc:  # pragma: no cover - optional dependency
    spconv = None
    _SPCONV_IMPORT_ERROR = exc
else:
    _SPCONV_IMPORT_ERROR = None


class EventEncoderSparse(nn.Module):
    """Event encoder built on sparse convolutions (spconv)."""

    def __init__(self, cfg, out_channels):
        super().__init__()
        if spconv is None:
            raise ImportError(
                "spconv is required for EventEncoderSparse. Please install spconv before using this encoder."
            ) from _SPCONV_IMPORT_ERROR

        self.cfg = cfg
        self.out_channels = out_channels

        voxel_threshold = float(getattr(cfg.MODEL.EVENT, "VOXEL_THRESHOLD", 0.0))
        max_points = getattr(cfg.MODEL.EVENT, "VOXEL_MAX_POINTS", None)
        if isinstance(max_points, float):
            max_points = int(max_points)
        self.voxelizer = EventVoxelizer(threshold=voxel_threshold, max_points=max_points)

        event_channels = getattr(cfg.MODEL.EVENT, "IN_CHANNELS", 0)
        if event_channels <= 0:
            event_channels = 2 * getattr(cfg.MODEL.EVENT, "BINS", 10)

        event_cameras_cfg = getattr(cfg.DATASET, "EVENT_CAMERAS", [])
        if event_cameras_cfg:
            num_event_cameras = len(event_cameras_cfg)
        else:
            num_event_cameras = len(cfg.IMAGE.NAMES)
        self.num_event_cameras = num_event_cameras

        in_channels = num_event_cameras * event_channels
        self.time_steps = int(getattr(cfg, "TIME_RECEPTIVE_FIELD", 1))

        gdsca_reduction = int(getattr(cfg.MODEL.EVENT, "GDSCA_REDUCTION", 16))

        self.backbone = SparseBackbone(in_channels, out_channels, gdsca_reduction)
        self.fusion_head = FusionHead(out_channels)

    def forward(self, x: torch.Tensor):
        """Accepts either flattened [B*S*N_evt, C, H, W] or structured [B,S,N_evt,C,H,W] inputs."""
        if x.dim() == 6:
            frames = x
            batch_size = x.shape[0]
            time_steps = x.shape[1]
        elif x.dim() == 4:
            total, channels, h, w = x.shape
            cams = self.num_event_cameras
            if total % cams != 0:
                raise ValueError(
                    f"Event encoder expected batch divisible by number of event cameras ({cams}), got {total}."
                )
            bs_times = total // cams
            time_steps = self.time_steps
            if bs_times % time_steps != 0:
                raise ValueError(
                    f"Cannot infer batch size: flattened frames {bs_times} not divisible by TIME_RECEPTIVE_FIELD={time_steps}."
                )
            batch_size = bs_times // time_steps
            frames = x.view(batch_size, time_steps, cams, channels, h, w)
        else:
            raise ValueError("EventEncoderSparse expects input with 4 or 6 dimensions.")

        voxelized = self.voxelizer(frames)
        voxelized["batch_size"] = batch_size
        voxelized["time_steps"] = time_steps
        voxelized["num_cameras"] = self.num_event_cameras

        sparse_feats = self.backbone(voxelized)
        bev, depth_logits = self.fusion_head(sparse_feats, voxelized)

        B = batch_size
        S = time_steps
        C = bev.shape[2]
        H, W = bev.shape[-2], bev.shape[-1]
        bev = bev.view(B * S, 1, C, H, W).repeat(1, self.num_event_cameras, 1, 1, 1)
        bev = bev.view(B * S * self.num_event_cameras, C, H, W)
        return bev, depth_logits


class EventVoxelizer(nn.Module):
    """Convert dense event frames into sparse voxel representation."""

    def __init__(self, threshold: float = 0.0, max_points: Optional[int] = None):
        super().__init__()
        self.threshold = float(threshold)
        self.max_points = max_points

    def forward(self, frames: torch.Tensor) -> dict:
        if frames.dim() != 6:
            raise ValueError(
                f"Expected event tensor with shape [B, S, N_evt, C_evt, H, W], got {tuple(frames.shape)}"
            )

        B, S, N_evt, C_evt, H, W = frames.shape
        dense = frames.reshape(B * S, N_evt * C_evt, H, W)

        if self.threshold > 0:
            mask = dense.abs().amax(dim=1) > self.threshold
        else:
            mask = dense.ne(0).any(dim=1)

        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            coords = torch.zeros((0, 4), dtype=torch.int32, device=frames.device)
            features = torch.zeros((0, N_evt * C_evt), dtype=frames.dtype, device=frames.device)
        else:
            batch_flat = nonzero[:, 0]
            y_idx = nonzero[:, 1]
            x_idx = nonzero[:, 2]

            batch_idx = torch.div(batch_flat, S, rounding_mode="floor")
            time_idx = batch_flat % S

            coords = torch.stack((batch_idx, time_idx, y_idx, x_idx), dim=1).to(torch.int32)
            features = dense[batch_flat, :, y_idx, x_idx]

            if self.max_points is not None and coords.shape[0] > self.max_points:
                keep = torch.randperm(coords.shape[0], device=frames.device)[: self.max_points]
                coords = coords[keep]
                features = features[keep]

        return {
            "features": features,
            "coords": coords,
            "batch_size": B,
            "time_steps": S,
            "num_cameras": N_evt,
            "channels": C_evt,
            "spatial_shape": (H, W),
        }


class SparseBackbone(nn.Module):
    """Two-stage sparse convolutional backbone operating on (time, y, x)."""

    def __init__(self, in_channels: int, hidden_channels: int, attn_reduction: int = 16):
        super().__init__()
        self.input_proj = spconv.SubMConv3d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.input_act = nn.ReLU(inplace=True)

        self.block1 = ResidualSubMBlock(hidden_channels, attn_reduction)
        self.down1 = ResidualDownBlock(hidden_channels, hidden_channels, attn_reduction)

    def forward(self, voxelized: Dict[str, torch.Tensor]) -> List[spconv.SparseConvTensor]:
        features = voxelized["features"]
        coords = voxelized["coords"]
        batch_size = voxelized["batch_size"]
        time_steps = voxelized["time_steps"]
        H, W = voxelized["spatial_shape"]

        spatial_shape = [time_steps, H, W]
        sparse_tensor = spconv.SparseConvTensor(
            features, coords.int(), spatial_shape=spatial_shape, batch_size=batch_size
        )

        proj = self.input_proj(sparse_tensor)
        if proj.features.numel() > 0:
            proj = proj.replace_feature(self.input_act(self.input_bn(proj.features)))

        stem = self.block1(proj)
        level1 = self.down1(stem)
        return [stem, level1]


class FusionHead(nn.Module):
    """Fuse multi-scale sparse features and return dense BEV tensor."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, sparse_feats: List[spconv.SparseConvTensor], meta: Dict[str, torch.Tensor]):
        batch_size = meta["batch_size"]
        time_steps = meta["time_steps"]

        dense_maps = []
        for tensor in sparse_feats:
            dense = tensor.dense()  # [B, C, T, H, W]
            if dense.size(2) > time_steps:
                dense = dense[:, :, :time_steps]
            dense = dense.permute(0, 2, 1, 3, 4).contiguous()
            dense_maps.append(dense)

        fused = dense_maps[0]
        for extra in dense_maps[1:]:
            if extra.shape[-2:] != fused.shape[-2:]:
                extra_flat = extra.view(
                    extra.shape[0] * extra.shape[1],
                    extra.shape[2],
                    extra.shape[3],
                    extra.shape[4],
                )
                resized = F.interpolate(
                    extra_flat,
                    size=fused.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                extra = resized.view(
                    extra.shape[0], extra.shape[1], extra.shape[2], fused.shape[-2], fused.shape[-1]
                )
            fused = fused + extra
        return fused, None


class ResidualSubMBlock(nn.Module):
    """Residual block with Submanifold sparse convolutions."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.attn = SparseGDSCA(channels, reduction)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x

        out = self.conv1(x)
        if out.features.numel() == 0:
            return identity
        out = out.replace_feature(self.act(self.bn1(out.features)))

        out = self.conv2(out)
        if out.features.numel() == 0:
            return identity
        out = out.replace_feature(self.bn2(out.features))
        out = self.attn(out)

        fused = out.features
        if identity.features.numel() > 0:
            fused = fused + identity.features
        fused = self.act(fused)
        return out.replace_feature(fused)


class ResidualDownBlock(nn.Module):
    """Residual downsampling block with stride-2 sparse convolutions on spatial dims."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.skip = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=(1, 2, 2),
            bias=False,
        )
        self.skip_bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.attn = SparseGDSCA(out_channels, reduction)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = self.skip(x)
        if identity.features.numel() > 0:
            identity = identity.replace_feature(self.skip_bn(identity.features))

        out = self.conv(x)
        if out.features.numel() == 0:
            return identity

        out = out.replace_feature(self.bn(out.features))
        out = self.attn(out)

        fused = out.features
        if identity.features.numel() > 0:
            fused = fused + identity.features
        fused = self.act(fused)
        return out.replace_feature(fused)
