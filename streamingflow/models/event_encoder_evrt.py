from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from evlearn.bundled.rtdetr_pytorch.nn.backbone.presnet import PResNet
    from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.hybrid_encoder import HybridEncoder
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "Failed to import EvRT-DETR modules. Please install evrt-detr (e.g. pip install -e ./evrt-detr)."
    ) from exc


def _parse_backbone_depth(backbone: str) -> int:
    if isinstance(backbone, int):
        return backbone
    if isinstance(backbone, str):
        if backbone.lower().startswith("resnet"):
            return int(backbone.lower().replace("resnet", ""))
        if backbone.lower().startswith("presnet"):
            return int(backbone.lower().replace("presnet", ""))
    raise ValueError(f"Unsupported backbone spec: {backbone}")


class EventEncoderEvRT(nn.Module):
    """Wraps EvRT-DETR backbone + HybridEncoder to produce stride-8 event features."""

    def __init__(self, cfg, out_channels: int):
        super().__init__()
        self.cfg = cfg
        self.out_channels = out_channels

        bins = getattr(cfg, "BINS", 10)
        in_channels = getattr(cfg, "IN_CHANNELS", 0)
        if in_channels <= 0:
            in_channels = 2 * bins
        backbone_depth = _parse_backbone_depth(getattr(cfg, "BACKBONE", 50))
        pretrained = getattr(cfg, "PRETRAINED", False)
        freeze_backbone = getattr(cfg, "FREEZE_BACKBONE", False)
        hybrid_hidden = getattr(cfg, "HIDDEN_DIM", 256)
        hybrid_layers = getattr(cfg, "NUM_ENCODER_LAYERS", 1)
        hybrid_heads = getattr(cfg, "NUM_HEADS", 8)

        # PresNet returns multi-scale features. We take stages with strides 8/16/32.
        return_idx = getattr(cfg, "RETURN_IDX", [1, 2, 3])

        self.backbone = PResNet(
            depth=backbone_depth,
            features_input=in_channels,
            return_idx=return_idx,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_channels_list = list(self.backbone.out_channels)

        self.hybrid_encoder = HybridEncoder(
            in_channels=in_channels_list,
            feat_strides=list(self.backbone.out_strides),
            hidden_dim=hybrid_hidden,
            num_encoder_layers=hybrid_layers,
            nhead=hybrid_heads,
        )

        fusion_width = len(in_channels_list) if getattr(cfg, "MULTISCALE_FUSION", "sum") == "concat" else 1

        # Compress multi-scale output back to the requested dimensionality.
        self.output_proj = nn.Conv2d(hybrid_hidden * fusion_width, out_channels, kernel_size=1)

        fusion_mode = getattr(cfg, "MULTISCALE_FUSION", "sum")
        if fusion_mode not in {"sum", "concat"}:
            raise ValueError(f"Unsupported MULTISCALE_FUSION={fusion_mode}")
        self.multiscale_fusion = fusion_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C_evt, H, W)

        Returns:
            Tensor of shape (B, out_channels, H/8, W/8)
        """
        feats: List[torch.Tensor] = self.backbone(x)
        encoded: List[torch.Tensor] = self.hybrid_encoder(feats)
        p3 = encoded[0]

        if self.multiscale_fusion == "sum" and len(encoded) > 1:
            fused = p3
            target_hw = p3.shape[-2:]
            for scale in encoded[1:]:
                fused = fused + F.interpolate(scale, size=target_hw, mode="bilinear", align_corners=False)
        elif self.multiscale_fusion == "concat" and len(encoded) > 1:
            resized = [p3]
            target_hw = p3.shape[-2:]
            for scale in encoded[1:]:
                resized.append(F.interpolate(scale, size=target_hw, mode="bilinear", align_corners=False))
            fused = torch.cat(resized, dim=1)
        else:
            fused = p3

        out = self.output_proj(fused)
        return out
