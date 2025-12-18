from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_tensor(tensor, name, allow_inf=False):
    """Validate tensor for NaN/Inf values and raise error if found.
    
    Args:
        tensor: Tensor to validate
        name: Name for error message
        allow_inf: If True, only check for NaN, not Inf
    """
    if torch.isnan(tensor).any():
        stats = f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        raise ValueError(f"NaN detected in {name}. Stats: {stats}, shape: {tensor.shape}")
    if not allow_inf and torch.isinf(tensor).any():
        stats = f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        raise ValueError(f"Inf detected in {name}. Stats: {stats}, shape: {tensor.shape}")

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
        # C_evt
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
        use_depth_head = getattr(cfg, "USE_DEPTH_HEAD", True)
        if use_depth_head:
            depth_bins = int(getattr(cfg, "DEPTH_BINS", 80))
            depth_hidden = int(getattr(cfg, "DEPTH_HEAD_CHANNELS", 128))
            skip_channels = hybrid_hidden if len(in_channels_list) > 1 else 0
            self.depth_head = EventDepthHead(
                in_channels=hybrid_hidden,
                skip_channels=skip_channels,
                hidden_channels=depth_hidden,
                depth_bins=depth_bins,
            )
        else:
            self.depth_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (B, C_evt, H, W)

        Returns:
            features: Tensor of shape (B, out_channels, H/8, W/8)
            depth_logits: Tensor of shape (B, depth_bins, H/8, W/8) or None
        """
        # Validate input
        _validate_tensor(x, "EventEncoderEvRT input")
        
        # Validate backbone parameters (only in training)
        if self.training:
            for name, param in self.backbone.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    stats = f"min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}"
                    raise ValueError(f"NaN/Inf in backbone parameter {name}. Stats: {stats}")
        
        # Validate BatchNorm running stats and fix numerical issues
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    if torch.isnan(module.running_mean).any() or torch.isinf(module.running_mean).any():
                        print(f"[WARNING] NaN/Inf in BatchNorm running_mean: {name}. Resetting to 0.")
                        module.running_mean.data = torch.where(
                            torch.isnan(module.running_mean) | torch.isinf(module.running_mean),
                            torch.zeros_like(module.running_mean),
                            module.running_mean
                        )
                if hasattr(module, 'running_var') and module.running_var is not None:
                    if torch.isnan(module.running_var).any() or torch.isinf(module.running_var).any():
                        print(f"[WARNING] NaN/Inf in BatchNorm running_var: {name}. Resetting to 1.")
                        module.running_var.data = torch.where(
                            torch.isnan(module.running_var) | torch.isinf(module.running_var),
                            torch.ones_like(module.running_var),
                            module.running_var
                        )
                    # 确保 running_var 不会太小（防止除以 0）
                    min_var = 1e-5  # 最小方差阈值
                    if (module.running_var < min_var).any():
                        print(f"[WARNING] BatchNorm running_var too small in {name}. Clamping to {min_var}.")
                        module.running_var.data = torch.clamp(module.running_var.data, min=min_var)
        
        feats: List[torch.Tensor] = self.backbone(x)
        for i, feat in enumerate(feats):
            _validate_tensor(feat, f"EventEncoderEvRT backbone output[{i}]")
        
        encoded: List[torch.Tensor] = self.hybrid_encoder(feats)
        for i, enc in enumerate(encoded):
            _validate_tensor(enc, f"EventEncoderEvRT hybrid_encoder output[{i}]")
        
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
        
        _validate_tensor(fused, "EventEncoderEvRT fused features")

        out = self.output_proj(fused)
        _validate_tensor(out, "EventEncoderEvRT output_proj output")
        
        depth_logits: Optional[torch.Tensor] = None
        if self.depth_head is not None:
            skip_feat = encoded[1] if len(encoded) > 1 else None
            depth_logits = self.depth_head(p3, skip_feat)
            if depth_logits is not None:
                _validate_tensor(depth_logits, "EventEncoderEvRT depth_head output")
        return out, depth_logits


class EventDepthHead(nn.Module):
    """Predict per-pixel depth logits from stride-8 event features."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        hidden_channels: int,
        depth_bins: int,
    ) -> None:
        super().__init__()
        if depth_bins <= 1:
            raise ValueError("depth_bins must be > 1 for depth distribution.")
        self.skip_proj: Optional[nn.Module] = None
        if skip_channels > 0:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
            )

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.refine_block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.output_proj = nn.Conv2d(hidden_channels, depth_bins, kernel_size=1)

    def forward(self, main: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_block(main)
        if skip is not None and self.skip_proj is not None:
            skip_resized = F.interpolate(skip, size=main.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip_resized)
        x = self.refine_block(x)
        return self.output_proj(x)
