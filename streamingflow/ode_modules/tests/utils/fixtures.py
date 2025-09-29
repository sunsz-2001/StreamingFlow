"""
Test fixtures and utilities for ODE modules testing.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace


def create_mock_config():
    """Create a mock configuration for testing."""
    cfg = SimpleNamespace()

    # Model config (matching source config.py)
    cfg.MODEL = SimpleNamespace()
    cfg.MODEL.IMPUTE = False
    cfg.MODEL.SOLVER = "euler"
    cfg.MODEL.ENCODER = SimpleNamespace()
    cfg.MODEL.ENCODER.OUT_CHANNELS = 64  # Match source: 64, not 256

    # Small encoder config (matching source config.py)
    cfg.MODEL.SMALL_ENCODER = SimpleNamespace()
    cfg.MODEL.SMALL_ENCODER.FILTER_SIZE = 64
    cfg.MODEL.SMALL_ENCODER.SKIPCO = False  # Match source: False, not True

    # Future prediction config
    cfg.MODEL.FUTURE_PRED = SimpleNamespace()
    cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP = False

    return cfg


def create_test_tensors():
    """Create standard test tensors for ODE modules."""
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    seq_len = 4

    # Standard feature tensor [B, C, H, W]
    feature_tensor = torch.randn(batch_size, channels, height, width)

    # Sequence tensor [B, T, C, H, W]
    sequence_tensor = torch.randn(batch_size, seq_len, channels, height, width)

    # Time stamps
    timestamps = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)

    # Target times
    target_times = torch.linspace(1.1, 2.0, 3).unsqueeze(0).repeat(batch_size, 1)

    return {
        'feature': feature_tensor,
        'sequence': sequence_tensor,
        'timestamps': timestamps,
        'target_times': target_times,
        'batch_size': batch_size,
        'channels': channels,
        'height': height,
        'width': width,
        'seq_len': seq_len
    }


class TestModelWrapper(nn.Module):
    """Wrapper for testing modules with standard forward pass."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def assert_tensor_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, \
        f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def assert_tensor_finite(tensor, name="tensor"):
    """Assert tensor contains finite values."""
    assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"


def assert_tensor_range(tensor, min_val=None, max_val=None, name="tensor"):
    """Assert tensor values are within expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"{name} has values below {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"{name} has values above {max_val}"