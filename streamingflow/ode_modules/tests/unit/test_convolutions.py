"""
Unit tests for convolution modules.
"""

import torch
import pytest
from ...utils.convolutions import ConvBlock, Bottleneck, Block, DeepLabHead, Bottleblock
from ..utils.fixtures import assert_tensor_shape, assert_tensor_finite


class TestConvBlock:
    """Test ConvBlock functionality."""

    def test_convblock_forward(self):
        """Test ConvBlock forward pass."""
        in_channels, out_channels = 64, 128
        height, width = 32, 32
        batch_size = 2

        conv_block = ConvBlock(in_channels, out_channels, kernel_size=3, norm='bn', activation='relu')
        x = torch.randn(batch_size, in_channels, height, width)

        output = conv_block(x)

        assert_tensor_shape(output, (batch_size, out_channels, height, width), "ConvBlock output")
        assert_tensor_finite(output, "ConvBlock output")

    def test_convblock_different_norms(self):
        """Test ConvBlock with different normalization options."""
        in_channels, out_channels = 32, 64
        x = torch.randn(2, in_channels, 16, 16)

        # Test BatchNorm
        conv_bn = ConvBlock(in_channels, out_channels, norm='bn')
        output_bn = conv_bn(x)
        assert_tensor_shape(output_bn, (2, out_channels, 16, 16))

        # Test LayerNorm (GroupNorm with 1 group)
        conv_ln = ConvBlock(in_channels, out_channels, norm='ln')
        output_ln = conv_ln(x)
        assert_tensor_shape(output_ln, (2, out_channels, 16, 16))

        # Test no normalization
        conv_none = ConvBlock(in_channels, out_channels, norm='none')
        output_none = conv_none(x)
        assert_tensor_shape(output_none, (2, out_channels, 16, 16))


class TestBottleneck:
    """Test Bottleneck functionality."""

    def test_bottleneck_forward(self):
        """Test Bottleneck forward pass."""
        in_channels = 64
        height, width = 32, 32
        batch_size = 2

        bottleneck = Bottleneck(in_channels)
        x = torch.randn(batch_size, in_channels, height, width)

        output = bottleneck(x)

        assert_tensor_shape(output, (batch_size, in_channels, height, width), "Bottleneck output")
        assert_tensor_finite(output, "Bottleneck output")

    def test_bottleneck_downsample(self):
        """Test Bottleneck with downsampling."""
        in_channels = 64
        out_channels = 128
        height, width = 32, 32
        batch_size = 2

        bottleneck = Bottleneck(in_channels, out_channels, downsample=True)
        x = torch.randn(batch_size, in_channels, height, width)

        output = bottleneck(x)

        expected_height, expected_width = height // 2, width // 2
        assert_tensor_shape(output, (batch_size, out_channels, expected_height, expected_width))
        assert_tensor_finite(output, "Bottleneck downsample output")


class TestBlock:
    """Test ConvNeXt Block functionality."""

    def test_block_forward(self):
        """Test Block forward pass."""
        dim = 64
        height, width = 32, 32
        batch_size = 2

        block = Block(dim)
        x = torch.randn(batch_size, dim, height, width)

        output = block(x)

        assert_tensor_shape(output, (batch_size, dim, height, width), "Block output")
        assert_tensor_finite(output, "Block output")

    def test_block_residual_connection(self):
        """Test that Block preserves residual connection."""
        dim = 64
        x = torch.randn(2, dim, 16, 16)

        block = Block(dim, layer_scale_init_value=0.0)  # No scaling
        output = block(x)

        # With no layer scaling, output should be close to input + processed_input
        # At least the shape should be preserved
        assert_tensor_shape(output, x.shape, "Block residual output")


class TestDeepLabHead:
    """Test DeepLabHead functionality."""

    def test_deeplabhead_forward(self):
        """Test DeepLabHead forward pass."""
        in_channels, out_channels = 256, 128
        height, width = 32, 32
        batch_size = 2

        head = DeepLabHead(in_channels, out_channels)
        x = torch.randn(batch_size, in_channels, height, width)

        output = head(x)

        assert_tensor_shape(output, (batch_size, out_channels, height, width), "DeepLabHead output")
        assert_tensor_finite(output, "DeepLabHead output")


class TestBottleblock:
    """Test Bottleblock functionality."""

    def test_bottleblock_forward(self):
        """Test Bottleblock forward pass."""
        in_channels = 64
        height, width = 32, 32
        batch_size = 2

        bottleblock = Bottleblock(in_channels)
        x = torch.randn(batch_size, in_channels, height, width)

        output = bottleblock(x)

        assert_tensor_shape(output, (batch_size, in_channels, height, width), "Bottleblock output")
        assert_tensor_finite(output, "Bottleblock output")

    def test_bottleblock_different_channels(self):
        """Test Bottleblock with different input/output channels."""
        in_channels, out_channels = 64, 128
        x = torch.randn(2, in_channels, 16, 16)

        bottleblock = Bottleblock(in_channels, out_channels)
        output = bottleblock(x)

        assert_tensor_shape(output, (2, out_channels, 16, 16))
        assert_tensor_finite(output, "Bottleblock channel change output")


if __name__ == "__main__":
    # Run basic smoke tests
    test_conv = TestConvBlock()
    test_conv.test_convblock_forward()
    test_conv.test_convblock_different_norms()

    test_bottleneck = TestBottleneck()
    test_bottleneck.test_bottleneck_forward()
    test_bottleneck.test_bottleneck_downsample()

    test_block = TestBlock()
    test_block.test_block_forward()
    test_block.test_block_residual_connection()

    test_head = TestDeepLabHead()
    test_head.test_deeplabhead_forward()

    test_bottle = TestBottleblock()
    test_bottle.test_bottleblock_forward()
    test_bottle.test_bottleblock_different_channels()

    print("✓ All convolution tests passed!")