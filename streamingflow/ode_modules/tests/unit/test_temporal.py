"""
Unit tests for temporal modules.
"""

import torch
from ...utils.temporal import SpatialGRU
from ..utils.fixtures import assert_tensor_shape, assert_tensor_finite


class TestSpatialGRU:
    """Test SpatialGRU functionality."""

    def test_spatial_gru_forward(self):
        """Test SpatialGRU forward pass."""
        input_dim, hidden_dim = 64, 64
        batch_size, seq_len = 2, 4
        height, width = 32, 32

        gru = SpatialGRU(input_dim, hidden_dim)

        # Input sequence [B, T, C, H, W]
        input_sequence = torch.randn(batch_size, seq_len, input_dim, height, width)
        # Hidden state [B, C, H, W]
        hidden_state = torch.randn(batch_size, hidden_dim, height, width)

        output = gru(input_sequence, hidden_state)

        expected_shape = (batch_size, seq_len, hidden_dim, height, width)
        assert_tensor_shape(output, expected_shape, "SpatialGRU output")
        assert_tensor_finite(output, "SpatialGRU output")

    def test_spatial_gru_different_dims(self):
        """Test SpatialGRU with different input/hidden dimensions."""
        input_dim, hidden_dim = 32, 64
        batch_size, seq_len = 1, 3
        height, width = 16, 16

        gru = SpatialGRU(input_dim, hidden_dim)

        input_sequence = torch.randn(batch_size, seq_len, input_dim, height, width)
        hidden_state = torch.randn(batch_size, hidden_dim, height, width)

        output = gru(input_sequence, hidden_state)

        expected_shape = (batch_size, seq_len, hidden_dim, height, width)
        assert_tensor_shape(output, expected_shape, "SpatialGRU different dims output")
        assert_tensor_finite(output, "SpatialGRU different dims output")

    def test_spatial_gru_gate_values(self):
        """Test that GRU gates produce values in expected ranges."""
        input_dim = hidden_dim = 32
        gru = SpatialGRU(input_dim, hidden_dim)

        # Small test case
        input_sequence = torch.randn(1, 2, input_dim, 8, 8)
        hidden_state = torch.randn(1, hidden_dim, 8, 8)

        # Test that forward pass doesn't explode
        output = gru(input_sequence, hidden_state)

        # Check output is reasonable (not too large)
        assert output.abs().max() < 100, "SpatialGRU output values too large"
        assert_tensor_finite(output, "SpatialGRU gate test output")

    def test_spatial_gru_sequential_consistency(self):
        """Test that SpatialGRU processes sequences consistently."""
        input_dim = hidden_dim = 16
        gru = SpatialGRU(input_dim, hidden_dim)

        # Process sequence all at once
        input_sequence = torch.randn(1, 3, input_dim, 4, 4)
        hidden_state = torch.randn(1, hidden_dim, 4, 4)
        output_batch = gru(input_sequence, hidden_state)

        # Process sequence step by step
        h = hidden_state.clone()
        outputs_step = []
        for t in range(3):
            single_input = input_sequence[:, t:t+1]  # [B, 1, C, H, W]
            output_step = gru(single_input, h)
            outputs_step.append(output_step[:, 0])  # Take the single timestep output
            h = output_step[:, 0]  # Update hidden state

        outputs_step = torch.stack(outputs_step, dim=1)

        # Results should be very similar (allowing for small numerical differences)
        assert torch.allclose(output_batch, outputs_step, atol=1e-5), \
            "SpatialGRU batch and step-by-step processing should be consistent"


if __name__ == "__main__":
    test_gru = TestSpatialGRU()
    test_gru.test_spatial_gru_forward()
    test_gru.test_spatial_gru_different_dims()
    test_gru.test_spatial_gru_gate_values()
    test_gru.test_spatial_gru_sequential_consistency()

    print("✓ All temporal tests passed!")