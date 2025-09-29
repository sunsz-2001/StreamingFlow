"""
Unit tests for ODE cell modules.
"""

import torch
from ...cells.gru_ode_cells import SpatialGRUODECell, DualGRUODECell
from ...cells.gru_cells import SpatialGRUCell, DualGRUCell
from ...cells.observation_cell import GRUObservationCell
from ..utils.fixtures import assert_tensor_shape, assert_tensor_finite


class TestSpatialGRUODECell:
    """Test SpatialGRUODECell functionality."""

    def test_spatial_gru_ode_cell_forward(self):
        """Test SpatialGRUODECell forward pass."""
        input_size, hidden_size = 64, 64
        batch_size = 2
        height, width = 16, 16

        cell = SpatialGRUODECell(input_size, hidden_size)

        x = torch.randn(batch_size, input_size, height, width)
        state = torch.randn(batch_size, hidden_size, height, width)

        output = cell(x, state)

        # Output should be the derivative (change in state)
        assert_tensor_shape(output, (batch_size, hidden_size, height, width), "SpatialGRUODECell output")
        assert_tensor_finite(output, "SpatialGRUODECell output")

    def test_spatial_gru_ode_cell_derivative(self):
        """Test that SpatialGRUODECell produces reasonable derivatives."""
        input_size = hidden_size = 32
        cell = SpatialGRUODECell(input_size, hidden_size)

        x = torch.randn(1, input_size, 8, 8)
        state = torch.randn(1, hidden_size, 8, 8)

        # Get derivative
        dh = cell(x, state)

        # Derivative should be reasonable in magnitude
        assert dh.abs().max() < 10, "SpatialGRUODECell derivative too large"
        assert_tensor_finite(dh, "SpatialGRUODECell derivative")


class TestDualGRUODECell:
    """Test DualGRUODECell functionality."""

    def test_dual_gru_ode_cell_forward(self):
        """Test DualGRUODECell forward pass."""
        input_size, hidden_size = 64, 64
        batch_size = 2
        n_present = 3
        height, width = 16, 16

        cell = DualGRUODECell(input_size, hidden_size)

        x = torch.randn(batch_size, 1, input_size, height, width)
        state = torch.randn(batch_size, n_present, hidden_size, height, width)

        output = cell(x, state)

        # Output should be the change in state
        assert_tensor_shape(output, (batch_size, hidden_size, height, width), "DualGRUODECell output")
        assert_tensor_finite(output, "DualGRUODECell output")


class TestSpatialGRUCell:
    """Test SpatialGRUCell functionality."""

    def test_spatial_gru_cell_forward(self):
        """Test SpatialGRUCell forward pass."""
        input_size, hidden_size = 64, 64
        batch_size = 2
        height, width = 16, 16

        cell = SpatialGRUCell(input_size, hidden_size)

        x = torch.randn(batch_size, input_size, height, width)
        state = torch.randn(batch_size, hidden_size, height, width)

        output = cell(x, state)

        # Output should be the updated state
        assert_tensor_shape(output, (batch_size, hidden_size, height, width), "SpatialGRUCell output")
        assert_tensor_finite(output, "SpatialGRUCell output")


class TestDualGRUCell:
    """Test DualGRUCell functionality."""

    def test_dual_gru_cell_forward(self):
        """Test DualGRUCell forward pass."""
        input_size, hidden_size = 64, 64
        batch_size = 2
        n_present = 3
        height, width = 16, 16

        cell = DualGRUCell(input_size, hidden_size)

        x = torch.randn(batch_size, 1, input_size, height, width)
        state = torch.randn(batch_size, n_present, hidden_size, height, width)

        output = cell(x, state)

        # Output should be the updated state
        assert_tensor_shape(output, (batch_size, hidden_size, height, width), "DualGRUCell output")
        assert_tensor_finite(output, "DualGRUCell output")


class TestGRUObservationCell:
    """Test GRUObservationCell functionality."""

    def test_gru_observation_cell_forward(self):
        """Test GRUObservationCell forward pass."""
        input_size, hidden_size = 64, 64
        batch_size = 2
        height, width = 16, 16

        cell = GRUObservationCell(input_size, hidden_size)

        state = torch.randn(batch_size, hidden_size, height, width)
        p = torch.randn(batch_size, hidden_size, height, width)  # Placeholder
        X_obs = torch.randn(batch_size, input_size, height, width)

        updated_state, loss = cell(state, p, X_obs)

        # Check updated state
        assert_tensor_shape(updated_state, (batch_size, hidden_size, height, width),
                          "GRUObservationCell updated state")
        assert_tensor_finite(updated_state, "GRUObservationCell updated state")

        # Loss should be None in current implementation
        assert loss is None, "GRUObservationCell loss should be None"


if __name__ == "__main__":
    # Test SpatialGRUODECell
    test_spatial_ode = TestSpatialGRUODECell()
    test_spatial_ode.test_spatial_gru_ode_cell_forward()
    test_spatial_ode.test_spatial_gru_ode_cell_derivative()

    # Test DualGRUODECell
    test_dual_ode = TestDualGRUODECell()
    test_dual_ode.test_dual_gru_ode_cell_forward()

    # Test SpatialGRUCell
    test_spatial = TestSpatialGRUCell()
    test_spatial.test_spatial_gru_cell_forward()

    # Test DualGRUCell
    test_dual = TestDualGRUCell()
    test_dual.test_dual_gru_cell_forward()

    # Test GRUObservationCell
    test_obs = TestGRUObservationCell()
    test_obs.test_gru_observation_cell_forward()

    print("✓ All cell tests passed!")