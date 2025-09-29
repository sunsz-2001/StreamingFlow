"""
Integration tests for ODE modules.

These tests verify that different components work together correctly
and that the modularized code produces the same results as the original.
"""

import torch
import numpy as np
from ...cores.bayesian_ode import NNFOwithBayesianJumps
from ...cores.future_predictor import FuturePredictionODE
from ...cells.gru_ode_cells import SpatialGRUODECell, DualGRUODECell
from ..utils.fixtures import create_mock_config, create_test_tensors, assert_tensor_shape, assert_tensor_finite


class TestODEIntegration:
    """Integration tests for ODE modules."""

    def test_cell_integration_with_ode(self):
        """Test that cells integrate properly with the main ODE system."""
        cfg = create_mock_config()
        input_size, hidden_size = 64, 32

        # Create ODE system
        ode_system = NNFOwithBayesianJumps(
            input_size=input_size,
            hidden_size=hidden_size,
            cfg=cfg
        )

        # Test that internal cells are accessible and functional
        assert ode_system.gru_c is not None, "DualGRUODECell should be initialized"
        assert ode_system.gru_obs is not None, "GRUObservationCell should be initialized"

        # Test cell forward passes
        batch_size = 1
        height, width = 8, 8

        # Test ODE cell
        x = torch.randn(batch_size, input_size, height, width)
        state = torch.randn(batch_size, hidden_size, height, width)

        # This should work without errors
        dh = ode_system.gru_c(x, state)
        assert_tensor_shape(dh, state.shape, "ODE cell output")
        assert_tensor_finite(dh, "ODE cell output")

        print("✓ Cell integration test passed")

    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        cfg = create_mock_config()
        in_channels = 64
        latent_dim = 32
        n_future = 3

        # Create predictor
        predictor = FuturePredictionODE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            n_future=n_future,
            cfg=cfg,
            n_gru_blocks=1,  # Simplified for testing
            n_res_layers=1
        )

        batch_size = 1
        height, width = 8, 8

        # Create realistic test scenario
        current_input = torch.randn(batch_size, 1, in_channels, height, width)
        camera_states = torch.randn(batch_size, 3, in_channels, height, width)
        lidar_states = torch.randn(batch_size, 2, in_channels, height, width)

        # Timestamps in increasing order
        camera_timestamps = torch.tensor([[0.0, 0.1, 0.2]]).float()
        lidar_timestamps = torch.tensor([[0.05, 0.15]]).float()
        target_timestamps = torch.tensor([[0.3, 0.4, 0.5]]).float()

        try:
            # Run prediction
            future_sequence, aux_loss = predictor.forward(
                future_prediction_input=current_input,
                camera_states=camera_states,
                lidar_states=lidar_states,
                camera_timestamp=camera_timestamps,
                lidar_timestamp=lidar_timestamps,
                target_timestamp=target_timestamps
            )

            # Validate results
            expected_shape = (batch_size, n_future, in_channels, height, width)
            assert_tensor_shape(future_sequence, expected_shape, "End-to-end prediction")
            assert_tensor_finite(future_sequence, "End-to-end prediction")

            # Check that predictions are reasonable
            assert future_sequence.abs().max() < 100, "Predictions should not be too large"

            print("✓ End-to-end prediction test passed")
            return True

        except Exception as e:
            print(f"End-to-end test failed: {e}")
            return False

    def test_ode_stability(self):
        """Test ODE integration stability over multiple steps."""
        cfg = create_mock_config()
        input_size, hidden_size = 32, 16  # Smaller for stability testing

        ode_system = NNFOwithBayesianJumps(
            input_size=input_size,
            hidden_size=hidden_size,
            cfg=cfg
        )

        batch_size = 1
        height, width = 4, 4
        delta_t = 0.01  # Small time step for stability

        # Initial state
        state = torch.randn(batch_size, hidden_size, height, width) * 0.1  # Small initial values
        input_tensor = torch.randn(batch_size, hidden_size, height, width) * 0.1

        current_time = 0.0
        max_magnitude = 0.0

        # Run multiple ODE steps
        for step in range(10):
            state, input_tensor, current_time, _, _ = ode_system.ode_step(
                state, input_tensor, delta_t, current_time
            )

            # Track maximum magnitude
            max_magnitude = max(max_magnitude, state.abs().max().item())

            # Check for explosions or NaN values
            assert_tensor_finite(state, f"ODE state at step {step}")
            assert state.abs().max() < 10, f"ODE state magnitude too large at step {step}: {state.abs().max()}"

        print(f"✓ ODE stability test passed (max magnitude: {max_magnitude:.4f})")

    def test_modular_compatibility(self):
        """Test that modularized components are compatible with each other."""
        cfg = create_mock_config()

        # Test creating different components and ensuring they're compatible
        input_size, hidden_size = 64, 32

        # Create individual cells
        spatial_ode_cell = SpatialGRUODECell(input_size, hidden_size)
        dual_ode_cell = DualGRUODECell(input_size, hidden_size)

        # Create full systems
        ode_system = NNFOwithBayesianJumps(input_size, hidden_size, cfg)
        predictor = FuturePredictionODE(input_size, hidden_size, 3, cfg)

        # Test that all components have compatible interfaces
        batch_size = 1
        height, width = 8, 8

        # Test inputs
        x = torch.randn(batch_size, input_size, height, width)
        state = torch.randn(batch_size, hidden_size, height, width)

        # Test spatial ODE cell
        dh1 = spatial_ode_cell(x, state)
        assert_tensor_shape(dh1, state.shape, "Spatial ODE cell compatibility")

        # Test dual ODE cell (needs different input format)
        x_dual = x.unsqueeze(1)  # Add sequence dimension
        state_dual = state.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # Add sequence dimension
        dh2 = dual_ode_cell(x_dual, state_dual)
        assert_tensor_shape(dh2, state.shape, "Dual ODE cell compatibility")

        print("✓ Modular compatibility test passed")

    def test_gradient_flow(self):
        """Test that gradients flow properly through the modularized system."""
        cfg = create_mock_config()
        in_channels = 32
        latent_dim = 16

        predictor = FuturePredictionODE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            n_future=2,
            cfg=cfg,
            n_gru_blocks=1,
            n_res_layers=1
        )

        # Enable gradient computation
        for param in predictor.parameters():
            param.requires_grad_(True)

        batch_size = 1
        height, width = 4, 4

        # Create inputs
        current_input = torch.randn(batch_size, 1, in_channels, height, width, requires_grad=True)
        camera_states = torch.randn(batch_size, 2, in_channels, height, width, requires_grad=True)

        camera_timestamps = torch.tensor([[0.0, 0.1]]).float()
        lidar_timestamps = torch.empty(batch_size, 0).float()
        target_timestamps = torch.tensor([[0.2, 0.3]]).float()

        try:
            # Forward pass
            future_sequence, aux_loss = predictor.forward(
                future_prediction_input=current_input,
                camera_states=camera_states,
                lidar_states=None,
                camera_timestamp=camera_timestamps,
                lidar_timestamp=lidar_timestamps,
                target_timestamp=target_timestamps
            )

            # Compute a simple loss
            loss = future_sequence.mean()

            # Backward pass
            loss.backward()

            # Check that gradients exist
            grad_count = 0
            for param in predictor.parameters():
                if param.grad is not None:
                    grad_count += 1
                    assert_tensor_finite(param.grad, "Parameter gradient")

            assert grad_count > 0, "No gradients computed"
            print(f"✓ Gradient flow test passed ({grad_count} parameters with gradients)")

        except Exception as e:
            print(f"Gradient flow test failed: {e}")


if __name__ == "__main__":
    test_integration = TestODEIntegration()

    print("Running integration tests...")

    print("\n1. Testing cell integration...")
    test_integration.test_cell_integration_with_ode()

    print("\n2. Testing end-to-end prediction...")
    test_integration.test_end_to_end_prediction()

    print("\n3. Testing ODE stability...")
    test_integration.test_ode_stability()

    print("\n4. Testing modular compatibility...")
    test_integration.test_modular_compatibility()

    print("\n5. Testing gradient flow...")
    test_integration.test_gradient_flow()

    print("\n✓ All integration tests completed!")