"""
Quick smoke test for ODE modules.

This script runs a minimal set of tests to verify basic functionality
without running the full test suite.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Test basic imports
        from streamingflow.ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
        print("✓ Core modules import successfully")

        # Test component imports
        from streamingflow.ode_modules.cells import SpatialGRUODECell, DualGRUODECell
        from streamingflow.ode_modules.utils import ConvBlock, SpatialGRU
        print("✓ Component modules import successfully")

        # Test base interfaces
        from streamingflow.ode_modules.base import ODEPredictor, FuturePredictor
        print("✓ Base interfaces import successfully")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("Testing basic functionality...")

    try:
        from streamingflow.ode_modules.utils.convolutions import ConvBlock
        from streamingflow.ode_modules.cells.gru_ode_cells import SpatialGRUODECell

        # Test ConvBlock
        conv = ConvBlock(32, 64)
        x = torch.randn(1, 32, 8, 8)
        y = conv(x)
        assert y.shape == (1, 64, 8, 8), f"ConvBlock output shape mismatch: {y.shape}"
        print("✓ ConvBlock works")

        # Test SpatialGRUODECell
        cell = SpatialGRUODECell(32, 32)
        x = torch.randn(1, 32, 8, 8)
        state = torch.randn(1, 32, 8, 8)
        dh = cell(x, state)
        assert dh.shape == state.shape, f"SpatialGRUODECell output shape mismatch: {dh.shape}"
        print("✓ SpatialGRUODECell works")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_model_creation():
    """Test that main models can be created."""
    print("Testing model creation...")

    try:
        from streamingflow.ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
        from streamingflow.ode_modules.tests.utils.fixtures import create_mock_config

        cfg = create_mock_config()

        # Test NNFOwithBayesianJumps creation
        ode_model = NNFOwithBayesianJumps(
            input_size=64,
            hidden_size=32,
            cfg=cfg
        )
        print("✓ NNFOwithBayesianJumps created successfully")

        # Test FuturePredictionODE creation
        predictor = FuturePredictionODE(
            in_channels=64,
            latent_dim=32,
            n_future=3,
            cfg=cfg
        )
        print("✓ FuturePredictionODE created successfully")

        return True

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def main():
    """Run quick tests."""
    print("🚀 ODE Modules Quick Test")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Model Creation Test", test_model_creation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*40}")
    print(f"QUICK TEST SUMMARY")
    print(f"{'='*40}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("❌ Some quick tests failed!")
        return False
    else:
        print("✅ All quick tests passed!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)