# ODE Modules Test Suite

This directory contains comprehensive tests for the `ode_modules` package to ensure functionality and correctness after modularization.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_convolutions.py    # Convolution layer tests
│   ├── test_temporal.py        # Temporal processing tests
│   ├── test_cells.py           # ODE cell tests
│   ├── test_bayesian_ode.py    # NNFOwithBayesianJumps tests
│   └── test_future_predictor.py # FuturePredictionODE tests
├── integration/            # Integration tests
│   └── test_ode_integration.py  # End-to-end integration tests
├── utils/                  # Test utilities
│   └── fixtures.py             # Test fixtures and helpers
├── run_tests.py           # Main test runner
├── quick_test.py          # Quick smoke test
└── README.md              # This file
```

## Running Tests

### Quick Test (Recommended for first-time verification)

```bash
cd streamingflow/ode_modules/tests
python quick_test.py
```

This runs a minimal set of smoke tests to verify basic functionality.

### Full Test Suite

```bash
cd streamingflow/ode_modules/tests
python run_tests.py
```

This runs all unit tests and integration tests with detailed reporting.

### Individual Test Modules

You can also run individual test files:

```bash
# Test convolution components
python unit/test_convolutions.py

# Test temporal components
python unit/test_temporal.py

# Test ODE cells
python unit/test_cells.py

# Test Bayesian ODE
python unit/test_bayesian_ode.py

# Test Future Predictor
python unit/test_future_predictor.py

# Test integration
python integration/test_ode_integration.py
```

## Test Categories

### Unit Tests

- **Convolution Tests**: Verify that all convolution components (ConvBlock, Bottleneck, Block, etc.) work correctly
- **Temporal Tests**: Test temporal processing components like SpatialGRU
- **Cell Tests**: Test individual ODE cells (SpatialGRUODECell, DualGRUODECell, etc.)
- **Bayesian ODE Tests**: Test the core NNFOwithBayesianJumps functionality
- **Future Predictor Tests**: Test the FuturePredictionODE module

### Integration Tests

- **End-to-End Pipeline**: Test complete prediction pipeline from input to output
- **Component Interaction**: Verify that different components work together correctly
- **Gradient Flow**: Ensure gradients flow properly through the modularized system
- **Stability Tests**: Check numerical stability of ODE integration

## Test Coverage

The tests cover:

- ✅ Module imports and initialization
- ✅ Forward pass functionality
- ✅ Tensor shape validation
- ✅ Numerical stability checks
- ✅ Gradient computation
- ✅ Integration between components
- ✅ Error handling

## Requirements

The tests require:
- PyTorch
- NumPy
- The `ode_modules` package

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and that the parent directories are in the Python path.

2. **CUDA Errors**: Tests are designed to run on CPU. If you encounter CUDA-related errors, make sure tensors are on CPU.

3. **Memory Issues**: Some tests use relatively small tensor sizes to avoid memory issues. If you encounter OOM errors, try reducing tensor sizes in the test fixtures.

### Expected Behavior

- Most tests should pass consistently
- Some complex integration tests might occasionally fail due to numerical precision issues
- The quick test should always pass if the modules are correctly installed

## Adding New Tests

When adding new components to `ode_modules`, please add corresponding tests:

1. Create unit tests in the appropriate `test_*.py` file
2. Add integration tests if the component interacts with other modules
3. Update this README if new test categories are added
4. Ensure new tests follow the existing patterns and use the test utilities

## Notes

- Tests use mock configurations to avoid dependency on actual StreamingFlow configs
- Tensor sizes are kept small for fast execution
- Tests focus on correctness rather than performance
- Some tests may print warnings or informational messages - this is normal