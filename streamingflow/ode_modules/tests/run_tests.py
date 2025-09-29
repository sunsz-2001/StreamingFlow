"""
Test runner for ODE modules.

This script runs all tests for the ode_modules package and provides
a comprehensive test report.
"""

import sys
import traceback
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def run_test_module(module_path, module_name):
    """Run a test module and return results."""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print(f"{'='*60}")

    start_time = time.time()
    success = True
    error_msg = None

    try:
        # Import and run the test module
        exec(open(module_path).read())
        print(f"✓ {module_name} completed successfully")
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"✗ {module_name} failed: {error_msg}")
        print(f"Full traceback:")
        traceback.print_exc()

    end_time = time.time()
    duration = end_time - start_time

    return {
        'name': module_name,
        'success': success,
        'duration': duration,
        'error': error_msg
    }


def main():
    """Run all tests and generate report."""
    print("🧪 ODE Modules Test Suite")
    print("=" * 60)

    # Define test modules
    test_dir = Path(__file__).parent
    test_modules = [
        # Unit tests
        (test_dir / "unit" / "test_convolutions.py", "Convolution Tests"),
        (test_dir / "unit" / "test_temporal.py", "Temporal Tests"),
        (test_dir / "unit" / "test_cells.py", "Cell Tests"),
        (test_dir / "unit" / "test_bayesian_ode.py", "Bayesian ODE Tests"),
        (test_dir / "unit" / "test_future_predictor.py", "Future Predictor Tests"),

        # Integration tests
        (test_dir / "integration" / "test_ode_integration.py", "Integration Tests"),
    ]

    # Run all tests
    results = []
    total_start_time = time.time()

    for module_path, module_name in test_modules:
        if module_path.exists():
            result = run_test_module(module_path, module_name)
            results.append(result)
        else:
            print(f"⚠️  Warning: {module_path} not found, skipping...")

    total_duration = time.time() - total_start_time

    # Generate report
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed

    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.2f}s")

    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if not result['success']:
                print(f"  - {result['name']}: {result['error']}")

    # Detailed results
    print(f"\nDETAILED RESULTS:")
    print("-" * 60)
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{result['name']:<30} {status:>10} ({result['duration']:.2f}s)")

    # Exit with appropriate code
    if failed > 0:
        print(f"\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()