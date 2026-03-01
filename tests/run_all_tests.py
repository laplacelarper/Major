#!/usr/bin/env python3
"""
Test runner for the comprehensive testing suite.

This script runs all unit and integration tests without requiring pytest.
It provides a simple test framework using Python's unittest module.
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_and_run_tests():
    """Discover and run all tests in the tests directory"""
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests
    test_dir = Path(__file__).parent
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(discover_and_run_tests())
