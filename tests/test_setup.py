#!/usr/bin/env python3
"""Test script to verify the project setup and configuration system"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all configuration modules can be imported"""
    try:
        from src.config import Config, load_config, save_config, setup_logging, set_random_seeds
        print("✓ Configuration imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config_creation():
    """Test creating a default configuration"""
    try:
        from src.config import Config
        config = Config()
        print("✓ Default configuration created successfully")
        print(f"  - Image size: {config.data.image_size}")
        print(f"  - Model type: {config.model.model_type}")
        print(f"  - Random seed: {config.random_seed}")
        return True
    except Exception as e:
        print(f"✗ Configuration creation error: {e}")
        return False


def test_yaml_loading():
    """Test loading configuration from YAML file"""
    try:
        from src.config import load_config
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            config = load_config(config_path)
            print("✓ YAML configuration loaded successfully")
            print(f"  - Experiment name: {config.experiment_name}")
            print(f"  - Synthetic dataset size: {config.data.synthetic_dataset_size}")
            return True
        else:
            print("✗ Default YAML config file not found")
            return False
    except Exception as e:
        print(f"✗ YAML loading error: {e}")
        return False


def test_directory_creation():
    """Test directory creation functionality"""
    try:
        from src.config import Config
        config = Config()
        config.create_directories()
        
        # Check if directories were created
        required_dirs = [
            config.data_dir,
            config.output_dir,
            config.checkpoint_dir,
            config.logs_dir,
            config.config_dir
        ]
        
        all_exist = all(d.exists() for d in required_dirs)
        if all_exist:
            print("✓ All required directories created successfully")
            return True
        else:
            print("✗ Some directories were not created")
            return False
    except Exception as e:
        print(f"✗ Directory creation error: {e}")
        return False


def test_logging_setup():
    """Test logging configuration"""
    try:
        from src.config import Config, setup_logging
        config = Config()
        logger = setup_logging(config)
        logger.info("Test log message")
        print("✓ Logging setup successful")
        return True
    except Exception as e:
        print(f"✗ Logging setup error: {e}")
        return False


def test_random_seeds():
    """Test random seed setting"""
    try:
        from src.config import set_random_seeds
        import random
        import numpy as np
        
        # Set seeds
        set_random_seeds(42, deterministic=True)
        
        # Test reproducibility
        random_val1 = random.random()
        numpy_val1 = np.random.random()
        
        # Reset seeds
        set_random_seeds(42, deterministic=True)
        
        random_val2 = random.random()
        numpy_val2 = np.random.random()
        
        if random_val1 == random_val2 and numpy_val1 == numpy_val2:
            print("✓ Random seed reproducibility working")
            return True
        else:
            print("✗ Random seed reproducibility failed")
            return False
    except Exception as e:
        print(f"✗ Random seed test error: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Testing Project Setup ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Config Creation", test_config_creation),
        ("YAML Loading", test_yaml_loading),
        ("Directory Creation", test_directory_creation),
        ("Logging Setup", test_logging_setup),
        ("Random Seeds", test_random_seeds)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())