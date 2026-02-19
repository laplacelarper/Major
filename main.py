#!/usr/bin/env python3
"""Main entry point for the Physics-Informed Sonar Detection System"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import load_config, setup_logging, set_random_seeds, Config


def main():
    """Main function to run the sonar detection system"""
    parser = argparse.ArgumentParser(
        description="Physics-Informed Sonar Object Detection System"
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("configs/default.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "generate_data", "test_config"],
        default="test_config",
        help="Mode to run the system in"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config.exists():
            config = load_config(args.config)
            print(f"✓ Loaded configuration from {args.config}")
        else:
            print(f"Configuration file {args.config} not found, using defaults")
            config = Config()
        
        # Create necessary directories
        config.create_directories()
        print("✓ Created project directories")
        
        # Setup logging and random seeds
        logger = setup_logging(config)
        set_random_seeds(config.random_seed, config.deterministic)
        print("✓ Initialized logging and random seeds")
        
        # Run based on mode
        if args.mode == "test_config":
            test_configuration(config)
        elif args.mode == "train":
            print("Training mode not yet implemented")
        elif args.mode == "evaluate":
            print("Evaluation mode not yet implemented")
        elif args.mode == "generate_data":
            print("Data generation mode not yet implemented")
        
        logger.info("System initialization completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def test_configuration(config):
    """Test the configuration system"""
    print("\n=== Configuration Test ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Random seed: {config.random_seed}")
    print(f"Image size: {config.data.image_size}")
    print(f"Model type: {config.model.model_type}")
    print(f"Synthetic dataset size: {config.data.synthetic_dataset_size}")
    print(f"Output directory: {config.output_dir}")
    print(f"Data directory: {config.data_dir}")
    print("✓ Configuration system working correctly")


if __name__ == "__main__":
    main()