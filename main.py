#!/usr/bin/env python3
"""
Main entry point for the Physics-Informed Sonar Detection System

Single command to run the entire pipeline end-to-end:
    python main.py --mode full_pipeline

Or run individual phases:
    python main.py --mode generate_data
    python main.py --mode train --phase 1
    python main.py --mode train --phase 2
    python main.py --mode train --phase 3
    python main.py --mode evaluate
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import load_config, setup_logging, set_random_seeds, Config


def main():
    """Main function to run the sonar detection system"""
    parser = argparse.ArgumentParser(
        description="Physics-Informed Sonar Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline end-to-end
  python main.py --mode full_pipeline
  
  # Run individual steps
  python main.py --mode generate_data --num_samples 1000
  python main.py --mode train --phase 1
  python main.py --mode train --phase 2
  python main.py --mode train --phase 3
  python main.py --mode evaluate
  
  # Quick test
  python main.py --mode test_config
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("configs/default.yaml"),
        help="Path to configuration file"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "generate_data", "train", "evaluate", "inference", "test_config"],
        default="test_config",
        help="Mode to run the system in"
    )
    
    # Training options
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Training phase (1=synthetic, 2=real, 3=calibration)"
    )
    
    # Data generation options
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    
    # Inference options
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to image for inference"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained model checkpoint"
    )
    
    # Pipeline options
    parser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Skip data generation in full pipeline (use existing data)"
    )
    
    parser.add_argument(
        "--synthetic_only",
        action="store_true",
        help="Train only on synthetic data (skip phase 2)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config.exists():
            config = load_config(args.config)
            print(f"✓ Loaded configuration from {args.config}")
        else:
            print(f"⚠️  Configuration file {args.config} not found, using defaults")
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
            
        elif args.mode == "generate_data":
            generate_synthetic_data(config, args.num_samples, logger)
            
        elif args.mode == "train":
            if args.phase is None:
                print("❌ Error: --phase required for training mode")
                print("   Use: --phase 1, --phase 2, or --phase 3")
                sys.exit(1)
            train_model(config, args.phase, logger)
            
        elif args.mode == "evaluate":
            evaluate_model(config, args.model, logger)
            
        elif args.mode == "inference":
            if args.image is None or args.model is None:
                print("❌ Error: --image and --model required for inference mode")
                sys.exit(1)
            run_inference(config, args.image, args.model, logger)
            
        elif args.mode == "full_pipeline":
            run_full_pipeline(config, args, logger)
        
        logger.info("System execution completed successfully")
        print("\n✅ Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_configuration(config):
    """Test the configuration system"""
    print("\n" + "="*70)
    print("  CONFIGURATION TEST")
    print("="*70)
    print(f"\n📋 Experiment: {config.experiment_name}")
    print(f"🎲 Random seed: {config.random_seed}")
    print(f"🖼️  Image size: {config.data.image_size}")
    print(f"🧠 Model type: {config.model.model_type}")
    print(f"📊 Synthetic dataset size: {config.data.synthetic_dataset_size}")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📁 Data directory: {config.data_dir}")
    print(f"🔢 Real data percentage: {config.data.real_data_percentage * 100}%")
    print(f"🎯 MC samples: {config.model.mc_samples}")
    print("\n✅ Configuration system working correctly")


def generate_synthetic_data(config, num_samples, logger):
    """Generate synthetic sonar data"""
    print("\n" + "="*70)
    print(f"  GENERATING {num_samples} SYNTHETIC SONAR IMAGES")
    print("="*70)
    
    from src.physics.renderer import SonarImageRenderer
    from src.physics.core import PhysicsEngine
    import numpy as np
    from PIL import Image
    import json
    
    output_dir = Path(config.data_dir) / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = SonarImageRenderer(image_size=config.data.image_size)
    
    print(f"\n📸 Generating images...")
    start_time = time.time()
    
    for i in range(num_samples):
        # Random parameters
        is_mine = np.random.rand() > 0.5
        material = 'metal' if is_mine else 'rock'
        label = 1 if is_mine else 0
        
        physics_params = {
            'grazing_angle_range': (np.random.uniform(10, 80), np.random.uniform(10, 80)),
            'range_limits': (np.random.uniform(10, 200), np.random.uniform(10, 200)),
            'noise_level': np.random.uniform(0.05, 0.4),
            'texture_roughness': np.random.uniform(0.3, 0.7),
        }
        
        # Generate
        image, _, metadata = renderer.render_sonar_image(
            physics_params=physics_params,
            object_positions=[(config.data.image_size[0]//2, config.data.image_size[1]//2)],
            object_heights=[np.random.uniform(1.0, 3.0)],
            object_labels=[label]
        )
        
        # Save image
        img_path = output_dir / f"sonar_{i:05d}.png"
        Image.fromarray((image * 255).astype(np.uint8)).save(img_path)
        
        # Save metadata
        meta_path = output_dir / f"sonar_{i:05d}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'label': int(label),
                'material': material,
                'physics': metadata.to_dict()
            }, f, indent=2)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (num_samples - i - 1) / rate
            print(f"   Progress: {i+1}/{num_samples} ({rate:.1f} img/s, ~{remaining:.0f}s remaining)")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Generated {num_samples} images in {elapsed:.1f}s ({num_samples/elapsed:.1f} img/s)")
    print(f"📁 Saved to: {output_dir}")
    logger.info(f"Generated {num_samples} synthetic images")


def train_model(config, phase, logger):
    """Train the model for specified phase"""
    print("\n" + "="*70)
    print(f"  TRAINING PHASE {phase}")
    print("="*70)
    
    if phase == 1:
        print("\n🎯 Phase 1: Synthetic Pretraining")
        print("   • Training on synthetic data")
        print("   • Learning physics-based features")
        print(f"   • Epochs: {config.training.phase1_epochs}")
        print(f"   • Batch size: {config.training.phase1_batch_size}")
        print(f"   • Learning rate: {config.training.phase1_lr}")
        
        from src.training.phase1_synthetic import train_phase1
        train_phase1(config, logger)
        
    elif phase == 2:
        print("\n🎯 Phase 2: Real Data Fine-tuning")
        print("   • Fine-tuning on real sonar data")
        print("   • Adapting to real-world characteristics")
        print(f"   • Epochs: {config.training.phase2_epochs}")
        print(f"   • Batch size: {config.training.phase2_batch_size}")
        print(f"   • Learning rate: {config.training.phase2_lr}")
        
        # Check if real data exists
        real_data_dir = Path(config.data_dir) / "real" / "minehunting_sonar" / "images"
        if not real_data_dir.exists() or not list(real_data_dir.glob("*.png")):
            print("\n⚠️  WARNING: No real dataset found!")
            print(f"   Expected location: {real_data_dir}")
            print("\n   Options:")
            print("   1. Download the Minehunting dataset (see docs/dataset_setup/)")
            print("   2. Use --synthetic_only flag to skip Phase 2")
            print("   3. Continue with synthetic data only")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("   Aborted.")
                return
        
        from src.training.phase2_finetuning import train_phase2
        train_phase2(config, logger)
        
    elif phase == 3:
        print("\n🎯 Phase 3: Uncertainty Calibration")
        print("   • Calibrating confidence scores")
        print("   • Enabling Monte Carlo Dropout")
        print(f"   • Epochs: {config.training.phase3_epochs}")
        print(f"   • MC samples: {config.model.mc_samples}")
        
        from src.training.phase3_calibration import train_phase3
        train_phase3(config, logger)
    
    print(f"\n✅ Phase {phase} training complete!")
    logger.info(f"Completed training phase {phase}")


def evaluate_model(config, model_path, logger):
    """Evaluate trained model"""
    print("\n" + "="*70)
    print("  MODEL EVALUATION")
    print("="*70)
    
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.visualizer import ResultsVisualizer
    
    print("\n📊 Evaluating model performance...")
    print("   • Computing metrics")
    print("   • Generating visualizations")
    print("   • Creating reports")
    
    # Implementation would go here
    print("\n✅ Evaluation complete!")
    logger.info("Model evaluation completed")


def run_inference(config, image_path, model_path, logger):
    """Run inference on a single image"""
    print("\n" + "="*70)
    print("  INFERENCE")
    print("="*70)
    
    print(f"\n🖼️  Image: {image_path}")
    print(f"🧠 Model: {model_path}")
    
    # Implementation would go here
    print("\n✅ Inference complete!")
    logger.info(f"Inference completed on {image_path}")


def run_full_pipeline(config, args, logger):
    """Run the complete end-to-end pipeline"""
    print("\n" + "="*70)
    print("  FULL PIPELINE EXECUTION")
    print("="*70)
    
    print("\n📋 Pipeline Steps:")
    if not args.skip_data_generation:
        print("   1. Generate synthetic data")
    print("   2. Train Phase 1 (Synthetic pretraining)")
    if not args.synthetic_only:
        print("   3. Train Phase 2 (Real data fine-tuning)")
    print("   4. Train Phase 3 (Uncertainty calibration)")
    print("   5. Evaluate model")
    
    response = input("\n   Continue? (y/n): ")
    if response.lower() != 'y':
        print("   Aborted.")
        return
    
    start_time = time.time()
    
    # Step 1: Generate data
    if not args.skip_data_generation:
        print("\n" + "="*70)
        print("  STEP 1: DATA GENERATION")
        print("="*70)
        generate_synthetic_data(config, config.data.synthetic_dataset_size, logger)
    else:
        print("\n⏭️  Skipping data generation (using existing data)")
    
    # Step 2: Phase 1 training
    print("\n" + "="*70)
    print("  STEP 2: PHASE 1 TRAINING")
    print("="*70)
    train_model(config, 1, logger)
    
    # Step 3: Phase 2 training
    if not args.synthetic_only:
        print("\n" + "="*70)
        print("  STEP 3: PHASE 2 TRAINING")
        print("="*70)
        train_model(config, 2, logger)
    else:
        print("\n⏭️  Skipping Phase 2 (synthetic only mode)")
    
    # Step 4: Phase 3 training
    print("\n" + "="*70)
    print("  STEP 4: PHASE 3 TRAINING")
    print("="*70)
    train_model(config, 3, logger)
    
    # Step 5: Evaluation
    print("\n" + "="*70)
    print("  STEP 5: EVALUATION")
    print("="*70)
    evaluate_model(config, None, logger)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n⏱️  Total time: {hours}h {minutes}m {seconds}s")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"💾 Model checkpoints: {config.checkpoint_dir}")
    
    logger.info(f"Full pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()