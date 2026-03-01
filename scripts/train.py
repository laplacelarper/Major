#!/usr/bin/env python
"""
Main training script that orchestrates all three phases

Requirements: 6.3, 6.4
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, setup_logging, set_seed
from src.training import (
    Phase1SyntheticTrainer,
    Phase2FineTuningTrainer,
    Phase3CalibrationTrainer
)
from src.validation import validate_configuration

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train sonar detection model with three-phase pipeline'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    # Phase selection
    parser.add_argument(
        '--phases',
        type=str,
        default='1,2,3',
        help='Phases to run (comma-separated, e.g., "1,2,3" or "2,3")'
    )
    
    # Checkpoint loading
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use for training'
    )
    
    # Validation
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    # Dry run
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without training'
    )
    
    return parser.parse_args()


def main():
    """Main training execution"""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    if args.seed is not None:
        config.random_seed = args.seed
    
    if args.device:
        config.device = args.device
    
    # Setup logging
    setup_logging(config)
    logger.info("=" * 80)
    logger.info("SONAR DETECTION MODEL TRAINING")
    logger.info("=" * 80)
    
    # Validate configuration
    logger.info("Validating configuration...")
    validation = validate_configuration(config)
    
    if not validation['is_valid']:
        logger.error("Configuration validation failed:")
        for error in validation['errors']:
            logger.error(f"  - {error}")
        if args.validate_config:
            sys.exit(1)
        else:
            logger.warning("Continuing despite validation errors...")
    else:
        logger.info("✓ Configuration validation passed")
    
    if args.validate_config:
        logger.info("Configuration validation complete. Exiting.")
        sys.exit(0)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model: {config.model.model_type}")
    logger.info(f"  Output mode: {config.model.output_mode}")
    logger.info(f"  Uncertainty: {config.model.use_uncertainty}")
    logger.info(f"  Random seed: {config.random_seed}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Output dir: {config.output_dir}")
    
    if args.dry_run:
        logger.info("\nDry run complete. Exiting.")
        sys.exit(0)
    
    # Create directories
    config.create_directories()
    
    # Set random seed
    set_seed(config.random_seed)
    logger.info(f"Random seed set to: {config.random_seed}")
    
    # Parse phases to run
    phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
    logger.info(f"Phases to run: {phases_to_run}")
    
    # Initialize checkpoint path
    checkpoint_path = args.resume
    
    try:
        # Phase 1: Synthetic pretraining
        if 1 in phases_to_run:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: SYNTHETIC PRETRAINING")
            logger.info("=" * 80)
            
            phase1_trainer = Phase1SyntheticTrainer(config)
            
            if checkpoint_path:
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                phase1_trainer.load_checkpoint(checkpoint_path)
            
            phase1_results = phase1_trainer.train()
            checkpoint_path = phase1_results.get('best_checkpoint')
            
            logger.info("Phase 1 complete!")
            logger.info(f"  Best validation loss: {phase1_results.get('best_val_loss', 'N/A'):.4f}")
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Phase 2: Real data fine-tuning
        if 2 in phases_to_run:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: REAL DATA FINE-TUNING")
            logger.info("=" * 80)
            
            if not checkpoint_path:
                logger.error("No checkpoint available for Phase 2. Run Phase 1 first or provide --resume")
                sys.exit(1)
            
            phase2_trainer = Phase2FineTuningTrainer(config)
            phase2_trainer.load_checkpoint(checkpoint_path)
            
            phase2_results = phase2_trainer.train()
            checkpoint_path = phase2_results.get('best_checkpoint')
            
            logger.info("Phase 2 complete!")
            logger.info(f"  Best validation loss: {phase2_results.get('best_val_loss', 'N/A'):.4f}")
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Phase 3: Uncertainty calibration
        if 3 in phases_to_run:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3: UNCERTAINTY CALIBRATION")
            logger.info("=" * 80)
            
            if not checkpoint_path:
                logger.error("No checkpoint available for Phase 3. Run Phases 1-2 first or provide --resume")
                sys.exit(1)
            
            phase3_trainer = Phase3CalibrationTrainer(config)
            phase3_trainer.load_checkpoint(checkpoint_path)
            
            phase3_results = phase3_trainer.train()
            checkpoint_path = phase3_results.get('best_checkpoint')
            
            logger.info("Phase 3 complete!")
            logger.info(f"  Expected Calibration Error: {phase3_results.get('ece', 'N/A'):.4f}")
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final model checkpoint: {checkpoint_path}")
        logger.info(f"Logs saved to: {config.logs_dir}")
        logger.info(f"Outputs saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
