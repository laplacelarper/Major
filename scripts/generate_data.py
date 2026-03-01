#!/usr/bin/env python
"""
Standalone synthetic data generation script

Requirements: 1.1, 6.5
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, setup_logging, set_seed
from src.physics import SonarImageGenerator

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic sonar images'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='Number of images to generate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Directory to save generated images'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Statistics and validation
    parser.add_argument(
        '--compute-stats',
        action='store_true',
        help='Compute and save dataset statistics'
    )
    
    parser.add_argument(
        '--validate-quality',
        action='store_true',
        help='Validate data quality'
    )
    
    # Visualization
    parser.add_argument(
        '--save-samples',
        type=int,
        default=10,
        help='Number of sample images to save with metadata'
    )
    
    return parser.parse_args()


def compute_dataset_statistics(images, metadata_list):
    """Compute statistics for generated dataset"""
    import numpy as np
    
    stats = {
        'num_images': len(images),
        'image_shape': images[0].shape if len(images) > 0 else None,
        'intensity_stats': {
            'mean': float(np.mean([img.mean() for img in images])),
            'std': float(np.std([img.mean() for img in images])),
            'min': float(np.min([img.min() for img in images])),
            'max': float(np.max([img.max() for img in images]))
        },
        'physics_parameters': {}
    }
    
    # Compute statistics for each physics parameter
    if metadata_list:
        param_names = metadata_list[0].keys()
        for param_name in param_names:
            values = [m[param_name] for m in metadata_list]
            stats['physics_parameters'][param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return stats


def validate_data_quality(images, metadata_list):
    """Validate quality of generated data"""
    import numpy as np
    
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check for NaN or Inf values
    for i, img in enumerate(images):
        if np.isnan(img).any():
            validation['errors'].append(f"Image {i} contains NaN values")
            validation['is_valid'] = False
        
        if np.isinf(img).any():
            validation['errors'].append(f"Image {i} contains Inf values")
            validation['is_valid'] = False
    
    # Check intensity range
    for i, img in enumerate(images):
        if img.min() < 0 or img.max() > 1:
            validation['warnings'].append(f"Image {i} intensity out of [0, 1] range")
    
    # Check for constant images
    for i, img in enumerate(images):
        if img.std() < 1e-6:
            validation['warnings'].append(f"Image {i} appears to be constant")
    
    # Check metadata completeness
    if metadata_list:
        expected_keys = metadata_list[0].keys()
        for i, metadata in enumerate(metadata_list):
            if set(metadata.keys()) != set(expected_keys):
                validation['errors'].append(f"Metadata {i} has inconsistent keys")
                validation['is_valid'] = False
    
    return validation


def main():
    """Main data generation execution"""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config
    config.data.synthetic_dataset_size = args.num_images
    config.random_seed = args.seed
    
    # Setup logging
    setup_logging(config)
    logger.info("=" * 80)
    logger.info("SYNTHETIC DATA GENERATION")
    logger.info("=" * 80)
    
    logger.info(f"Number of images: {args.num_images}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create generator
        logger.info("\nInitializing generator...")
        generator = SonarImageGenerator(config.physics)
        logger.info("✓ Generator initialized")
        
        # Generate images
        logger.info(f"\nGenerating {args.num_images} images...")
        
        images = []
        metadata_list = []
        
        for i in range(args.num_images):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  Progress: {i + 1}/{args.num_images}")
            
            # Generate image
            image, metadata = generator.generate()
            images.append(image)
            metadata_list.append(metadata)
            
            # Save image
            image_path = output_dir / f"image_{i:05d}.png"
            generator.save_image(image, image_path)
            
            # Save metadata
            metadata_path = output_dir / f"metadata_{i:05d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info("✓ Generation complete")
        
        # Compute statistics
        if args.compute_stats:
            logger.info("\nComputing dataset statistics...")
            stats = compute_dataset_statistics(images, metadata_list)
            
            stats_path = output_dir / 'dataset_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✓ Statistics saved to: {stats_path}")
            
            # Print summary
            logger.info("\nDataset Statistics:")
            logger.info(f"  Number of images: {stats['num_images']}")
            logger.info(f"  Image shape: {stats['image_shape']}")
            logger.info(f"  Mean intensity: {stats['intensity_stats']['mean']:.4f}")
            logger.info(f"  Std intensity: {stats['intensity_stats']['std']:.4f}")
        
        # Validate quality
        if args.validate_quality:
            logger.info("\nValidating data quality...")
            validation = validate_data_quality(images, metadata_list)
            
            if validation['is_valid']:
                logger.info("✓ Data quality validation passed")
            else:
                logger.error("✗ Data quality validation failed")
                for error in validation['errors']:
                    logger.error(f"  - {error}")
            
            if validation['warnings']:
                logger.warning("Warnings:")
                for warning in validation['warnings']:
                    logger.warning(f"  - {warning}")
            
            validation_path = output_dir / 'quality_validation.json'
            with open(validation_path, 'w') as f:
                json.dump(validation, f, indent=2)
            
            logger.info(f"Validation report saved to: {validation_path}")
        
        # Save sample images with metadata
        if args.save_samples > 0:
            logger.info(f"\nSaving {args.save_samples} sample visualizations...")
            
            samples_dir = output_dir / 'samples'
            samples_dir.mkdir(exist_ok=True)
            
            num_samples = min(args.save_samples, len(images))
            
            for i in range(num_samples):
                sample_info = {
                    'image_path': f"image_{i:05d}.png",
                    'metadata': metadata_list[i],
                    'statistics': {
                        'mean': float(images[i].mean()),
                        'std': float(images[i].std()),
                        'min': float(images[i].min()),
                        'max': float(images[i].max())
                    }
                }
                
                sample_path = samples_dir / f"sample_{i:03d}.json"
                with open(sample_path, 'w') as f:
                    json.dump(sample_info, f, indent=2)
            
            logger.info(f"✓ Samples saved to: {samples_dir}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Generated {len(images)} images")
        logger.info(f"Saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("\nData generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nData generation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
