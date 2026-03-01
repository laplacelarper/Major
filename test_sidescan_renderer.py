#!/usr/bin/env python3
"""
Test script for the new side-scan sonar renderer
Generates sample images and compares them with real data
"""

import numpy as np
from pathlib import Path
from src.physics.sidescan_renderer import SideScanRenderer, SideScanParams, generate_realistic_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_rendering():
    """Test basic image rendering"""
    logger.info("Testing basic side-scan sonar rendering...")
    
    renderer = SideScanRenderer()
    
    # Test 1: Empty seabed (no objects)
    image, label = renderer.render(objects=None, random_seed=42)
    assert image.shape == (512, 512, 3), f"Expected shape (512, 512, 3), got {image.shape}"
    assert image.dtype == np.uint8, f"Expected dtype uint8, got {image.dtype}"
    assert label == 0, f"Expected label 0 for empty seabed, got {label}"
    logger.info("✓ Empty seabed rendering works")
    
    # Test 2: Single rock
    objects = [{'type': 'rock', 'x': 0.5, 'y': 0.5, 'size': 15}]
    image, label = renderer.render(objects=objects, random_seed=42)
    assert label == 0, f"Expected label 0 for rock, got {label}"
    logger.info("✓ Rock rendering works")
    
    # Test 3: Single mine
    objects = [{'type': 'mine', 'x': 0.5, 'y': 0.5, 'size': 15}]
    image, label = renderer.render(objects=objects, random_seed=42)
    assert label == 1, f"Expected label 1 for mine, got {label}"
    logger.info("✓ Mine rendering works")
    
    # Test 4: Multiple objects
    objects = [
        {'type': 'rock', 'x': 0.3, 'y': 0.3, 'size': 12},
        {'type': 'mine', 'x': 0.7, 'y': 0.7, 'size': 18},
        {'type': 'rock', 'x': 0.5, 'y': 0.8, 'size': 10}
    ]
    image, label = renderer.render(objects=objects, random_seed=42)
    assert label == 1, f"Expected label 1 (mine present), got {label}"
    logger.info("✓ Multiple objects rendering works")
    
    # Test 5: Image statistics
    logger.info(f"  Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}, std={image.std():.1f}")
    assert image.min() >= 0 and image.max() <= 255, "Image values out of range"
    logger.info("✓ Image value ranges correct")


def test_dataset_generation():
    """Test dataset generation"""
    logger.info("\nTesting dataset generation...")
    
    num_samples = 50
    images, labels = generate_realistic_dataset(num_samples=num_samples, random_seed=42)
    
    assert images.shape == (num_samples, 512, 512, 3), f"Expected shape ({num_samples}, 512, 512, 3), got {images.shape}"
    assert labels.shape == (num_samples,), f"Expected shape ({num_samples},), got {labels.shape}"
    assert np.all((labels == 0) | (labels == 1)), "Labels should be 0 or 1"
    
    logger.info(f"✓ Generated {num_samples} images")
    logger.info(f"  Label distribution: {np.sum(labels == 0)} rocks, {np.sum(labels == 1)} mines")
    logger.info(f"  Image stats: min={images.min()}, max={images.max()}, mean={images.mean():.1f}, std={images.std():.1f}")


def test_reproducibility():
    """Test that random seed produces reproducible results"""
    logger.info("\nTesting reproducibility...")
    
    renderer = SideScanRenderer()
    objects = [{'type': 'mine', 'x': 0.5, 'y': 0.5, 'size': 15}]
    
    # Generate two images with same seed
    image1, label1 = renderer.render(objects=objects, random_seed=123)
    image2, label2 = renderer.render(objects=objects, random_seed=123)
    
    assert np.allclose(image1, image2), "Images with same seed should be identical"
    assert label1 == label2, "Labels with same seed should be identical"
    logger.info("✓ Reproducibility verified")


def test_image_characteristics():
    """Test that generated images have realistic characteristics"""
    logger.info("\nTesting image characteristics...")
    
    # Generate a batch
    images, labels = generate_realistic_dataset(num_samples=100, random_seed=42)
    
    # Check intensity distribution
    mean_intensity = images.mean()
    std_intensity = images.std()
    
    logger.info(f"  Mean intensity: {mean_intensity:.1f} (expected ~80-120)")
    logger.info(f"  Std intensity: {std_intensity:.1f} (expected ~30-50)")
    
    # Check that images have variation (not uniform)
    assert std_intensity > 20, "Images should have sufficient variation"
    logger.info("✓ Images have realistic intensity variation")
    
    # Check that mines and rocks have different characteristics
    mine_images = images[labels == 1]
    rock_images = images[labels == 0]
    
    if len(mine_images) > 0 and len(rock_images) > 0:
        mine_mean = mine_images.mean()
        rock_mean = rock_images.mean()
        logger.info(f"  Mine images mean: {mine_mean:.1f}")
        logger.info(f"  Rock images mean: {rock_mean:.1f}")
        logger.info("✓ Mine and rock images have different characteristics")


def save_sample_images():
    """Save sample images for visual inspection"""
    logger.info("\nSaving sample images...")
    
    output_dir = Path('demo_outputs/sidescan_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = SideScanRenderer()
    
    # Sample 1: Empty seabed
    image, _ = renderer.render(objects=None, random_seed=42)
    from PIL import Image as PILImage
    PILImage.fromarray(image).save(output_dir / 'sample_empty.png')
    logger.info(f"  Saved: sample_empty.png")
    
    # Sample 2: Single rock
    objects = [{'type': 'rock', 'x': 0.5, 'y': 0.5, 'size': 15}]
    image, _ = renderer.render(objects=objects, random_seed=42)
    PILImage.fromarray(image).save(output_dir / 'sample_rock.png')
    logger.info(f"  Saved: sample_rock.png")
    
    # Sample 3: Single mine
    objects = [{'type': 'mine', 'x': 0.5, 'y': 0.5, 'size': 15}]
    image, _ = renderer.render(objects=objects, random_seed=42)
    PILImage.fromarray(image).save(output_dir / 'sample_mine.png')
    logger.info(f"  Saved: sample_mine.png")
    
    # Sample 4: Multiple objects
    objects = [
        {'type': 'rock', 'x': 0.3, 'y': 0.3, 'size': 12},
        {'type': 'mine', 'x': 0.7, 'y': 0.7, 'size': 18},
        {'type': 'rock', 'x': 0.5, 'y': 0.8, 'size': 10}
    ]
    image, _ = renderer.render(objects=objects, random_seed=42)
    PILImage.fromarray(image).save(output_dir / 'sample_multiple.png')
    logger.info(f"  Saved: sample_multiple.png")
    
    logger.info(f"✓ Sample images saved to {output_dir}")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Side-Scan Sonar Renderer Tests")
    logger.info("=" * 60)
    
    try:
        test_basic_rendering()
        test_dataset_generation()
        test_reproducibility()
        test_image_characteristics()
        save_sample_images()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
