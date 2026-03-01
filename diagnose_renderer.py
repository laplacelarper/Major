#!/usr/bin/env python3
"""Diagnose the synthetic data generation issue"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from src.physics.renderer import SonarImageRenderer

# Create renderer
renderer = SonarImageRenderer(image_size=(512, 512))

# Test with simple parameters
physics_params = {
    'grazing_angle_range': (10, 80),
    'range_limits': (10, 200),
    'noise_level': 0.2,
    'texture_roughness': 0.5,
    'cosine_exponent': 4.0,
    'base_intensity': 0.5,
    'attenuation_coefficient': 2.0,
}

print("Generating test image...")
image, label, metadata = renderer.render_sonar_image(
    physics_params=physics_params,
    object_positions=[(256, 256)],
    object_heights=[2.0],
    object_labels=[1],
    random_seed=42
)

print(f"\nImage statistics:")
print(f"  Shape: {image.shape}")
print(f"  Dtype: {image.dtype}")
print(f"  Range: [{image.min():.6f}, {image.max():.6f}]")
print(f"  Mean: {image.mean():.6f}")
print(f"  Std: {image.std():.6f}")
print(f"  Label: {label}")

# Convert to uint8 and save
image_uint8 = (image * 255).astype(np.uint8)
print(f"\nAfter uint8 conversion:")
print(f"  Range: [{image_uint8.min()}, {image_uint8.max()}]")
print(f"  Mean: {image_uint8.mean():.2f}")
print(f"  Non-zero pixels: {np.sum(image_uint8 > 0)} ({np.sum(image_uint8 > 0) / image_uint8.size * 100:.2f}%)")

# Save test image
cv2.imwrite('test_render.png', image_uint8)
print(f"\nSaved test image to: test_render.png")

# Compare with real image
real_img = cv2.imread('data/real/minehunting_sonar/2015/0001_2015.jpg', cv2.IMREAD_GRAYSCALE)
print(f"\nReal image for comparison:")
print(f"  Range: [{real_img.min()}, {real_img.max()}]")
print(f"  Mean: {real_img.mean():.2f}")
print(f"  Std: {real_img.std():.2f}")

print(f"\nProblem identified:")
print(f"  Synthetic mean: {image_uint8.mean():.2f} (should be ~40-50)")
print(f"  Real mean: {real_img.mean():.2f}")
print(f"  Ratio: {image_uint8.mean() / real_img.mean():.4f}x")
