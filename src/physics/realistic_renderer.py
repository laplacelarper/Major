#!/usr/bin/env python3
"""
Realistic Sonar Image Renderer

Generates synthetic sonar images that match real minehunting sonar characteristics:
- 1024×1024 resolution (matching real data)
- RGB color output (matching real data)
- Complex texture patterns
- Realistic noise and artifacts
- Darker overall appearance
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RealisticPhysicsParams:
    """Physics parameters for realistic sonar rendering"""
    # Image properties
    image_size: Tuple[int, int] = (1024, 1024)
    
    # Sonar characteristics
    grazing_angle_deg: float = 45.0
    range_m: float = 100.0
    frequency_khz: float = 300.0
    
    # Backscatter
    cosine_exponent: float = 2.5  # Lower for more realistic
    base_intensity: float = 0.3   # Darker baseline
    
    # Attenuation
    attenuation_coefficient: float = 1.5
    
    # Noise
    speckle_noise_level: float = 0.4
    gaussian_noise_level: float = 0.1
    
    # Texture
    seabed_roughness: float = 0.6
    texture_scale: float = 20.0
    
    # Objects
    target_material: str = 'metal'  # 'metal' or 'rock'
    object_brightness: float = 0.7
    shadow_intensity: float = 0.15


class RealisticSonarRenderer:
    """Render realistic sonar images matching real minehunting data"""
    
    def __init__(self, image_size: Tuple[int, int] = (1024, 1024)):
        self.image_size = image_size
        self.width, self.height = image_size
    
    def render_image(self, params: RealisticPhysicsParams) -> Tuple[np.ndarray, int]:
        """
        Render a realistic sonar image
        
        Returns:
            Tuple of (image_array, label)
            - image_array: RGB image (1024, 1024, 3) with values 0-255
            - label: 1 for mine, 0 for rock
        """
        # Create base seabed
        seabed = self._create_seabed_texture(params.seabed_roughness, params.texture_scale)
        
        # Add range-based attenuation
        range_map = self._create_range_attenuation(params.attenuation_coefficient)
        seabed = seabed * range_map
        
        # Add object (mine or rock)
        label = 1 if params.target_material == 'metal' else 0
        seabed = self._add_object(seabed, params)
        
        # Add realistic noise
        seabed = self._add_realistic_noise(seabed, params)
        
        # Normalize and convert to RGB
        seabed = np.clip(seabed, 0, 1)
        
        # Create RGB image (matching real sonar color appearance)
        rgb_image = self._create_rgb_sonar(seabed)
        
        # Convert to 0-255 range
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image, label
    
    def _create_seabed_texture(self, roughness: float, scale: float) -> np.ndarray:
        """Create realistic seabed texture using Perlin-like noise"""
        # Create base texture with multiple scales
        texture = np.zeros((self.height, self.width))
        
        # Multi-scale noise for realistic appearance
        for octave in range(4):
            freq = 2 ** octave
            amp = 0.5 ** octave
            
            # Create noise at this scale
            noise = np.random.rand(self.height // (scale // freq), 
                                   self.width // (scale // freq))
            
            # Interpolate to full size
            noise = cv2.resize(noise, (self.width, self.height), 
                             interpolation=cv2.INTER_CUBIC)
            
            texture += noise * amp
        
        # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
        
        # Apply roughness
        texture = texture ** (1 / (roughness + 0.5))
        
        # Scale to reasonable intensity
        texture = texture * 0.4 + 0.1  # Range: 0.1-0.5
        
        return texture
    
    def _create_range_attenuation(self, coefficient: float) -> np.ndarray:
        """Create range-based attenuation (darker at edges)"""
        y = np.linspace(0, 1, self.height)
        x = np.linspace(0, 1, self.width)
        yy, xx = np.meshgrid(y, x)
        
        # Distance from top (sonar position)
        distance = yy
        
        # Attenuation: 1/R^coefficient
        attenuation = 1.0 / (1.0 + coefficient * distance)
        
        return attenuation
    
    def _add_object(self, image: np.ndarray, params: RealisticPhysicsParams) -> np.ndarray:
        """Add mine or rock object to image"""
        image = image.copy()
        
        # Object position (center-ish)
        obj_y = int(self.height * 0.4)
        obj_x = int(self.width * 0.5)
        obj_size = 60
        
        # Create object signature
        yy, xx = np.ogrid[-obj_size:obj_size, -obj_size:obj_size]
        distance = np.sqrt(xx**2 + yy**2)
        
        # Object brightness depends on material
        if params.target_material == 'metal':
            # Mines: bright, sharp signature
            obj_signature = np.exp(-(distance ** 2) / (obj_size ** 2 / 4))
            obj_brightness = 0.8
        else:
            # Rocks: dimmer, softer signature
            obj_signature = np.exp(-(distance ** 2) / (obj_size ** 2 / 2))
            obj_brightness = 0.4
        
        # Add object to image
        y1, y2 = max(0, obj_y - obj_size), min(self.height, obj_y + obj_size)
        x1, x2 = max(0, obj_x - obj_size), min(self.width, obj_x + obj_size)
        
        dy1, dy2 = max(0, obj_size - obj_y), obj_size + min(self.height - obj_y, obj_size)
        dx1, dx2 = max(0, obj_size - obj_x), obj_size + min(self.width - obj_x, obj_size)
        
        image[y1:y2, x1:x2] += obj_signature[dy1:dy2, dx1:dx2] * obj_brightness
        
        # Add shadow behind object
        shadow_y_start = obj_y + obj_size
        shadow_y_end = min(self.height, shadow_y_start + 100)
        shadow_x_start = max(0, obj_x - 30)
        shadow_x_end = min(self.width, obj_x + 30)
        
        if shadow_y_end > shadow_y_start:
            image[shadow_y_start:shadow_y_end, shadow_x_start:shadow_x_end] *= (1 - params.shadow_intensity)
        
        return image
    
    def _add_realistic_noise(self, image: np.ndarray, params: RealisticPhysicsParams) -> np.ndarray:
        """Add realistic sonar noise"""
        image = image.copy()
        
        # Speckle noise (multiplicative)
        speckle = np.random.gamma(shape=2, scale=0.5, size=image.shape)
        image = image * (1 + params.speckle_noise_level * (speckle - 1))
        
        # Gaussian noise (additive)
        gaussian = np.random.normal(0, params.gaussian_noise_level, image.shape)
        image = image + gaussian
        
        # Clipping
        image = np.clip(image, 0, 1)
        
        return image
    
    def _create_rgb_sonar(self, grayscale: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB with sonar-like coloring"""
        # Create RGB image
        rgb = np.zeros((self.height, self.width, 3))
        
        # Sonar typically appears as:
        # - Dark areas: low intensity (blue-ish)
        # - Medium areas: medium intensity (green-ish)
        # - Bright areas: high intensity (red-ish)
        
        # Red channel: high intensity
        rgb[:, :, 0] = grayscale ** 1.2
        
        # Green channel: medium intensity
        rgb[:, :, 1] = grayscale ** 1.0
        
        # Blue channel: low intensity
        rgb[:, :, 2] = grayscale ** 0.8
        
        # Darken overall to match real data
        rgb = rgb * 0.6
        
        return rgb


def generate_realistic_dataset(num_samples: int = 100, 
                               output_dir: str = 'data/synthetic_realistic') -> None:
    """Generate realistic synthetic sonar dataset"""
    from pathlib import Path
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = RealisticSonarRenderer(image_size=(1024, 1024))
    
    print(f"Generating {num_samples} realistic sonar images...")
    
    for i in range(num_samples):
        # Random parameters
        is_mine = np.random.rand() > 0.5
        material = 'metal' if is_mine else 'rock'
        
        params = RealisticPhysicsParams(
            grazing_angle_deg=np.random.uniform(20, 70),
            range_m=np.random.uniform(50, 150),
            seabed_roughness=np.random.uniform(0.4, 0.8),
            speckle_noise_level=np.random.uniform(0.2, 0.5),
            target_material=material
        )
        
        # Render
        image, label = renderer.render_image(params)
        
        # Save image
        img_path = output_dir / f"sonar_{i:05d}.jpg"
        Image.fromarray(image).save(img_path, quality=95)
        
        # Save metadata
        meta_path = output_dir / f"sonar_{i:05d}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'label': int(label),
                'material': material,
                'grazing_angle': params.grazing_angle_deg,
                'range_m': params.range_m,
                'seabed_roughness': params.seabed_roughness,
                'noise_level': params.speckle_noise_level
            }, f, indent=2)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{num_samples}")
    
    print(f"✅ Generated {num_samples} realistic images in {output_dir}")


if __name__ == "__main__":
    # Test
    print("\n" + "="*70)
    print("  REALISTIC SONAR RENDERER TEST")
    print("="*70)
    
    renderer = RealisticSonarRenderer(image_size=(1024, 1024))
    
    # Generate test images
    params_mine = RealisticPhysicsParams(target_material='metal')
    params_rock = RealisticPhysicsParams(target_material='rock')
    
    img_mine, label_mine = renderer.render_image(params_mine)
    img_rock, label_rock = renderer.render_image(params_rock)
    
    print(f"\n✅ Generated realistic sonar images")
    print(f"   Mine image: {img_mine.shape}, label={label_mine}")
    print(f"   Rock image: {img_rock.shape}, label={label_rock}")
    
    # Save samples
    Image.fromarray(img_mine).save('demo_outputs/realistic_mine.jpg')
    Image.fromarray(img_rock).save('demo_outputs/realistic_rock.jpg')
    print(f"\n✅ Saved samples to demo_outputs/")
    
    # Generate full dataset
    print(f"\nGenerating realistic dataset...")
    generate_realistic_dataset(num_samples=50, output_dir='data/synthetic_realistic')
