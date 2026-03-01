#!/usr/bin/env python3
"""
Side-Scan Sonar Image Renderer

Generates synthetic side-scan sonar images that accurately match real minehunting sonar data.
Side-scan sonar creates a top-down view of the seabed by sweeping a sonar beam across it.

Key characteristics of side-scan sonar:
- Horizontal beam sweeps across seabed (creates the "scan" lines)
- Vertical beam creates range information
- Objects appear as bright spots with acoustic shadows
- Seabed appears as grainy texture (speckle noise from coherent imaging)
- Range-based intensity falloff (closer = brighter)
- Frequency-dependent attenuation
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class SideScanParams:
    """Parameters for side-scan sonar rendering"""
    # Image properties
    image_width: int = 512
    image_height: int = 512
    
    # Sonar characteristics
    frequency_khz: float = 300.0  # Typical minehunting sonar
    beam_width_deg: float = 2.0   # Narrow beam for high resolution
    
    # Range characteristics
    range_start_m: float = 10.0   # Closest range
    range_end_m: float = 200.0    # Farthest range
    
    # Intensity parameters
    base_intensity: float = 0.4   # Baseline seabed intensity (darker)
    max_intensity: float = 0.9    # Maximum possible intensity
    
    # Attenuation (frequency-dependent absorption in water)
    attenuation_db_per_km: float = 15.0  # Typical for 300 kHz
    
    # Noise characteristics
    speckle_level: float = 0.3    # Multiplicative speckle noise
    gaussian_noise_level: float = 0.05  # Additive Gaussian noise
    
    # Seabed characteristics
    seabed_roughness: float = 0.6  # 0=smooth, 1=rough
    texture_scale: float = 20.0    # Texture feature size in pixels
    
    # Object characteristics
    object_brightness_mine: float = 0.85  # Mines are very bright
    object_brightness_rock: float = 0.65  # Rocks are moderately bright
    shadow_darkness: float = 0.15  # Acoustic shadow darkness
    shadow_length_factor: float = 2.0  # How long shadows are


class SideScanRenderer:
    """Render realistic side-scan sonar images"""
    
    def __init__(self, params: Optional[SideScanParams] = None):
        """Initialize renderer with parameters"""
        self.params = params or SideScanParams()
        self.width = self.params.image_width
        self.height = self.params.image_height
    
    def render(
        self,
        objects: Optional[List[Dict[str, Any]]] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Render a side-scan sonar image
        
        Args:
            objects: List of objects with keys:
                - 'x': x position (0-1 normalized)
                - 'y': y position (0-1 normalized)
                - 'type': 'mine' or 'rock'
                - 'size': object size in pixels (optional)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image_array, label)
            - image_array: RGB image (512, 512, 3) with values 0-255
            - label: 1 if any mine present, 0 otherwise
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create base seabed texture
        seabed = self._create_seabed_texture()
        
        # Apply range-based attenuation (closer = brighter)
        seabed = self._apply_range_attenuation(seabed)
        
        # Add objects if provided
        label = 0
        if objects:
            for obj in objects:
                seabed = self._add_object(seabed, obj)
                if obj.get('type') == 'mine':
                    label = 1
        
        # Add realistic noise
        seabed = self._add_speckle_noise(seabed)
        seabed = self._add_gaussian_noise(seabed)
        
        # Normalize to 0-255 range
        seabed = np.clip(seabed * 255, 0, 255).astype(np.uint8)
        
        # Convert to RGB (side-scan sonar is typically displayed as grayscale)
        image_rgb = np.stack([seabed, seabed, seabed], axis=2)
        
        return image_rgb, label
    
    def _create_seabed_texture(self) -> np.ndarray:
        """Create realistic seabed texture using Perlin-like noise"""
        # Create multi-octave noise for natural-looking texture
        texture = np.zeros((self.height, self.width))
        
        # Multiple octaves of noise for fractal-like appearance
        for octave in range(4):
            scale = self.params.texture_scale / (2 ** octave)
            amplitude = 0.5 ** octave
            
            # Generate gradient noise
            noise = self._generate_perlin_noise(scale)
            texture += amplitude * noise
        
        # Normalize to 0-1 range
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        
        # Apply roughness modulation
        texture = self.params.base_intensity + texture * (1 - self.params.base_intensity) * self.params.seabed_roughness
        
        return np.clip(texture, 0, 1)
    
    def _generate_perlin_noise(self, scale: float) -> np.ndarray:
        """Generate Perlin-like noise at given scale"""
        # Simple implementation using interpolated random gradients
        grid_size = max(1, int(self.width / scale))
        
        # Create random gradient grid
        gradients = np.random.randn(grid_size + 1, grid_size + 1)
        
        # Interpolate to full resolution
        x = np.linspace(0, grid_size, self.width)
        y = np.linspace(0, grid_size, self.height)
        xx, yy = np.meshgrid(x, y)
        
        # Bilinear interpolation
        xi = np.floor(xx).astype(int)
        yi = np.floor(yy).astype(int)
        xf = xx - xi
        yf = yy - yi
        
        # Clamp indices
        xi = np.clip(xi, 0, grid_size - 1)
        yi = np.clip(yi, 0, grid_size - 1)
        
        # Get corner values
        v00 = gradients[yi, xi]
        v10 = gradients[yi, xi + 1]
        v01 = gradients[yi + 1, xi]
        v11 = gradients[yi + 1, xi + 1]
        
        # Smooth interpolation
        u = xf * xf * (3 - 2 * xf)
        v = yf * yf * (3 - 2 * yf)
        
        # Bilinear interpolation
        v0 = v00 * (1 - u) + v10 * u
        v1 = v01 * (1 - u) + v11 * u
        noise = v0 * (1 - v) + v1 * v
        
        return noise
    
    def _apply_range_attenuation(self, seabed: np.ndarray) -> np.ndarray:
        """Apply range-based attenuation (closer = brighter)"""
        # Create range map (0 at top = close, 1 at bottom = far)
        range_map = np.linspace(0, 1, self.height)[:, np.newaxis]
        
        # Convert to actual range in meters
        range_m = self.params.range_start_m + range_map * (self.params.range_end_m - self.params.range_start_m)
        
        # Calculate attenuation in dB
        # Attenuation = 20*log10(R) + absorption*R
        # Simplified: use exponential decay
        attenuation_factor = np.exp(-range_map * self.params.attenuation_db_per_km / 20)
        
        # Apply attenuation
        seabed = seabed * attenuation_factor
        
        return seabed
    
    def _add_object(self, seabed: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        """Add an object (mine or rock) to the image"""
        # Get object parameters
        obj_type = obj.get('type', 'rock')
        x_norm = obj.get('x', 0.5)
        y_norm = obj.get('y', 0.5)
        size = obj.get('size', 15)
        
        # Convert normalized coordinates to pixel coordinates
        x_pix = int(x_norm * self.width)
        y_pix = int(y_norm * self.height)
        
        # Get object brightness
        brightness = self.params.object_brightness_mine if obj_type == 'mine' else self.params.object_brightness_rock
        
        # Create object signature (Gaussian blob)
        yy, xx = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((xx - x_pix)**2 + (yy - y_pix)**2)
        
        # Gaussian object signature
        obj_signature = brightness * np.exp(-(dist**2) / (2 * size**2))
        
        # Add object to seabed
        seabed = np.maximum(seabed, obj_signature)
        
        # Add acoustic shadow (behind the object)
        shadow_start = int(y_pix + size)
        shadow_end = min(self.height, int(y_pix + size * self.params.shadow_length_factor))
        
        if shadow_start < self.height:
            # Create shadow mask
            shadow_mask = np.zeros((self.height, self.width))
            
            # Shadow is a cone behind the object
            for y in range(shadow_start, shadow_end):
                # Shadow width increases with distance from object
                shadow_width = size * (y - y_pix) / (shadow_end - y_pix)
                x_min = max(0, int(x_pix - shadow_width))
                x_max = min(self.width, int(x_pix + shadow_width))
                
                # Fade shadow with distance
                fade = 1 - (y - shadow_start) / (shadow_end - shadow_start)
                shadow_mask[y, x_min:x_max] = fade
            
            # Apply shadow (darken the region)
            seabed = seabed * (1 - shadow_mask * self.params.shadow_darkness)
        
        return seabed
    
    def _add_speckle_noise(self, seabed: np.ndarray) -> np.ndarray:
        """Add multiplicative speckle noise (characteristic of coherent sonar)"""
        # Rayleigh-distributed speckle noise
        speckle = np.random.rayleigh(scale=self.params.speckle_level, size=seabed.shape)
        
        # Normalize to have mean around 1.0
        speckle = speckle / np.mean(speckle)
        
        # Apply multiplicative noise
        seabed = seabed * speckle
        
        return np.clip(seabed, 0, 1)
    
    def _add_gaussian_noise(self, seabed: np.ndarray) -> np.ndarray:
        """Add additive Gaussian noise"""
        noise = np.random.normal(0, self.params.gaussian_noise_level, seabed.shape)
        seabed = seabed + noise
        
        return np.clip(seabed, 0, 1)


def generate_realistic_dataset(
    num_samples: int,
    output_dir: Optional[str] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of realistic side-scan sonar images
    
    Args:
        num_samples: Number of images to generate
        output_dir: Optional directory to save images
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (images, labels)
        - images: Array of shape (num_samples, 512, 512, 3)
        - labels: Array of shape (num_samples,) with 0/1 labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    renderer = SideScanRenderer()
    images = []
    labels = []
    
    logger.info(f"Generating {num_samples} realistic side-scan sonar images")
    
    for i in range(num_samples):
        # Randomly decide if this image has a mine
        has_mine = np.random.rand() < 0.4  # 40% mines, 60% rocks
        
        # Generate random objects
        objects = []
        num_objects = np.random.randint(1, 4)  # 1-3 objects per image
        
        for _ in range(num_objects):
            obj_type = 'mine' if (has_mine and len(objects) == 0) else np.random.choice(['mine', 'rock'], p=[0.3, 0.7])
            
            objects.append({
                'type': obj_type,
                'x': np.random.uniform(0.1, 0.9),
                'y': np.random.uniform(0.2, 0.9),
                'size': np.random.uniform(10, 25)
            })
            
            if obj_type == 'mine':
                has_mine = True
        
        # Render image
        seed = (random_seed + i) if random_seed else None
        image, label = renderer.render(objects, random_seed=seed)
        
        images.append(image)
        labels.append(label)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} images")
    
    images_array = np.stack(images, axis=0)
    labels_array = np.array(labels)
    
    logger.info(f"Dataset generation complete. "
               f"Mines: {np.sum(labels_array)}, Rocks: {np.sum(labels_array == 0)}")
    
    return images_array, labels_array
