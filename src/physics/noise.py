"""Noise and texture generation for sonar simulation"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def generate_speckle_noise(
    image_shape: Tuple[int, int],
    noise_type: str = "rayleigh",
    noise_level: float = 0.2,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate multiplicative speckle noise using Rayleigh or Gamma distributions.
    
    Speckle noise is characteristic of coherent imaging systems like sonar,
    where constructive and destructive interference creates a granular pattern.
    
    Args:
        image_shape: Output noise shape (height, width)
        noise_type: Type of noise distribution ("rayleigh" or "gamma")
        noise_level: Noise intensity level (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Multiplicative noise array (values around 1.0), shape image_shape
        
    Requirements: 1.5 - Multiplicative speckle noise using Rayleigh/Gamma distributions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    height, width = image_shape
    
    if noise_type.lower() == "rayleigh":
        # Rayleigh distribution for speckle noise
        # Scale parameter controls the spread of the distribution
        scale = noise_level * 0.5  # Adjust scale based on noise level
        noise = np.random.rayleigh(scale=scale, size=(height, width))
        
        # Normalize to have mean around 1.0 for multiplicative noise
        noise = noise / np.mean(noise)
        
    elif noise_type.lower() == "gamma":
        # Gamma distribution for speckle noise
        # Shape and scale parameters control the distribution
        shape = 1.0 / (noise_level**2)  # Lower noise_level = higher shape = less variance
        scale = noise_level
        noise = np.random.gamma(shape=shape, scale=scale, size=(height, width))
        
        # Normalize to have mean around 1.0 for multiplicative noise
        noise = noise / np.mean(noise)
        
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Use 'rayleigh' or 'gamma'")
    
    # Ensure noise values are positive and reasonable for multiplication
    noise = np.clip(noise, 0.1, 3.0)
    
    logger.debug(f"Generated {noise_type} speckle noise with level {noise_level}, "
                f"noise range: [{noise.min():.3f}, {noise.max():.3f}], "
                f"mean: {noise.mean():.3f}")
    
    return noise


def generate_seabed_texture(
    image_shape: Tuple[int, int],
    roughness: float = 0.5,
    texture_scale: float = 10.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate seabed texture using procedural noise.
    
    Creates realistic seabed texture patterns using Perlin-like noise
    to simulate variations in seabed composition and roughness.
    
    Args:
        image_shape: Output texture shape (height, width)
        roughness: Texture roughness level (0-1)
        texture_scale: Scale of texture features (larger = coarser texture)
        random_seed: Random seed for reproducibility
        
    Returns:
        Texture intensity array (0-1), shape image_shape
        
    Requirements: 1.5 - Seabed texture generation with procedural noise
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    height, width = image_shape
    
    # Generate multi-octave noise for realistic texture
    texture = np.zeros((height, width), dtype=np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Generate multiple octaves of noise with different frequencies
    num_octaves = 4
    amplitude = 1.0
    frequency = 1.0 / texture_scale
    
    for octave in range(num_octaves):
        # Generate noise for this octave using simple gradient noise approximation
        octave_noise = _generate_gradient_noise(
            x_coords * frequency,
            y_coords * frequency,
            random_seed=(random_seed + octave) if random_seed else None
        )
        
        # Add to texture with decreasing amplitude
        texture += amplitude * octave_noise
        
        # Update parameters for next octave
        amplitude *= 0.5  # Decrease amplitude
        frequency *= 2.0  # Increase frequency
    
    # Normalize texture to [0, 1] range
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    
    # Apply roughness scaling
    # Higher roughness = more variation, lower roughness = smoother
    texture_mean = 0.5
    texture = texture_mean + (texture - texture_mean) * roughness
    
    # Ensure values are in [0, 1] range
    texture = np.clip(texture, 0.0, 1.0)
    
    logger.debug(f"Generated seabed texture with roughness {roughness}, scale {texture_scale}, "
                f"texture range: [{texture.min():.3f}, {texture.max():.3f}]")
    
    return texture


def _generate_gradient_noise(
    x: np.ndarray,
    y: np.ndarray,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate gradient noise (simplified Perlin noise approximation).
    
    Args:
        x: X coordinate array
        y: Y coordinate array
        random_seed: Random seed for reproducibility
        
    Returns:
        Noise array with same shape as input coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get integer and fractional parts of coordinates
    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)
    x_frac = x - x_int
    y_frac = y - y_int
    
    # Generate random gradients at grid points
    # Use a simple hash function for reproducible random gradients
    def hash_coords(xi, yi):
        # Simple hash function for coordinate pairs
        return ((xi * 73856093) ^ (yi * 19349663)) % 1000000
    
    # Get gradients at four corners of each cell
    grad_00 = _get_gradient(hash_coords(x_int, y_int))
    grad_10 = _get_gradient(hash_coords(x_int + 1, y_int))
    grad_01 = _get_gradient(hash_coords(x_int, y_int + 1))
    grad_11 = _get_gradient(hash_coords(x_int + 1, y_int + 1))
    
    # Calculate dot products with distance vectors
    dot_00 = grad_00[..., 0] * x_frac + grad_00[..., 1] * y_frac
    dot_10 = grad_10[..., 0] * (x_frac - 1) + grad_10[..., 1] * y_frac
    dot_01 = grad_01[..., 0] * x_frac + grad_01[..., 1] * (y_frac - 1)
    dot_11 = grad_11[..., 0] * (x_frac - 1) + grad_11[..., 1] * (y_frac - 1)
    
    # Smooth interpolation using fade function
    u = _fade(x_frac)
    v = _fade(y_frac)
    
    # Bilinear interpolation
    noise = _lerp(v, _lerp(u, dot_00, dot_10), _lerp(u, dot_01, dot_11))
    
    return noise


def _get_gradient(hash_val: np.ndarray) -> np.ndarray:
    """Get unit gradient vectors from hash values."""
    # Convert hash to angle
    angles = (hash_val % 8) * np.pi / 4
    
    # Create gradient vectors
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    
    return gradients


def _fade(t: np.ndarray) -> np.ndarray:
    """Fade function for smooth interpolation: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def randomize_parameters(
    base_params: dict,
    variation_ranges: dict,
    random_seed: Optional[int] = None
) -> dict:
    """
    Randomize physics parameters for domain variation.
    
    Creates parameter variations to increase diversity in synthetic data
    and improve model generalization.
    
    Args:
        base_params: Base parameter values
        variation_ranges: Ranges for parameter variation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with randomized parameters
        
    Requirements: 1.5, 7.3 - Parameter randomization for domain variation
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    randomized_params = base_params.copy()
    
    for param_name, base_value in base_params.items():
        if param_name in variation_ranges:
            var_range = variation_ranges[param_name]
            
            if isinstance(var_range, (list, tuple)) and len(var_range) == 2:
                # Range specified as [min, max]
                min_val, max_val = var_range
                randomized_params[param_name] = np.random.uniform(min_val, max_val)
                
            elif isinstance(var_range, (int, float)):
                # Range specified as +/- percentage of base value
                variation = base_value * var_range * (2 * np.random.random() - 1)
                randomized_params[param_name] = base_value + variation
                
            else:
                logger.warning(f"Invalid variation range for parameter {param_name}: {var_range}")
    
    logger.debug(f"Randomized {len(variation_ranges)} parameters")
    
    return randomized_params


def apply_multiplicative_noise(
    image: np.ndarray,
    noise_array: np.ndarray,
    noise_strength: float = 1.0
) -> np.ndarray:
    """
    Apply multiplicative noise to an image.
    
    Args:
        image: Input image array
        noise_array: Multiplicative noise array (values around 1.0)
        noise_strength: Strength of noise application (0-1)
        
    Returns:
        Noisy image array
    """
    # Blend between original image and noisy image
    noisy_image = image * (1 - noise_strength + noise_strength * noise_array)
    
    # Ensure output is in valid range
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    
    return noisy_image


def generate_combined_texture_noise(
    image_shape: Tuple[int, int],
    texture_params: dict,
    noise_params: dict,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate both seabed texture and speckle noise in one call.
    
    Args:
        image_shape: Output shape (height, width)
        texture_params: Parameters for texture generation
        noise_params: Parameters for noise generation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texture_array, noise_array)
    """
    # Generate texture
    texture = generate_seabed_texture(
        image_shape=image_shape,
        roughness=texture_params.get('roughness', 0.5),
        texture_scale=texture_params.get('texture_scale', 10.0),
        random_seed=random_seed
    )
    
    # Generate noise with different seed to avoid correlation
    noise_seed = (random_seed + 1000) if random_seed is not None else None
    noise = generate_speckle_noise(
        image_shape=image_shape,
        noise_type=noise_params.get('noise_type', 'rayleigh'),
        noise_level=noise_params.get('noise_level', 0.2),
        random_seed=noise_seed
    )
    
    return texture, noise