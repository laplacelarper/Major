"""Core physics calculations for sonar simulation"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_backscatter_intensity(
    grazing_angle_deg: np.ndarray,
    cosine_exponent: float = 4.0,
    base_intensity: float = 0.5
) -> np.ndarray:
    """
    Calculate backscatter intensity using cosⁿ(grazing_angle) formula.
    
    This implements the Lambert's law approximation for acoustic backscattering
    from the seabed, where intensity is proportional to cosⁿ(θ) where θ is
    the grazing angle.
    
    Args:
        grazing_angle_deg: Grazing angles in degrees, shape (H, W)
        cosine_exponent: Exponent n in cosⁿ(θ) formula (typically 2-8)
        base_intensity: Base intensity multiplier (0-1)
        
    Returns:
        Backscatter intensity array, same shape as input
        
    Requirements: 1.2 - Backscatter intensity proportional to cosⁿ(grazing_angle)
    """
    # Convert degrees to radians
    grazing_angle_rad = np.deg2rad(grazing_angle_deg)
    
    # Ensure angles are in valid range [0, 90] degrees
    grazing_angle_rad = np.clip(grazing_angle_rad, 0, np.pi/2)
    
    # Calculate cosine of grazing angle
    cos_grazing = np.cos(grazing_angle_rad)
    
    # Apply cosⁿ(θ) formula
    intensity = base_intensity * np.power(cos_grazing, cosine_exponent)
    
    # Ensure intensity is in valid range [0, 1]
    intensity = np.clip(intensity, 0.0, 1.0)
    
    logger.debug(f"Calculated backscatter intensity with exponent {cosine_exponent}, "
                f"intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    return intensity


def calculate_range_attenuation(
    range_map: np.ndarray,
    attenuation_coefficient: float = 2.0,
    reference_range: float = 1.0
) -> np.ndarray:
    """
    Calculate range-based attenuation using 1/R² relationship.
    
    Implements the geometric spreading loss for acoustic waves in water,
    where intensity decreases with the square of distance (or higher power
    depending on attenuation_coefficient).
    
    Args:
        range_map: Distance map in meters, shape (H, W)
        attenuation_coefficient: Exponent for range attenuation (typically 2.0)
        reference_range: Reference range for normalization (meters)
        
    Returns:
        Attenuation factor array (0-1), same shape as input
        
    Requirements: 1.3 - Range-based attenuation using 1/R² relationship
    """
    # Ensure positive ranges and avoid division by zero
    range_map = np.maximum(range_map, 0.1)  # Minimum 0.1m range
    
    # Calculate attenuation factor: 1/R^n
    attenuation = np.power(reference_range / range_map, attenuation_coefficient)
    
    # Normalize to [0, 1] range
    attenuation = np.clip(attenuation, 0.0, 1.0)
    
    logger.debug(f"Calculated range attenuation with coefficient {attenuation_coefficient}, "
                f"range: [{range_map.min():.1f}, {range_map.max():.1f}]m, "
                f"attenuation: [{attenuation.min():.3f}, {attenuation.max():.3f}]")
    
    return attenuation


def generate_acoustic_shadows(
    object_positions: np.ndarray,
    object_heights: np.ndarray,
    sonar_position: Tuple[float, float],
    image_shape: Tuple[int, int],
    shadow_length_factor: float = 3.0,
    shadow_intensity_factor: float = 0.1
) -> np.ndarray:
    """
    Generate acoustic shadows using geometric approximations.
    
    Creates shadow regions behind elevated objects based on geometric
    ray-tracing approximation. Shadows extend from objects in the direction
    opposite to the sonar beam.
    
    Args:
        object_positions: Object center positions as (x, y) coordinates, shape (N, 2)
        object_heights: Object heights above seabed, shape (N,)
        sonar_position: Sonar position as (x, y) tuple
        image_shape: Output image shape (height, width)
        shadow_length_factor: Multiplier for shadow length relative to object height
        shadow_intensity_factor: Shadow intensity (0=black, 1=no shadow)
        
    Returns:
        Shadow mask array (0-1), shape image_shape
        
    Requirements: 1.4 - Acoustic shadow generation using geometric approximations
    """
    height, width = image_shape
    shadow_mask = np.ones((height, width), dtype=np.float32)
    
    if len(object_positions) == 0:
        return shadow_mask
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    sonar_x, sonar_y = sonar_position
    
    for i, (obj_pos, obj_height) in enumerate(zip(object_positions, object_heights)):
        obj_x, obj_y = obj_pos
        
        # Calculate direction vector from sonar to object
        dx = obj_x - sonar_x
        dy = obj_y - sonar_y
        distance_to_object = np.sqrt(dx**2 + dy**2)
        
        if distance_to_object < 1e-6:  # Skip if object is at sonar position
            continue
            
        # Normalize direction vector
        dx_norm = dx / distance_to_object
        dy_norm = dy / distance_to_object
        
        # Calculate shadow length based on object height and geometry
        shadow_length = obj_height * shadow_length_factor
        
        # Calculate shadow end position
        shadow_end_x = obj_x + dx_norm * shadow_length
        shadow_end_y = obj_y + dy_norm * shadow_length
        
        # Create shadow region using line drawing approximation
        shadow_mask = _draw_shadow_region(
            shadow_mask,
            (obj_x, obj_y),
            (shadow_end_x, shadow_end_y),
            obj_height,
            shadow_intensity_factor
        )
    
    logger.debug(f"Generated acoustic shadows for {len(object_positions)} objects, "
                f"shadow coverage: {(1 - shadow_mask).sum() / shadow_mask.size * 100:.1f}%")
    
    return shadow_mask


def _draw_shadow_region(
    shadow_mask: np.ndarray,
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    object_height: float,
    shadow_intensity: float
) -> np.ndarray:
    """
    Draw a shadow region between two points with width based on object height.
    
    Args:
        shadow_mask: Current shadow mask to modify
        start_pos: Shadow start position (x, y)
        end_pos: Shadow end position (x, y)
        object_height: Height of object casting shadow
        shadow_intensity: Shadow intensity factor
        
    Returns:
        Updated shadow mask
    """
    height, width = shadow_mask.shape
    
    # Convert positions to integer pixel coordinates
    start_x, start_y = int(np.clip(start_pos[0], 0, width-1)), int(np.clip(start_pos[1], 0, height-1))
    end_x, end_y = int(np.clip(end_pos[0], 0, width-1)), int(np.clip(end_pos[1], 0, height-1))
    
    # Calculate shadow width based on object height (simple approximation)
    shadow_width = max(1, int(object_height * 0.5))
    
    # Use Bresenham-like algorithm to draw shadow line with width
    dx = abs(end_x - start_x)
    dy = abs(end_y - start_y)
    
    if dx == 0 and dy == 0:
        return shadow_mask
    
    # Determine step directions
    sx = 1 if start_x < end_x else -1
    sy = 1 if start_y < end_y else -1
    
    # Draw shadow line with width
    if dx > dy:
        # More horizontal than vertical
        for x in range(start_x, end_x + sx, sx):
            if 0 <= x < width:
                # Calculate corresponding y coordinate
                t = (x - start_x) / max(dx, 1)
                y = int(start_y + t * (end_y - start_y))
                
                # Apply shadow with width
                for dy_offset in range(-shadow_width, shadow_width + 1):
                    y_shadow = y + dy_offset
                    if 0 <= y_shadow < height:
                        shadow_mask[y_shadow, x] = min(shadow_mask[y_shadow, x], shadow_intensity)
    else:
        # More vertical than horizontal
        for y in range(start_y, end_y + sy, sy):
            if 0 <= y < height:
                # Calculate corresponding x coordinate
                t = (y - start_y) / max(dy, 1)
                x = int(start_x + t * (end_x - start_x))
                
                # Apply shadow with width
                for dx_offset in range(-shadow_width, shadow_width + 1):
                    x_shadow = x + dx_offset
                    if 0 <= x_shadow < width:
                        shadow_mask[y, x_shadow] = min(shadow_mask[y, x_shadow], shadow_intensity)
    
    return shadow_mask


def create_range_map(
    image_shape: Tuple[int, int],
    sonar_position: Tuple[float, float],
    range_min: float = 10.0,
    range_max: float = 200.0
) -> np.ndarray:
    """
    Create a range map showing distance from sonar to each pixel.
    
    Args:
        image_shape: Output image shape (height, width)
        sonar_position: Sonar position as (x, y) tuple
        range_min: Minimum range value (meters)
        range_max: Maximum range value (meters)
        
    Returns:
        Range map array in meters, shape image_shape
    """
    height, width = image_shape
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distance from sonar position
    sonar_x, sonar_y = sonar_position
    distances = np.sqrt((x_coords - sonar_x)**2 + (y_coords - sonar_y)**2)
    
    # Scale distances to physical range
    # Assume image coordinates map linearly to physical range
    max_image_distance = np.sqrt(height**2 + width**2)
    range_map = range_min + (range_max - range_min) * (distances / max_image_distance)
    
    return range_map


def create_grazing_angle_map(
    image_shape: Tuple[int, int],
    sonar_position: Tuple[float, float],
    seabed_depth: float = 50.0,
    sonar_altitude: float = 10.0
) -> np.ndarray:
    """
    Create a grazing angle map for the sonar beam geometry.
    
    Args:
        image_shape: Output image shape (height, width)
        sonar_position: Sonar position as (x, y) tuple
        seabed_depth: Depth of seabed below sonar (meters)
        sonar_altitude: Sonar altitude above seabed (meters)
        
    Returns:
        Grazing angle map in degrees, shape image_shape
    """
    height, width = image_shape
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate horizontal distance from sonar
    sonar_x, sonar_y = sonar_position
    horizontal_distances = np.sqrt((x_coords - sonar_x)**2 + (y_coords - sonar_y)**2)
    
    # Convert image coordinates to physical distances (simple linear mapping)
    max_image_distance = np.sqrt(height**2 + width**2)
    physical_distances = horizontal_distances * (200.0 / max_image_distance)  # Scale to ~200m max range
    
    # Calculate grazing angles using geometry
    # grazing_angle = arctan(sonar_altitude / horizontal_distance)
    grazing_angles_rad = np.arctan2(sonar_altitude, physical_distances)
    grazing_angles_deg = np.rad2deg(grazing_angles_rad)
    
    # Ensure angles are in valid range [0, 90]
    grazing_angles_deg = np.clip(grazing_angles_deg, 0.0, 90.0)
    
    return grazing_angles_deg