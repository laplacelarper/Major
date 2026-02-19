"""Image rendering pipeline for synthetic sonar data generation"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
# import cv2  # Optional dependency

from .calculations import (
    calculate_backscatter_intensity,
    calculate_range_attenuation,
    generate_acoustic_shadows,
    create_range_map,
    create_grazing_angle_map
)
from .noise import (
    generate_seabed_texture,
    generate_speckle_noise,
    apply_multiplicative_noise,
    randomize_parameters
)

logger = logging.getLogger(__name__)


@dataclass
class PhysicsMetadata:
    """Metadata structure for physics parameters used in image generation"""
    grazing_angle_deg: float
    seabed_roughness: float
    range_m: float
    noise_level: float
    target_material: str
    frequency_khz: float
    beam_width_deg: float
    cosine_exponent: float
    base_intensity: float
    shadow_length_factor: float
    shadow_intensity_factor: float
    attenuation_coefficient: float
    texture_scale: float
    noise_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for model input"""
        # Convert categorical variables to numerical
        material_encoding = {
            'metal': 0.0,
            'rock': 1.0, 
            'sand': 2.0,
            'mud': 3.0
        }
        
        noise_encoding = {
            'rayleigh': 0.0,
            'gamma': 1.0
        }
        
        vector = np.array([
            self.grazing_angle_deg / 90.0,  # Normalize to [0, 1]
            self.seabed_roughness,  # Already [0, 1]
            self.range_m / 200.0,  # Normalize to [0, 1] assuming max 200m
            self.noise_level,  # Already [0, 1]
            material_encoding.get(self.target_material, 0.0),
            self.frequency_khz / 500.0,  # Normalize to [0, 1] assuming max 500kHz
            self.beam_width_deg / 10.0,  # Normalize to [0, 1] assuming max 10 degrees
        ])
        
        return vector.astype(np.float32)


class SonarImageRenderer:
    """
    Renders 512x512 grayscale sonar images using physics-based calculations.
    
    Requirements: 1.1, 7.3 - Create 512x512 grayscale image renderer using NumPy operations
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the sonar image renderer.
        
        Args:
            image_size: Output image size (height, width)
        """
        self.image_size = image_size
        self.height, self.width = image_size
        
        logger.info(f"Initialized SonarImageRenderer with size {image_size}")
    
    def render_sonar_image(
        self,
        physics_params: Dict[str, Any],
        object_positions: Optional[List[Tuple[float, float]]] = None,
        object_heights: Optional[List[float]] = None,
        object_labels: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, int, PhysicsMetadata]:
        """
        Render a complete sonar image with physics effects.
        
        Args:
            physics_params: Dictionary of physics parameters
            object_positions: List of object positions as (x, y) tuples
            object_heights: List of object heights above seabed
            object_labels: List of object labels (0=rock, 1=mine)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image_array, label, metadata)
            
        Requirements: 1.1 - 512x512 grayscale image renderer
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Set default object parameters if not provided
        if object_positions is None:
            object_positions = []
        if object_heights is None:
            object_heights = []
        if object_labels is None:
            object_labels = []
        
        # Ensure consistent list lengths
        num_objects = len(object_positions)
        if len(object_heights) != num_objects:
            object_heights = [1.0] * num_objects
        if len(object_labels) != num_objects:
            object_labels = [0] * num_objects  # Default to rock
        
        # Extract physics parameters with defaults
        sonar_position = physics_params.get('sonar_position', (self.width // 4, self.height // 2))
        grazing_angle_range = physics_params.get('grazing_angle_range', (10.0, 80.0))
        range_limits = physics_params.get('range_limits', (10.0, 200.0))
        cosine_exponent = physics_params.get('cosine_exponent', 4.0)
        base_intensity = physics_params.get('base_intensity', 0.5)
        attenuation_coefficient = physics_params.get('attenuation_coefficient', 2.0)
        shadow_length_factor = physics_params.get('shadow_length_factor', 3.0)
        shadow_intensity_factor = physics_params.get('shadow_intensity_factor', 0.1)
        
        # Texture and noise parameters
        texture_roughness = physics_params.get('texture_roughness', 0.5)
        texture_scale = physics_params.get('texture_scale', 10.0)
        noise_type = physics_params.get('noise_type', 'rayleigh')
        noise_level = physics_params.get('noise_level', 0.2)
        
        # Additional metadata parameters
        frequency_khz = physics_params.get('frequency_khz', 300.0)
        beam_width_deg = physics_params.get('beam_width_deg', 2.0)
        target_material = physics_params.get('target_material', 'rock')
        
        # Step 1: Create base geometry maps
        range_map = create_range_map(self.image_size, sonar_position, *range_limits)
        grazing_angle_map = create_grazing_angle_map(
            self.image_size, 
            sonar_position,
            seabed_depth=50.0,
            sonar_altitude=10.0
        )
        
        # Step 2: Generate seabed texture
        seabed_texture = generate_seabed_texture(
            self.image_size,
            roughness=texture_roughness,
            texture_scale=texture_scale,
            random_seed=random_seed
        )
        
        # Step 3: Calculate backscatter intensity
        backscatter = calculate_backscatter_intensity(
            grazing_angle_map,
            cosine_exponent=cosine_exponent,
            base_intensity=base_intensity
        )
        
        # Step 4: Apply range attenuation
        range_attenuation = calculate_range_attenuation(
            range_map,
            attenuation_coefficient=attenuation_coefficient
        )
        
        # Step 5: Generate acoustic shadows
        shadow_mask = generate_acoustic_shadows(
            np.array(object_positions) if object_positions else np.array([]).reshape(0, 2),
            np.array(object_heights) if object_heights else np.array([]),
            sonar_position,
            self.image_size,
            shadow_length_factor=shadow_length_factor,
            shadow_intensity_factor=shadow_intensity_factor
        )
        
        # Step 6: Combine physics effects
        # Base image = texture * backscatter * range_attenuation * shadows
        base_image = seabed_texture * backscatter * range_attenuation * shadow_mask
        
        # Step 7: Add objects to the scene
        if object_positions:
            base_image = self._add_objects_to_image(
                base_image,
                object_positions,
                object_heights,
                object_labels
            )
        
        # Step 8: Apply speckle noise
        speckle_noise = generate_speckle_noise(
            self.image_size,
            noise_type=noise_type,
            noise_level=noise_level,
            random_seed=(random_seed + 1) if random_seed else None
        )
        
        final_image = apply_multiplicative_noise(base_image, speckle_noise, noise_strength=1.0)
        
        # Step 9: Normalize to [0, 1] and convert to uint8 for display
        final_image = np.clip(final_image, 0.0, 1.0)
        
        # Step 10: Create metadata
        metadata = PhysicsMetadata(
            grazing_angle_deg=float(np.mean(grazing_angle_map)),
            seabed_roughness=texture_roughness,
            range_m=float(np.mean(range_map)),
            noise_level=noise_level,
            target_material=target_material,
            frequency_khz=frequency_khz,
            beam_width_deg=beam_width_deg,
            cosine_exponent=cosine_exponent,
            base_intensity=base_intensity,
            shadow_length_factor=shadow_length_factor,
            shadow_intensity_factor=shadow_intensity_factor,
            attenuation_coefficient=attenuation_coefficient,
            texture_scale=texture_scale,
            noise_type=noise_type
        )
        
        # Determine overall label (1 if any mine present, 0 otherwise)
        overall_label = 1 if any(label == 1 for label in object_labels) else 0
        
        logger.debug(f"Rendered sonar image with {num_objects} objects, label={overall_label}")
        
        return final_image, overall_label, metadata
    
    def _add_objects_to_image(
        self,
        base_image: np.ndarray,
        object_positions: List[Tuple[float, float]],
        object_heights: List[float],
        object_labels: List[int]
    ) -> np.ndarray:
        """
        Add objects (mines/rocks) to the sonar image.
        
        Args:
            base_image: Base sonar image
            object_positions: Object positions as (x, y) tuples
            object_heights: Object heights above seabed
            object_labels: Object labels (0=rock, 1=mine)
            
        Returns:
            Image with objects added
        """
        image_with_objects = base_image.copy()
        
        for pos, height, label in zip(object_positions, object_heights, object_labels):
            x, y = pos
            
            # Convert to integer pixel coordinates
            px = int(np.clip(x, 0, self.width - 1))
            py = int(np.clip(y, 0, self.height - 1))
            
            # Object size based on height and type
            if label == 1:  # Mine
                object_size = max(3, int(height * 2))  # Mines are more compact
                object_intensity = 0.9  # High reflectivity
            else:  # Rock
                object_size = max(2, int(height * 1.5))  # Rocks are more spread out
                object_intensity = 0.7  # Medium reflectivity
            
            # Draw object as a filled circle
            y_coords, x_coords = np.ogrid[:self.height, :self.width]
            mask = (x_coords - px)**2 + (y_coords - py)**2 <= object_size**2
            
            # Apply object intensity with some randomness
            object_pattern = object_intensity * (0.8 + 0.4 * np.random.random())
            image_with_objects[mask] = np.maximum(image_with_objects[mask], object_pattern)
        
        return image_with_objects


class ImageExporter:
    """
    Handles exporting sonar images and metadata to disk.
    
    Requirements: 1.1, 7.3 - Image export functionality with PNG format support
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the image exporter.
        
        Args:
            output_dir: Directory to save images and metadata
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Initialized ImageExporter with output directory: {self.output_dir}")
    
    def save_image_with_metadata(
        self,
        image: np.ndarray,
        label: int,
        metadata: PhysicsMetadata,
        image_id: str,
        save_png: bool = True,
        save_npy: bool = False
    ) -> Dict[str, Path]:
        """
        Save sonar image and associated metadata to disk.
        
        Args:
            image: Sonar image array (0-1 float values)
            label: Image label (0=no mine, 1=mine present)
            metadata: Physics metadata object
            image_id: Unique identifier for the image
            save_png: Whether to save as PNG image
            save_npy: Whether to save as NumPy array
            
        Returns:
            Dictionary with paths to saved files
            
        Requirements: 1.1 - Image export functionality with PNG format support
        """
        saved_paths = {}
        
        # Save image in PNG format
        if save_png:
            # Convert to 8-bit grayscale
            image_uint8 = (image * 255).astype(np.uint8)
            png_path = self.output_dir / "images" / f"{image_id}.png"
            
            # Try to use OpenCV if available, otherwise use PIL or basic numpy save
            try:
                import cv2
                success = cv2.imwrite(str(png_path), image_uint8)
                if success:
                    saved_paths['png'] = png_path
                    logger.debug(f"Saved PNG image with OpenCV: {png_path}")
                else:
                    logger.error(f"Failed to save PNG image with OpenCV: {png_path}")
            except ImportError:
                # Fallback to PIL if available
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(image_uint8, mode='L')
                    pil_image.save(png_path)
                    saved_paths['png'] = png_path
                    logger.debug(f"Saved PNG image with PIL: {png_path}")
                except ImportError:
                    # Final fallback: save as numpy array with .png extension
                    np.save(png_path.with_suffix('.npy'), image_uint8)
                    saved_paths['png'] = png_path.with_suffix('.npy')
                    logger.warning(f"Saved as NumPy array (no PNG library available): {png_path.with_suffix('.npy')}")
        
        # Save image as NumPy array
        if save_npy:
            npy_path = self.output_dir / "images" / f"{image_id}.npy"
            np.save(npy_path, image)
            saved_paths['npy'] = npy_path
            logger.debug(f"Saved NumPy array: {npy_path}")
        
        # Save metadata as JSON
        metadata_dict = {
            'image_id': image_id,
            'label': label,
            'physics_parameters': metadata.to_dict(),
            'metadata_vector': metadata.to_vector().tolist(),
            'image_shape': list(image.shape),
            'image_stats': {
                'min': float(image.min()),
                'max': float(image.max()),
                'mean': float(image.mean()),
                'std': float(image.std())
            }
        }
        
        json_path = self.output_dir / "metadata" / f"{image_id}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        saved_paths['metadata'] = json_path
        
        logger.debug(f"Saved metadata: {json_path}")
        
        return saved_paths
    
    def save_batch_summary(
        self,
        batch_info: Dict[str, Any],
        summary_filename: str = "batch_summary.json"
    ) -> Path:
        """
        Save summary information for a batch of generated images.
        
        Args:
            batch_info: Dictionary with batch generation information
            summary_filename: Name of summary file
            
        Returns:
            Path to saved summary file
        """
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(batch_info, f, indent=2)
        
        logger.info(f"Saved batch summary: {summary_path}")
        return summary_path


def generate_random_scene_parameters(
    physics_config: Dict[str, Any],
    num_objects_range: Tuple[int, int] = (0, 3),
    mine_probability: float = 0.3,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate random scene parameters for diverse synthetic data.
    
    Args:
        physics_config: Base physics configuration
        num_objects_range: Range for number of objects in scene
        mine_probability: Probability that an object is a mine
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with randomized scene parameters
        
    Requirements: 7.3 - Parameter randomization for domain variation
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomize physics parameters
    variation_ranges = {
        'cosine_exponent': physics_config.get('cosine_exponent_range', (2.0, 8.0)),
        'base_intensity': physics_config.get('base_intensity_range', (0.3, 0.8)),
        'texture_roughness': physics_config.get('texture_roughness_range', (0.2, 0.8)),
        'texture_scale': physics_config.get('texture_scale_range', (5.0, 20.0)),
        'noise_level': physics_config.get('noise_level_range', (0.1, 0.4)),
        'frequency_khz': physics_config.get('frequency_khz_range', (100.0, 500.0)),
        'beam_width_deg': physics_config.get('beam_width_deg_range', (1.0, 5.0))
    }
    
    # Generate random parameters within ranges
    scene_params = {}
    for param, (min_val, max_val) in variation_ranges.items():
        scene_params[param] = np.random.uniform(min_val, max_val)
    
    # Add fixed parameters
    scene_params.update({
        'sonar_position': (128, 256),  # Fixed sonar position
        'range_limits': (10.0, 200.0),
        'attenuation_coefficient': 2.0,
        'shadow_length_factor': 3.0,
        'shadow_intensity_factor': 0.1,
        'noise_type': np.random.choice(['rayleigh', 'gamma'])
    })
    
    # Generate random objects
    num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
    object_positions = []
    object_heights = []
    object_labels = []
    
    for _ in range(num_objects):
        # Random position (avoid edges and sonar position)
        x = np.random.uniform(200, 500)  # Right side of image
        y = np.random.uniform(50, 450)   # Avoid top/bottom edges
        object_positions.append((x, y))
        
        # Random height
        height = np.random.uniform(0.5, 3.0)
        object_heights.append(height)
        
        # Random label (mine vs rock)
        is_mine = np.random.random() < mine_probability
        object_labels.append(1 if is_mine else 0)
        
        # Set target material based on label
        if is_mine:
            scene_params['target_material'] = 'metal'
        else:
            scene_params['target_material'] = np.random.choice(['rock', 'sand'])
    
    scene_params.update({
        'object_positions': object_positions,
        'object_heights': object_heights,
        'object_labels': object_labels
    })
    
    return scene_params