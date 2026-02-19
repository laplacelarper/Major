"""Core physics engine for synthetic sonar data generation"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from .renderer import SonarImageRenderer, ImageExporter, PhysicsMetadata, generate_random_scene_parameters
from .calculations import *
from .noise import *

logger = logging.getLogger(__name__)


class PhysicsEngine:
    """
    Main physics engine for synthetic sonar data generation.
    
    Combines all physics calculations, noise generation, and image rendering
    into a unified interface for creating realistic sonar imagery.
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.3 - Complete physics-informed synthetic data generation
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the physics engine.
        
        Args:
            image_size: Output image size (height, width)
            output_dir: Directory for saving generated images (optional)
        """
        self.image_size = image_size
        self.renderer = SonarImageRenderer(image_size)
        
        if output_dir:
            self.exporter = ImageExporter(output_dir)
        else:
            self.exporter = None
        
        logger.info(f"Initialized PhysicsEngine with image size {image_size}")
    
    def generate_single_image(
        self,
        physics_params: Optional[Dict[str, Any]] = None,
        object_positions: Optional[List[Tuple[float, float]]] = None,
        object_heights: Optional[List[float]] = None,
        object_labels: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, int, PhysicsMetadata]:
        """
        Generate a single synthetic sonar image.
        
        Args:
            physics_params: Physics parameters (will use defaults if None)
            object_positions: Object positions as (x, y) tuples
            object_heights: Object heights above seabed
            object_labels: Object labels (0=rock, 1=mine)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image_array, label, metadata)
            
        Requirements: 1.1 - Generate 512x512 grayscale PNG images with realistic sonar characteristics
        """
        if physics_params is None:
            # Use default parameters
            physics_params = self._get_default_physics_params()
        
        # Generate the image using the renderer
        image, label, metadata = self.renderer.render_sonar_image(
            physics_params=physics_params,
            object_positions=object_positions,
            object_heights=object_heights,
            object_labels=object_labels,
            random_seed=random_seed
        )
        
        logger.debug(f"Generated single sonar image with label {label}")
        
        return image, label, metadata
    
    def generate_dataset(
        self,
        num_samples: int,
        physics_config: Dict[str, Any],
        save_to_disk: bool = True,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[PhysicsMetadata]]:
        """
        Generate a complete dataset of synthetic sonar images.
        
        Args:
            num_samples: Number of images to generate
            physics_config: Configuration for physics parameter ranges
            save_to_disk: Whether to save images to disk
            random_seed: Base random seed for reproducibility
            
        Returns:
            Tuple of (images_array, labels_array, metadata_list)
            
        Requirements: 1.1, 7.3 - Generate synthetic dataset with parameter variation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        images = []
        labels = []
        metadata_list = []
        
        logger.info(f"Generating dataset with {num_samples} synthetic sonar images")
        
        for i in range(num_samples):
            # Generate random scene parameters
            sample_seed = (random_seed + i) if random_seed else None
            scene_params = generate_random_scene_parameters(
                physics_config=physics_config,
                random_seed=sample_seed
            )
            
            # Extract object parameters
            object_positions = scene_params.pop('object_positions', [])
            object_heights = scene_params.pop('object_heights', [])
            object_labels = scene_params.pop('object_labels', [])
            
            # Generate image
            image, label, metadata = self.generate_single_image(
                physics_params=scene_params,
                object_positions=object_positions,
                object_heights=object_heights,
                object_labels=object_labels,
                random_seed=sample_seed
            )
            
            images.append(image)
            labels.append(label)
            metadata_list.append(metadata)
            
            # Save to disk if requested
            if save_to_disk and self.exporter:
                image_id = f"synthetic_{i:06d}"
                self.exporter.save_image_with_metadata(
                    image=image,
                    label=label,
                    metadata=metadata,
                    image_id=image_id
                )
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} images")
        
        # Convert to numpy arrays
        images_array = np.stack(images, axis=0)
        labels_array = np.array(labels)
        
        # Save batch summary if saving to disk
        if save_to_disk and self.exporter:
            batch_info = {
                'num_samples': num_samples,
                'image_size': self.image_size,
                'physics_config': physics_config,
                'random_seed': random_seed,
                'label_distribution': {
                    'no_mine': int(np.sum(labels_array == 0)),
                    'mine_present': int(np.sum(labels_array == 1))
                },
                'dataset_stats': {
                    'mean_intensity': float(images_array.mean()),
                    'std_intensity': float(images_array.std()),
                    'min_intensity': float(images_array.min()),
                    'max_intensity': float(images_array.max())
                }
            }
            self.exporter.save_batch_summary(batch_info)
        
        logger.info(f"Dataset generation complete. "
                   f"Mine samples: {np.sum(labels_array == 1)}, "
                   f"No-mine samples: {np.sum(labels_array == 0)}")
        
        return images_array, labels_array, metadata_list
    
    def _get_default_physics_params(self) -> Dict[str, Any]:
        """Get default physics parameters for image generation."""
        return {
            'sonar_position': (self.image_size[1] // 4, self.image_size[0] // 2),
            'cosine_exponent': 4.0,
            'base_intensity': 0.5,
            'range_limits': (10.0, 200.0),
            'attenuation_coefficient': 2.0,
            'shadow_length_factor': 3.0,
            'shadow_intensity_factor': 0.1,
            'texture_roughness': 0.5,
            'texture_scale': 10.0,
            'noise_type': 'rayleigh',
            'noise_level': 0.2,
            'frequency_khz': 300.0,
            'beam_width_deg': 2.0,
            'target_material': 'rock'
        }
    
    def validate_physics_calculations(self) -> Dict[str, bool]:
        """
        Validate physics calculations with known test cases.
        
        Returns:
            Dictionary with validation results for each physics component
        """
        validation_results = {}
        
        try:
            # Test backscatter calculation
            test_angles = np.array([[0, 30, 60, 90]])
            backscatter = calculate_backscatter_intensity(test_angles, cosine_exponent=2.0)
            # At 0 degrees (normal incidence), should have maximum intensity
            # At 90 degrees (grazing), should have minimum intensity
            validation_results['backscatter'] = (
                backscatter[0, 0] > backscatter[0, -1] and
                np.all(backscatter >= 0) and np.all(backscatter <= 1)
            )
            
            # Test range attenuation
            test_ranges = np.array([[10, 50, 100, 200]])
            attenuation = calculate_range_attenuation(test_ranges, attenuation_coefficient=2.0)
            # Closer ranges should have less attenuation (higher values)
            validation_results['range_attenuation'] = (
                attenuation[0, 0] > attenuation[0, -1] and
                np.all(attenuation >= 0) and np.all(attenuation <= 1)
            )
            
            # Test shadow generation
            object_pos = np.array([[256, 256]])
            object_heights = np.array([2.0])
            shadows = generate_acoustic_shadows(
                object_pos, object_heights, (128, 256), (512, 512)
            )
            # Shadow mask should be mostly 1.0 with some regions < 1.0
            validation_results['acoustic_shadows'] = (
                shadows.shape == (512, 512) and
                np.all(shadows >= 0) and np.all(shadows <= 1) and
                np.any(shadows < 1.0)
            )
            
            # Test noise generation
            noise = generate_speckle_noise((100, 100), noise_type='rayleigh', noise_level=0.2)
            validation_results['speckle_noise'] = (
                noise.shape == (100, 100) and
                np.all(noise > 0) and
                0.5 < np.mean(noise) < 1.5  # Should be around 1.0 for multiplicative noise
            )
            
            # Test texture generation
            texture = generate_seabed_texture((100, 100), roughness=0.5, texture_scale=10.0)
            validation_results['seabed_texture'] = (
                texture.shape == (100, 100) and
                np.all(texture >= 0) and np.all(texture <= 1)
            )
            
        except Exception as e:
            logger.error(f"Physics validation failed: {e}")
            validation_results['error'] = str(e)
        
        # Log validation results
        passed_tests = sum(1 for result in validation_results.values() if result is True)
        total_tests = len([k for k in validation_results.keys() if k != 'error'])
        
        logger.info(f"Physics validation: {passed_tests}/{total_tests} tests passed")
        for test_name, result in validation_results.items():
            if test_name != 'error':
                status = "PASS" if result else "FAIL"
                logger.info(f"  {test_name}: {status}")
        
        return validation_results
    
    def get_physics_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the valid ranges for all physics parameters.
        
        Returns:
            Dictionary mapping parameter names to (min, max) ranges
        """
        return {
            'cosine_exponent': (1.0, 10.0),
            'base_intensity': (0.1, 1.0),
            'range_min': (1.0, 50.0),
            'range_max': (50.0, 500.0),
            'attenuation_coefficient': (1.0, 4.0),
            'shadow_length_factor': (1.0, 10.0),
            'shadow_intensity_factor': (0.0, 0.5),
            'texture_roughness': (0.0, 1.0),
            'texture_scale': (1.0, 50.0),
            'noise_level': (0.0, 1.0),
            'frequency_khz': (50.0, 1000.0),
            'beam_width_deg': (0.5, 10.0),
            'grazing_angle_deg': (0.0, 90.0)
        }