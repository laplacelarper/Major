"""Data preprocessing and augmentation transforms for sonar images"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import cv2

from ..config.config import Config


logger = logging.getLogger(__name__)


class SonarImageNormalize:
    """Normalize sonar images to specified range"""
    
    def __init__(self, mean: float = 0.5, std: float = 0.5, input_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize normalization transform
        
        Args:
            mean: Target mean value
            std: Target standard deviation
            input_range: Expected input range (min, max)
        """
        self.mean = mean
        self.std = std
        self.input_range = input_range
    
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Apply normalization to image"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Ensure input is in expected range
        image = torch.clamp(image, self.input_range[0], self.input_range[1])
        
        # Normalize to [-1, 1] range if using mean=0.5, std=0.5
        image = (image - self.mean) / self.std
        
        return image


class SonarImageDenormalize:
    """Denormalize sonar images back to [0, 1] range"""
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply denormalization to image"""
        image = image * self.std + self.mean
        return torch.clamp(image, 0.0, 1.0)


class RandomRotation:
    """Random rotation with sonar-specific considerations"""
    
    def __init__(self, degrees: float = 30.0, probability: float = 0.5):
        """
        Initialize random rotation
        
        Args:
            degrees: Maximum rotation angle in degrees
            probability: Probability of applying rotation
        """
        self.degrees = degrees
        self.probability = probability
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to image"""
        if random.random() > self.probability:
            return image
        
        # Random angle
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Apply rotation
        if image.dim() == 2:
            image = image.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False
        
        # Use torchvision's functional rotation
        image = transforms.functional.rotate(
            image, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0
        )
        
        if squeeze_after:
            image = image.squeeze(0)
        
        return image


class RandomFlip:
    """Random horizontal and vertical flips"""
    
    def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.3):
        """
        Initialize random flip
        
        Args:
            horizontal_prob: Probability of horizontal flip
            vertical_prob: Probability of vertical flip
        """
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random flips to image"""
        # Horizontal flip
        if random.random() < self.horizontal_prob:
            image = torch.flip(image, dims=[-1])
        
        # Vertical flip
        if random.random() < self.vertical_prob:
            image = torch.flip(image, dims=[-2])
        
        return image


class SonarNoiseInjection:
    """Inject sonar-specific noise patterns"""
    
    def __init__(
        self,
        speckle_prob: float = 0.3,
        speckle_intensity: float = 0.1,
        gaussian_prob: float = 0.2,
        gaussian_std: float = 0.05
    ):
        """
        Initialize noise injection
        
        Args:
            speckle_prob: Probability of adding speckle noise
            speckle_intensity: Intensity of speckle noise
            gaussian_prob: Probability of adding Gaussian noise
            gaussian_std: Standard deviation of Gaussian noise
        """
        self.speckle_prob = speckle_prob
        self.speckle_intensity = speckle_intensity
        self.gaussian_prob = gaussian_prob
        self.gaussian_std = gaussian_std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply noise injection to image"""
        # Speckle noise (multiplicative)
        if random.random() < self.speckle_prob:
            speckle = torch.randn_like(image) * self.speckle_intensity + 1.0
            image = image * speckle
        
        # Gaussian noise (additive)
        if random.random() < self.gaussian_prob:
            noise = torch.randn_like(image) * self.gaussian_std
            image = image + noise
        
        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image


class RandomBrightnessContrast:
    """Random brightness and contrast adjustments"""
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.4
    ):
        """
        Initialize brightness/contrast adjustment
        
        Args:
            brightness_range: Range for brightness multiplier
            contrast_range: Range for contrast multiplier
            probability: Probability of applying adjustment
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.probability = probability
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness/contrast adjustment"""
        if random.random() > self.probability:
            return image
        
        # Random brightness and contrast factors
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        # Apply adjustments
        image = image * brightness  # Brightness
        image = (image - 0.5) * contrast + 0.5  # Contrast around midpoint
        
        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image


class RandomElasticDeformation:
    """Random elastic deformation for sonar images"""
    
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, probability: float = 0.2):
        """
        Initialize elastic deformation
        
        Args:
            alpha: Deformation strength
            sigma: Gaussian kernel standard deviation
            probability: Probability of applying deformation
        """
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random elastic deformation"""
        if random.random() > self.probability:
            return image
        
        # Convert to numpy for OpenCV operations
        if image.dim() == 3:
            image_np = image.squeeze(0).numpy()
        else:
            image_np = image.numpy()
        
        h, w = image_np.shape
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)) * self.alpha
        dy = np.random.uniform(-1, 1, (h, w)) * self.alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply deformation
        deformed = cv2.remap(image_np, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Convert back to tensor
        result = torch.from_numpy(deformed).float()
        if image.dim() == 3:
            result = result.unsqueeze(0)
        
        return result


class MetadataEncoder:
    """Encode physics metadata for auxiliary model inputs"""
    
    def __init__(self, config: Config):
        """
        Initialize metadata encoder
        
        Args:
            config: System configuration
        """
        self.config = config
        self.metadata_dim = config.model.metadata_dim
        
        # Define normalization ranges for metadata features
        self.feature_ranges = {
            'grazing_angle_deg': (0.0, 90.0),
            'seabed_roughness': (0.0, 1.0),
            'range_m': (10.0, 200.0),
            'noise_level': (0.0, 1.0),
            'frequency_khz': (50.0, 1000.0),
            'beam_width_deg': (0.5, 10.0),
            'target_material_encoded': (0.0, 1.0)
        }
    
    def __call__(self, metadata: Dict) -> torch.Tensor:
        """Encode metadata dictionary as normalized tensor"""
        if not metadata or not self.config.model.use_physics_metadata:
            return torch.zeros(self.metadata_dim, dtype=torch.float32)
        
        features = []
        
        for feature_name, (min_val, max_val) in self.feature_ranges.items():
            if feature_name == 'target_material_encoded':
                # Special handling for material encoding
                material = metadata.get('target_material', 'rock')
                if material == 'metal':
                    value = 1.0
                elif material == 'sand':
                    value = 0.5
                else:  # rock or unknown
                    value = 0.0
            else:
                # Get raw value with default
                raw_value = metadata.get(feature_name, (min_val + max_val) / 2)
                # Normalize to [0, 1] range
                value = (float(raw_value) - min_val) / (max_val - min_val)
                value = np.clip(value, 0.0, 1.0)
            
            features.append(value)
        
        # Pad or truncate to expected dimension
        while len(features) < self.metadata_dim:
            features.append(0.0)
        features = features[:self.metadata_dim]
        
        return torch.tensor(features, dtype=torch.float32)


class SonarAugmentationPipeline:
    """Complete augmentation pipeline for sonar images"""
    
    def __init__(self, config: Config, phase: str = "train"):
        """
        Initialize augmentation pipeline
        
        Args:
            config: System configuration
            phase: Training phase ("train", "val", "test")
        """
        self.config = config
        self.phase = phase
        self.metadata_encoder = MetadataEncoder(config)
        
        # Build transform pipeline based on phase
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> List:
        """Build list of transforms based on configuration and phase"""
        transforms_list = []
        
        if self.phase == "train" and self.config.data.use_augmentation:
            # Training augmentations
            transforms_list.extend([
                RandomRotation(
                    degrees=self.config.data.rotation_range,
                    probability=0.5
                ),
                RandomFlip(
                    horizontal_prob=self.config.data.flip_probability,
                    vertical_prob=0.3
                ),
                SonarNoiseInjection(
                    speckle_prob=self.config.data.noise_injection_prob,
                    speckle_intensity=0.1,
                    gaussian_prob=0.2,
                    gaussian_std=0.05
                ),
                RandomBrightnessContrast(
                    brightness_range=(0.8, 1.2),
                    contrast_range=(0.8, 1.2),
                    probability=0.4
                ),
                RandomElasticDeformation(
                    alpha=30.0,
                    sigma=3.0,
                    probability=0.15
                )
            ])
        
        # Always apply normalization
        if self.config.data.normalize_images:
            transforms_list.append(
                SonarImageNormalize(
                    mean=self.config.data.normalization_mean,
                    std=self.config.data.normalization_std
                )
            )
        
        return transforms_list
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply complete preprocessing pipeline to a sample"""
        # Extract components
        image = sample['image']
        metadata_dict = sample.get('metadata_dict', {})
        
        # Apply image transforms
        for transform in self.transforms:
            image = transform(image)
        
        # Encode metadata
        metadata_tensor = self.metadata_encoder(metadata_dict)
        
        # Update sample
        sample['image'] = image
        sample['metadata'] = metadata_tensor
        
        return sample


def create_transforms(config: Config) -> Dict[str, SonarAugmentationPipeline]:
    """
    Create transform pipelines for different phases
    
    Args:
        config: System configuration
    
    Returns:
        Dictionary of transform pipelines for each phase
    """
    return {
        'train': SonarAugmentationPipeline(config, 'train'),
        'val': SonarAugmentationPipeline(config, 'val'),
        'test': SonarAugmentationPipeline(config, 'test')
    }


class BatchCollator:
    """Custom collate function for sonar data batches"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        # Separate components
        images = []
        labels = []
        metadata = []
        image_ids = []
        sources = []
        metadata_dicts = []
        
        for sample in batch:
            images.append(sample['image'])
            labels.append(sample['label'])
            metadata.append(sample['metadata'])
            image_ids.append(sample['image_id'])
            sources.append(sample['source'])
            metadata_dicts.append(sample['metadata_dict'])
        
        # Stack tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)
        metadata_tensor = torch.stack(metadata)
        
        return {
            'images': images_tensor,
            'labels': labels_tensor,
            'metadata': metadata_tensor,
            'image_ids': image_ids,
            'sources': sources,
            'metadata_dicts': metadata_dicts
        }


def create_dataloaders_with_transforms(
    datasets: Dict[str, torch.utils.data.Dataset],
    config: Config
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders with appropriate transforms and collation
    
    Args:
        datasets: Dictionary of datasets for each split
        config: System configuration
    
    Returns:
        Dictionary of DataLoaders for each split
    """
    collator = BatchCollator(config)
    dataloaders = {}
    
    # Batch sizes for different phases
    batch_sizes = {
        'train': config.training.phase1_batch_size,
        'val': config.training.phase1_batch_size,
        'test': config.training.phase1_batch_size
    }
    
    for split, dataset in datasets.items():
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_sizes.get(split, config.training.phase1_batch_size),
            shuffle=(split == 'train'),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=(split == 'train'),
            collate_fn=collator
        )
    
    return dataloaders