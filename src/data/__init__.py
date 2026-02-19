"""Data handling modules for sonar detection system"""

from .synthetic_dataset import (
    SyntheticSonarDataset,
    create_synthetic_data_splits,
    create_synthetic_dataloaders
)
from .real_dataset import (
    RealSonarDataset,
    MinehuntingSonarDataset,
    CMREMuscleSASDataset,
    RealDatasetManager,
    create_combined_dataset
)
from .transforms import (
    SonarImageNormalize,
    SonarImageDenormalize,
    RandomRotation,
    RandomFlip,
    SonarNoiseInjection,
    RandomBrightnessContrast,
    RandomElasticDeformation,
    MetadataEncoder,
    SonarAugmentationPipeline,
    BatchCollator,
    create_transforms,
    create_dataloaders_with_transforms
)
from .data_loader import (
    SonarDataManager,
    create_data_manager
)

__all__ = [
    # Synthetic dataset
    'SyntheticSonarDataset',
    'create_synthetic_data_splits',
    'create_synthetic_dataloaders',
    
    # Real datasets
    'RealSonarDataset',
    'MinehuntingSonarDataset',
    'CMREMuscleSASDataset',
    'RealDatasetManager',
    'create_combined_dataset',
    
    # Transforms and preprocessing
    'SonarImageNormalize',
    'SonarImageDenormalize',
    'RandomRotation',
    'RandomFlip',
    'SonarNoiseInjection',
    'RandomBrightnessContrast',
    'RandomElasticDeformation',
    'MetadataEncoder',
    'SonarAugmentationPipeline',
    'BatchCollator',
    'create_transforms',
    'create_dataloaders_with_transforms',
    
    # Main data manager
    'SonarDataManager',
    'create_data_manager'
]