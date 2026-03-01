"""Synthetic dataset loader for physics-informed sonar images"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from ..config.config import Config


logger = logging.getLogger(__name__)


class SyntheticSonarDataset(Dataset):
    """Dataset class for synthetic sonar images with physics metadata"""
    
    def __init__(
        self,
        data_dir: Path,
        config: Config,
        split: str = "train",
        transform=None,
        load_metadata: bool = True
    ):
        """
        Initialize synthetic sonar dataset
        
        Args:
            data_dir: Directory containing synthetic data
            config: System configuration
            split: Dataset split ("train", "val", "test")
            transform: Optional data transforms
            load_metadata: Whether to load physics metadata
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        self.load_metadata = load_metadata
        
        # Initialize data structures
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.metadata: List[Dict] = []
        self.image_ids: List[str] = []
        
        # Load dataset
        self._load_dataset()
        self._validate_dataset()
        
        logger.info(f"Loaded {len(self)} synthetic samples for {split} split")
    
    def _load_dataset(self):
        """Load synthetic dataset from disk"""
        synthetic_dir = self.data_dir / "synthetic"
        
        if not synthetic_dir.exists():
            raise FileNotFoundError(f"Synthetic data directory not found: {synthetic_dir}")
        
        # Look for images and metadata files
        image_files = list(synthetic_dir.glob("*.png"))
        metadata_file = synthetic_dir / "metadata.json"
        
        if not image_files:
            raise ValueError(f"No PNG images found in {synthetic_dir}")
        
        # Load metadata if available
        metadata_dict = {}
        if metadata_file.exists() and self.load_metadata:
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                logger.info(f"Loaded metadata for {len(metadata_dict)} images")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                metadata_dict = {}
        
        # Process each image file
        for img_path in sorted(image_files):
            image_id = img_path.stem
            
            # Extract label from filename (assuming format: label_id.png)
            try:
                if '_' in image_id:
                    label_str = image_id.split('_')[0]
                    label = int(label_str)
                else:
                    # Default to 0 if no label in filename
                    label = 0
            except ValueError:
                logger.warning(f"Could not extract label from {image_id}, using 0")
                label = 0
            
            # Get metadata for this image
            img_metadata = metadata_dict.get(image_id, {})
            
            # Add to dataset
            self.image_paths.append(img_path)
            self.labels.append(label)
            self.metadata.append(img_metadata)
            self.image_ids.append(image_id)
    
    def _validate_dataset(self):
        """Validate dataset integrity"""
        if len(self.image_paths) == 0:
            raise ValueError("No valid images found in dataset")
        
        # Check that all lists have same length
        lengths = [
            len(self.image_paths),
            len(self.labels),
            len(self.metadata),
            len(self.image_ids)
        ]
        
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("Inconsistent dataset lengths")
        
        # Validate image files exist
        missing_files = [p for p in self.image_paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing image files: {missing_files[:5]}...")
        
        # Check label distribution
        unique_labels = set(self.labels)
        label_counts = {label: self.labels.count(label) for label in unique_labels}
        logger.info(f"Label distribution: {label_counts}")
        
        # Validate metadata if loaded
        if self.load_metadata and self.metadata:
            metadata_keys = set()
            for meta in self.metadata:
                metadata_keys.update(meta.keys())
            logger.info(f"Available metadata keys: {sorted(metadata_keys)}")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, Dict]]:
        """Get a single sample from the dataset"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = self._load_image(image_path)
            
            # Get label and metadata
            label = self.labels[idx]
            metadata = self.metadata[idx].copy() if self.metadata[idx] else {}
            image_id = self.image_ids[idx]
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            # Convert to tensor if not already
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            
            # Ensure correct shape (C, H, W)
            if image.dim() == 2:
                image = image.unsqueeze(0)  # Add channel dimension
            
            # Prepare metadata tensor if available
            metadata_tensor = self._encode_metadata(metadata)
            
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'metadata': metadata_tensor,
                'metadata_dict': metadata,
                'image_id': image_id,
                'source': 'synthetic'
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({self.image_ids[idx]}): {e}")
            raise
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize if needed
                target_size = self.config.data.image_size
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                image = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1] range
                if self.config.data.normalize_images:
                    image = image / 255.0
                
                return image
                
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def _encode_metadata(self, metadata: Dict) -> torch.Tensor:
        """Encode physics metadata as tensor"""
        if not metadata or not self.config.model.use_physics_metadata:
            # Return zero tensor if no metadata
            return torch.zeros(self.config.model.metadata_dim, dtype=torch.float32)
        
        # Define expected metadata keys and their default values
        expected_keys = {
            'grazing_angle_deg': 45.0,
            'seabed_roughness': 0.5,
            'range_m': 100.0,
            'noise_level': 0.2,
            'frequency_khz': 300.0,
            'beam_width_deg': 3.0,
            'target_material_encoded': 0.0  # 0=rock, 1=metal, 0.5=sand
        }
        
        # Extract values with defaults
        values = []
        for key, default_val in expected_keys.items():
            if key == 'target_material_encoded':
                # Special handling for material encoding
                material = metadata.get('target_material', 'rock')
                if material == 'metal':
                    values.append(1.0)
                elif material == 'sand':
                    values.append(0.5)
                else:  # rock or unknown
                    values.append(0.0)
            else:
                values.append(float(metadata.get(key, default_val)))
        
        return torch.tensor(values, dtype=torch.float32)
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample"""
        return {
            'index': idx,
            'image_path': str(self.image_paths[idx]),
            'label': self.labels[idx],
            'metadata': self.metadata[idx],
            'image_id': self.image_ids[idx]
        }
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in the dataset"""
        unique_labels = set(self.labels)
        return {label: self.labels.count(label) for label in sorted(unique_labels)}


def create_synthetic_data_splits(
    data_dir: Path,
    config: Config,
    transform_train=None,
    transform_val=None,
    transform_test=None
) -> Tuple[SyntheticSonarDataset, SyntheticSonarDataset, SyntheticSonarDataset]:
    """
    Create train/validation/test splits for synthetic data
    
    Args:
        data_dir: Directory containing synthetic data
        config: System configuration
        transform_train: Transforms for training data
        transform_val: Transforms for validation data
        transform_test: Transforms for test data
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create full dataset
    full_dataset = SyntheticSonarDataset(
        data_dir=data_dir,
        config=config,
        split="full",
        transform=None,  # Apply transforms later
        load_metadata=True
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.data.train_split * total_size)
    val_size = int(config.data.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f"Splitting {total_size} samples: train={train_size}, val={val_size}, test={test_size}")
    
    # Create random splits
    generator = torch.Generator().manual_seed(config.random_seed)
    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create subset datasets
    train_dataset = SyntheticSonarSubset(
        full_dataset, train_indices.indices, "train", transform_train
    )
    val_dataset = SyntheticSonarSubset(
        full_dataset, val_indices.indices, "val", transform_val
    )
    test_dataset = SyntheticSonarSubset(
        full_dataset, test_indices.indices, "test", transform_test
    )
    
    return train_dataset, val_dataset, test_dataset


class SyntheticSonarSubset(Dataset):
    """Subset of SyntheticSonarDataset with specific indices"""
    
    def __init__(self, dataset: SyntheticSonarDataset, indices: List[int], split: str, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.split = split
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, Dict]]:
        # Get sample from original dataset
        original_idx = self.indices[idx]
        sample = self.dataset[original_idx]
        
        # Apply subset-specific transform if provided
        # Note: transform should operate on the entire sample dict, not just the image
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get label distribution for this subset"""
        labels = [self.dataset.labels[idx] for idx in self.indices]
        unique_labels = set(labels)
        return {label: labels.count(label) for label in sorted(unique_labels)}


def create_synthetic_dataloaders(
    data_dir: Path,
    config: Config,
    transform_train=None,
    transform_val=None,
    transform_test=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for synthetic data splits
    
    Args:
        data_dir: Directory containing synthetic data
        config: System configuration
        transform_train: Transforms for training data
        transform_val: Transforms for validation data
        transform_test: Transforms for test data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_synthetic_data_splits(
        data_dir, config, transform_train, transform_val, transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.phase1_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.phase1_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.phase1_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader