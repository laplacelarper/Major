#!/usr/bin/env python3
"""
Real Minehunting Sonar Dataset Loader

Loads multi-year sonar data (2010, 2015, 2017, 2018, 2021) with bounding box annotations.
Each year folder contains:
  - *.jpg: Sonar images
  - *.txt: Bounding box annotations (label, x_center, y_center, width, height)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Dict, Optional
import json

logger = logging.getLogger(__name__)


class MinehuntingAnnotation:
    """Parse and store bounding box annotation"""
    
    def __init__(self, label: int, x_center: float, y_center: float, 
                 width: float, height: float):
        self.label = label  # 0=rock, 1=mine
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
    
    def to_dict(self):
        return {
            'label': self.label,
            'x_center': self.x_center,
            'y_center': self.y_center,
            'width': self.width,
            'height': self.height
        }
    
    @staticmethod
    def from_line(line: str):
        """Parse annotation from txt line"""
        parts = line.strip().split()
        if len(parts) >= 5:
            return MinehuntingAnnotation(
                label=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4])
            )
        return None


class MinehuntingSonarDataset(Dataset):
    """
    Real Minehunting Sonar Dataset
    
    Loads sonar images from multiple years with bounding box annotations.
    Each image can have multiple objects (mines/rocks).
    """
    
    def __init__(self, 
                 data_dir: Path = Path('data/real/minehunting_sonar'),
                 years: Optional[List[int]] = None,
                 image_size: Tuple[int, int] = (512, 512),
                 transform=None,
                 use_image_level_labels: bool = True):
        """
        Args:
            data_dir: Root directory containing year folders
            years: List of years to include (default: all available)
            image_size: Target image size
            transform: Image transforms
            use_image_level_labels: If True, use image-level labels (any mine=1, all rocks=0)
                                   If False, use object-level labels
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.use_image_level_labels = use_image_level_labels
        
        # Available years
        available_years = [2010, 2015, 2017, 2018, 2021]
        self.years = years if years else available_years
        
        # Load all images and annotations
        self.samples = []
        self._load_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples from years {self.years}")
    
    def _load_dataset(self):
        """Load all images and annotations from year folders"""
        for year in self.years:
            year_dir = self.data_dir / str(year)
            if not year_dir.exists():
                logger.warning(f"Year directory not found: {year_dir}")
                continue
            
            # Find all jpg files
            image_files = sorted(year_dir.glob('*.jpg'))
            
            for image_file in image_files:
                # Get corresponding txt file
                txt_file = image_file.with_suffix('.txt')
                
                if not txt_file.exists():
                    logger.warning(f"Annotation file not found: {txt_file}")
                    continue
                
                # Parse annotations
                annotations = self._parse_annotations(txt_file)
                if not annotations:
                    continue
                
                self.samples.append({
                    'image_path': image_file,
                    'txt_path': txt_file,
                    'year': year,
                    'annotations': annotations,
                    'image_id': image_file.stem
                })
    
    def _parse_annotations(self, txt_file: Path) -> List[MinehuntingAnnotation]:
        """Parse bounding box annotations from txt file"""
        annotations = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    ann = MinehuntingAnnotation.from_line(line)
                    if ann is not None:
                        annotations.append(ann)
        except Exception as e:
            logger.error(f"Error parsing {txt_file}: {e}")
        
        return annotations
    
    def _get_image_level_label(self, annotations: List[MinehuntingAnnotation]) -> int:
        """Get image-level label: 1 if any mine, 0 if all rocks"""
        for ann in annotations:
            if ann.label == 1:  # Mine found
                return 1
        return 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('L')  # Grayscale
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Get label
        if self.use_image_level_labels:
            label = self._get_image_level_label(sample['annotations'])
        else:
            # Use first object's label (for object-level training)
            label = sample['annotations'][0].label if sample['annotations'] else 0
        
        # Apply transforms if any
        if self.transform:
            image_array = self.transform(image_array)
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0)  # Add channel dim
        
        # Convert annotations to dict for JSON serialization
        annotations_dict = [ann.to_dict() for ann in sample['annotations']]
        
        return {
            'image': image_tensor,
            'label': torch.LongTensor([label])[0],
            'image_id': sample['image_id'],
            'year': sample['year'],
            'num_objects': len(sample['annotations'])
        }


class MinehuntingDataManager:
    """Manage Minehunting dataset loading and splitting"""
    
    def __init__(self,
                 data_dir: Path = Path('data/real/minehunting_sonar'),
                 image_size: Tuple[int, int] = (512, 512),
                 batch_size: int = 16,
                 num_workers: int = 0,
                 transform=None):
        """
        Args:
            data_dir: Root directory
            image_size: Target image size
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            transform: Image transforms
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        # Load full dataset
        self.full_dataset = MinehuntingSonarDataset(
            data_dir=data_dir,
            image_size=image_size,
            transform=transform
        )
        
        logger.info(f"Loaded {len(self.full_dataset)} samples")
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.full_dataset),
            'years': {},
            'label_distribution': {'0': 0, '1': 0},
            'objects_per_image': []
        }
        
        for sample in self.full_dataset.samples:
            year = sample['year']
            if year not in stats['years']:
                stats['years'][year] = 0
            stats['years'][year] += 1
            
            # Count labels
            label = self.full_dataset._get_image_level_label(sample['annotations'])
            stats['label_distribution'][str(label)] += 1
            
            # Count objects
            stats['objects_per_image'].append(len(sample['annotations']))
        
        stats['avg_objects_per_image'] = np.mean(stats['objects_per_image'])
        
        return stats
    
    def get_train_val_test_split(self, 
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split dataset into train/val/test
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        np.random.seed(random_seed)
        
        n_samples = len(self.full_dataset)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(self.full_dataset, train_indices)
        val_dataset = Subset(self.full_dataset, val_indices)
        test_dataset = Subset(self.full_dataset, test_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        logger.info(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        return train_loader, val_loader, test_loader
    
    def get_year_split_loaders(self, 
                               test_year: int = 2021,
                               val_ratio: float = 0.15,
                               batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split by year: use one year for testing, split others for train/val
        
        Args:
            test_year: Year to use for testing
            val_ratio: Proportion of training data for validation
            batch_size: Override batch size
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or self.batch_size
        
        # Separate by year
        test_indices = []
        train_val_indices = []
        
        for idx, sample in enumerate(self.full_dataset.samples):
            if sample['year'] == test_year:
                test_indices.append(idx)
            else:
                train_val_indices.append(idx)
        
        # Split train/val
        np.random.seed(42)
        train_val_indices = np.random.permutation(train_val_indices)
        val_size = int(len(train_val_indices) * val_ratio)
        
        train_indices = train_val_indices[val_size:]
        val_indices = train_val_indices[:val_size]
        
        # Create dataloaders
        from torch.utils.data import Subset
        
        train_loader = DataLoader(
            Subset(self.full_dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            Subset(self.full_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            Subset(self.full_dataset, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        logger.info(f"Year split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)} (test_year={test_year})")
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("  MINEHUNTING DATASET LOADER TEST")
    print("="*70)
    
    # Load dataset
    manager = MinehuntingDataManager(
        batch_size=8
    )
    
    # Print stats
    stats = manager.get_dataset_stats()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Years: {stats['years']}")
    print(f"   Label distribution: {stats['label_distribution']}")
    print(f"   Avg objects per image: {stats['avg_objects_per_image']:.2f}")
    
    # Test loading
    print(f"\n🔄 Testing data loading...")
    train_loader, val_loader, test_loader = manager.get_train_val_test_split()
    
    batch = next(iter(train_loader))
    print(f"   Batch shape: {batch['image'].shape}")
    print(f"   Labels: {batch['label']}")
    print(f"   Image IDs: {batch['image_id']}")
    
    print(f"\n✅ Dataset loader working correctly!")
