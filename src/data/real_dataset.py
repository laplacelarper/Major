"""Real dataset integration for public sonar datasets"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import csv

from ..config.config import Config


logger = logging.getLogger(__name__)


class RealSonarDataset(Dataset):
    """Base class for real sonar datasets"""
    
    def __init__(
        self,
        data_dir: Path,
        config: Config,
        dataset_name: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize real sonar dataset
        
        Args:
            data_dir: Directory containing real data
            config: System configuration
            dataset_name: Name of the dataset
            split: Dataset split ("train", "val", "test")
            transform: Optional data transforms
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Dataset citation information
        self.citation_info = self._get_citation_info()
        
        # Initialize data structures
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.metadata: List[Dict] = []
        self.image_ids: List[str] = []
        
        # Load dataset
        self._load_dataset()
        self._validate_dataset()
        
        logger.info(f"Loaded {len(self)} real samples from {dataset_name} for {split} split")
    
    def _get_citation_info(self) -> Dict[str, str]:
        """Get citation information for the dataset"""
        citations = {
            "minehunting_sonar": {
                "name": "Minehunting Sonar Image Dataset",
                "source": "Naval Research Laboratory",
                "url": "https://www.nrl.navy.mil/",
                "description": "Side-scan sonar images for mine detection research",
                "license": "Public Domain (U.S. Government Work)"
            },
            "cmre_muscle_sas": {
                "name": "CMRE MUSCLE SAS Dataset",
                "source": "NATO Centre for Maritime Research and Experimentation",
                "url": "https://www.cmre.nato.int/",
                "description": "Synthetic Aperture Sonar images from MUSCLE project",
                "license": "Research Use Only"
            }
        }
        
        return citations.get(self.dataset_name, {
            "name": f"Unknown Dataset: {self.dataset_name}",
            "source": "Unknown",
            "url": "",
            "description": "No description available",
            "license": "Unknown"
        })
    
    def _load_dataset(self):
        """Load real dataset from disk - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
    def _validate_dataset(self):
        """Validate dataset integrity"""
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {self.dataset_name} dataset")
        
        # Check that all lists have same length
        lengths = [
            len(self.image_paths),
            len(self.labels),
            len(self.metadata),
            len(self.image_ids)
        ]
        
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Inconsistent dataset lengths in {self.dataset_name}")
        
        # Validate image files exist
        missing_files = [p for p in self.image_paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing image files in {self.dataset_name}: {missing_files[:5]}...")
        
        # Check label distribution
        unique_labels = set(self.labels)
        label_counts = {label: self.labels.count(label) for label in unique_labels}
        logger.info(f"{self.dataset_name} label distribution: {label_counts}")
        
        # Log citation information
        logger.info(f"Dataset: {self.citation_info['name']}")
        logger.info(f"Source: {self.citation_info['source']}")
    
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
            
            # Prepare metadata tensor (real datasets typically have limited metadata)
            metadata_tensor = self._encode_metadata(metadata)
            
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'metadata': metadata_tensor,
                'metadata_dict': metadata,
                'image_id': image_id,
                'source': self.dataset_name
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {self.dataset_name}: {e}")
            raise
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to target size
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
        """Encode metadata as tensor (real datasets have limited metadata)"""
        # Real datasets typically don't have physics metadata
        # Return zero tensor to maintain compatibility
        return torch.zeros(self.config.model.metadata_dim, dtype=torch.float32)
    
    def get_citation_info(self) -> Dict[str, str]:
        """Get citation information for this dataset"""
        return self.citation_info.copy()


class MinehuntingSonarDataset(RealSonarDataset):
    """Minehunting Sonar Image Dataset loader"""
    
    def __init__(self, data_dir: Path, config: Config, split: str = "train", transform=None, max_samples: Optional[int] = None):
        super().__init__(data_dir, config, "minehunting_sonar", split, transform, max_samples)
    
    def _load_dataset(self):
        """Load Minehunting sonar dataset"""
        dataset_dir = self.data_dir / "real" / "minehunting_sonar"
        
        if not dataset_dir.exists():
            logger.warning(f"Minehunting dataset directory not found: {dataset_dir}")
            return
        
        # Look for standard directory structure
        # Assuming structure: minehunting_sonar/images/ and minehunting_sonar/labels.csv
        images_dir = dataset_dir / "images"
        labels_file = dataset_dir / "labels.csv"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return
        
        # Load labels if available
        labels_dict = {}
        if labels_file.exists():
            try:
                with open(labels_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        image_id = row.get('image_id', row.get('filename', ''))
                        label = int(row.get('label', row.get('class', 0)))
                        labels_dict[image_id] = label
                logger.info(f"Loaded labels for {len(labels_dict)} images")
            except Exception as e:
                logger.warning(f"Failed to load labels file: {e}")
        
        # Load images
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.tiff"))
        
        for img_path in sorted(image_files)[:self.max_samples]:
            image_id = img_path.stem
            
            # Get label from labels file or infer from filename
            if image_id in labels_dict:
                label = labels_dict[image_id]
            else:
                # Try to infer from filename (e.g., mine_001.png or rock_001.png)
                if 'mine' in image_id.lower():
                    label = 1
                elif 'rock' in image_id.lower() or 'clutter' in image_id.lower():
                    label = 0
                else:
                    label = 0  # Default to non-mine
            
            # Add to dataset
            self.image_paths.append(img_path)
            self.labels.append(label)
            self.metadata.append({})  # No physics metadata for real data
            self.image_ids.append(image_id)


class CMREMuscleSASDataset(RealSonarDataset):
    """CMRE MUSCLE SAS Dataset loader"""
    
    def __init__(self, data_dir: Path, config: Config, split: str = "train", transform=None, max_samples: Optional[int] = None):
        super().__init__(data_dir, config, "cmre_muscle_sas", split, transform, max_samples)
    
    def _load_dataset(self):
        """Load CMRE MUSCLE SAS dataset"""
        dataset_dir = self.data_dir / "real" / "cmre_muscle_sas"
        
        if not dataset_dir.exists():
            logger.warning(f"CMRE MUSCLE dataset directory not found: {dataset_dir}")
            return
        
        # Look for standard directory structure
        images_dir = dataset_dir / "images"
        annotations_file = dataset_dir / "annotations.json"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return
        
        # Load annotations if available
        annotations_dict = {}
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r') as f:
                    annotations = json.load(f)
                    for ann in annotations.get('annotations', []):
                        image_id = ann.get('image_id', '')
                        label = ann.get('category_id', 0)
                        annotations_dict[image_id] = label
                logger.info(f"Loaded annotations for {len(annotations_dict)} images")
            except Exception as e:
                logger.warning(f"Failed to load annotations file: {e}")
        
        # Load images
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.tiff"))
        
        for img_path in sorted(image_files)[:self.max_samples]:
            image_id = img_path.stem
            
            # Get label from annotations or infer from filename
            if image_id in annotations_dict:
                label = annotations_dict[image_id]
            else:
                # Try to infer from filename or directory structure
                if 'target' in image_id.lower() or 'mine' in image_id.lower():
                    label = 1
                else:
                    label = 0  # Default to non-mine
            
            # Add to dataset
            self.image_paths.append(img_path)
            self.labels.append(label)
            self.metadata.append({})  # No physics metadata for real data
            self.image_ids.append(image_id)


class RealDatasetManager:
    """Manager for real sonar datasets with usage tracking and citation management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.usage_tracker = {}
        self.citations = {}
        self.max_real_data_percentage = config.data.real_data_percentage
        
    def load_real_datasets(
        self,
        data_dir: Path,
        synthetic_dataset_size: int,
        transform=None
    ) -> Tuple[List[Dataset], Dict[str, str]]:
        """
        Load real datasets with 30% usage limitation
        
        Args:
            data_dir: Directory containing real data
            synthetic_dataset_size: Size of synthetic dataset for percentage calculation
            transform: Optional data transforms
        
        Returns:
            Tuple of (list of datasets, citation information)
        """
        datasets = []
        all_citations = {}
        
        # Calculate maximum real data samples allowed
        max_real_samples = int(synthetic_dataset_size * self.max_real_data_percentage / (1 - self.max_real_data_percentage))
        
        logger.info(f"Maximum real data samples allowed: {max_real_samples} ({self.max_real_data_percentage*100}%)")
        
        # Load each configured real dataset
        for dataset_name in self.config.data.real_datasets:
            try:
                dataset = self._load_single_dataset(
                    data_dir, dataset_name, transform, max_real_samples // len(self.config.data.real_datasets)
                )
                
                if len(dataset) > 0:
                    datasets.append(dataset)
                    all_citations[dataset_name] = dataset.get_citation_info()
                    self.usage_tracker[dataset_name] = len(dataset)
                    logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Log usage statistics
        total_real_samples = sum(len(d) for d in datasets)
        actual_percentage = total_real_samples / (synthetic_dataset_size + total_real_samples) * 100
        
        logger.info(f"Total real data samples: {total_real_samples}")
        logger.info(f"Actual real data percentage: {actual_percentage:.1f}%")
        
        if actual_percentage > self.max_real_data_percentage * 100:
            logger.warning(f"Real data percentage ({actual_percentage:.1f}%) exceeds limit ({self.max_real_data_percentage*100}%)")
        
        return datasets, all_citations
    
    def _load_single_dataset(
        self,
        data_dir: Path,
        dataset_name: str,
        transform=None,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Load a single real dataset"""
        if dataset_name == "minehunting_sonar":
            return MinehuntingSonarDataset(data_dir, self.config, transform=transform, max_samples=max_samples)
        elif dataset_name == "cmre_muscle_sas":
            return CMREMuscleSASDataset(data_dir, self.config, transform=transform, max_samples=max_samples)
        else:
            raise ValueError(f"Unknown real dataset: {dataset_name}")
    
    def get_usage_report(self) -> Dict[str, Union[int, float]]:
        """Get usage statistics for real datasets"""
        total_samples = sum(self.usage_tracker.values())
        
        report = {
            'total_real_samples': total_samples,
            'datasets': self.usage_tracker.copy(),
            'max_allowed_percentage': self.max_real_data_percentage * 100
        }
        
        return report
    
    def get_all_citations(self) -> Dict[str, Dict[str, str]]:
        """Get citation information for all loaded datasets"""
        return self.citations.copy()
    
    def export_citations(self, output_path: Path):
        """Export citation information to file"""
        citation_data = {
            'usage_date': str(Path.cwd()),
            'datasets': self.citations,
            'usage_statistics': self.get_usage_report()
        }
        
        with open(output_path, 'w') as f:
            json.dump(citation_data, f, indent=2)
        
        logger.info(f"Exported citation information to {output_path}")


def create_combined_dataset(
    data_dir: Path,
    config: Config,
    synthetic_dataset_size: int,
    transform_synthetic=None,
    transform_real=None
) -> Tuple[ConcatDataset, Dict[str, str]]:
    """
    Create combined dataset with synthetic and real data
    
    Args:
        data_dir: Directory containing data
        config: System configuration
        synthetic_dataset_size: Size of synthetic dataset
        transform_synthetic: Transforms for synthetic data
        transform_real: Transforms for real data
    
    Returns:
        Tuple of (combined dataset, citation information)
    """
    from .synthetic_dataset import SyntheticSonarDataset
    
    # Load synthetic dataset
    synthetic_dataset = SyntheticSonarDataset(
        data_dir=data_dir,
        config=config,
        transform=transform_synthetic
    )
    
    # Load real datasets
    real_manager = RealDatasetManager(config)
    real_datasets, citations = real_manager.load_real_datasets(
        data_dir, len(synthetic_dataset), transform_real
    )
    
    # Combine all datasets
    all_datasets = [synthetic_dataset] + real_datasets
    combined_dataset = ConcatDataset(all_datasets)
    
    logger.info(f"Created combined dataset with {len(combined_dataset)} total samples")
    logger.info(f"Synthetic: {len(synthetic_dataset)}, Real: {sum(len(d) for d in real_datasets)}")
    
    return combined_dataset, citations