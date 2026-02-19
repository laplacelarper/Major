"""Main data loading interface for the sonar detection system"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from ..config.config import Config
from .synthetic_dataset import SyntheticSonarDataset, create_synthetic_data_splits
from .real_dataset import RealDatasetManager, create_combined_dataset
from .transforms import create_transforms, BatchCollator


logger = logging.getLogger(__name__)


class SonarDataManager:
    """Main data manager for the sonar detection system"""
    
    def __init__(self, config: Config):
        """
        Initialize data manager
        
        Args:
            config: System configuration
        """
        self.config = config
        self.transforms = create_transforms(config)
        self.collator = BatchCollator(config)
        self.real_manager = RealDatasetManager(config)
        
        # Storage for loaded datasets and dataloaders
        self.datasets = {}
        self.dataloaders = {}
        self.citations = {}
        
        logger.info("Initialized SonarDataManager")
    
    def load_synthetic_data(
        self,
        data_dir: Path,
        create_splits: bool = True
    ) -> Union[SyntheticSonarDataset, Tuple[SyntheticSonarDataset, SyntheticSonarDataset, SyntheticSonarDataset]]:
        """
        Load synthetic sonar dataset
        
        Args:
            data_dir: Directory containing synthetic data
            create_splits: Whether to create train/val/test splits
        
        Returns:
            Single dataset or tuple of (train, val, test) datasets
        """
        logger.info("Loading synthetic sonar dataset...")
        
        if create_splits:
            train_dataset, val_dataset, test_dataset = create_synthetic_data_splits(
                data_dir=data_dir,
                config=self.config,
                transform_train=self.transforms['train'],
                transform_val=self.transforms['val'],
                transform_test=self.transforms['test']
            )
            
            self.datasets.update({
                'synthetic_train': train_dataset,
                'synthetic_val': val_dataset,
                'synthetic_test': test_dataset
            })
            
            logger.info(f"Created synthetic data splits: train={len(train_dataset)}, "
                       f"val={len(val_dataset)}, test={len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
        else:
            dataset = SyntheticSonarDataset(
                data_dir=data_dir,
                config=self.config,
                transform=self.transforms['train']
            )
            
            self.datasets['synthetic_full'] = dataset
            logger.info(f"Loaded full synthetic dataset: {len(dataset)} samples")
            
            return dataset
    
    def load_real_data(self, data_dir: Path) -> Tuple[List[torch.utils.data.Dataset], Dict[str, str]]:
        """
        Load real sonar datasets
        
        Args:
            data_dir: Directory containing real data
        
        Returns:
            Tuple of (list of datasets, citation information)
        """
        logger.info("Loading real sonar datasets...")
        
        # Get synthetic dataset size for percentage calculation
        synthetic_size = self.config.data.synthetic_dataset_size
        if 'synthetic_full' in self.datasets:
            synthetic_size = len(self.datasets['synthetic_full'])
        elif 'synthetic_train' in self.datasets:
            synthetic_size = len(self.datasets['synthetic_train'])
        
        real_datasets, citations = self.real_manager.load_real_datasets(
            data_dir=data_dir,
            synthetic_dataset_size=synthetic_size,
            transform=self.transforms['train']
        )
        
        self.citations.update(citations)
        
        # Store individual real datasets
        for i, dataset in enumerate(real_datasets):
            dataset_name = f"real_{dataset.dataset_name}"
            self.datasets[dataset_name] = dataset
        
        logger.info(f"Loaded {len(real_datasets)} real datasets")
        
        return real_datasets, citations
    
    def create_combined_dataset(
        self,
        data_dir: Path,
        use_real_data: bool = True
    ) -> Tuple[torch.utils.data.Dataset, Dict[str, str]]:
        """
        Create combined dataset with synthetic and optionally real data
        
        Args:
            data_dir: Directory containing data
            use_real_data: Whether to include real data
        
        Returns:
            Tuple of (combined dataset, citation information)
        """
        logger.info("Creating combined dataset...")
        
        # Load synthetic data if not already loaded
        if 'synthetic_full' not in self.datasets:
            self.load_synthetic_data(data_dir, create_splits=False)
        
        synthetic_dataset = self.datasets['synthetic_full']
        all_datasets = [synthetic_dataset]
        citations = {}
        
        # Add real data if requested
        if use_real_data:
            real_datasets, real_citations = self.load_real_data(data_dir)
            all_datasets.extend(real_datasets)
            citations.update(real_citations)
        
        # Create combined dataset
        combined_dataset = ConcatDataset(all_datasets)
        
        self.datasets['combined'] = combined_dataset
        self.citations.update(citations)
        
        logger.info(f"Created combined dataset with {len(combined_dataset)} total samples")
        
        return combined_dataset, citations
    
    def create_dataloaders(
        self,
        data_dir: Path,
        phase: str = "phase1",
        use_real_data: bool = True,
        create_combined: bool = False
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for training phases
        
        Args:
            data_dir: Directory containing data
            phase: Training phase ("phase1", "phase2", "phase3")
            use_real_data: Whether to include real data
            create_combined: Whether to create combined dataset or separate splits
        
        Returns:
            Dictionary of DataLoaders
        """
        logger.info(f"Creating dataloaders for {phase}...")
        
        dataloaders = {}
        
        if create_combined:
            # Create combined dataset and split it
            combined_dataset, _ = self.create_combined_dataset(data_dir, use_real_data)
            
            # Create train/val/test splits
            total_size = len(combined_dataset)
            train_size = int(self.config.data.train_split * total_size)
            val_size = int(self.config.data.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            generator = torch.Generator().manual_seed(self.config.random_seed)
            train_dataset, val_dataset, test_dataset = random_split(
                combined_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )
            
            datasets = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
        else:
            # Load synthetic data with splits
            train_dataset, val_dataset, test_dataset = self.load_synthetic_data(data_dir, create_splits=True)
            
            datasets = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
        
        # Get batch sizes based on phase
        batch_sizes = self._get_batch_sizes(phase)
        
        # Create dataloaders
        for split, dataset in datasets.items():
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_sizes[split],
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=(split == 'train'),
                collate_fn=self.collator
            )
        
        self.dataloaders.update(dataloaders)
        
        logger.info(f"Created dataloaders: {list(dataloaders.keys())}")
        
        return dataloaders
    
    def _get_batch_sizes(self, phase: str) -> Dict[str, int]:
        """Get batch sizes for different phases and splits"""
        if phase == "phase1":
            return {
                'train': self.config.training.phase1_batch_size,
                'val': self.config.training.phase1_batch_size,
                'test': self.config.training.phase1_batch_size
            }
        elif phase == "phase2":
            return {
                'train': self.config.training.phase2_batch_size,
                'val': self.config.training.phase2_batch_size,
                'test': self.config.training.phase2_batch_size
            }
        elif phase == "phase3":
            return {
                'train': self.config.training.phase3_batch_size,
                'val': self.config.training.phase3_batch_size,
                'test': self.config.training.phase3_batch_size
            }
        else:
            return {
                'train': self.config.training.phase1_batch_size,
                'val': self.config.training.phase1_batch_size,
                'test': self.config.training.phase1_batch_size
            }
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about loaded datasets"""
        info = {}
        
        for name, dataset in self.datasets.items():
            if hasattr(dataset, 'get_label_distribution'):
                label_dist = dataset.get_label_distribution()
            else:
                label_dist = {}
            
            info[name] = {
                'size': len(dataset),
                'label_distribution': label_dist,
                'type': type(dataset).__name__
            }
        
        return info
    
    def get_usage_report(self) -> Dict:
        """Get comprehensive usage report"""
        report = {
            'datasets': self.get_dataset_info(),
            'real_data_usage': self.real_manager.get_usage_report(),
            'citations': self.citations,
            'configuration': {
                'synthetic_dataset_size': self.config.data.synthetic_dataset_size,
                'real_data_percentage': self.config.data.real_data_percentage,
                'image_size': self.config.data.image_size,
                'use_augmentation': self.config.data.use_augmentation
            }
        }
        
        return report
    
    def export_usage_report(self, output_path: Path):
        """Export usage report to file"""
        import json
        
        report = self.get_usage_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported usage report to {output_path}")
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate integrity of loaded datasets"""
        results = {}
        
        for name, dataset in self.datasets.items():
            try:
                # Try to load a few samples
                if len(dataset) > 0:
                    sample = dataset[0]
                    if len(dataset) > 1:
                        sample2 = dataset[min(1, len(dataset) - 1)]
                    
                    # Check sample structure
                    required_keys = ['image', 'label', 'metadata', 'image_id', 'source']
                    has_required_keys = all(key in sample for key in required_keys)
                    
                    # Check tensor shapes
                    image_shape_ok = sample['image'].dim() >= 2
                    label_type_ok = isinstance(sample['label'], torch.Tensor)
                    
                    results[name] = has_required_keys and image_shape_ok and label_type_ok
                else:
                    results[name] = False
                    
            except Exception as e:
                logger.error(f"Validation failed for {name}: {e}")
                results[name] = False
        
        return results


def create_data_manager(config: Config) -> SonarDataManager:
    """
    Factory function to create a configured data manager
    
    Args:
        config: System configuration
    
    Returns:
        Configured SonarDataManager instance
    """
    return SonarDataManager(config)