"""Reproducibility and validation tools"""

import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ReproducibilityValidator:
    """
    Validate reproducibility and determinism
    
    Requirements: 6.1, 6.2, 8.3
    """
    
    def __init__(self, config=None):
        self.config = config
        self.validation_results = {}
    
    def validate_deterministic_run(
        self,
        run1_results: Dict[str, float],
        run2_results: Dict[str, float],
        tolerance: float = 1e-6
    ) -> Dict[str, any]:
        """
        Validate that two runs with same seed produce identical results
        
        Args:
            run1_results: Results from first run
            run2_results: Results from second run
            tolerance: Numerical tolerance for floating point comparison
        
        Returns:
            Validation results
        """
        validation = {
            'is_deterministic': True,
            'differences': {},
            'max_difference': 0.0,
            'matching_metrics': [],
            'differing_metrics': []
        }
        
        # Compare each metric
        for metric_name in run1_results.keys():
            if metric_name not in run2_results:
                validation['is_deterministic'] = False
                validation['differing_metrics'].append(metric_name)
                logger.warning(f"Metric {metric_name} missing in run 2")
                continue
            
            val1 = run1_results[metric_name]
            val2 = run2_results[metric_name]
            
            # Skip non-numeric values
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
            
            difference = abs(val1 - val2)
            validation['differences'][metric_name] = difference
            
            if difference > tolerance:
                validation['is_deterministic'] = False
                validation['differing_metrics'].append(metric_name)
                logger.warning(f"Metric {metric_name} differs: {val1} vs {val2} (diff={difference})")
            else:
                validation['matching_metrics'].append(metric_name)
            
            validation['max_difference'] = max(validation['max_difference'], difference)
        
        if validation['is_deterministic']:
            logger.info("✓ Deterministic validation passed")
        else:
            logger.error(f"✗ Deterministic validation failed: {len(validation['differing_metrics'])} metrics differ")
        
        return validation
    
    def validate_dataset_splits(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        total_size: int
    ) -> Dict[str, any]:
        """
        Validate dataset splits for correctness
        
        Args:
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            total_size: Total dataset size
        
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for overlaps
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap:
            validation['is_valid'] = False
            validation['errors'].append(f"Train-Val overlap: {len(train_val_overlap)} samples")
            logger.error(f"Train-Val overlap detected: {len(train_val_overlap)} samples")
        
        if train_test_overlap:
            validation['is_valid'] = False
            validation['errors'].append(f"Train-Test overlap: {len(train_test_overlap)} samples")
            logger.error(f"Train-Test overlap detected: {len(train_test_overlap)} samples")
        
        if val_test_overlap:
            validation['is_valid'] = False
            validation['errors'].append(f"Val-Test overlap: {len(val_test_overlap)} samples")
            logger.error(f"Val-Test overlap detected: {len(val_test_overlap)} samples")
        
        # Check coverage
        all_indices = train_set | val_set | test_set
        if len(all_indices) != total_size:
            validation['warnings'].append(f"Coverage mismatch: {len(all_indices)} vs {total_size}")
            logger.warning(f"Dataset coverage: {len(all_indices)}/{total_size}")
        
        # Check for duplicates within splits
        if len(train_indices) != len(train_set):
            validation['errors'].append(f"Duplicate indices in train set")
            validation['is_valid'] = False
        
        if len(val_indices) != len(val_set):
            validation['errors'].append(f"Duplicate indices in val set")
            validation['is_valid'] = False
        
        if len(test_indices) != len(test_set):
            validation['errors'].append(f"Duplicate indices in test set")
            validation['is_valid'] = False
        
        # Statistics
        validation['statistics'] = {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'total_size': total_size,
            'train_ratio': len(train_indices) / total_size,
            'val_ratio': len(val_indices) / total_size,
            'test_ratio': len(test_indices) / total_size
        }
        
        if validation['is_valid']:
            logger.info("✓ Dataset split validation passed")
        else:
            logger.error(f"✗ Dataset split validation failed: {len(validation['errors'])} errors")
        
        return validation
    
    def validate_test_set_protection(
        self,
        test_indices: np.ndarray,
        accessed_indices: np.ndarray
    ) -> Dict[str, any]:
        """
        Validate that test set was not accessed during training
        
        Args:
            test_indices: Test set indices
            accessed_indices: Indices accessed during training
        
        Returns:
            Validation results
        """
        test_set = set(test_indices)
        accessed_set = set(accessed_indices)
        
        contamination = test_set & accessed_set
        
        validation = {
            'is_protected': len(contamination) == 0,
            'contaminated_samples': len(contamination),
            'contamination_ratio': len(contamination) / len(test_set) if len(test_set) > 0 else 0
        }
        
        if validation['is_protected']:
            logger.info("✓ Test set protection validated")
        else:
            logger.error(f"✗ Test set contamination detected: {len(contamination)} samples")
        
        return validation
    
    def compute_dataset_hash(self, data: np.ndarray) -> str:
        """
        Compute hash of dataset for integrity checking
        
        Args:
            data: Dataset array
        
        Returns:
            SHA256 hash string
        """
        data_bytes = data.tobytes()
        hash_obj = hashlib.sha256(data_bytes)
        return hash_obj.hexdigest()
    
    def validate_configuration(self, config) -> Dict[str, any]:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration object
        
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate data splits
        try:
            total_split = config.data.train_split + config.data.val_split + config.data.test_split
            if abs(total_split - 1.0) > 1e-6:
                validation['errors'].append(f"Data splits sum to {total_split}, not 1.0")
                validation['is_valid'] = False
        except AttributeError:
            validation['warnings'].append("Could not validate data splits")
        
        # Validate real data percentage
        try:
            if not 0.0 <= config.data.real_data_percentage <= 0.3:
                validation['errors'].append(f"Real data percentage {config.data.real_data_percentage} exceeds 30% limit")
                validation['is_valid'] = False
        except AttributeError:
            validation['warnings'].append("Could not validate real data percentage")
        
        # Validate learning rates
        try:
            if config.training.phase2_lr >= config.training.phase1_lr:
                validation['warnings'].append("Phase 2 LR should be lower than Phase 1 LR")
            
            if config.training.phase3_lr >= config.training.phase2_lr:
                validation['warnings'].append("Phase 3 LR should be lower than Phase 2 LR")
        except AttributeError:
            validation['warnings'].append("Could not validate learning rates")
        
        # Validate image size
        try:
            if config.data.image_size[0] != config.data.image_size[1]:
                validation['errors'].append("Only square images are supported")
                validation['is_valid'] = False
        except (AttributeError, IndexError):
            validation['warnings'].append("Could not validate image size")
        
        # Validate model type
        try:
            valid_models = ["unet", "resnet18", "efficientnet-b0"]
            if config.model.model_type not in valid_models:
                validation['errors'].append(f"Invalid model type: {config.model.model_type}")
                validation['is_valid'] = False
        except AttributeError:
            validation['warnings'].append("Could not validate model type")
        
        # Validate physics parameters
        try:
            if config.physics.range_min_m >= config.physics.range_max_m:
                validation['errors'].append("Range min must be less than range max")
                validation['is_valid'] = False
        except AttributeError:
            validation['warnings'].append("Could not validate physics parameters")
        
        if validation['is_valid']:
            logger.info("✓ Configuration validation passed")
        else:
            logger.error(f"✗ Configuration validation failed: {len(validation['errors'])} errors")
        
        return validation
    
    def save_validation_report(self, filepath: Path):
        """Save validation results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        logger.info(f"Validation report saved to {filepath}")


def validate_deterministic_run(
    run1_results: Dict[str, float],
    run2_results: Dict[str, float],
    tolerance: float = 1e-6
) -> Dict[str, any]:
    """
    Standalone function to validate deterministic runs
    
    Args:
        run1_results: Results from first run
        run2_results: Results from second run
        tolerance: Numerical tolerance
    
    Returns:
        Validation results
    """
    validator = ReproducibilityValidator()
    return validator.validate_deterministic_run(run1_results, run2_results, tolerance)


def validate_dataset_splits(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    total_size: int
) -> Dict[str, any]:
    """
    Standalone function to validate dataset splits
    
    Args:
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        total_size: Total dataset size
    
    Returns:
        Validation results
    """
    validator = ReproducibilityValidator()
    return validator.validate_dataset_splits(train_indices, val_indices, test_indices, total_size)


def validate_configuration(config) -> Dict[str, any]:
    """
    Standalone function to validate configuration
    
    Args:
        config: Configuration object
    
    Returns:
        Validation results
    """
    validator = ReproducibilityValidator()
    return validator.validate_configuration(config)
