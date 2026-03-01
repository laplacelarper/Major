"""Validation and comparison modules"""

from .comparison import (
    ModelComparison,
    compare_models,
    statistical_significance_test
)
from .reproducibility import (
    ReproducibilityValidator,
    validate_deterministic_run,
    validate_dataset_splits,
    validate_configuration
)

__all__ = [
    # Comparison
    'ModelComparison',
    'compare_models',
    'statistical_significance_test',
    
    # Reproducibility
    'ReproducibilityValidator',
    'validate_deterministic_run',
    'validate_dataset_splits',
    'validate_configuration'
]
