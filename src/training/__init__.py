"""Training modules for the sonar detection system"""

from .trainer import Trainer
from .phase1_synthetic import Phase1SyntheticTrainer
from .phase2_finetuning import Phase2FineTuningTrainer
from .phase3_calibration import Phase3CalibrationTrainer
from .utils import (
    EarlyStopping,
    CheckpointManager,
    TrainingMetrics,
    LearningRateScheduler
)

__all__ = [
    'Trainer',
    'Phase1SyntheticTrainer',
    'Phase2FineTuningTrainer',
    'Phase3CalibrationTrainer',
    'EarlyStopping',
    'CheckpointManager',
    'TrainingMetrics',
    'LearningRateScheduler'
]
