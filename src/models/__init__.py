"""Sonar detection models package"""

from .base import BaseSonarModel, ModelOutput, ClassificationHead, SegmentationHead
from .unet import UNet
from .resnet import ResNet18
from .efficientnet import EfficientNetB0
from .uncertainty import UncertaintyEstimator, MCDropout, MCDropout2d
from .factory import (
    ModelFactory, 
    create_model_from_config, 
    get_model_parameter_count,
    freeze_model_layers,
    unfreeze_all_layers
)

__all__ = [
    # Base classes
    "BaseSonarModel",
    "ModelOutput", 
    "ClassificationHead",
    "SegmentationHead",
    
    # Model architectures
    "UNet",
    "ResNet18", 
    "EfficientNetB0",
    
    # Uncertainty estimation
    "UncertaintyEstimator",
    "MCDropout",
    "MCDropout2d",
    
    # Factory and utilities
    "ModelFactory",
    "create_model_from_config",
    "get_model_parameter_count",
    "freeze_model_layers",
    "unfreeze_all_layers"
]