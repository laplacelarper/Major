"""Model factory for creating sonar detection models"""

import torch.nn as nn
from typing import Dict, Any, Type
import sys
from pathlib import Path

# Add config to path for imports
config_path = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(config_path))

from config import ModelConfig
from .base import BaseSonarModel
from .unet import UNet
from .resnet import ResNet18
from .efficientnet import EfficientNetB0


class ModelFactory:
    """Factory class for creating sonar detection models"""
    
    # Registry of available models
    _models: Dict[str, Type[BaseSonarModel]] = {
        "unet": UNet,
        "resnet18": ResNet18,
        "efficientnet-b0": EfficientNetB0
    }
    
    @classmethod
    def create_model(
        cls, 
        model_config: ModelConfig, 
        **kwargs
    ) -> BaseSonarModel:
        """
        Create a model instance based on configuration
        
        Args:
            model_config: Model configuration object
            **kwargs: Additional keyword arguments to override config
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        model_type = model_config.model_type.lower()
        
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Available models: {available_models}"
            )
        
        # Prepare model arguments
        model_args = {
            "num_classes": model_config.num_classes,
            "input_channels": model_config.input_channels,
            "dropout_rate": model_config.dropout_rate,
            "use_physics_metadata": model_config.use_physics_metadata,
            "metadata_dim": model_config.metadata_dim,
            "output_mode": model_config.output_mode
        }
        
        # Override with any additional kwargs
        model_args.update(kwargs)
        
        # Create and return model
        model_class = cls._models[model_type]
        return model_class(**model_args)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseSonarModel]):
        """
        Register a new model type
        
        Args:
            name: Model name identifier
            model_class: Model class that inherits from BaseSonarModel
        """
        if not issubclass(model_class, BaseSonarModel):
            raise ValueError("Model class must inherit from BaseSonarModel")
        
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dictionary with model information
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        model_class = cls._models[model_type]
        
        return {
            "name": model_type,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "description": model_class.__doc__ or "No description available"
        }


def create_model_from_config(config) -> BaseSonarModel:
    """
    Convenience function to create model from full config object
    
    Args:
        config: Full configuration object with model attribute
        
    Returns:
        Initialized model instance
    """
    return ModelFactory.create_model(config.model)


def get_model_parameter_count(model: nn.Module) -> Dict[str, int]:
    """
    Get parameter count statistics for a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0
    }


def freeze_model_layers(model: nn.Module, num_layers_to_freeze: int = 0) -> nn.Module:
    """
    Freeze early layers of a model for fine-tuning
    
    Args:
        model: PyTorch model to freeze layers in
        num_layers_to_freeze: Number of early layers to freeze
        
    Returns:
        Model with frozen layers
    """
    if num_layers_to_freeze <= 0:
        return model
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Freeze the first num_layers_to_freeze layers
    layers_frozen = 0
    for name, param in named_params:
        if layers_frozen < num_layers_to_freeze:
            param.requires_grad = False
            layers_frozen += 1
        else:
            break
    
    print(f"Frozen {layers_frozen} layers out of {len(named_params)} total parameters")
    return model


def unfreeze_all_layers(model: nn.Module) -> nn.Module:
    """
    Unfreeze all layers in a model
    
    Args:
        model: PyTorch model to unfreeze
        
    Returns:
        Model with all layers unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True
    
    return model