"""Base model classes for sonar detection system"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Standardized model output structure"""
    predictions: torch.Tensor  # Main predictions (classification or segmentation)
    features: Optional[torch.Tensor] = None  # Intermediate features
    auxiliary_outputs: Optional[Dict[str, torch.Tensor]] = None  # Additional outputs
    metadata: Optional[Dict[str, Any]] = None  # Model metadata


class BaseSonarModel(nn.Module, ABC):
    """Abstract base class for all sonar detection models"""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout_rate: float = 0.1,
        use_physics_metadata: bool = True,
        metadata_dim: int = 7,
        output_mode: str = "classification"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.use_physics_metadata = use_physics_metadata
        self.metadata_dim = metadata_dim
        self.output_mode = output_mode
        
        # Validate output mode
        if output_mode not in ["classification", "segmentation"]:
            raise ValueError(f"Output mode must be 'classification' or 'segmentation', got {output_mode}")
        
        # Initialize metadata processor first if needed
        if use_physics_metadata:
            self._build_metadata_processor()
        else:
            self.metadata_output_dim = 0
        
        # Initialize components
        self._build_backbone()
        self._build_heads()
    
    @abstractmethod
    def _build_backbone(self):
        """Build the main feature extraction backbone"""
        pass
    
    @abstractmethod
    def _build_heads(self):
        """Build task-specific output heads"""
        pass
    
    def _build_metadata_processor(self):
        """Build metadata processing layers"""
        self.metadata_processor = nn.Sequential(
            nn.Linear(self.metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        self.metadata_output_dim = 32
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image"""
        pass
    
    def process_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        """Process physics metadata if available"""
        if not self.use_physics_metadata:
            return None
        return self.metadata_processor(metadata)
    
    def forward(
        self, 
        x: torch.Tensor, 
        metadata: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """Forward pass with optional metadata"""
        # Extract image features
        features = self.extract_features(x)
        
        # Process metadata if provided
        metadata_features = None
        if metadata is not None and self.use_physics_metadata:
            metadata_features = self.process_metadata(metadata)
        
        # Generate predictions
        predictions = self._forward_head(features, metadata_features)
        
        return ModelOutput(
            predictions=predictions,
            features=features,
            auxiliary_outputs={"metadata_features": metadata_features} if metadata_features is not None else None,
            metadata={
                "model_type": self.__class__.__name__,
                "output_mode": self.output_mode,
                "input_shape": x.shape
            }
        )
    
    @abstractmethod
    def _forward_head(
        self, 
        features: torch.Tensor, 
        metadata_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through task-specific heads"""
        pass
    
    def forward_with_uncertainty(
        self, 
        x: torch.Tensor, 
        metadata: Optional[torch.Tensor] = None,
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Monte Carlo Dropout uncertainty estimation
        
        Args:
            x: Input tensor
            metadata: Optional physics metadata
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        # Temporarily enable MC dropout
        original_training = self.training
        self.eval()  # Set to eval but dropout will remain active if MC dropout is used
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x, metadata)
                predictions.append(output.predictions)
        
        # Restore original training mode
        self.train(original_training)
        
        # Stack predictions and calculate statistics
        predictions = torch.stack(predictions, dim=0)
        mean_predictions = torch.mean(predictions, dim=0)
        
        # Calculate uncertainty based on output mode
        if self.output_mode == "classification":
            # Use entropy as uncertainty measure
            probs = torch.softmax(mean_predictions, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            uncertainty = entropy
        else:  # segmentation
            # Use variance as uncertainty measure
            variance = torch.var(predictions, dim=0)
            uncertainty = variance
        
        return mean_predictions, uncertainty
        
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "output_mode": self.output_mode,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "uses_metadata": self.use_physics_metadata,
            "dropout_rate": self.dropout_rate
        }


class ClassificationHead(nn.Module):
    """Classification head for binary mine/rock detection"""
    
    def __init__(
        self, 
        feature_dim: int, 
        num_classes: int = 2, 
        dropout_rate: float = 0.1,
        metadata_dim: Optional[int] = None
    ):
        super().__init__()
        
        # Calculate input dimension after global average pooling
        pooled_feature_dim = feature_dim
        if metadata_dim is not None:
            pooled_feature_dim += metadata_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(pooled_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        metadata_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through classification head"""
        # Global average pooling
        pooled_features = nn.functional.adaptive_avg_pool2d(features, 1)
        pooled_features = pooled_features.flatten(1)
        
        # Concatenate with metadata if available
        if metadata_features is not None:
            pooled_features = torch.cat([pooled_features, metadata_features], dim=1)
        
        return self.classifier(pooled_features)


class SegmentationHead(nn.Module):
    """Segmentation head for pixel-wise predictions"""
    
    def __init__(
        self, 
        feature_dim: int, 
        num_classes: int = 2, 
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        metadata_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through segmentation head"""
        # Note: metadata_features not used in segmentation head
        # Could be incorporated through spatial broadcasting if needed
        return self.decoder(features)