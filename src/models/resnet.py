"""ResNet18 implementation for sonar detection"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseSonarModel, ClassificationHead, SegmentationHead


class BasicBlock(nn.Module):
    """Basic residual block for ResNet18"""
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(nn.functional.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout2(nn.functional.relu(out))
        return out


class ResNet18(BaseSonarModel):
    """ResNet18 architecture for sonar detection"""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout_rate: float = 0.1,
        use_physics_metadata: bool = True,
        metadata_dim: int = 7,
        output_mode: str = "classification"
    ):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout_rate=dropout_rate,
            use_physics_metadata=use_physics_metadata,
            metadata_dim=metadata_dim,
            output_mode=output_mode
        )
    
    def _build_backbone(self):
        """Build ResNet18 backbone"""
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, kernel_size=7, 
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Feature dimension for heads
        self.feature_dim = 512 * BasicBlock.expansion
    
    def _make_layer(
        self, 
        block: nn.Module, 
        out_channels: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """Create a residual layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.dropout_rate))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _build_heads(self):
        """Build task-specific heads"""
        metadata_dim = self.metadata_output_dim if self.use_physics_metadata else None
        
        if self.output_mode == "classification":
            self.head = ClassificationHead(
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                metadata_dim=metadata_dim
            )
        else:  # segmentation
            # For segmentation, we need to upsample features
            self.segmentation_upsampler = nn.Sequential(
                nn.ConvTranspose2d(self.feature_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            
            self.head = SegmentationHead(
                feature_dim=16,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using ResNet18 backbone"""
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # For segmentation, upsample features to match input resolution
        if self.output_mode == "segmentation":
            x = self.segmentation_upsampler(x)
        
        return x
    
    def _forward_head(
        self, 
        features: torch.Tensor, 
        metadata_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through task-specific head"""
        return self.head(features, metadata_features)