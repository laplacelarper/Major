"""EfficientNet-B0 implementation for sonar detection"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .base import BaseSonarModel, ClassificationHead, SegmentationHead


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.dropout_rate = dropout_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout for residual connection
        if self.use_residual and dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze and Excitation
        se_weight = self.se(x)
        x = x * se_weight
        
        # Output projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            x = self.dropout(x)
            x = x + identity
        
        return x


class EfficientNetB0(BaseSonarModel):
    """EfficientNet-B0 architecture for sonar detection"""
    
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
        """Build EfficientNet-B0 backbone"""
        # EfficientNet-B0 configuration
        # (expand_ratio, channels, repeats, stride, kernel_size)
        mb_config = [
            (1, 16, 1, 1, 3),   # Stage 1
            (6, 24, 2, 2, 3),   # Stage 2
            (6, 40, 2, 2, 5),   # Stage 3
            (6, 80, 3, 2, 3),   # Stage 4
            (6, 112, 3, 1, 5),  # Stage 5
            (6, 192, 4, 2, 5),  # Stage 6
            (6, 320, 1, 1, 3),  # Stage 7
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in mb_config:
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        dropout_rate=self.dropout_rate
                    )
                )
                in_channels = out_channels
        
        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )
        
        # Feature dimension for heads
        self.feature_dim = 1280
    
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
                nn.ConvTranspose2d(self.feature_dim, 320, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(320),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(320, 192, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(192),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(192, 112, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(112),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(112, 80, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(80),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(80, 40, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(40),
                nn.SiLU(inplace=True)
            )
            
            self.head = SegmentationHead(
                feature_dim=40,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using EfficientNet-B0 backbone"""
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        
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