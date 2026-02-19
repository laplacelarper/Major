"""U-Net implementation for sonar detection"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseSonarModel, ClassificationHead, SegmentationHead


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseSonarModel):
    """U-Net architecture for sonar detection"""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout_rate: float = 0.1,
        use_physics_metadata: bool = True,
        metadata_dim: int = 7,
        output_mode: str = "classification",
        base_channels: int = 64
    ):
        self.base_channels = base_channels
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout_rate=dropout_rate,
            use_physics_metadata=use_physics_metadata,
            metadata_dim=metadata_dim,
            output_mode=output_mode
        )
    
    def _build_backbone(self):
        """Build U-Net encoder-decoder backbone"""
        # Encoder
        self.inc = DoubleConv(self.input_channels, self.base_channels, self.dropout_rate)
        self.down1 = Down(self.base_channels, self.base_channels * 2, self.dropout_rate)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4, self.dropout_rate)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8, self.dropout_rate)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16, self.dropout_rate)
        
        # Decoder
        self.up1 = Up(self.base_channels * 16, self.base_channels * 8, self.dropout_rate)
        self.up2 = Up(self.base_channels * 8, self.base_channels * 4, self.dropout_rate)
        self.up3 = Up(self.base_channels * 4, self.base_channels * 2, self.dropout_rate)
        self.up4 = Up(self.base_channels * 2, self.base_channels, self.dropout_rate)
        
        # Feature dimension for heads
        self.feature_dim = self.base_channels
    
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
            self.head = SegmentationHead(
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using U-Net encoder-decoder"""
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return x
    
    def _forward_head(
        self, 
        features: torch.Tensor, 
        metadata_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through task-specific head"""
        return self.head(features, metadata_features)