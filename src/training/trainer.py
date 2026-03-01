"""Base trainer class for sonar detection system"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import Config
from .utils import EarlyStopping, CheckpointManager, TrainingMetrics, LearningRateScheduler

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """Base trainer class for all training phases"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            config: System configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        
        # Setup device
        if device is None:
            if config.device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping = None
        self.checkpoint_manager = None
        self.metrics = TrainingMetrics()
        
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def setup_training(self):
        """Setup training components (optimizer, criterion, etc.)"""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        pass
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, any]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Setup training components
        self.setup_training()
        
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics.update(epoch_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_results(epoch, epoch_metrics, epoch_time)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    epoch_metrics,
                    is_best=is_best
                )
            
            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['val_loss'], epoch):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        training_time = time.time() - training_start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Save training history
        if self.checkpoint_manager is not None:
            history_path = self.checkpoint_manager.checkpoint_dir / 'training_history.json'
            self.metrics.save_history(history_path)
        
        return {
            'history': self.metrics.get_history(),
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }
    
    def _log_epoch_results(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """Log epoch results"""
        log_str = f"Epoch {epoch+1}: "
        log_str += f"train_loss={metrics.get('train_loss', 0):.4f}, "
        log_str += f"val_loss={metrics.get('val_loss', 0):.4f}, "
        log_str += f"train_acc={metrics.get('train_acc', 0):.4f}, "
        log_str += f"val_acc={metrics.get('val_acc', 0):.4f}, "
        log_str += f"lr={self.optimizer.param_groups[0]['lr']:.6f}, "
        log_str += f"time={epoch_time:.2f}s"
        
        logger.info(log_str)
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            metadata: Optional metadata
        
        Returns:
            Loss value
        """
        return self.criterion(outputs, labels)
    
    def compute_accuracy(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute accuracy
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
        
        Returns:
            Accuracy value
        """
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class classification
            _, predicted = torch.max(outputs, 1)
        else:
            # Binary classification
            predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()
        
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def save_model(self, save_path: Path):
        """Save model weights"""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")
    
    def load_model(self, load_path: Path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        logger.info(f"Loaded model from {load_path}")
