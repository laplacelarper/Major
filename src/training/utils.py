"""Training utilities for checkpoint management, early stopping, and metrics tracking"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation score improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best: {self.best_score:.6f} at epoch {self.best_epoch})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save best checkpoint
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.checkpoints: List[Dict] = []
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Track checkpoints
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best
        })
        
        # Remove old checkpoints if needed
        if not self.save_best_only and len(self.checkpoints) > self.max_checkpoints:
            self._remove_old_checkpoints()
        
        return checkpoint_path
    
    def _remove_old_checkpoints(self):
        """Remove oldest checkpoints"""
        # Sort by epoch
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove oldest non-best checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints[0]
            if not oldest['is_best']:
                oldest['path'].unlink(missing_ok=True)
                self.checkpoints.pop(0)
                logger.info(f"Removed old checkpoint: {oldest['path']}")
            else:
                break
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Dict:
        """
        Load a checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Path to checkpoint (default: best_model.pth)
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


class TrainingMetrics:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Compute average metrics for the epoch"""
        epoch_metrics = {}
        for key, values in self.metrics.items():
            if values:
                epoch_metrics[key] = np.mean(values)
        
        self.epoch_metrics = epoch_metrics
        return epoch_metrics
    
    def reset(self):
        """Reset metrics for new epoch"""
        self.metrics = defaultdict(list)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full training history"""
        return dict(self.metrics)
    
    def save_history(self, save_path: Path):
        """Save training history to file"""
        history = {
            'metrics': {k: [float(v) for v in vals] for k, vals in self.metrics.items()},
            'epoch_metrics': {k: float(v) for k, v in self.epoch_metrics.items()}
        }
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved training history to {save_path}")


class LearningRateScheduler:
    """Learning rate scheduling"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'cosine',
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_epochs: int = 0,
        total_epochs: int = 100
    ):
        """
        Initialize learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            mode: Scheduling mode ('cosine', 'step', 'plateau')
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
        """
        self.optimizer = optimizer
        self.mode = mode
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[Dict] = None):
        """Update learning rate"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        elif self.mode == 'cosine':
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.mode == 'step':
            # Step decay
            decay_epochs = [self.total_epochs // 3, 2 * self.total_epochs // 3]
            lr = self.initial_lr
            for decay_epoch in decay_epochs:
                if self.current_epoch >= decay_epoch:
                    lr *= 0.1
        else:
            lr = self.initial_lr
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
