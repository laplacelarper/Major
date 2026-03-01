"""Phase 1: Synthetic pretraining trainer"""

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import Config
from .trainer import Trainer
from .utils import EarlyStopping, CheckpointManager, LearningRateScheduler

logger = logging.getLogger(__name__)


class Phase1SyntheticTrainer(Trainer):
    """
    Phase 1: Synthetic pretraining trainer
    
    Trains model exclusively on synthetic data with:
    - Heavy augmentation
    - Early stopping on validation loss
    - Checkpoint saving
    - Learning rate scheduling
    
    Requirements: 4.1, 4.4
    """
    
    def __init__(self, model: nn.Module, config: Config, device=None):
        super().__init__(model, config, device)
        
        self.num_epochs = config.training.phase1_epochs
        self.learning_rate = config.training.phase1_lr
        self.batch_size = config.training.phase1_batch_size
        self.weight_decay = config.training.phase1_weight_decay
        
        logger.info("Initialized Phase 1: Synthetic Pretraining")
    
    def setup_training(self):
        """Setup training components for Phase 1"""
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss criterion
        # Use CrossEntropyLoss for multi-class (including 2-class) classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=1e-4,
            mode='min',
            verbose=True
        )
        
        # Checkpoint manager
        checkpoint_dir = self.config.checkpoint_dir / "phase1_synthetic"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,
            save_best_only=False,
            monitor='val_loss',
            mode='min'
        )
        
        logger.info(f"Setup Phase 1 training: lr={self.learning_rate}, "
                   f"epochs={self.num_epochs}, batch_size={self.batch_size}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch on synthetic data
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if metadata is not None and self.config.model.use_physics_metadata:
                model_output = self.model(images, metadata)
            else:
                model_output = self.model(images)
            
            # Extract predictions from ModelOutput
            if hasattr(model_output, 'predictions'):
                outputs = model_output.predictions
            else:
                outputs = model_output
            
            # For CrossEntropyLoss, outputs should be [batch, num_classes] and labels should be [batch] with class indices
            # No need to squeeze or convert labels to float
            
            loss = self.compute_loss(outputs, labels, metadata)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self.compute_accuracy(outputs, labels.long())
            
            # Update metrics
            total_loss += loss.item()
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            
            for batch in pbar:
                # Get data
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
                
                # Forward pass
                if metadata is not None and self.config.model.use_physics_metadata:
                    model_output = self.model(images, metadata)
                else:
                    model_output = self.model(images)
                
                # Extract predictions from ModelOutput
                if hasattr(model_output, 'predictions'):
                    outputs = model_output.predictions
                else:
                    outputs = model_output
                
                # For CrossEntropyLoss, outputs should be [batch, num_classes] and labels should be [batch] with class indices
                # No need to squeeze or convert labels to float
                
                loss = self.compute_loss(outputs, labels, metadata)
                
                # Compute accuracy
                accuracy = self.compute_accuracy(outputs, labels.long())
                
                # Update metrics
                total_loss += loss.item()
                total_correct += accuracy * labels.size(0)
                total_samples += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}'
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
    
    def run(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Run Phase 1 training
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training results
        """
        logger.info("="*80)
        logger.info("PHASE 1: SYNTHETIC PRETRAINING")
        logger.info("="*80)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info("="*80)
        
        # Run training
        results = self.train(train_loader, val_loader, self.num_epochs)
        
        logger.info("="*80)
        logger.info("PHASE 1 COMPLETED")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Training time: {results['training_time']:.2f}s")
        logger.info("="*80)
        
        return results


def train_phase1(config: Config, logger):
    """
    Convenience function to run Phase 1 training
    
    Args:
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Training results dictionary
    """
    from pathlib import Path
    from ..models.factory import create_model_from_config
    from ..data.data_loader import create_data_manager
    
    logger.info("Starting Phase 1: Synthetic Pretraining")
    
    # Create model
    model = create_model_from_config(config)
    logger.info(f"Created model: {config.model.model_type}")
    
    # Create data manager and loaders
    data_manager = create_data_manager(config)
    dataloaders = data_manager.create_dataloaders(
        data_dir=config.data_dir,
        phase='phase1',
        use_real_data=False,
        create_combined=False
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    logger.info(f"Created data loaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Phase1SyntheticTrainer(model, config)
    trainer.setup_training()
    
    # Run training
    results = trainer.run(train_loader, val_loader)
    
    logger.info("Phase 1 training completed successfully")
    
    return results
