"""Phase 2: Real data fine-tuning trainer"""

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import Config
from .trainer import Trainer
from .utils import CheckpointManager

logger = logging.getLogger(__name__)


class Phase2FineTuningTrainer(Trainer):
    """
    Phase 2: Real data fine-tuning trainer
    
    Fine-tunes pretrained model on real sonar data with:
    - Frozen early layers
    - Very low learning rate
    - Minimal augmentation
    
    Requirements: 4.2, 4.5
    """
    
    def __init__(self, model: nn.Module, config: Config, device=None):
        super().__init__(model, config, device)
        
        self.num_epochs = config.training.phase2_epochs
        self.learning_rate = config.training.phase2_lr
        self.batch_size = config.training.phase2_batch_size
        self.freeze_layers = config.training.phase2_freeze_layers
        
        logger.info("Initialized Phase 2: Real Data Fine-tuning")
    
    def setup_training(self):
        """Setup training components for Phase 2"""
        # Freeze early layers
        self._freeze_early_layers()
        
        # Optimizer - only for unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Loss criterion
        if self.config.model.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler - plateau for fine-tuning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-7
        )
        
        # Checkpoint manager
        checkpoint_dir = self.config.checkpoint_dir / "phase2_finetuning"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=3,
            save_best_only=False,
            monitor='val_loss',
            mode='min'
        )
        
        logger.info(f"Setup Phase 2 training: lr={self.learning_rate}, "
                   f"epochs={self.num_epochs}, frozen_layers={self.freeze_layers}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def _freeze_early_layers(self):
        """Freeze early layers of the model"""
        # Get model layers
        if hasattr(self.model, 'backbone'):
            # Model with backbone attribute
            backbone = self.model.backbone
            
            # Freeze first N layers
            layer_count = 0
            for name, module in backbone.named_children():
                if layer_count < self.freeze_layers:
                    for param in module.parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")
                    layer_count += 1
                else:
                    break
        else:
            # Freeze first N modules
            modules = list(self.model.children())
            for i in range(min(self.freeze_layers, len(modules))):
                for param in modules[i].parameters():
                    param.requires_grad = False
                logger.info(f"Frozen module {i}")
        
        # Log frozen/trainable parameter counts
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch on real data
        
        Args:
            train_loader: Training data loader (real data)
            epoch: Current epoch number
        
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Keep frozen layers in eval mode
        if hasattr(self.model, 'backbone'):
            for name, module in self.model.backbone.named_children():
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train-Real]")
        
        for batch in pbar:
            # Get data
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Real data may not have metadata - use zero tensor
            if metadata is not None and self.config.model.use_physics_metadata:
                outputs = self.model(images, metadata)
            else:
                outputs = self.model(images)
            
            # Compute loss
            if self.config.model.num_classes == 2:
                outputs = outputs.squeeze()
                labels = labels.float()
            
            loss = self.compute_loss(outputs, labels, metadata)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
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
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val-Real]")
            
            for batch in pbar:
                # Get data
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
                
                # Forward pass
                if metadata is not None and self.config.model.use_physics_metadata:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
                # Compute loss
                if self.config.model.num_classes == 2:
                    outputs = outputs.squeeze()
                    labels = labels.float()
                
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
        Run Phase 2 fine-tuning
        
        Args:
            train_loader: Training data loader (real data)
            val_loader: Validation data loader (real data)
        
        Returns:
            Training results
        """
        logger.info("="*80)
        logger.info("PHASE 2: REAL DATA FINE-TUNING")
        logger.info("="*80)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Frozen layers: {self.freeze_layers}")
        logger.info("="*80)
        
        # Run training
        results = self.train(train_loader, val_loader, self.num_epochs)
        
        logger.info("="*80)
        logger.info("PHASE 2 COMPLETED")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Training time: {results['training_time']:.2f}s")
        logger.info("="*80)
        
        return results


def train_phase2(config: Config, logger):
    """
    Convenience function to run Phase 2 training
    
    Args:
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Training results dictionary
    """
    from pathlib import Path
    import torch
    from ..models.factory import create_model_from_config
    from ..data.data_loader import create_data_manager
    
    logger.info("Starting Phase 2: Real Data Fine-tuning")
    
    # Load pretrained model from Phase 1
    checkpoint_path = config.checkpoint_dir / "phase1_synthetic" / "best_model.pth"
    
    if not checkpoint_path.exists():
        logger.warning(f"No Phase 1 checkpoint found at {checkpoint_path}")
        logger.warning("Creating new model instead")
        model = create_model_from_config(config)
    else:
        logger.info(f"Loading Phase 1 checkpoint from {checkpoint_path}")
        model = create_model_from_config(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model: {config.model.model_type}")
    
    # Create data manager and loaders
    data_manager = create_data_manager(config)
    dataloaders = data_manager.create_dataloaders(
        data_dir=config.data_dir,
        phase='phase2',
        use_real_data=True,
        create_combined=True
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    logger.info(f"Created data loaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Phase2FineTuningTrainer(model, config)
    trainer.setup_training()
    
    # Run training
    results = trainer.run(train_loader, val_loader)
    
    logger.info("Phase 2 training completed successfully")
    
    return results
