"""Phase 3: Uncertainty calibration trainer"""

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..config.config import Config
from .trainer import Trainer
from .utils import CheckpointManager

logger = logging.getLogger(__name__)


class Phase3CalibrationTrainer(Trainer):
    """
    Phase 3: Uncertainty calibration trainer
    
    Calibrates uncertainty estimates with:
    - Dropout enabled during validation
    - Multiple forward passes for uncertainty
    - Confidence vs correctness tracking
    
    Requirements: 4.3, 3.5
    """
    
    def __init__(self, model: nn.Module, config: Config, device=None):
        super().__init__(model, config, device)
        
        self.num_epochs = config.training.phase3_epochs
        self.learning_rate = config.training.phase3_lr
        self.batch_size = config.training.phase3_batch_size
        self.mc_samples = config.model.mc_samples
        
        logger.info("Initialized Phase 3: Uncertainty Calibration")
    
    def setup_training(self):
        """Setup training components for Phase 3"""
        # Optimizer - very low learning rate for calibration
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6
        )
        
        # Loss criterion
        if self.config.model.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # No scheduler for calibration phase
        self.scheduler = None
        
        # Checkpoint manager
        checkpoint_dir = self.config.checkpoint_dir / "phase3_calibration"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=3,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        
        logger.info(f"Setup Phase 3 training: lr={self.learning_rate}, "
                   f"epochs={self.num_epochs}, mc_samples={self.mc_samples}")
    
    def enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with uncertainty calibration
        
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
        total_uncertainty = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train-Calib]")
        
        for batch in pbar:
            # Get data
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
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
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self.compute_accuracy(outputs, labels.long())
            
            # Compute uncertainty (variance of predictions)
            with torch.no_grad():
                uncertainty = self._compute_uncertainty(images, metadata)
            
            # Update metrics
            total_loss += loss.item()
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
            total_uncertainty += uncertainty.mean().item() * labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'unc': f'{uncertainty.mean().item():.4f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        avg_uncertainty = total_uncertainty / total_samples
        
        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'train_uncertainty': avg_uncertainty
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch with uncertainty estimation
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Validation metrics including uncertainty
        """
        self.model.eval()
        self.enable_dropout()  # Keep dropout enabled for uncertainty
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_uncertainty = 0.0
        
        # Track confidence vs correctness for calibration
        confidences = []
        correctness = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val-Calib]")
            
            for batch in pbar:
                # Get data
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                metadata = batch['metadata'].to(self.device) if self.config.model.use_physics_metadata else None
                
                # Multiple forward passes for uncertainty
                predictions = []
                for _ in range(self.mc_samples):
                    if metadata is not None and self.config.model.use_physics_metadata:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    
                    if self.config.model.num_classes == 2:
                        outputs = outputs.squeeze()
                    
                    predictions.append(torch.sigmoid(outputs).cpu().numpy())
                
                # Compute mean and uncertainty
                predictions = np.array(predictions)
                mean_pred = predictions.mean(axis=0)
                uncertainty = predictions.std(axis=0)
                
                # Compute loss on mean prediction
                mean_pred_tensor = torch.from_numpy(mean_pred).to(self.device)
                labels_float = labels.float()
                loss = self.criterion(mean_pred_tensor, labels_float)
                
                # Compute accuracy
                predicted_labels = (mean_pred > 0.5).astype(int)
                correct = (predicted_labels == labels.cpu().numpy()).astype(float)
                accuracy = correct.mean()
                
                # Track confidence vs correctness
                confidence = np.abs(mean_pred - 0.5) * 2  # Scale to [0, 1]
                confidences.extend(confidence.tolist())
                correctness.extend(correct.tolist())
                
                # Update metrics
                total_loss += loss.item()
                total_correct += accuracy * labels.size(0)
                total_samples += labels.size(0)
                total_uncertainty += uncertainty.mean() * labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'unc': f'{uncertainty.mean():.4f}'
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        avg_uncertainty = total_uncertainty / total_samples
        
        # Compute calibration metrics
        calibration_error = self._compute_calibration_error(confidences, correctness)
        
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'val_uncertainty': avg_uncertainty,
            'calibration_error': calibration_error
        }
    
    def _compute_uncertainty(self, images: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """
        Compute uncertainty using Monte Carlo Dropout
        
        Args:
            images: Input images
            metadata: Optional metadata
        
        Returns:
            Uncertainty estimates
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                if metadata is not None and self.config.model.use_physics_metadata:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
                if self.config.model.num_classes == 2:
                    outputs = outputs.squeeze()
                
                predictions.append(torch.sigmoid(outputs))
        
        # Stack predictions and compute variance
        predictions = torch.stack(predictions)
        uncertainty = predictions.std(dim=0)
        
        return uncertainty
    
    def _compute_calibration_error(self, confidences: list, correctness: list, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            confidences: List of confidence scores
            correctness: List of correctness indicators
            n_bins: Number of bins for calibration
        
        Returns:
            Expected Calibration Error
        """
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            # Get samples in this bin
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            
            if mask.sum() > 0:
                bin_confidence = confidences[mask].mean()
                bin_accuracy = correctness[mask].mean()
                bin_size = mask.sum()
                
                # Add to ECE
                ece += (bin_size / len(confidences)) * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def run(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Run Phase 3 calibration
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training results
        """
        logger.info("="*80)
        logger.info("PHASE 3: UNCERTAINTY CALIBRATION")
        logger.info("="*80)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"MC samples: {self.mc_samples}")
        logger.info("="*80)
        
        # Run training
        results = self.train(train_loader, val_loader, self.num_epochs)
        
        logger.info("="*80)
        logger.info("PHASE 3 COMPLETED")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Training time: {results['training_time']:.2f}s")
        logger.info("="*80)
        
        return results


def train_phase3(config: Config, logger):
    """
    Convenience function to run Phase 3 training
    
    Args:
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Training results dictionary
    """
    from pathlib import Path
    from ..models.factory import create_model_from_config
    from ..data.data_loader import create_data_manager
    
    logger.info("Starting Phase 3: Uncertainty Calibration")
    
    # Load pretrained model from Phase 2 (or Phase 1 if Phase 2 was skipped)
    checkpoint_path = config.checkpoint_dir / "phase2_finetuning" / "best_model.pth"
    
    if not checkpoint_path.exists():
        checkpoint_path = config.checkpoint_dir / "phase1_synthetic" / "best_model.pth"
        logger.info(f"Phase 2 checkpoint not found, using Phase 1 checkpoint")
    
    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint found, creating new model")
        model = create_model_from_config(config)
    else:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model = create_model_from_config(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model: {config.model.model_type}")
    
    # Create data manager and loaders
    data_manager = create_data_manager(config)
    dataloaders = data_manager.create_dataloaders(
        data_dir=config.data_dir,
        phase='phase3',
        use_real_data=True,
        create_combined=True
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    logger.info(f"Created data loaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Phase3CalibrationTrainer(model, config)
    trainer.setup_training()
    
    # Run training
    results = trainer.run(train_loader, val_loader)
    
    logger.info("Phase 3 training completed successfully")
    
    return results
