"""Monte Carlo Dropout uncertainty estimation for sonar detection models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
from .base import BaseSonarModel, ModelOutput


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer that remains active during inference"""
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode"""
        return F.dropout(x, p=self.p, training=True)


class MCDropout2d(nn.Module):
    """Monte Carlo Dropout2d layer that remains active during inference"""
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D dropout regardless of training mode"""
        return F.dropout2d(x, p=self.p, training=True)


class UncertaintyEstimator:
    """Monte Carlo Dropout uncertainty estimator for sonar detection models"""
    
    def __init__(
        self, 
        model: BaseSonarModel, 
        num_samples: int = 20,
        device: Optional[torch.device] = None
    ):
        """
        Initialize uncertainty estimator
        
        Args:
            model: Trained sonar detection model
            num_samples: Number of Monte Carlo samples for uncertainty estimation
            device: Device to run inference on
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device or torch.device('cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable Monte Carlo dropout by replacing standard dropout layers
        self._enable_mc_dropout()
    
    def _enable_mc_dropout(self):
        """Replace standard dropout layers with Monte Carlo dropout"""
        def replace_dropout(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Dropout):
                    setattr(module, name, MCDropout(child.p))
                elif isinstance(child, nn.Dropout2d):
                    setattr(module, name, MCDropout2d(child.p))
                else:
                    replace_dropout(child)
        
        replace_dropout(self.model)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        metadata: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            metadata: Optional physics metadata tensor
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        self.model.eval()  # Set to eval mode but dropout will still be active
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x, metadata)
                predictions.append(output.predictions)
        
        # Stack predictions: (num_samples, batch_size, ...)
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean_predictions = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        
        # For classification, use entropy as uncertainty measure
        if self.model.output_mode == "classification":
            # Apply softmax to get probabilities
            probs = F.softmax(mean_predictions, dim=-1)
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            uncertainty = entropy
        else:  # segmentation
            # Use variance as uncertainty measure
            uncertainty = variance
        
        return mean_predictions, uncertainty
    
    def predict_with_detailed_uncertainty(
        self, 
        x: torch.Tensor, 
        metadata: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with detailed uncertainty metrics
        
        Args:
            x: Input tensor
            metadata: Optional physics metadata tensor
            
        Returns:
            Dictionary with detailed uncertainty information
        """
        self.model.eval()
        
        predictions = []
        features_list = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x, metadata)
                predictions.append(output.predictions)
                if output.features is not None:
                    features_list.append(output.features)
        
        # Stack predictions and features
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, ...)
        
        # Calculate statistics
        mean_predictions = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        result = {
            "mean_predictions": mean_predictions,
            "variance": variance,
            "std": std,
            "all_predictions": predictions
        }
        
        if self.model.output_mode == "classification":
            # Classification-specific metrics
            probs = F.softmax(mean_predictions, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            
            # Mutual information (epistemic uncertainty)
            mean_entropy = torch.mean(
                -torch.sum(F.softmax(predictions, dim=-1) * 
                          torch.log(F.softmax(predictions, dim=-1) + 1e-8), dim=-1), 
                dim=0
            )
            mutual_info = entropy - mean_entropy
            
            result.update({
                "probabilities": probs,
                "entropy": entropy,
                "mutual_information": mutual_info,
                "epistemic_uncertainty": mutual_info,
                "aleatoric_uncertainty": mean_entropy
            })
        else:
            # Segmentation-specific metrics
            result.update({
                "pixel_uncertainty": variance,
                "mean_pixel_uncertainty": torch.mean(variance, dim=[1, 2, 3])
            })
        
        # Feature uncertainty if available
        if features_list:
            features = torch.stack(features_list, dim=0)
            feature_variance = torch.var(features, dim=0)
            result["feature_uncertainty"] = torch.mean(feature_variance, dim=[1, 2, 3])
        
        return result
    
    def calibrate_uncertainty(
        self, 
        dataloader: torch.utils.data.DataLoader,
        confidence_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates on validation data
        
        Args:
            dataloader: Validation data loader
            confidence_levels: Confidence levels to evaluate
            
        Returns:
            Dictionary with calibration metrics
        """
        self.model.eval()
        
        all_uncertainties = []
        all_correct = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    x, metadata, targets = batch
                    x, metadata, targets = x.to(self.device), metadata.to(self.device), targets.to(self.device)
                else:
                    x, targets = batch
                    x, targets = x.to(self.device), targets.to(self.device)
                    metadata = None
                
                # Get predictions with uncertainty
                mean_preds, uncertainty = self.predict_with_uncertainty(x, metadata)
                
                if self.model.output_mode == "classification":
                    # Get predicted classes and confidence
                    probs = F.softmax(mean_preds, dim=-1)
                    confidence, predicted = torch.max(probs, dim=-1)
                    correct = (predicted == targets).float()
                    
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                    all_correct.extend(correct.cpu().numpy())
                    all_confidences.extend(confidence.cpu().numpy())
                else:
                    # For segmentation, use pixel-wise metrics
                    predicted = torch.argmax(mean_preds, dim=1)
                    correct = (predicted == targets).float()
                    pixel_uncertainty = torch.mean(uncertainty, dim=[1, 2])
                    
                    all_uncertainties.extend(pixel_uncertainty.cpu().numpy())
                    all_correct.extend(torch.mean(correct, dim=[1, 2]).cpu().numpy())
        
        # Convert to numpy arrays
        uncertainties = np.array(all_uncertainties)
        correct = np.array(all_correct)
        
        # Calculate calibration metrics
        calibration_metrics = {}
        
        # Expected Calibration Error (ECE)
        if self.model.output_mode == "classification":
            confidences = np.array(all_confidences)
            ece = self._calculate_ece(confidences, correct)
            calibration_metrics["expected_calibration_error"] = ece
        
        # Uncertainty-based calibration
        # Sort by uncertainty (low to high)
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_correct = correct[sorted_indices]
        
        # Calculate accuracy for different uncertainty thresholds
        for conf_level in confidence_levels:
            threshold_idx = int(conf_level * len(sorted_uncertainties))
            if threshold_idx > 0:
                high_confidence_accuracy = np.mean(sorted_correct[:threshold_idx])
                calibration_metrics[f"accuracy_at_{conf_level}_confidence"] = high_confidence_accuracy
        
        # Correlation between uncertainty and correctness
        correlation = np.corrcoef(uncertainties, 1 - correct)[0, 1]  # 1 - correct for error
        calibration_metrics["uncertainty_error_correlation"] = correlation
        
        return calibration_metrics
    
    def _calculate_ece(self, confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def generate_uncertainty_heatmap(
        self, 
        x: torch.Tensor, 
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate uncertainty heatmap for visualization
        
        Args:
            x: Input tensor (single image)
            metadata: Optional physics metadata
            
        Returns:
            Uncertainty heatmap tensor
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        if metadata is not None and metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        
        _, uncertainty = self.predict_with_uncertainty(x, metadata)
        
        if self.model.output_mode == "classification":
            # For classification, uncertainty is per-sample
            # Create a uniform heatmap
            heatmap = uncertainty.unsqueeze(-1).unsqueeze(-1).expand(-1, x.size(2), x.size(3))
        else:
            # For segmentation, uncertainty is already spatial
            if uncertainty.dim() == 4:  # (batch, classes, h, w)
                heatmap = torch.mean(uncertainty, dim=1)  # Average over classes
            else:
                heatmap = uncertainty
        
        return heatmap.squeeze(0)  # Remove batch dimension