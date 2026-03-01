"""Uncertainty evaluation framework with calibration curves and reliability diagrams"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class CalibrationCurve:
    """
    Generate calibration curves for uncertainty estimates
    
    Requirements: 5.4
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        """Reset calibration data"""
        self.confidences = []
        self.correctness = []
    
    def update(self, confidences: np.ndarray, correctness: np.ndarray):
        """
        Update with new data
        
        Args:
            confidences: Prediction confidences [0, 1]
            correctness: Binary correctness indicators
        """
        self.confidences.extend(confidences.flatten().tolist())
        self.correctness.extend(correctness.flatten().tolist())
    
    def compute(self) -> Dict[str, np.ndarray]:
        """
        Compute calibration curve data
        
        Returns:
            Dictionary with bin_confidences, bin_accuracies, bin_counts
        """
        confidences = np.array(self.confidences)
        correctness = np.array(self.correctness)
        
        # Create bins
        bins = np.linspace(0, 1, self.n_bins + 1)
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(self.n_bins):
            # Get samples in this bin
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correctness[mask].mean()
                bin_count = mask.sum()
            else:
                bin_conf = (bins[i] + bins[i + 1]) / 2
                bin_acc = 0.0
                bin_count = 0
            
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            bin_counts.append(bin_count)
        
        return {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
            'bins': bins
        }
    
    def compute_ece(self) -> float:
        """Compute Expected Calibration Error"""
        data = self.compute()
        bin_confidences = data['bin_confidences']
        bin_accuracies = data['bin_accuracies']
        bin_counts = data['bin_counts']
        
        total_samples = sum(bin_counts)
        ece = sum(
            (count / total_samples) * abs(conf - acc)
            for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
            if count > 0
        )
        
        return ece


class ReliabilityDiagram:
    """
    Generate reliability diagrams for model calibration
    
    Requirements: 5.4
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_curve = CalibrationCurve(n_bins)
    
    def update(self, confidences: np.ndarray, correctness: np.ndarray):
        """Update with new data"""
        self.calibration_curve.update(confidences, correctness)
    
    def compute(self) -> Dict[str, any]:
        """
        Compute reliability diagram data
        
        Returns:
            Dictionary with calibration data and metrics
        """
        calib_data = self.calibration_curve.compute()
        ece = self.calibration_curve.compute_ece()
        
        # Compute additional metrics
        bin_confidences = calib_data['bin_confidences']
        bin_accuracies = calib_data['bin_accuracies']
        bin_counts = calib_data['bin_counts']
        
        # Maximum Calibration Error (MCE)
        mce = max(
            abs(conf - acc)
            for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
            if count > 0
        ) if any(count > 0 for count in bin_counts) else 0.0
        
        # Average Confidence
        all_confidences = np.array(self.calibration_curve.confidences)
        avg_confidence = all_confidences.mean() if len(all_confidences) > 0 else 0.0
        
        # Average Accuracy
        all_correctness = np.array(self.calibration_curve.correctness)
        avg_accuracy = all_correctness.mean() if len(all_correctness) > 0 else 0.0
        
        return {
            **calib_data,
            'ece': ece,
            'mce': mce,
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'confidence_accuracy_gap': avg_confidence - avg_accuracy
        }


class UncertaintyEvaluator:
    """
    Comprehensive uncertainty evaluation
    
    Requirements: 3.5, 5.4
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_curve = CalibrationCurve(n_bins)
        self.reliability_diagram = ReliabilityDiagram(n_bins)
        
        self.uncertainties = []
        self.predictions = []
        self.labels = []
        self.confidences = []
    
    def update(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ):
        """
        Update with new predictions
        
        Args:
            uncertainties: Prediction uncertainties
            predictions: Predicted labels
            labels: Ground truth labels
            confidences: Prediction confidences (optional)
        """
        self.uncertainties.extend(uncertainties.flatten().tolist())
        self.predictions.extend(predictions.flatten().tolist())
        self.labels.extend(labels.flatten().tolist())
        
        # Compute correctness
        correctness = (predictions.flatten() == labels.flatten()).astype(float)
        
        # Use confidences or convert from uncertainties
        if confidences is not None:
            conf = confidences.flatten()
        else:
            conf = 1 - uncertainties.flatten()  # Convert uncertainty to confidence
        
        self.confidences.extend(conf.tolist())
        
        # Update calibration and reliability
        self.calibration_curve.update(conf, correctness)
        self.reliability_diagram.update(conf, correctness)
    
    def evaluate(self) -> Dict[str, any]:
        """
        Perform comprehensive uncertainty evaluation
        
        Returns:
            Dictionary with all uncertainty metrics and calibration data
        """
        uncertainties = np.array(self.uncertainties)
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        confidences = np.array(self.confidences)
        correctness = (predictions == labels).astype(float)
        
        # Basic uncertainty statistics
        metrics = {
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std(),
            'min_uncertainty': uncertainties.min(),
            'max_uncertainty': uncertainties.max(),
            'median_uncertainty': np.median(uncertainties)
        }
        
        # Uncertainty for correct vs incorrect predictions
        correct_mask = correctness == 1
        incorrect_mask = correctness == 0
        
        if correct_mask.sum() > 0:
            metrics['mean_uncertainty_correct'] = uncertainties[correct_mask].mean()
            metrics['std_uncertainty_correct'] = uncertainties[correct_mask].std()
        
        if incorrect_mask.sum() > 0:
            metrics['mean_uncertainty_incorrect'] = uncertainties[incorrect_mask].mean()
            metrics['std_uncertainty_incorrect'] = uncertainties[incorrect_mask].std()
        
        # Uncertainty-accuracy correlation
        if len(uncertainties) > 1:
            # Negative correlation expected (high uncertainty = low accuracy)
            metrics['uncertainty_accuracy_correlation'] = -np.corrcoef(uncertainties, correctness)[0, 1]
        
        # Calibration metrics
        reliability_data = self.reliability_diagram.compute()
        metrics.update({
            'expected_calibration_error': reliability_data['ece'],
            'maximum_calibration_error': reliability_data['mce'],
            'avg_confidence': reliability_data['avg_confidence'],
            'avg_accuracy': reliability_data['avg_accuracy'],
            'confidence_accuracy_gap': reliability_data['confidence_accuracy_gap']
        })
        
        # Add calibration curve data
        metrics['calibration_data'] = reliability_data
        
        logger.info(f"Uncertainty Evaluation: ECE={reliability_data['ece']:.4f}, "
                   f"MCE={reliability_data['mce']:.4f}, "
                   f"Mean Unc={metrics['mean_uncertainty']:.4f}")
        
        return metrics
    
    def get_calibration_data(self) -> Dict[str, np.ndarray]:
        """Get calibration curve data for plotting"""
        return self.calibration_curve.compute()
    
    def get_reliability_data(self) -> Dict[str, any]:
        """Get reliability diagram data for plotting"""
        return self.reliability_diagram.compute()
