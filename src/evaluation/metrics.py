"""Core metrics calculation for classification and segmentation"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Calculate classification metrics
    
    Requirements: 5.1, 5.2, 5.3
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.labels = []
        self.confidences = []
    
    def update(self, predictions: np.ndarray, labels: np.ndarray, confidences: Optional[np.ndarray] = None):
        """
        Update metrics with new predictions
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            confidences: Prediction confidences (optional)
        """
        self.predictions.extend(predictions.flatten().tolist())
        self.labels.extend(labels.flatten().tolist())
        
        if confidences is not None:
            self.confidences.extend(confidences.flatten().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all classification metrics
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Confusion matrix components
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False alarms per image
        total_images = len(predictions)
        false_alarms_per_image = fp / total_images if total_images > 0 else 0.0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False negative rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'false_alarms_per_image': false_alarms_per_image,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(total_images)
        }
        
        logger.info(f"Classification Metrics: Acc={accuracy:.4f}, P={precision:.4f}, "
                   f"R={recall:.4f}, F1={f1_score:.4f}")
        
        return metrics
    
    def compute_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Compute metrics for each class"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        unique_classes = np.unique(labels)
        per_class_metrics = {}
        
        for cls in unique_classes:
            # Binary metrics for this class vs rest
            cls_predictions = (predictions == cls).astype(int)
            cls_labels = (labels == cls).astype(int)
            
            tp = np.sum((cls_predictions == 1) & (cls_labels == 1))
            tn = np.sum((cls_predictions == 0) & (cls_labels == 0))
            fp = np.sum((cls_predictions == 1) & (cls_labels == 0))
            fn = np.sum((cls_predictions == 0) & (cls_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[int(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(np.sum(cls_labels))
            }
        
        return per_class_metrics


class SegmentationMetrics:
    """
    Calculate segmentation metrics (IoU, Dice)
    
    Requirements: 5.2
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.target_sum = np.zeros(self.num_classes)
        self.pred_sum = np.zeros(self.num_classes)
    
    def update(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Update metrics with new predictions
        
        Args:
            predictions: Predicted segmentation masks
            labels: Ground truth segmentation masks
        """
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        for cls in range(self.num_classes):
            pred_cls = (predictions == cls)
            label_cls = (labels == cls)
            
            self.intersection[cls] += np.sum(pred_cls & label_cls)
            self.union[cls] += np.sum(pred_cls | label_cls)
            self.target_sum[cls] += np.sum(label_cls)
            self.pred_sum[cls] += np.sum(pred_cls)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics
        
        Returns:
            Dictionary of metrics
        """
        # IoU (Intersection over Union)
        iou_per_class = self.intersection / (self.union + 1e-10)
        mean_iou = np.mean(iou_per_class)
        
        # Dice coefficient
        dice_per_class = (2 * self.intersection) / (self.target_sum + self.pred_sum + 1e-10)
        mean_dice = np.mean(dice_per_class)
        
        # Pixel accuracy
        pixel_accuracy = np.sum(self.intersection) / (np.sum(self.target_sum) + 1e-10)
        
        metrics = {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'pixel_accuracy': pixel_accuracy
        }
        
        # Add per-class metrics
        for cls in range(self.num_classes):
            metrics[f'iou_class_{cls}'] = iou_per_class[cls]
            metrics[f'dice_class_{cls}'] = dice_per_class[cls]
        
        logger.info(f"Segmentation Metrics: mIoU={mean_iou:.4f}, mDice={mean_dice:.4f}, "
                   f"PixAcc={pixel_accuracy:.4f}")
        
        return metrics


class UncertaintyMetrics:
    """
    Calculate uncertainty-related metrics
    
    Requirements: 5.4, 3.5
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.uncertainties = []
        self.confidences = []
        self.correctness = []
        self.predictions = []
        self.labels = []
    
    def update(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ):
        """
        Update metrics with new data
        
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
        correct = (predictions.flatten() == labels.flatten()).astype(float)
        self.correctness.extend(correct.tolist())
        
        if confidences is not None:
            self.confidences.extend(confidences.flatten().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute uncertainty metrics
        
        Returns:
            Dictionary of metrics
        """
        uncertainties = np.array(self.uncertainties)
        correctness = np.array(self.correctness)
        
        # Mean uncertainty
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
        # Uncertainty for correct vs incorrect predictions
        correct_mask = correctness == 1
        incorrect_mask = correctness == 0
        
        mean_uncertainty_correct = np.mean(uncertainties[correct_mask]) if correct_mask.sum() > 0 else 0.0
        mean_uncertainty_incorrect = np.mean(uncertainties[incorrect_mask]) if incorrect_mask.sum() > 0 else 0.0
        
        # Correlation between uncertainty and correctness
        # Higher uncertainty should correlate with incorrect predictions
        correlation = -np.corrcoef(uncertainties, correctness)[0, 1] if len(uncertainties) > 1 else 0.0
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(uncertainties, correctness)
        
        metrics = {
            'mean_uncertainty': mean_uncertainty,
            'std_uncertainty': std_uncertainty,
            'mean_uncertainty_correct': mean_uncertainty_correct,
            'mean_uncertainty_incorrect': mean_uncertainty_incorrect,
            'uncertainty_correctness_correlation': correlation,
            'expected_calibration_error': ece
        }
        
        logger.info(f"Uncertainty Metrics: Mean={mean_uncertainty:.4f}, "
                   f"Correct={mean_uncertainty_correct:.4f}, "
                   f"Incorrect={mean_uncertainty_incorrect:.4f}, ECE={ece:.4f}")
        
        return metrics
    
    def _compute_ece(self, uncertainties: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error
        
        Args:
            uncertainties: Prediction uncertainties
            correctness: Correctness indicators
            n_bins: Number of bins
        
        Returns:
            Expected Calibration Error
        """
        # Convert uncertainty to confidence (1 - uncertainty)
        confidences = 1 - uncertainties
        
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


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    task_type: str = 'classification'
) -> Dict[str, float]:
    """
    Compute all relevant metrics
    
    Args:
        predictions: Predicted labels or masks
        labels: Ground truth labels or masks
        uncertainties: Prediction uncertainties (optional)
        confidences: Prediction confidences (optional)
        task_type: 'classification' or 'segmentation'
    
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    # Classification metrics
    if task_type == 'classification':
        clf_metrics = ClassificationMetrics()
        clf_metrics.update(predictions, labels, confidences)
        all_metrics.update(clf_metrics.compute())
    
    # Segmentation metrics
    elif task_type == 'segmentation':
        seg_metrics = SegmentationMetrics()
        seg_metrics.update(predictions, labels)
        all_metrics.update(seg_metrics.compute())
    
    # Uncertainty metrics
    if uncertainties is not None:
        unc_metrics = UncertaintyMetrics()
        unc_metrics.update(uncertainties, predictions, labels, confidences)
        all_metrics.update(unc_metrics.compute())
    
    return all_metrics
