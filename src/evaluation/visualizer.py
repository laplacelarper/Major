"""Visualization system for evaluation results"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Generate visualizations for evaluation results
    
    Requirements: 5.5, 6.4
    """
    
    def __init__(self, config=None):
        self.config = config
        self.output_dir = Path("outputs/visualizations") if config is None else config.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Generate confusion matrix visualization
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            save_path: Path to save the plot
        
        Returns:
            Dictionary with confusion matrix data
        """
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Compute confusion matrix
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        logger.info(f"Confusion Matrix:\n{confusion_matrix}")
        
        return {
            'confusion_matrix': confusion_matrix,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def plot_roc_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Generate ROC curve data
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            confidences: Prediction confidences
            save_path: Path to save the plot
        
        Returns:
            Dictionary with ROC curve data
        """
        if confidences is None:
            confidences = predictions.astype(float)
        
        # Sort by confidence
        sorted_indices = np.argsort(confidences.flatten())[::-1]
        sorted_labels = labels.flatten()[sorted_indices]
        
        # Compute TPR and FPR at different thresholds
        tpr_list = []
        fpr_list = []
        
        total_positives = np.sum(sorted_labels == 1)
        total_negatives = np.sum(sorted_labels == 0)
        
        tp = 0
        fp = 0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / total_positives if total_positives > 0 else 0
            fpr = fp / total_negatives if total_negatives > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Compute AUC using trapezoidal rule
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        auc = np.trapz(tpr_array, fpr_array)
        
        logger.info(f"ROC AUC: {auc:.4f}")
        
        return {
            'fpr': fpr_array,
            'tpr': tpr_array,
            'auc': float(auc)
        }
    
    def plot_uncertainty_heatmap(
        self,
        uncertainties: np.ndarray,
        images: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Generate uncertainty heatmap visualization
        
        Args:
            uncertainties: Prediction uncertainties (can be per-pixel or per-image)
            images: Original images (optional)
            save_path: Path to save the plot
        
        Returns:
            Dictionary with heatmap data
        """
        # Reshape uncertainties if needed
        if uncertainties.ndim == 1:
            # Per-image uncertainties
            uncertainty_map = uncertainties
        else:
            # Per-pixel uncertainties
            uncertainty_map = uncertainties
        
        stats = {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'min_uncertainty': float(np.min(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'median_uncertainty': float(np.median(uncertainties))
        }
        
        logger.info(f"Uncertainty Stats: Mean={stats['mean_uncertainty']:.4f}, "
                   f"Std={stats['std_uncertainty']:.4f}")
        
        return {
            'uncertainty_map': uncertainty_map,
            'stats': stats
        }
    
    def visualize_synthetic_sample(
        self,
        image: np.ndarray,
        metadata: Dict[str, float],
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Visualize synthetic image with physics parameters
        
        Args:
            image: Synthetic sonar image
            metadata: Physics parameters dictionary
            save_path: Path to save the visualization
        
        Returns:
            Dictionary with visualization data
        """
        logger.info(f"Synthetic Sample - Physics Parameters:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return {
            'image': image,
            'metadata': metadata,
            'image_shape': image.shape,
            'image_stats': {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image))
            }
        }
    
    def visualize_detection_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        label: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Create detection overlay visualization
        
        Args:
            image: Original sonar image
            prediction: Predicted label or mask
            label: Ground truth label or mask (optional)
            uncertainty: Prediction uncertainty (optional)
            save_path: Path to save the visualization
        
        Returns:
            Dictionary with overlay data
        """
        result = {
            'image': image,
            'prediction': prediction
        }
        
        if label is not None:
            result['label'] = label
            result['correct'] = bool(np.array_equal(prediction, label))
        
        if uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        logger.info(f"Detection Overlay - Prediction: {prediction.flatten()[0] if prediction.size > 0 else 'N/A'}")
        
        return result


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Standalone function to plot confusion matrix
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        save_path: Path to save the plot
    
    Returns:
        Dictionary with confusion matrix data
    """
    visualizer = ResultVisualizer()
    return visualizer.plot_confusion_matrix(predictions, labels, save_path)


def plot_roc_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Standalone function to plot ROC curve
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        confidences: Prediction confidences
        save_path: Path to save the plot
    
    Returns:
        Dictionary with ROC curve data
    """
    visualizer = ResultVisualizer()
    return visualizer.plot_roc_curve(predictions, labels, confidences, save_path)


def plot_uncertainty_heatmap(
    uncertainties: np.ndarray,
    images: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Standalone function to plot uncertainty heatmap
    
    Args:
        uncertainties: Prediction uncertainties
        images: Original images (optional)
        save_path: Path to save the plot
    
    Returns:
        Dictionary with heatmap data
    """
    visualizer = ResultVisualizer()
    return visualizer.plot_uncertainty_heatmap(uncertainties, images, save_path)
