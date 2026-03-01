"""Metrics reporting and export system"""

import logging
import json
import csv
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsReporter:
    """
    Export and report evaluation metrics
    
    Requirements: 5.5, 6.4
    """
    
    def __init__(self, config=None):
        self.config = config
        self.output_dir = Path("outputs/reports") if config is None else config.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_csv(
        self,
        metrics: Dict[str, Any],
        filepath: Optional[Path] = None,
        append: bool = False
    ) -> Path:
        """
        Export metrics to CSV file
        
        Args:
            metrics: Dictionary of metrics
            filepath: Output file path (optional)
            append: Whether to append to existing file
        
        Returns:
            Path to the saved CSV file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"metrics_{timestamp}.csv"
        
        # Flatten nested dictionaries
        flat_metrics = self._flatten_dict(metrics)
        
        # Write to CSV
        mode = 'a' if append and filepath.exists() else 'w'
        with open(filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_metrics.keys())
            
            if mode == 'w':
                writer.writeheader()
            
            writer.writerow(flat_metrics)
        
        logger.info(f"Metrics exported to CSV: {filepath}")
        return filepath
    
    def export_json(
        self,
        metrics: Dict[str, Any],
        filepath: Optional[Path] = None,
        indent: int = 2
    ) -> Path:
        """
        Export metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            filepath: Output file path (optional)
            indent: JSON indentation level
        
        Returns:
            Path to the saved JSON file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"metrics_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_serializable(metrics)
        
        # Add metadata
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': serializable_metrics
        }
        
        # Write to JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=indent)
        
        logger.info(f"Metrics exported to JSON: {filepath}")
        return filepath
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive text report
        
        Args:
            metrics: Dictionary of metrics
            report_name: Name for the report
        
        Returns:
            Report as formatted string
        """
        if report_name is None:
            report_name = f"Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        lines = []
        lines.append("=" * 80)
        lines.append(report_name.center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # Classification metrics
        if 'accuracy' in metrics:
            lines.append("CLASSIFICATION METRICS")
            lines.append("-" * 80)
            lines.append(f"  Accuracy:              {metrics.get('accuracy', 0):.4f}")
            lines.append(f"  Precision:             {metrics.get('precision', 0):.4f}")
            lines.append(f"  Recall:                {metrics.get('recall', 0):.4f}")
            lines.append(f"  F1-Score:              {metrics.get('f1_score', 0):.4f}")
            lines.append(f"  Specificity:           {metrics.get('specificity', 0):.4f}")
            lines.append(f"  False Alarms/Image:    {metrics.get('false_alarms_per_image', 0):.4f}")
            lines.append("")
            
            # Confusion matrix
            lines.append("  Confusion Matrix:")
            lines.append(f"    True Positives:      {metrics.get('true_positives', 0)}")
            lines.append(f"    True Negatives:      {metrics.get('true_negatives', 0)}")
            lines.append(f"    False Positives:     {metrics.get('false_positives', 0)}")
            lines.append(f"    False Negatives:     {metrics.get('false_negatives', 0)}")
            lines.append("")
        
        # Segmentation metrics
        if 'mean_iou' in metrics:
            lines.append("SEGMENTATION METRICS")
            lines.append("-" * 80)
            lines.append(f"  Mean IoU:              {metrics.get('mean_iou', 0):.4f}")
            lines.append(f"  Mean Dice:             {metrics.get('mean_dice', 0):.4f}")
            lines.append(f"  Pixel Accuracy:        {metrics.get('pixel_accuracy', 0):.4f}")
            lines.append("")
        
        # Uncertainty metrics
        if 'mean_uncertainty' in metrics:
            lines.append("UNCERTAINTY METRICS")
            lines.append("-" * 80)
            lines.append(f"  Mean Uncertainty:      {metrics.get('mean_uncertainty', 0):.4f}")
            lines.append(f"  Std Uncertainty:       {metrics.get('std_uncertainty', 0):.4f}")
            lines.append(f"  Unc. (Correct):        {metrics.get('mean_uncertainty_correct', 0):.4f}")
            lines.append(f"  Unc. (Incorrect):      {metrics.get('mean_uncertainty_incorrect', 0):.4f}")
            lines.append(f"  Unc-Acc Correlation:   {metrics.get('uncertainty_accuracy_correlation', 0):.4f}")
            lines.append("")
        
        # Calibration metrics
        if 'expected_calibration_error' in metrics:
            lines.append("CALIBRATION METRICS")
            lines.append("-" * 80)
            lines.append(f"  Expected Calib. Error: {metrics.get('expected_calibration_error', 0):.4f}")
            lines.append(f"  Maximum Calib. Error:  {metrics.get('maximum_calibration_error', 0):.4f}")
            lines.append(f"  Avg Confidence:        {metrics.get('avg_confidence', 0):.4f}")
            lines.append(f"  Avg Accuracy:          {metrics.get('avg_accuracy', 0):.4f}")
            lines.append(f"  Conf-Acc Gap:          {metrics.get('confidence_accuracy_gap', 0):.4f}")
            lines.append("")
        
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        logger.info(f"\n{report_text}")
        
        return report_text
    
    def save_report(
        self,
        metrics: Dict[str, Any],
        filepath: Optional[Path] = None,
        report_name: Optional[str] = None
    ) -> Path:
        """
        Generate and save text report to file
        
        Args:
            metrics: Dictionary of metrics
            filepath: Output file path (optional)
            report_name: Name for the report
        
        Returns:
            Path to the saved report file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"report_{timestamp}.txt"
        
        report_text = self.generate_report(metrics, report_name)
        
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to: {filepath}")
        return filepath
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(v[0], (int, float, str)):
                # Skip complex nested structures
                continue
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


def export_metrics_csv(
    metrics: Dict[str, Any],
    filepath: Optional[Path] = None,
    append: bool = False
) -> Path:
    """
    Standalone function to export metrics to CSV
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path (optional)
        append: Whether to append to existing file
    
    Returns:
        Path to the saved CSV file
    """
    reporter = MetricsReporter()
    return reporter.export_csv(metrics, filepath, append)


def export_metrics_json(
    metrics: Dict[str, Any],
    filepath: Optional[Path] = None,
    indent: int = 2
) -> Path:
    """
    Standalone function to export metrics to JSON
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path (optional)
        indent: JSON indentation level
    
    Returns:
        Path to the saved JSON file
    """
    reporter = MetricsReporter()
    return reporter.export_json(metrics, filepath, indent)
