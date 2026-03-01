"""Model comparison system for evaluating different training approaches"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare different model training approaches
    
    Requirements: 5.5
    """
    
    def __init__(self, config=None):
        self.config = config
        self.results = {}
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Add model results for comparison
        
        Args:
            model_name: Identifier for the model/approach
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata (training config, etc.)
        """
        self.results[model_name] = {
            'metrics': metrics,
            'metadata': metadata or {}
        }
        logger.info(f"Added results for model: {model_name}")
    
    def compare_synthetic_vs_finetuned(
        self,
        synthetic_only_metrics: Dict[str, float],
        finetuned_metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Compare synthetic-only vs synthetic+real fine-tuned models
        
        Args:
            synthetic_only_metrics: Metrics from synthetic-only training
            finetuned_metrics: Metrics from fine-tuned model
        
        Returns:
            Comparison results with improvements
        """
        comparison = {
            'synthetic_only': synthetic_only_metrics,
            'finetuned': finetuned_metrics,
            'improvements': {},
            'degradations': {}
        }
        
        # Calculate improvements for each metric
        for metric_name in synthetic_only_metrics.keys():
            if metric_name in finetuned_metrics:
                synthetic_val = synthetic_only_metrics[metric_name]
                finetuned_val = finetuned_metrics[metric_name]
                
                # Skip non-numeric values
                if not isinstance(synthetic_val, (int, float)) or not isinstance(finetuned_val, (int, float)):
                    continue
                
                improvement = finetuned_val - synthetic_val
                improvement_pct = (improvement / synthetic_val * 100) if synthetic_val != 0 else 0
                
                comparison['improvements'][metric_name] = {
                    'absolute': improvement,
                    'percentage': improvement_pct,
                    'synthetic_value': synthetic_val,
                    'finetuned_value': finetuned_val
                }
                
                # Track degradations
                if improvement < 0:
                    comparison['degradations'][metric_name] = improvement_pct
        
        logger.info(f"Comparison complete: {len(comparison['improvements'])} metrics compared")
        logger.info(f"Improvements: {len([k for k, v in comparison['improvements'].items() if v['absolute'] > 0])}")
        logger.info(f"Degradations: {len(comparison['degradations'])}")
        
        return comparison
    
    def compare_with_without_uncertainty(
        self,
        without_uncertainty_metrics: Dict[str, float],
        with_uncertainty_metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Compare models with and without uncertainty estimation
        
        Args:
            without_uncertainty_metrics: Metrics without uncertainty
            with_uncertainty_metrics: Metrics with uncertainty estimation
        
        Returns:
            Comparison results
        """
        comparison = {
            'without_uncertainty': without_uncertainty_metrics,
            'with_uncertainty': with_uncertainty_metrics,
            'differences': {}
        }
        
        # Calculate differences
        for metric_name in without_uncertainty_metrics.keys():
            if metric_name in with_uncertainty_metrics:
                without_val = without_uncertainty_metrics[metric_name]
                with_val = with_uncertainty_metrics[metric_name]
                
                if not isinstance(without_val, (int, float)) or not isinstance(with_val, (int, float)):
                    continue
                
                difference = with_val - without_val
                difference_pct = (difference / without_val * 100) if without_val != 0 else 0
                
                comparison['differences'][metric_name] = {
                    'absolute': difference,
                    'percentage': difference_pct,
                    'without_value': without_val,
                    'with_value': with_val
                }
        
        logger.info(f"Uncertainty comparison complete: {len(comparison['differences'])} metrics compared")
        
        return comparison
    
    def compare_all(self) -> Dict[str, any]:
        """
        Compare all added model results
        
        Returns:
            Comprehensive comparison of all models
        """
        if len(self.results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        comparison = {
            'models': list(self.results.keys()),
            'metric_comparison': {},
            'best_per_metric': {},
            'worst_per_metric': {}
        }
        
        # Get all unique metrics
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result['metrics'].keys())
        
        # Compare each metric across models
        for metric_name in all_metrics:
            metric_values = {}
            
            for model_name, result in self.results.items():
                if metric_name in result['metrics']:
                    value = result['metrics'][metric_name]
                    if isinstance(value, (int, float)):
                        metric_values[model_name] = value
            
            if len(metric_values) > 0:
                comparison['metric_comparison'][metric_name] = metric_values
                
                # Find best and worst
                best_model = max(metric_values.items(), key=lambda x: x[1])
                worst_model = min(metric_values.items(), key=lambda x: x[1])
                
                comparison['best_per_metric'][metric_name] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }
                comparison['worst_per_metric'][metric_name] = {
                    'model': worst_model[0],
                    'value': worst_model[1]
                }
        
        logger.info(f"Compared {len(self.results)} models across {len(all_metrics)} metrics")
        
        return comparison
    
    def generate_comparison_report(self) -> str:
        """Generate text report of all comparisons"""
        comparison = self.compare_all()
        
        if not comparison:
            return "Insufficient data for comparison"
        
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL COMPARISON REPORT".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Models Compared: {', '.join(comparison['models'])}")
        lines.append("")
        
        lines.append("BEST PERFORMANCE PER METRIC")
        lines.append("-" * 80)
        for metric_name, data in comparison['best_per_metric'].items():
            lines.append(f"  {metric_name:30s}: {data['model']:20s} ({data['value']:.4f})")
        lines.append("")
        
        lines.append("METRIC COMPARISON")
        lines.append("-" * 80)
        for metric_name, values in comparison['metric_comparison'].items():
            lines.append(f"  {metric_name}:")
            for model_name, value in values.items():
                lines.append(f"    {model_name:25s}: {value:.4f}")
        lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        logger.info(f"\n{report}")
        
        return report


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    config=None
) -> Dict[str, any]:
    """
    Standalone function to compare multiple models
    
    Args:
        model_results: Dictionary mapping model names to their metrics
        config: Optional configuration
    
    Returns:
        Comparison results
    """
    comparison = ModelComparison(config)
    
    for model_name, metrics in model_results.items():
        comparison.add_result(model_name, metrics)
    
    return comparison.compare_all()


def statistical_significance_test(
    results_a: np.ndarray,
    results_b: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Perform statistical significance test between two sets of results
    
    Args:
        results_a: Results from model A (multiple runs)
        results_b: Results from model B (multiple runs)
        alpha: Significance level
    
    Returns:
        Test results with p-value and significance
    """
    # Compute means and standard deviations
    mean_a = np.mean(results_a)
    mean_b = np.mean(results_b)
    std_a = np.std(results_a, ddof=1)
    std_b = np.std(results_b, ddof=1)
    
    n_a = len(results_a)
    n_b = len(results_b)
    
    # Compute t-statistic (Welch's t-test for unequal variances)
    pooled_std = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
    
    if pooled_std == 0:
        t_statistic = 0
        p_value = 1.0
    else:
        t_statistic = (mean_a - mean_b) / pooled_std
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = ((std_a**2 / n_a + std_b**2 / n_b)**2) / \
             ((std_a**2 / n_a)**2 / (n_a - 1) + (std_b**2 / n_b)**2 / (n_b - 1))
        
        # Approximate p-value using normal distribution
        # For more accurate results, use scipy.stats.t.sf
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_statistic / np.sqrt(2))))
    
    is_significant = p_value < alpha
    
    result = {
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'std_a': float(std_a),
        'std_b': float(std_b),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'alpha': alpha,
        'is_significant': is_significant,
        'difference': float(mean_a - mean_b),
        'effect_size': float((mean_a - mean_b) / np.sqrt((std_a**2 + std_b**2) / 2)) if (std_a**2 + std_b**2) > 0 else 0
    }
    
    logger.info(f"Statistical test: t={t_statistic:.4f}, p={p_value:.4f}, "
               f"significant={is_significant}")
    
    return result
