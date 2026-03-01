"""Evaluation modules for the sonar detection system"""

from .metrics import (
    ClassificationMetrics,
    SegmentationMetrics,
    UncertaintyMetrics,
    compute_all_metrics
)
from .uncertainty_eval import (
    UncertaintyEvaluator,
    CalibrationCurve,
    ReliabilityDiagram
)
from .visualizer import (
    ResultVisualizer,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_uncertainty_heatmap
)
from .reporter import (
    MetricsReporter,
    export_metrics_csv,
    export_metrics_json
)

__all__ = [
    # Metrics
    'ClassificationMetrics',
    'SegmentationMetrics',
    'UncertaintyMetrics',
    'compute_all_metrics',
    
    # Uncertainty evaluation
    'UncertaintyEvaluator',
    'CalibrationCurve',
    'ReliabilityDiagram',
    
    # Visualization
    'ResultVisualizer',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_uncertainty_heatmap',
    
    # Reporting
    'MetricsReporter',
    'export_metrics_csv',
    'export_metrics_json'
]
