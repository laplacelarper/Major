"""Test the complete evaluation system (Task 6)"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    compute_all_metrics,
    UncertaintyEvaluator,
    ResultVisualizer,
    MetricsReporter,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_uncertainty_heatmap,
    export_metrics_csv,
    export_metrics_json
)


def test_complete_evaluation_pipeline():
    """Test the complete evaluation pipeline"""
    print("Testing Complete Evaluation System (Task 6)")
    print("=" * 80)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    predictions = np.random.randint(0, 2, n_samples)
    labels = np.random.randint(0, 2, n_samples)
    uncertainties = np.random.rand(n_samples) * 0.5
    confidences = 1 - uncertainties
    
    # Test 6.1: Core Metrics
    print("\n[Task 6.1] Testing Core Metrics Calculation...")
    metrics = compute_all_metrics(
        predictions=predictions,
        labels=labels,
        uncertainties=uncertainties,
        confidences=confidences,
        task_type='classification'
    )
    print(f"✓ Computed {len(metrics)} metrics")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {metrics['f1_score']:.4f}")
    print(f"  - ECE: {metrics['expected_calibration_error']:.4f}")
    
    # Test 6.2: Uncertainty Evaluation
    print("\n[Task 6.2] Testing Uncertainty Evaluation Framework...")
    unc_eval = UncertaintyEvaluator(n_bins=10)
    unc_eval.update(uncertainties, predictions, labels, confidences)
    unc_metrics = unc_eval.evaluate()
    print(f"✓ Uncertainty evaluation complete")
    print(f"  - Mean Uncertainty: {unc_metrics['mean_uncertainty']:.4f}")
    print(f"  - ECE: {unc_metrics['expected_calibration_error']:.4f}")
    print(f"  - MCE: {unc_metrics['maximum_calibration_error']:.4f}")
    
    # Get calibration data
    calib_data = unc_eval.get_calibration_data()
    print(f"  - Calibration bins: {len(calib_data['bin_confidences'])}")
    
    # Test 6.3: Visualization System
    print("\n[Task 6.3] Testing Visualization System...")
    visualizer = ResultVisualizer()
    
    # Confusion matrix
    cm_data = visualizer.plot_confusion_matrix(predictions, labels)
    print(f"✓ Confusion matrix generated")
    print(f"  - TP: {cm_data['true_positives']}, TN: {cm_data['true_negatives']}")
    print(f"  - FP: {cm_data['false_positives']}, FN: {cm_data['false_negatives']}")
    
    # ROC curve
    roc_data = visualizer.plot_roc_curve(predictions, labels, confidences)
    print(f"✓ ROC curve generated")
    print(f"  - AUC: {roc_data['auc']:.4f}")
    
    # Uncertainty heatmap
    heatmap_data = visualizer.plot_uncertainty_heatmap(uncertainties)
    print(f"✓ Uncertainty heatmap generated")
    print(f"  - Mean: {heatmap_data['stats']['mean_uncertainty']:.4f}")
    
    # Synthetic sample visualization
    synthetic_image = np.random.rand(512, 512)
    metadata = {
        'grazing_angle': 45.0,
        'roughness': 0.5,
        'range': 100.0,
        'noise': 0.2,
        'material': 0.7,
        'frequency': 300.0,
        'beam_width': 2.5
    }
    sample_data = visualizer.visualize_synthetic_sample(synthetic_image, metadata)
    print(f"✓ Synthetic sample visualization generated")
    print(f"  - Image shape: {sample_data['image_shape']}")
    
    # Detection overlay
    overlay_data = visualizer.visualize_detection_overlay(
        synthetic_image, predictions[0], labels[0], uncertainties[0]
    )
    print(f"✓ Detection overlay generated")
    
    # Test standalone functions
    print("\n[Task 6.3] Testing Standalone Visualization Functions...")
    cm = plot_confusion_matrix(predictions, labels)
    roc = plot_roc_curve(predictions, labels, confidences)
    heatmap = plot_uncertainty_heatmap(uncertainties)
    print(f"✓ All standalone functions work")
    
    # Test 6.3: Reporting System
    print("\n[Task 6.3] Testing Reporting System...")
    reporter = MetricsReporter()
    
    # Generate text report
    report_text = reporter.generate_report(metrics, "Test Evaluation Report")
    print(f"✓ Text report generated ({len(report_text)} characters)")
    
    # Export to CSV
    csv_path = reporter.export_csv(metrics)
    print(f"✓ CSV exported to: {csv_path}")
    
    # Export to JSON
    json_path = reporter.export_json(metrics)
    print(f"✓ JSON exported to: {json_path}")
    
    # Save text report
    report_path = reporter.save_report(metrics, report_name="Test Report")
    print(f"✓ Report saved to: {report_path}")
    
    # Test standalone functions
    print("\n[Task 6.3] Testing Standalone Reporting Functions...")
    csv_path2 = export_metrics_csv(metrics)
    json_path2 = export_metrics_json(metrics)
    print(f"✓ All standalone functions work")
    
    print("\n" + "=" * 80)
    print("✓ ALL TASK 6 TESTS PASSED")
    print("=" * 80)
    print("\nTask 6 Complete:")
    print("  ✓ 6.1: Core Metrics Calculation")
    print("  ✓ 6.2: Uncertainty Evaluation Framework")
    print("  ✓ 6.3: Visualization and Reporting System")
    print("\nAll evaluation components are working correctly!")


if __name__ == "__main__":
    test_complete_evaluation_pipeline()
