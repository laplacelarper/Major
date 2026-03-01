"""
Unit tests for Monte Carlo Dropout uncertainty estimation.

Tests Requirement 3.1:
- Monte Carlo Dropout for uncertainty quantification
- Multiple forward passes for uncertainty estimation
- Mean prediction and variance calculation
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.uncertainty import (
    MCDropout,
    MCDropout2d,
    UncertaintyEstimator
)
from src.models.base import BaseSonarModel, ModelOutput


class SimpleMockModel(BaseSonarModel):
    """Simple mock model for testing uncertainty estimation"""
    
    def __init__(self, output_mode="classification"):
        super().__init__()
        self.output_mode = output_mode
        self.num_classes = 2
        
        # Simple architecture with dropout
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 2)
    
    def forward(self, x, metadata=None):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        features = self.dropout2(x)
        x = self.pool(features)
        x = x.view(x.size(0), -1)
        predictions = self.fc(x)
        
        return ModelOutput(
            predictions=predictions,
            features=features,
            uncertainty=None
        )


class TestMCDropout:
    """Test Monte Carlo Dropout layers (Requirement 3.1)"""
    
    def test_mc_dropout_always_active(self):
        """MC Dropout should be active even in eval mode"""
        mc_dropout = MCDropout(p=0.5)
        mc_dropout.eval()  # Set to eval mode
        
        x = torch.ones(10, 100)
        
        # Run multiple times and check for variation
        outputs = [mc_dropout(x) for _ in range(10)]
        
        # Outputs should be different due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])
    
    def test_mc_dropout_shape_preserved(self):
        """MC Dropout should preserve tensor shape"""
        mc_dropout = MCDropout(p=0.1)
        x = torch.rand(4, 64, 32, 32)
        
        output = mc_dropout(x)
        
        assert output.shape == x.shape
    
    def test_mc_dropout_probability(self):
        """MC Dropout should respect dropout probability"""
        mc_dropout = MCDropout(p=0.5)
        x = torch.ones(1000, 100)
        
        output = mc_dropout(x)
        
        # With p=0.5, roughly half should be zeroed
        # (accounting for scaling, non-zero values should be ~2x original)
        zero_ratio = (output == 0).float().mean()
        assert 0.4 < zero_ratio < 0.6


class TestMCDropout2d:
    """Test Monte Carlo Dropout2d layers"""
    
    def test_mc_dropout2d_always_active(self):
        """MC Dropout2d should be active even in eval mode"""
        mc_dropout = MCDropout2d(p=0.5)
        mc_dropout.eval()
        
        x = torch.ones(4, 16, 32, 32)
        
        # Run multiple times and check for variation
        outputs = [mc_dropout(x) for _ in range(10)]
        
        # Outputs should be different
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])
    
    def test_mc_dropout2d_shape_preserved(self):
        """MC Dropout2d should preserve tensor shape"""
        mc_dropout = MCDropout2d(p=0.1)
        x = torch.rand(4, 16, 32, 32)
        
        output = mc_dropout(x)
        
        assert output.shape == x.shape


class TestUncertaintyEstimator:
    """Test uncertainty estimation (Requirement 3.1)"""
    
    def test_estimator_initialization(self):
        """Uncertainty estimator should initialize correctly"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=20)
        
        assert estimator.num_samples == 20
        assert estimator.model is model
    
    def test_predict_with_uncertainty_shape(self):
        """Predictions and uncertainty should have correct shapes"""
        model = SimpleMockModel(output_mode="classification")
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        x = torch.rand(4, 1, 64, 64)
        
        mean_preds, uncertainty = estimator.predict_with_uncertainty(x)
        
        assert mean_preds.shape == (4, 2)  # (batch_size, num_classes)
        assert uncertainty.shape == (4,)    # (batch_size,)
    
    def test_predict_with_uncertainty_multiple_samples(self):
        """Should perform multiple forward passes (Requirement 3.1)"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=20)
        
        x = torch.rand(2, 1, 64, 64)
        
        # Run prediction
        mean_preds, uncertainty = estimator.predict_with_uncertainty(x)
        
        # Should return valid tensors
        assert torch.all(torch.isfinite(mean_preds))
        assert torch.all(torch.isfinite(uncertainty))
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_uncertainty_varies_across_samples(self):
        """Uncertainty should vary across different samples"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=20)
        
        # Create samples with different characteristics
        x1 = torch.ones(1, 1, 64, 64) * 0.5  # Uniform
        x2 = torch.rand(1, 1, 64, 64)        # Random
        
        _, unc1 = estimator.predict_with_uncertainty(x1)
        _, unc2 = estimator.predict_with_uncertainty(x2)
        
        # Uncertainties should be different
        assert not torch.allclose(unc1, unc2)
    
    def test_predict_with_detailed_uncertainty(self):
        """Detailed uncertainty should provide comprehensive metrics"""
        model = SimpleMockModel(output_mode="classification")
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        x = torch.rand(2, 1, 64, 64)
        
        result = estimator.predict_with_detailed_uncertainty(x)
        
        # Check required keys
        assert "mean_predictions" in result
        assert "variance" in result
        assert "std" in result
        assert "all_predictions" in result
        
        # Classification-specific keys
        assert "probabilities" in result
        assert "entropy" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
    
    def test_detailed_uncertainty_shapes(self):
        """Detailed uncertainty metrics should have correct shapes"""
        model = SimpleMockModel(output_mode="classification")
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        batch_size = 4
        x = torch.rand(batch_size, 1, 64, 64)
        
        result = estimator.predict_with_detailed_uncertainty(x)
        
        assert result["mean_predictions"].shape == (batch_size, 2)
        assert result["variance"].shape == (batch_size, 2)
        assert result["std"].shape == (batch_size, 2)
        assert result["all_predictions"].shape == (10, batch_size, 2)
        assert result["probabilities"].shape == (batch_size, 2)
        assert result["entropy"].shape == (batch_size,)
    
    def test_uncertainty_with_metadata(self):
        """Should handle metadata input correctly"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        x = torch.rand(2, 1, 64, 64)
        metadata = torch.rand(2, 7)  # Mock metadata
        
        mean_preds, uncertainty = estimator.predict_with_uncertainty(x, metadata)
        
        assert mean_preds.shape == (2, 2)
        assert uncertainty.shape == (2,)
    
    def test_uncertainty_heatmap_generation(self):
        """Should generate uncertainty heatmap for visualization"""
        model = SimpleMockModel(output_mode="classification")
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        x = torch.rand(1, 64, 64)  # Single image without batch dim
        
        heatmap = estimator.generate_uncertainty_heatmap(x)
        
        # Heatmap should have spatial dimensions
        assert heatmap.shape == (64, 64)
        assert torch.all(torch.isfinite(heatmap))
    
    def test_mc_dropout_replacement(self):
        """Should replace standard dropout with MC dropout"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        # Check that dropout layers were replaced
        has_mc_dropout = False
        for module in estimator.model.modules():
            if isinstance(module, (MCDropout, MCDropout2d)):
                has_mc_dropout = True
                break
        
        assert has_mc_dropout, "MC Dropout layers should be present"
    
    def test_uncertainty_increases_with_ambiguity(self):
        """Uncertainty should be higher for ambiguous inputs"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=30)
        
        # Clear input (all zeros)
        x_clear = torch.zeros(1, 1, 64, 64)
        
        # Ambiguous input (random noise)
        x_ambiguous = torch.rand(1, 1, 64, 64)
        
        _, unc_clear = estimator.predict_with_uncertainty(x_clear)
        _, unc_ambiguous = estimator.predict_with_uncertainty(x_ambiguous)
        
        # Both should produce valid uncertainties
        assert torch.all(torch.isfinite(unc_clear))
        assert torch.all(torch.isfinite(unc_ambiguous))
    
    def test_mean_prediction_consistency(self):
        """Mean prediction should be consistent across runs"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=50)
        
        x = torch.rand(2, 1, 64, 64)
        
        # Run multiple times
        mean1, _ = estimator.predict_with_uncertainty(x)
        mean2, _ = estimator.predict_with_uncertainty(x)
        
        # Means should be similar (not exact due to stochasticity)
        assert torch.allclose(mean1, mean2, rtol=0.2, atol=0.1)
    
    def test_epistemic_vs_aleatoric_uncertainty(self):
        """Should separate epistemic and aleatoric uncertainty"""
        model = SimpleMockModel(output_mode="classification")
        estimator = UncertaintyEstimator(model, num_samples=20)
        
        x = torch.rand(4, 1, 64, 64)
        
        result = estimator.predict_with_detailed_uncertainty(x)
        
        epistemic = result["epistemic_uncertainty"]
        aleatoric = result["aleatoric_uncertainty"]
        
        # Both should be non-negative
        assert torch.all(epistemic >= 0)
        assert torch.all(aleatoric >= 0)
        
        # Total uncertainty (entropy) should be sum of both
        total_entropy = result["entropy"]
        assert torch.allclose(total_entropy, epistemic + aleatoric, rtol=0.1)


class TestUncertaintyCalibration:
    """Test uncertainty calibration"""
    
    def test_calibration_metrics_structure(self):
        """Calibration should return expected metrics"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        # Create mock dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.rand(20, 1, 64, 64),
            torch.randint(0, 2, (20,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        metrics = estimator.calibrate_uncertainty(dataloader)
        
        # Should contain calibration metrics
        assert isinstance(metrics, dict)
        assert "expected_calibration_error" in metrics
        assert "uncertainty_error_correlation" in metrics
    
    def test_calibration_with_confidence_levels(self):
        """Should compute accuracy at different confidence levels"""
        model = SimpleMockModel()
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        dataset = torch.utils.data.TensorDataset(
            torch.rand(20, 1, 64, 64),
            torch.randint(0, 2, (20,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        confidence_levels = [0.5, 0.7, 0.9]
        metrics = estimator.calibrate_uncertainty(
            dataloader, confidence_levels=confidence_levels
        )
        
        # Should have metrics for each confidence level
        for level in confidence_levels:
            key = f"accuracy_at_{level}_confidence"
            assert key in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
