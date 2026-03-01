"""
Integration tests for the complete training and evaluation pipeline.

Tests Requirements 7.4, 6.5:
- Full pipeline integration from data to evaluation
- Performance benchmarking for synthetic generation
- Model training and evaluation workflow validation
- Error handling without silent failures
"""

import pytest
import torch
import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.physics.core import PhysicsEngine
from src.models.factory import ModelFactory
from src.models.uncertainty import UncertaintyEstimator
from src.evaluation.metrics import compute_all_metrics
from src.training.utils import EarlyStopping


class TestSyntheticDataGeneration:
    """Test synthetic data generation pipeline (Requirement 7.4)"""
    
    def test_single_image_generation(self):
        """Should generate a single synthetic image successfully"""
        engine = PhysicsEngine(image_size=(512, 512))
        
        image, label, metadata = engine.generate_single_image(random_seed=42)
        
        assert image.shape == (512, 512)
        assert label in [0, 1]
        assert metadata is not None
        assert np.all(image >= 0.0) and np.all(image <= 1.0)
    
    def test_batch_generation_performance(self):
        """Benchmark synthetic image generation time (Requirement 7.4)"""
        engine = PhysicsEngine(image_size=(512, 512))
        
        num_samples = 10
        physics_config = {
            'cosine_exponent_range': (2.0, 8.0),
            'texture_roughness_range': (0.2, 0.8),
            'noise_level_range': (0.1, 0.4)
        }
        
        start_time = time.time()
        images, labels, metadata_list = engine.generate_dataset(
            num_samples=num_samples,
            physics_config=physics_config,
            save_to_disk=False,
            random_seed=42
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        time_per_image = generation_time / num_samples
        
        # Verify outputs
        assert images.shape == (num_samples, 512, 512)
        assert labels.shape == (num_samples,)
        assert len(metadata_list) == num_samples
        
        # Performance check: should generate reasonably fast
        # Allow up to 1 second per image (very generous for CI)
        assert time_per_image < 1.0, f"Generation too slow: {time_per_image:.3f}s per image"
        
        print(f"\nGeneration performance: {time_per_image:.3f}s per image")
    
    def test_physics_validation(self):
        """Validate physics calculations are correct"""
        engine = PhysicsEngine(image_size=(512, 512))
        
        validation_results = engine.validate_physics_calculations()
        
        # All physics components should pass validation
        assert validation_results.get('backscatter', False)
        assert validation_results.get('range_attenuation', False)
        assert validation_results.get('acoustic_shadows', False)
        assert validation_results.get('speckle_noise', False)
        assert validation_results.get('seabed_texture', False)
    
    def test_parameter_ranges(self):
        """Verify parameter ranges are valid"""
        engine = PhysicsEngine(image_size=(512, 512))
        
        param_ranges = engine.get_physics_parameter_ranges()
        
        # Check that ranges are sensible
        assert param_ranges['cosine_exponent'][0] < param_ranges['cosine_exponent'][1]
        assert param_ranges['noise_level'][0] >= 0.0
        assert param_ranges['noise_level'][1] <= 1.0
        assert param_ranges['grazing_angle_deg'][1] == 90.0


class TestModelTrainingPipeline:
    """Test model training pipeline integration"""
    
    def test_model_creation(self):
        """Should create models successfully"""
        config = Config()
        
        for model_type in ['unet', 'resnet18', 'efficientnet-b0']:
            model = ModelFactory.create_model(model_type, config)
            assert model is not None
            assert hasattr(model, 'forward')
    
    def test_model_forward_pass(self):
        """Model should perform forward pass correctly"""
        config = Config()
        model = ModelFactory.create_model('unet', config)
        
        # Create dummy input
        x = torch.rand(2, 1, 512, 512)
        metadata = torch.rand(2, config.model.metadata_dim)
        
        # Forward pass
        output = model(x, metadata)
        
        assert output.predictions is not None
        assert output.predictions.shape[0] == 2  # Batch size
    
    def test_training_step(self):
        """Should perform a single training step"""
        config = Config()
        model = ModelFactory.create_model('resnet18', config)
        
        # Create dummy data
        x = torch.rand(4, 1, 512, 512)
        targets = torch.randint(0, 2, (4,))
        metadata = torch.rand(4, config.model.metadata_dim)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x, metadata)
        loss = criterion(output.predictions, targets)
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss)
    
    def test_early_stopping(self):
        """Early stopping should work correctly"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Simulate improving validation loss
        assert not early_stopping(0.5)
        assert not early_stopping(0.4)
        assert not early_stopping(0.3)
        
        # Simulate plateau
        assert not early_stopping(0.3)
        assert not early_stopping(0.3)
        assert not early_stopping(0.3)
        
        # Should trigger early stopping after patience
        assert early_stopping(0.3)


class TestEvaluationPipeline:
    """Test evaluation pipeline integration"""
    
    def test_metrics_computation(self):
        """Should compute all metrics correctly"""
        # Create dummy predictions and labels
        predictions = np.random.randint(0, 2, 100)
        labels = np.random.randint(0, 2, 100)
        uncertainties = np.random.rand(100) * 0.5
        confidences = 1 - uncertainties
        
        metrics = compute_all_metrics(
            predictions=predictions,
            labels=labels,
            uncertainties=uncertainties,
            confidences=confidences,
            task_type='classification'
        )
        
        # Check required metrics exist
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'expected_calibration_error' in metrics
        
        # All metrics should be finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite"
    
    def test_uncertainty_evaluation_integration(self):
        """Should integrate uncertainty estimation with evaluation"""
        config = Config()
        model = ModelFactory.create_model('resnet18', config)
        
        # Create uncertainty estimator
        estimator = UncertaintyEstimator(model, num_samples=10)
        
        # Create dummy data
        x = torch.rand(4, 1, 512, 512)
        
        # Get predictions with uncertainty
        mean_preds, uncertainty = estimator.predict_with_uncertainty(x)
        
        assert mean_preds.shape == (4, 2)
        assert uncertainty.shape == (4,)
        assert torch.all(uncertainty >= 0)


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline (Requirement 6.5)"""
    
    def test_full_pipeline_small_scale(self):
        """Test complete pipeline with small dataset"""
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Step 1: Generate synthetic data
            engine = PhysicsEngine(image_size=(128, 128), output_dir=tmpdir / "data")
            
            physics_config = {
                'cosine_exponent_range': (2.0, 8.0),
                'texture_roughness_range': (0.2, 0.8),
                'noise_level_range': (0.1, 0.4)
            }
            
            images, labels, metadata_list = engine.generate_dataset(
                num_samples=20,
                physics_config=physics_config,
                save_to_disk=False,
                random_seed=42
            )
            
            assert images.shape == (20, 128, 128)
            
            # Step 2: Create model
            config = Config()
            config.data.image_size = (128, 128)
            model = ModelFactory.create_model('resnet18', config)
            
            # Step 3: Prepare data
            X = torch.from_numpy(images).float().unsqueeze(1)  # Add channel dim
            y = torch.from_numpy(labels).long()
            
            # Split into train/val
            train_X, val_X = X[:15], X[15:]
            train_y, val_y = y[:15], y[15:]
            
            # Step 4: Train for a few steps
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(2):
                optimizer.zero_grad()
                output = model(train_X, None)
                loss = criterion(output.predictions, train_y)
                loss.backward()
                optimizer.step()
            
            # Step 5: Evaluate
            model.eval()
            with torch.no_grad():
                output = model(val_X, None)
                val_preds = torch.argmax(output.predictions, dim=1)
            
            # Step 6: Compute metrics
            metrics = compute_all_metrics(
                predictions=val_preds.numpy(),
                labels=val_y.numpy(),
                uncertainties=np.random.rand(len(val_y)) * 0.3,
                confidences=np.random.rand(len(val_y)) * 0.7 + 0.3,
                task_type='classification'
            )
            
            # Verify metrics are computed
            assert 'accuracy' in metrics
            assert 'f1_score' in metrics
            
            print(f"\nEnd-to-end test metrics: Accuracy={metrics['accuracy']:.3f}")
    
    def test_error_handling_invalid_input(self):
        """Should handle invalid inputs gracefully (Requirement 6.5)"""
        config = Config()
        model = ModelFactory.create_model('resnet18', config)
        
        # Test with wrong input shape
        with pytest.raises((RuntimeError, ValueError)):
            x = torch.rand(2, 3, 512, 512)  # Wrong number of channels
            model(x, None)
    
    def test_error_handling_missing_data(self):
        """Should handle missing data gracefully (Requirement 6.5)"""
        engine = PhysicsEngine(image_size=(512, 512))
        
        # Test with invalid physics parameters
        try:
            invalid_params = {
                'cosine_exponent': -1.0,  # Invalid negative value
                'noise_level': 2.0  # Invalid > 1.0
            }
            image, label, metadata = engine.generate_single_image(
                physics_params=invalid_params
            )
            # Should either handle gracefully or use defaults
            assert image.shape == (512, 512)
        except Exception as e:
            # If it raises an exception, it should be informative
            assert len(str(e)) > 0


class TestPerformanceBenchmarks:
    """Performance benchmarking tests (Requirement 7.4)"""
    
    def test_inference_speed(self):
        """Benchmark model inference speed"""
        config = Config()
        model = ModelFactory.create_model('efficientnet-b0', config)
        model.eval()
        
        # Warm-up
        x = torch.rand(1, 1, 512, 512)
        with torch.no_grad():
            _ = model(x, None)
        
        # Benchmark
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x, None)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # Should be reasonably fast (< 1 second per image on CPU)
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s per image"
        
        print(f"\nInference performance: {avg_time:.3f}s per image")
    
    def test_uncertainty_estimation_speed(self):
        """Benchmark uncertainty estimation speed"""
        config = Config()
        model = ModelFactory.create_model('resnet18', config)
        estimator = UncertaintyEstimator(model, num_samples=20)
        
        x = torch.rand(1, 1, 512, 512)
        
        start_time = time.time()
        mean_preds, uncertainty = estimator.predict_with_uncertainty(x)
        end_time = time.time()
        
        estimation_time = end_time - start_time
        
        # Should complete in reasonable time (< 20 seconds for 20 samples)
        assert estimation_time < 20.0, f"Uncertainty estimation too slow: {estimation_time:.3f}s"
        
        print(f"\nUncertainty estimation: {estimation_time:.3f}s for 20 samples")
    
    def test_batch_processing_efficiency(self):
        """Batch processing should be more efficient than sequential"""
        config = Config()
        model = ModelFactory.create_model('resnet18', config)
        model.eval()
        
        # Sequential processing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(4):
                x = torch.rand(1, 1, 512, 512)
                _ = model(x, None)
        sequential_time = time.time() - start_time
        
        # Batch processing
        start_time = time.time()
        with torch.no_grad():
            x_batch = torch.rand(4, 1, 512, 512)
            _ = model(x_batch, None)
        batch_time = time.time() - start_time
        
        # Batch should be faster than sequential
        assert batch_time < sequential_time
        
        speedup = sequential_time / batch_time
        print(f"\nBatch processing speedup: {speedup:.2f}x")


class TestReproducibility:
    """Test reproducibility of results"""
    
    def test_deterministic_data_generation(self):
        """Same seed should produce same data"""
        engine1 = PhysicsEngine(image_size=(128, 128))
        engine2 = PhysicsEngine(image_size=(128, 128))
        
        physics_config = {
            'cosine_exponent_range': (2.0, 8.0),
            'texture_roughness_range': (0.2, 0.8)
        }
        
        images1, labels1, _ = engine1.generate_dataset(
            num_samples=5,
            physics_config=physics_config,
            save_to_disk=False,
            random_seed=42
        )
        
        images2, labels2, _ = engine2.generate_dataset(
            num_samples=5,
            physics_config=physics_config,
            save_to_disk=False,
            random_seed=42
        )
        
        np.testing.assert_array_equal(images1, images2)
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_deterministic_model_initialization(self):
        """Same seed should produce same model initialization"""
        config = Config()
        config.random_seed = 42
        
        torch.manual_seed(42)
        model1 = ModelFactory.create_model('resnet18', config)
        
        torch.manual_seed(42)
        model2 = ModelFactory.create_model('resnet18', config)
        
        # Check that parameters are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
