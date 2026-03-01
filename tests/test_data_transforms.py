"""
Unit tests for data preprocessing and augmentation transforms.

Tests data loading, preprocessing, and augmentation pipeline.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.transforms import (
    SonarImageNormalize,
    SonarImageDenormalize,
    RandomRotation,
    RandomFlip,
    SonarNoiseInjection,
    RandomBrightnessContrast,
    MetadataEncoder,
    SonarAugmentationPipeline
)


class TestSonarImageNormalize:
    """Test image normalization"""
    
    def test_normalize_shape_preserved(self):
        """Normalization should preserve image shape"""
        image = torch.rand(1, 512, 512)
        normalizer = SonarImageNormalize(mean=0.5, std=0.5)
        
        normalized = normalizer(image)
        
        assert normalized.shape == image.shape
    
    def test_normalize_range(self):
        """Normalized image should be in expected range"""
        image = torch.rand(1, 512, 512)
        normalizer = SonarImageNormalize(mean=0.5, std=0.5)
        
        normalized = normalizer(image)
        
        # With mean=0.5, std=0.5, output should be roughly in [-1, 1]
        assert normalized.min() >= -2.0
        assert normalized.max() <= 2.0
    
    def test_normalize_numpy_input(self):
        """Should handle numpy array input"""
        image = np.random.rand(512, 512).astype(np.float32)
        normalizer = SonarImageNormalize(mean=0.5, std=0.5)
        
        normalized = normalizer(image)
        
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == image.shape


class TestSonarImageDenormalize:
    """Test image denormalization"""
    
    def test_denormalize_inverts_normalize(self):
        """Denormalization should invert normalization"""
        image = torch.rand(1, 512, 512)
        normalizer = SonarImageNormalize(mean=0.5, std=0.5)
        denormalizer = SonarImageDenormalize(mean=0.5, std=0.5)
        
        normalized = normalizer(image)
        denormalized = denormalizer(normalized)
        
        torch.testing.assert_close(denormalized, image, rtol=1e-5, atol=1e-5)
    
    def test_denormalize_range(self):
        """Denormalized image should be in [0, 1] range"""
        image = torch.randn(1, 512, 512)  # Can be outside [0, 1]
        denormalizer = SonarImageDenormalize(mean=0.5, std=0.5)
        
        denormalized = denormalizer(image)
        
        assert denormalized.min() >= 0.0
        assert denormalized.max() <= 1.0


class TestRandomRotation:
    """Test random rotation augmentation"""
    
    def test_rotation_shape_preserved(self):
        """Rotation should preserve image shape"""
        image = torch.rand(1, 512, 512)
        rotator = RandomRotation(degrees=30.0, probability=1.0)
        
        rotated = rotator(image)
        
        assert rotated.shape == image.shape
    
    def test_rotation_probability(self):
        """Rotation should respect probability parameter"""
        image = torch.rand(1, 512, 512)
        rotator_never = RandomRotation(degrees=30.0, probability=0.0)
        
        rotated = rotator_never(image)
        
        # With probability=0, should return unchanged
        torch.testing.assert_close(rotated, image)
    
    def test_rotation_changes_image(self):
        """Rotation should change the image"""
        image = torch.rand(1, 512, 512)
        rotator = RandomRotation(degrees=45.0, probability=1.0)
        
        rotated = rotator(image)
        
        # Should be different from original
        assert not torch.allclose(rotated, image)


class TestRandomFlip:
    """Test random flip augmentation"""
    
    def test_flip_shape_preserved(self):
        """Flip should preserve image shape"""
        image = torch.rand(1, 512, 512)
        flipper = RandomFlip(horizontal_prob=1.0, vertical_prob=1.0)
        
        flipped = flipper(image)
        
        assert flipped.shape == image.shape
    
    def test_horizontal_flip(self):
        """Horizontal flip should flip along width dimension"""
        image = torch.rand(1, 512, 512)
        flipper = RandomFlip(horizontal_prob=1.0, vertical_prob=0.0)
        
        flipped = flipper(image)
        expected = torch.flip(image, dims=[-1])
        
        torch.testing.assert_close(flipped, expected)
    
    def test_vertical_flip(self):
        """Vertical flip should flip along height dimension"""
        image = torch.rand(1, 512, 512)
        flipper = RandomFlip(horizontal_prob=0.0, vertical_prob=1.0)
        
        flipped = flipper(image)
        expected = torch.flip(image, dims=[-2])
        
        torch.testing.assert_close(flipped, expected)


class TestSonarNoiseInjection:
    """Test noise injection augmentation"""
    
    def test_noise_shape_preserved(self):
        """Noise injection should preserve image shape"""
        image = torch.rand(1, 512, 512)
        noise_injector = SonarNoiseInjection(
            speckle_prob=1.0, gaussian_prob=1.0
        )
        
        noisy = noise_injector(image)
        
        assert noisy.shape == image.shape
    
    def test_noise_range_preserved(self):
        """Noisy image should remain in [0, 1] range"""
        image = torch.rand(1, 512, 512)
        noise_injector = SonarNoiseInjection(
            speckle_prob=1.0, speckle_intensity=0.5,
            gaussian_prob=1.0, gaussian_std=0.1
        )
        
        noisy = noise_injector(image)
        
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0
    
    def test_noise_changes_image(self):
        """Noise injection should change the image"""
        torch.manual_seed(42)
        image = torch.rand(1, 512, 512)
        noise_injector = SonarNoiseInjection(
            speckle_prob=1.0, gaussian_prob=1.0
        )
        
        noisy = noise_injector(image)
        
        # Should be different from original
        assert not torch.allclose(noisy, image)


class TestRandomBrightnessContrast:
    """Test brightness/contrast augmentation"""
    
    def test_brightness_contrast_shape_preserved(self):
        """Brightness/contrast adjustment should preserve shape"""
        image = torch.rand(1, 512, 512)
        adjuster = RandomBrightnessContrast(probability=1.0)
        
        adjusted = adjuster(image)
        
        assert adjusted.shape == image.shape
    
    def test_brightness_contrast_range_preserved(self):
        """Adjusted image should remain in [0, 1] range"""
        image = torch.rand(1, 512, 512)
        adjuster = RandomBrightnessContrast(
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            probability=1.0
        )
        
        adjusted = adjuster(image)
        
        assert adjusted.min() >= 0.0
        assert adjusted.max() <= 1.0
    
    def test_brightness_contrast_probability(self):
        """Should respect probability parameter"""
        image = torch.rand(1, 512, 512)
        adjuster = RandomBrightnessContrast(probability=0.0)
        
        adjusted = adjuster(image)
        
        # With probability=0, should return unchanged
        torch.testing.assert_close(adjusted, image)


class TestMetadataEncoder:
    """Test metadata encoding (Requirement 3.1)"""
    
    def test_encoder_output_shape(self):
        """Encoder should output correct shape"""
        config = Config()
        encoder = MetadataEncoder(config)
        
        metadata = {
            'grazing_angle_deg': 45.0,
            'seabed_roughness': 0.5,
            'range_m': 100.0,
            'noise_level': 0.2
        }
        
        encoded = encoder(metadata)
        
        assert encoded.shape == (config.model.metadata_dim,)
        assert isinstance(encoded, torch.Tensor)
    
    def test_encoder_normalization(self):
        """Encoded values should be normalized"""
        config = Config()
        encoder = MetadataEncoder(config)
        
        metadata = {
            'grazing_angle_deg': 90.0,  # Max value
            'seabed_roughness': 1.0,    # Max value
            'range_m': 200.0,           # Max value
            'noise_level': 1.0          # Max value
        }
        
        encoded = encoder(metadata)
        
        # All values should be in [0, 1] range
        assert torch.all(encoded >= 0.0)
        assert torch.all(encoded <= 1.0)
    
    def test_encoder_material_encoding(self):
        """Material should be encoded correctly"""
        config = Config()
        encoder = MetadataEncoder(config)
        
        # Test different materials
        metadata_metal = {'target_material': 'metal'}
        metadata_rock = {'target_material': 'rock'}
        metadata_sand = {'target_material': 'sand'}
        
        encoded_metal = encoder(metadata_metal)
        encoded_rock = encoder(metadata_rock)
        encoded_sand = encoder(metadata_sand)
        
        # Metal should have highest encoding
        assert encoded_metal[-1] > encoded_sand[-1]
        assert encoded_sand[-1] > encoded_rock[-1]
    
    def test_encoder_empty_metadata(self):
        """Should handle empty metadata gracefully"""
        config = Config()
        encoder = MetadataEncoder(config)
        
        encoded = encoder({})
        
        assert encoded.shape == (config.model.metadata_dim,)
        # Should return zeros or defaults
        assert torch.all(torch.isfinite(encoded))


class TestSonarAugmentationPipeline:
    """Test complete augmentation pipeline"""
    
    def test_pipeline_train_phase(self):
        """Training pipeline should apply augmentations"""
        config = Config()
        config.data.use_augmentation = True
        pipeline = SonarAugmentationPipeline(config, phase='train')
        
        sample = {
            'image': torch.rand(1, 512, 512),
            'metadata_dict': {'grazing_angle_deg': 45.0}
        }
        
        augmented = pipeline(sample)
        
        assert 'image' in augmented
        assert 'metadata' in augmented
        assert augmented['image'].shape == (1, 512, 512)
    
    def test_pipeline_val_phase(self):
        """Validation pipeline should not apply augmentations"""
        config = Config()
        pipeline = SonarAugmentationPipeline(config, phase='val')
        
        sample = {
            'image': torch.rand(1, 512, 512),
            'metadata_dict': {}
        }
        
        processed = pipeline(sample)
        
        assert 'image' in processed
        assert 'metadata' in processed
    
    def test_pipeline_test_phase(self):
        """Test pipeline should not apply augmentations"""
        config = Config()
        pipeline = SonarAugmentationPipeline(config, phase='test')
        
        sample = {
            'image': torch.rand(1, 512, 512),
            'metadata_dict': {}
        }
        
        processed = pipeline(sample)
        
        assert 'image' in processed
        assert 'metadata' in processed
    
    def test_pipeline_preserves_sample_structure(self):
        """Pipeline should preserve sample dictionary structure"""
        config = Config()
        pipeline = SonarAugmentationPipeline(config, phase='train')
        
        sample = {
            'image': torch.rand(1, 512, 512),
            'metadata_dict': {'grazing_angle_deg': 45.0},
            'label': 1,
            'image_id': 'test_001'
        }
        
        processed = pipeline(sample)
        
        # Should preserve all original keys
        assert 'label' in processed
        assert 'image_id' in processed
        assert processed['label'] == 1
        assert processed['image_id'] == 'test_001'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
