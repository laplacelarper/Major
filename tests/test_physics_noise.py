"""
Unit tests for noise and texture generation module.

Tests Requirement 1.5:
- Multiplicative speckle noise using Rayleigh/Gamma distributions
- Seabed texture generation with procedural noise
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.noise import (
    generate_speckle_noise,
    generate_seabed_texture,
    randomize_parameters,
    apply_multiplicative_noise,
    generate_combined_texture_noise
)


class TestSpeckleNoise:
    """Test speckle noise generation (Requirement 1.5)"""
    
    def test_rayleigh_noise_shape(self):
        """Rayleigh noise should have correct shape"""
        shape = (100, 100)
        noise = generate_speckle_noise(shape, noise_type='rayleigh')
        
        assert noise.shape == shape
    
    def test_gamma_noise_shape(self):
        """Gamma noise should have correct shape"""
        shape = (100, 100)
        noise = generate_speckle_noise(shape, noise_type='gamma')
        
        assert noise.shape == shape
    
    def test_rayleigh_noise_multiplicative(self):
        """Rayleigh noise should be multiplicative (mean around 1.0)"""
        shape = (1000, 1000)
        noise = generate_speckle_noise(shape, noise_type='rayleigh', noise_level=0.2)
        
        # Mean should be close to 1.0 for multiplicative noise
        assert 0.8 < np.mean(noise) < 1.2
    
    def test_gamma_noise_multiplicative(self):
        """Gamma noise should be multiplicative (mean around 1.0)"""
        shape = (1000, 1000)
        noise = generate_speckle_noise(shape, noise_type='gamma', noise_level=0.2)
        
        # Mean should be close to 1.0 for multiplicative noise
        assert 0.8 < np.mean(noise) < 1.2
    
    def test_noise_level_effect(self):
        """Higher noise level should create more variation"""
        shape = (1000, 1000)
        
        noise_low = generate_speckle_noise(shape, noise_type='rayleigh', noise_level=0.1)
        noise_high = generate_speckle_noise(shape, noise_type='rayleigh', noise_level=0.5)
        
        # Higher noise level should have higher standard deviation
        assert np.std(noise_high) > np.std(noise_low)
    
    def test_noise_positive_values(self):
        """Noise should have positive values only"""
        shape = (100, 100)
        noise = generate_speckle_noise(shape, noise_type='rayleigh')
        
        assert np.all(noise > 0)
    
    def test_noise_reproducibility(self):
        """Same seed should produce same noise"""
        shape = (100, 100)
        
        noise1 = generate_speckle_noise(shape, noise_type='rayleigh', random_seed=42)
        noise2 = generate_speckle_noise(shape, noise_type='rayleigh', random_seed=42)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_invalid_noise_type(self):
        """Invalid noise type should raise error"""
        shape = (100, 100)
        
        with pytest.raises(ValueError):
            generate_speckle_noise(shape, noise_type='invalid')


class TestSeabedTexture:
    """Test seabed texture generation (Requirement 1.5)"""
    
    def test_texture_shape(self):
        """Texture should have correct shape"""
        shape = (100, 100)
        texture = generate_seabed_texture(shape)
        
        assert texture.shape == shape
    
    def test_texture_range(self):
        """Texture should be in [0, 1] range"""
        shape = (100, 100)
        texture = generate_seabed_texture(shape)
        
        assert np.all(texture >= 0.0)
        assert np.all(texture <= 1.0)
    
    def test_texture_roughness_effect(self):
        """Higher roughness should create more variation"""
        shape = (500, 500)
        
        texture_smooth = generate_seabed_texture(shape, roughness=0.1)
        texture_rough = generate_seabed_texture(shape, roughness=0.9)
        
        # Higher roughness should have higher standard deviation
        assert np.std(texture_rough) > np.std(texture_smooth)
    
    def test_texture_scale_effect(self):
        """Different scales should create different texture patterns"""
        shape = (100, 100)
        
        texture_fine = generate_seabed_texture(shape, texture_scale=5.0, random_seed=42)
        texture_coarse = generate_seabed_texture(shape, texture_scale=20.0, random_seed=42)
        
        # Different scales should produce different patterns
        assert not np.allclose(texture_fine, texture_coarse)
    
    def test_texture_reproducibility(self):
        """Same seed should produce same texture"""
        shape = (100, 100)
        
        texture1 = generate_seabed_texture(shape, random_seed=42)
        texture2 = generate_seabed_texture(shape, random_seed=42)
        
        np.testing.assert_array_equal(texture1, texture2)
    
    def test_texture_spatial_coherence(self):
        """Texture should have spatial coherence (not pure random)"""
        shape = (100, 100)
        texture = generate_seabed_texture(shape, texture_scale=10.0)
        
        # Check that neighboring pixels are correlated
        # Calculate correlation between adjacent pixels
        horizontal_diff = np.abs(texture[:, 1:] - texture[:, :-1])
        
        # Mean difference should be small (indicating coherence)
        assert np.mean(horizontal_diff) < 0.2


class TestParameterRandomization:
    """Test parameter randomization for domain variation"""
    
    def test_randomize_with_ranges(self):
        """Parameters should be randomized within specified ranges"""
        base_params = {
            'param1': 5.0,
            'param2': 10.0
        }
        
        variation_ranges = {
            'param1': [2.0, 8.0],
            'param2': [5.0, 15.0]
        }
        
        randomized = randomize_parameters(base_params, variation_ranges, random_seed=42)
        
        assert 2.0 <= randomized['param1'] <= 8.0
        assert 5.0 <= randomized['param2'] <= 15.0
    
    def test_randomize_with_percentage(self):
        """Parameters should be randomized with percentage variation"""
        base_params = {
            'param1': 10.0
        }
        
        variation_ranges = {
            'param1': 0.2  # ±20%
        }
        
        randomized = randomize_parameters(base_params, variation_ranges, random_seed=42)
        
        # Should be within ±20% of base value
        assert 8.0 <= randomized['param1'] <= 12.0
    
    def test_randomize_reproducibility(self):
        """Same seed should produce same randomization"""
        base_params = {'param1': 5.0}
        variation_ranges = {'param1': [2.0, 8.0]}
        
        rand1 = randomize_parameters(base_params, variation_ranges, random_seed=42)
        rand2 = randomize_parameters(base_params, variation_ranges, random_seed=42)
        
        assert rand1['param1'] == rand2['param1']
    
    def test_randomize_preserves_unspecified_params(self):
        """Parameters not in variation_ranges should remain unchanged"""
        base_params = {
            'param1': 5.0,
            'param2': 10.0
        }
        
        variation_ranges = {
            'param1': [2.0, 8.0]
        }
        
        randomized = randomize_parameters(base_params, variation_ranges)
        
        assert randomized['param2'] == 10.0


class TestMultiplicativeNoiseApplication:
    """Test applying multiplicative noise to images"""
    
    def test_apply_noise_shape(self):
        """Output should have same shape as input"""
        image = np.random.rand(100, 100)
        noise = np.ones((100, 100)) * 1.2
        
        result = apply_multiplicative_noise(image, noise)
        
        assert result.shape == image.shape
    
    def test_apply_noise_range(self):
        """Output should be in [0, 1] range"""
        image = np.random.rand(100, 100)
        noise = np.random.rand(100, 100) * 0.5 + 0.75  # [0.75, 1.25]
        
        result = apply_multiplicative_noise(image, noise)
        
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_noise_strength_effect(self):
        """Noise strength should control the amount of noise applied"""
        image = np.ones((100, 100)) * 0.5
        noise = np.ones((100, 100)) * 1.5
        
        result_weak = apply_multiplicative_noise(image, noise, noise_strength=0.1)
        result_strong = apply_multiplicative_noise(image, noise, noise_strength=1.0)
        
        # Stronger noise should deviate more from original
        assert np.abs(result_strong - image).mean() > np.abs(result_weak - image).mean()
    
    def test_zero_noise_strength(self):
        """Zero noise strength should return original image"""
        image = np.random.rand(100, 100)
        noise = np.random.rand(100, 100) * 2.0
        
        result = apply_multiplicative_noise(image, noise, noise_strength=0.0)
        
        np.testing.assert_allclose(result, image, rtol=1e-5)


class TestCombinedTextureNoise:
    """Test combined texture and noise generation"""
    
    def test_combined_generation_shapes(self):
        """Both outputs should have correct shape"""
        shape = (100, 100)
        
        texture, noise = generate_combined_texture_noise(
            shape,
            texture_params={'roughness': 0.5, 'texture_scale': 10.0},
            noise_params={'noise_type': 'rayleigh', 'noise_level': 0.2}
        )
        
        assert texture.shape == shape
        assert noise.shape == shape
    
    def test_combined_generation_ranges(self):
        """Outputs should be in valid ranges"""
        shape = (100, 100)
        
        texture, noise = generate_combined_texture_noise(
            shape,
            texture_params={'roughness': 0.5, 'texture_scale': 10.0},
            noise_params={'noise_type': 'rayleigh', 'noise_level': 0.2}
        )
        
        assert np.all(texture >= 0.0) and np.all(texture <= 1.0)
        assert np.all(noise > 0.0)
    
    def test_combined_generation_independence(self):
        """Texture and noise should be independent (different patterns)"""
        shape = (100, 100)
        
        texture, noise = generate_combined_texture_noise(
            shape,
            texture_params={'roughness': 0.5, 'texture_scale': 10.0},
            noise_params={'noise_type': 'rayleigh', 'noise_level': 0.2},
            random_seed=42
        )
        
        # Normalize both to [0, 1] for comparison
        texture_norm = (texture - texture.min()) / (texture.max() - texture.min())
        noise_norm = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Should not be highly correlated
        correlation = np.corrcoef(texture_norm.flatten(), noise_norm.flatten())[0, 1]
        assert abs(correlation) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
