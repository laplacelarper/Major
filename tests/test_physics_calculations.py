"""
Unit tests for physics calculations module.

Tests Requirements 1.2, 1.3, 1.4:
- Backscatter intensity proportional to cosⁿ(grazing_angle)
- Acoustic shadows using geometric approximations
- Range-based attenuation following 1/R²
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.calculations import (
    calculate_backscatter_intensity,
    calculate_range_attenuation,
    generate_acoustic_shadows,
    create_range_map,
    create_grazing_angle_map
)


class TestBackscatterIntensity:
    """Test backscatter intensity calculations (Requirement 1.2)"""
    
    def test_backscatter_at_normal_incidence(self):
        """At 0 degrees (normal incidence), backscatter should be maximum"""
        angles = np.array([[0.0, 45.0, 90.0]])
        intensity = calculate_backscatter_intensity(angles, cosine_exponent=2.0)
        
        # At 0 degrees, cos(0) = 1, so intensity should be highest
        assert intensity[0, 0] > intensity[0, 1]
        assert intensity[0, 1] > intensity[0, 2]
    
    def test_backscatter_at_grazing_angle(self):
        """At 90 degrees (grazing), backscatter should be minimum"""
        angles = np.array([[90.0]])
        intensity = calculate_backscatter_intensity(angles, cosine_exponent=4.0)
        
        # At 90 degrees, cos(90) = 0, so intensity should be very low
        assert intensity[0, 0] < 0.1
    
    def test_backscatter_cosine_law(self):
        """Verify cosⁿ(θ) relationship"""
        angles = np.array([[0.0, 30.0, 60.0]])
        exponent = 2.0
        base = 1.0
        
        intensity = calculate_backscatter_intensity(
            angles, cosine_exponent=exponent, base_intensity=base
        )
        
        # Manually calculate expected values
        expected = base * np.power(np.cos(np.deg2rad(angles)), exponent)
        
        np.testing.assert_allclose(intensity, expected, rtol=1e-5)
    
    def test_backscatter_output_range(self):
        """Backscatter intensity should be in [0, 1] range"""
        angles = np.random.uniform(0, 90, size=(10, 10))
        intensity = calculate_backscatter_intensity(angles)
        
        assert np.all(intensity >= 0.0)
        assert np.all(intensity <= 1.0)
    
    def test_backscatter_exponent_effect(self):
        """Higher exponent should create steeper falloff"""
        angles = np.array([[45.0]])
        
        intensity_low = calculate_backscatter_intensity(angles, cosine_exponent=2.0)
        intensity_high = calculate_backscatter_intensity(angles, cosine_exponent=8.0)
        
        # Higher exponent should give lower intensity at 45 degrees
        assert intensity_high[0, 0] < intensity_low[0, 0]


class TestRangeAttenuation:
    """Test range-based attenuation calculations (Requirement 1.4)"""
    
    def test_attenuation_inverse_square_law(self):
        """Verify 1/R² relationship"""
        ranges = np.array([[10.0, 20.0, 40.0]])
        attenuation = calculate_range_attenuation(
            ranges, attenuation_coefficient=2.0, reference_range=10.0
        )
        
        # At reference range, attenuation should be 1.0
        assert np.isclose(attenuation[0, 0], 1.0, rtol=1e-5)
        
        # At 2x range, attenuation should be 1/4 (inverse square)
        assert np.isclose(attenuation[0, 1], 0.25, rtol=1e-5)
        
        # At 4x range, attenuation should be 1/16
        assert np.isclose(attenuation[0, 2], 0.0625, rtol=1e-5)
    
    def test_attenuation_decreases_with_range(self):
        """Attenuation should decrease with increasing range"""
        ranges = np.array([[10.0, 50.0, 100.0, 200.0]])
        attenuation = calculate_range_attenuation(ranges)
        
        # Verify monotonic decrease
        assert attenuation[0, 0] > attenuation[0, 1]
        assert attenuation[0, 1] > attenuation[0, 2]
        assert attenuation[0, 2] > attenuation[0, 3]
    
    def test_attenuation_output_range(self):
        """Attenuation should be in [0, 1] range"""
        ranges = np.random.uniform(1, 500, size=(10, 10))
        attenuation = calculate_range_attenuation(ranges)
        
        assert np.all(attenuation >= 0.0)
        assert np.all(attenuation <= 1.0)
    
    def test_attenuation_coefficient_effect(self):
        """Higher coefficient should create stronger attenuation"""
        ranges = np.array([[50.0]])
        
        atten_low = calculate_range_attenuation(ranges, attenuation_coefficient=1.0)
        atten_high = calculate_range_attenuation(ranges, attenuation_coefficient=3.0)
        
        # Higher coefficient should give lower attenuation
        assert atten_high[0, 0] < atten_low[0, 0]
    
    def test_attenuation_handles_zero_range(self):
        """Should handle zero or very small ranges gracefully"""
        ranges = np.array([[0.0, 0.01, 0.05]])
        attenuation = calculate_range_attenuation(ranges)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(attenuation))
        assert np.all(attenuation >= 0.0)


class TestAcousticShadows:
    """Test acoustic shadow generation (Requirement 1.3)"""
    
    def test_shadow_generation_basic(self):
        """Basic shadow generation should work"""
        object_pos = np.array([[256.0, 256.0]])
        object_heights = np.array([2.0])
        sonar_pos = (128.0, 256.0)
        image_shape = (512, 512)
        
        shadow_mask = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape
        )
        
        assert shadow_mask.shape == image_shape
        assert np.all(shadow_mask >= 0.0)
        assert np.all(shadow_mask <= 1.0)
    
    def test_shadow_creates_dark_regions(self):
        """Shadows should create regions with values < 1.0"""
        object_pos = np.array([[300.0, 256.0]])
        object_heights = np.array([3.0])
        sonar_pos = (100.0, 256.0)
        image_shape = (512, 512)
        
        shadow_mask = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape
        )
        
        # Should have some shadow regions (values < 1.0)
        assert np.any(shadow_mask < 1.0)
        
        # Most of the image should not be shadowed
        assert np.mean(shadow_mask) > 0.9
    
    def test_shadow_direction(self):
        """Shadows should extend away from sonar"""
        object_pos = np.array([[256.0, 256.0]])
        object_heights = np.array([2.0])
        sonar_pos = (128.0, 256.0)
        image_shape = (512, 512)
        
        shadow_mask = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape
        )
        
        # Shadow should be on the far side of object from sonar
        # Check region behind object (x > 256)
        behind_object = shadow_mask[:, 300:400]
        in_front = shadow_mask[:, 150:200]
        
        # Behind object should have more shadowing
        assert np.mean(behind_object) < np.mean(in_front)
    
    def test_shadow_with_no_objects(self):
        """No objects should produce no shadows"""
        object_pos = np.array([]).reshape(0, 2)
        object_heights = np.array([])
        sonar_pos = (128.0, 256.0)
        image_shape = (512, 512)
        
        shadow_mask = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape
        )
        
        # Should be all ones (no shadows)
        assert np.all(shadow_mask == 1.0)
    
    def test_shadow_length_factor(self):
        """Shadow length should scale with length factor"""
        object_pos = np.array([[256.0, 256.0]])
        object_heights = np.array([2.0])
        sonar_pos = (128.0, 256.0)
        image_shape = (512, 512)
        
        shadow_short = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape,
            shadow_length_factor=1.0
        )
        
        shadow_long = generate_acoustic_shadows(
            object_pos, object_heights, sonar_pos, image_shape,
            shadow_length_factor=5.0
        )
        
        # Longer shadow factor should create more shadowed area
        assert np.sum(shadow_long < 1.0) > np.sum(shadow_short < 1.0)


class TestRangeMap:
    """Test range map creation"""
    
    def test_range_map_shape(self):
        """Range map should have correct shape"""
        image_shape = (512, 512)
        sonar_pos = (128, 256)
        
        range_map = create_range_map(image_shape, sonar_pos)
        
        assert range_map.shape == image_shape
    
    def test_range_map_increases_with_distance(self):
        """Range should increase with distance from sonar"""
        image_shape = (512, 512)
        sonar_pos = (256, 256)
        
        range_map = create_range_map(image_shape, sonar_pos)
        
        # Center should have lower range than corners
        center_range = range_map[256, 256]
        corner_range = range_map[0, 0]
        
        assert corner_range > center_range
    
    def test_range_map_values_in_bounds(self):
        """Range values should be within specified bounds"""
        image_shape = (512, 512)
        sonar_pos = (128, 256)
        range_min = 10.0
        range_max = 200.0
        
        range_map = create_range_map(
            image_shape, sonar_pos, range_min, range_max
        )
        
        assert np.all(range_map >= range_min)
        assert np.all(range_map <= range_max)


class TestGrazingAngleMap:
    """Test grazing angle map creation"""
    
    def test_grazing_angle_shape(self):
        """Grazing angle map should have correct shape"""
        image_shape = (512, 512)
        sonar_pos = (128, 256)
        
        angle_map = create_grazing_angle_map(image_shape, sonar_pos)
        
        assert angle_map.shape == image_shape
    
    def test_grazing_angle_range(self):
        """Grazing angles should be in [0, 90] degrees"""
        image_shape = (512, 512)
        sonar_pos = (128, 256)
        
        angle_map = create_grazing_angle_map(image_shape, sonar_pos)
        
        assert np.all(angle_map >= 0.0)
        assert np.all(angle_map <= 90.0)
    
    def test_grazing_angle_decreases_with_distance(self):
        """Grazing angle should decrease with horizontal distance"""
        image_shape = (512, 512)
        sonar_pos = (256, 256)
        
        angle_map = create_grazing_angle_map(image_shape, sonar_pos)
        
        # Near sonar should have higher grazing angle than far away
        near_angle = angle_map[256, 260]  # Close to sonar
        far_angle = angle_map[256, 500]   # Far from sonar
        
        assert near_angle > far_angle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
