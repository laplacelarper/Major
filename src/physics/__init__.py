"""Physics engine for synthetic sonar data generation"""

from .core import PhysicsEngine
from .calculations import (
    calculate_backscatter_intensity,
    calculate_range_attenuation,
    generate_acoustic_shadows,
    create_range_map,
    create_grazing_angle_map
)
from .noise import (
    generate_speckle_noise,
    generate_seabed_texture,
    randomize_parameters,
    apply_multiplicative_noise
)
from .renderer import (
    SonarImageRenderer,
    ImageExporter,
    PhysicsMetadata,
    generate_random_scene_parameters
)

__all__ = [
    'PhysicsEngine',
    'calculate_backscatter_intensity',
    'calculate_range_attenuation', 
    'generate_acoustic_shadows',
    'create_range_map',
    'create_grazing_angle_map',
    'generate_speckle_noise',
    'generate_seabed_texture',
    'randomize_parameters',
    'apply_multiplicative_noise',
    'SonarImageRenderer',
    'ImageExporter',
    'PhysicsMetadata',
    'generate_random_scene_parameters'
]