"""Configuration dataclasses for the sonar detection system"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os


@dataclass
class PhysicsConfig:
    """Configuration for physics-based synthetic data generation"""
    # Backscatter parameters
    cosine_exponent_range: Tuple[float, float] = (2.0, 8.0)
    base_intensity_range: Tuple[float, float] = (0.3, 0.8)
    
    # Range and attenuation
    range_min_m: float = 10.0
    range_max_m: float = 200.0
    attenuation_coefficient: float = 2.0  # for 1/R^n relationship
    
    # Acoustic shadow parameters
    shadow_length_factor: float = 3.0
    shadow_intensity_factor: float = 0.1
    
    # Noise parameters
    speckle_noise_type: str = "rayleigh"  # "rayleigh" or "gamma"
    noise_level_range: Tuple[float, float] = (0.1, 0.4)
    
    # Seabed texture
    texture_roughness_range: Tuple[float, float] = (0.2, 0.8)
    texture_scale_range: Tuple[float, float] = (5.0, 20.0)
    
    # Grazing angle range (degrees)
    grazing_angle_range: Tuple[float, float] = (10.0, 80.0)
    
    # Sonar frequency parameters
    frequency_khz_range: Tuple[float, float] = (100.0, 500.0)
    beam_width_deg_range: Tuple[float, float] = (1.0, 5.0)


@dataclass
class ModelConfig:
    """Configuration for CNN model architecture and uncertainty estimation"""
    # Model architecture
    model_type: str = "unet"  # "unet", "resnet18", "efficientnet-b0"
    num_classes: int = 2  # binary classification
    input_channels: int = 1  # grayscale images
    
    # Uncertainty estimation
    dropout_rate: float = 0.1
    mc_samples: int = 20
    use_uncertainty: bool = True
    
    # Auxiliary inputs
    use_physics_metadata: bool = True
    metadata_dim: int = 7  # number of physics parameters
    
    # Output modes
    output_mode: str = "classification"  # "classification" or "segmentation"


@dataclass
class TrainingConfig:
    """Configuration for the three-phase training pipeline"""
    # Phase 1: Synthetic pretraining
    phase1_epochs: int = 100
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 16
    phase1_weight_decay: float = 1e-4
    
    # Phase 2: Real data fine-tuning
    phase2_epochs: int = 50
    phase2_lr: float = 1e-5
    phase2_batch_size: int = 8
    phase2_freeze_layers: int = 3  # number of early layers to freeze
    
    # Phase 3: Uncertainty calibration
    phase3_epochs: int = 20
    phase3_lr: float = 1e-6
    phase3_batch_size: int = 8
    
    # General training parameters
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 5
    validation_split: float = 0.2
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    gradient_clip_norm: float = 1.0


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    # Image parameters
    image_size: Tuple[int, int] = (512, 512)
    image_channels: int = 1
    
    # Dataset sizes
    synthetic_dataset_size: int = 10000
    real_data_percentage: float = 0.3  # max 30% real data
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Augmentation parameters
    use_augmentation: bool = True
    rotation_range: float = 30.0  # degrees
    flip_probability: float = 0.5
    noise_injection_prob: float = 0.3
    
    # Preprocessing
    normalize_images: bool = True
    normalization_mean: float = 0.5
    normalization_std: float = 0.5
    
    # Real dataset sources
    real_datasets: List[str] = field(default_factory=lambda: [
        "minehunting_sonar",
        "cmre_muscle_sas"
    ])


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""
    # Sub-configurations
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    config_dir: Path = field(default_factory=lambda: Path("configs"))
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Experiment tracking
    experiment_name: str = "sonar_detection"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization to resolve paths and validate configuration"""
        # Convert string paths to Path objects and make them absolute
        for attr_name in ["data_dir", "output_dir", "checkpoint_dir", "logs_dir", "config_dir"]:
            path_value = getattr(self, attr_name)
            if isinstance(path_value, str):
                path_value = Path(path_value)
            
            # Make paths relative to project root if they're not absolute
            if not path_value.is_absolute():
                path_value = self.project_root / path_value
            
            setattr(self, attr_name, path_value)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate data splits sum to 1.0
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate real data percentage
        if not 0.0 <= self.data.real_data_percentage <= 0.3:
            raise ValueError("Real data percentage must be between 0.0 and 0.3 (30%)")
        
        # Validate image size
        if self.data.image_size[0] != self.data.image_size[1]:
            raise ValueError("Only square images are supported")
        
        # Validate model type
        valid_models = ["unet", "resnet18", "efficientnet-b0"]
        if self.model.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}")
        
        # Validate output mode
        valid_modes = ["classification", "segmentation"]
        if self.model.output_mode not in valid_modes:
            raise ValueError(f"Output mode must be one of {valid_modes}")
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.config_dir,
            self.data_dir / "synthetic",
            self.data_dir / "real",
            self.output_dir / "visualizations",
            self.output_dir / "metrics",
            self.output_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_run_dir(self) -> Path:
        """Get the directory for the current run"""
        run_name = self.run_name or f"run_{self.random_seed}"
        return self.output_dir / self.experiment_name / run_name
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization"""
        def convert_value(value):
            if isinstance(value, Path):
                return str(value)
            elif hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return {k: convert_value(v) for k, v in self.__dict__.items()}