# Physics-Informed Sonar Object Detection System

A machine learning pipeline for detecting mine-like objects versus natural seabed objects (rocks/clutter) from side-scan sonar images using physics-informed synthetic data generation and uncertainty-aware deep learning.

## Project Structure

```
physics-informed-sonar-detection/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration dataclasses
│   │   └── utils.py             # Config utilities (load/save/logging)
│   ├── data/                    # Data handling modules
│   ├── models/                  # Model architectures
│   ├── training/                # Training pipeline
│   ├── evaluation/              # Evaluation and metrics
│   └── utils/                   # General utilities
├── configs/                     # Configuration files
│   └── default.yaml            # Default configuration
├── data/                       # Data directory (created automatically)
│   ├── synthetic/              # Generated synthetic data
│   └── real/                   # Real sonar datasets
├── outputs/                    # Output directory (created automatically)
│   ├── visualizations/         # Generated plots and images
│   ├── metrics/               # Evaluation metrics
│   └── reports/               # Analysis reports
├── checkpoints/               # Model checkpoints (created automatically)
├── logs/                     # Log files (created automatically)
├── main.py                   # Main entry point
├── test_setup.py            # Setup verification script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Setup**
   ```bash
   python test_setup.py
   ```

3. **Run with Default Configuration**
   ```bash
   python main.py --mode test_config
   ```

## Configuration System

The system uses a hierarchical configuration system with YAML support:

- **PhysicsConfig**: Parameters for synthetic data generation
- **ModelConfig**: CNN architecture and uncertainty estimation settings
- **TrainingConfig**: Three-phase training pipeline parameters
- **DataConfig**: Data loading and preprocessing settings

### Example Configuration

```yaml
# Physics parameters
physics:
  grazing_angle_range: [10.0, 80.0]
  noise_level_range: [0.1, 0.4]
  
# Model settings
model:
  model_type: "unet"
  dropout_rate: 0.1
  mc_samples: 20
  
# Training phases
training:
  phase1_epochs: 100
  phase2_epochs: 50
  phase3_epochs: 20
```

## Key Features

- **Physics-Informed Synthetic Data**: Generates realistic sonar images using acoustic scattering approximations
- **Three-Phase Training**: Synthetic pretraining → Real data fine-tuning → Uncertainty calibration
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence estimation
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Reproducible Results**: Deterministic runs with seed management
- **Flexible Configuration**: YAML-based parameter management

## Requirements Addressed

This setup addresses the following requirements from the specification:

- **6.1**: Deterministic runs with random seed management
- **6.3**: Config-driven parameters without hardcoded paths
- **7.4**: Optimized for standard hardware compatibility

## Next Steps

After completing this setup task, the next tasks in the implementation plan are:

1. **Task 2**: Implement physics engine for synthetic data generation
2. **Task 3**: Create dataset loading and preprocessing system
3. **Task 4**: Implement CNN model architectures

## Development

To extend the system:

1. Add new configuration parameters to the appropriate dataclass in `src/config/config.py`
2. Update the default YAML configuration in `configs/default.yaml`
3. Implement new modules in the appropriate `src/` subdirectories
4. Use the logging system for debugging and monitoring

## Testing

Run the setup verification:
```bash
python test_setup.py
```

This will test:
- Module imports
- Configuration creation
- YAML loading
- Directory creation
- Logging setup
- Random seed reproducibility