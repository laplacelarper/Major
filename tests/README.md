# Comprehensive Testing Suite

This directory contains a comprehensive testing suite for the Physics-Informed Sonar Object Detection system.

## Test Coverage

### Task 9.1: Unit Tests for Core Components

#### Physics Engine Tests (`test_physics_calculations.py`)
Tests Requirements 1.2, 1.3, 1.4:
- **Backscatter Intensity** (Requirement 1.2)
  - Cosⁿ(grazing_angle) relationship validation
  - Normal incidence vs grazing angle behavior
  - Output range validation [0, 1]
  - Exponent effect on intensity falloff

- **Range Attenuation** (Requirement 1.4)
  - 1/R² inverse square law validation
  - Monotonic decrease with distance
  - Output range validation [0, 1]
  - Attenuation coefficient effects
  - Zero range handling

- **Acoustic Shadows** (Requirement 1.3)
  - Shadow generation with geometric approximations
  - Shadow direction relative to sonar position
  - Shadow length factor effects
  - Multiple object handling
  - Empty scene handling

- **Range and Grazing Angle Maps**
  - Correct spatial geometry
  - Value range validation
  - Distance-based calculations

#### Noise Generation Tests (`test_physics_noise.py`)
Tests Requirement 1.5:
- **Speckle Noise**
  - Rayleigh distribution implementation
  - Gamma distribution implementation
  - Multiplicative noise properties (mean ≈ 1.0)
  - Noise level effects on variance
  - Reproducibility with random seeds

- **Seabed Texture**
  - Procedural noise generation
  - Roughness parameter effects
  - Texture scale variations
  - Spatial coherence validation
  - Output range [0, 1]

- **Parameter Randomization**
  - Range-based randomization
  - Percentage-based variation
  - Reproducibility
  - Preservation of unspecified parameters

- **Noise Application**
  - Multiplicative noise application
  - Noise strength control
  - Output range preservation

#### Data Transform Tests (`test_data_transforms.py`)
Tests data preprocessing and augmentation:
- **Image Normalization**
  - Shape preservation
  - Range transformation
  - NumPy/Torch compatibility
  - Invertibility with denormalization

- **Augmentation Transforms**
  - Random rotation (shape preservation, probability control)
  - Random flips (horizontal/vertical)
  - Noise injection (speckle, Gaussian)
  - Brightness/contrast adjustment
  - Elastic deformation

- **Metadata Encoding** (Requirement 3.1)
  - Output shape validation
  - Value normalization [0, 1]
  - Material encoding (metal/rock/sand)
  - Empty metadata handling

- **Augmentation Pipeline**
  - Phase-specific behavior (train/val/test)
  - Sample structure preservation
  - Transform composition

#### Model Uncertainty Tests (`test_model_uncertainty.py`)
Tests Requirement 3.1:
- **Monte Carlo Dropout**
  - Always-active dropout in eval mode
  - Shape preservation
  - Dropout probability effects
  - 2D dropout for convolutional layers

- **Uncertainty Estimation**
  - Multiple forward passes (≥20 samples)
  - Mean prediction calculation
  - Variance/uncertainty computation
  - Detailed uncertainty metrics:
    - Epistemic uncertainty (model uncertainty)
    - Aleatoric uncertainty (data uncertainty)
    - Entropy-based measures
    - Feature uncertainty

- **Uncertainty Calibration**
  - Expected Calibration Error (ECE)
  - Confidence-accuracy correlation
  - Uncertainty-error correlation
  - Calibration at different confidence levels

- **Uncertainty Visualization**
  - Heatmap generation
  - Spatial uncertainty maps
  - Classification uncertainty display

### Task 9.2: Integration and End-to-End Tests

#### Integration Pipeline Tests (`test_integration_pipeline.py`)
Tests Requirements 7.4, 6.5:

- **Synthetic Data Generation Pipeline**
  - Single image generation
  - Batch generation performance benchmarking
  - Physics validation across all components
  - Parameter range validation

- **Model Training Pipeline**
  - Model creation for all architectures (U-Net, ResNet18, EfficientNet-B0)
  - Forward pass validation
  - Training step execution
  - Early stopping mechanism
  - Gradient flow validation

- **Evaluation Pipeline**
  - Comprehensive metrics computation
  - Uncertainty integration with evaluation
  - Metrics validation (accuracy, precision, recall, F1, ECE)

- **End-to-End Pipeline** (Requirement 6.5)
  - Complete workflow: data generation → training → evaluation
  - Error handling for invalid inputs
  - Graceful degradation with missing data
  - No silent failures

- **Performance Benchmarks** (Requirement 7.4)
  - Synthetic image generation speed
  - Model inference speed
  - Uncertainty estimation speed
  - Batch processing efficiency
  - CPU-only compatibility

- **Reproducibility**
  - Deterministic data generation with seeds
  - Deterministic model initialization
  - Consistent results across runs

## Running Tests

### Prerequisites
```bash
pip install pytest torch torchvision numpy opencv-python pillow
```

### Run All Tests
```bash
# Using pytest (recommended)
pytest tests/ -v

# Using unittest (no dependencies)
python tests/run_all_tests.py

# Run specific test file
pytest tests/test_physics_calculations.py -v

# Run specific test class
pytest tests/test_physics_calculations.py::TestBackscatterIntensity -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Individual Test Modules

```bash
# Physics calculations
pytest tests/test_physics_calculations.py -v

# Noise generation
pytest tests/test_physics_noise.py -v

# Data transforms
pytest tests/test_data_transforms.py -v

# Model uncertainty
pytest tests/test_model_uncertainty.py -v

# Integration tests
pytest tests/test_integration_pipeline.py -v
```

## Test Organization

```
tests/
├── README.md                          # This file
├── run_all_tests.py                   # Simple test runner
├── test_physics_calculations.py       # Physics engine unit tests
├── test_physics_noise.py              # Noise generation unit tests
├── test_data_transforms.py            # Data preprocessing tests
├── test_model_uncertainty.py          # Uncertainty estimation tests
├── test_integration_pipeline.py       # Integration and E2E tests
├── test_data_loading.py              # Existing data loading tests
├── test_evaluation_structure.py      # Existing evaluation tests
├── test_evaluation_system.py         # Existing evaluation tests
├── test_setup.py                     # Existing setup tests
└── test_validation_structure.py     # Existing validation tests
```

## Requirements Coverage

### Requirement 1.2: Backscatter Intensity
✅ Tested in `test_physics_calculations.py::TestBackscatterIntensity`
- Validates cosⁿ(grazing_angle) formula
- Tests normal incidence vs grazing behavior
- Verifies exponent effects

### Requirement 1.3: Acoustic Shadows
✅ Tested in `test_physics_calculations.py::TestAcousticShadows`
- Validates geometric shadow generation
- Tests shadow direction and length
- Verifies multiple object handling

### Requirement 1.4: Range Attenuation
✅ Tested in `test_physics_calculations.py::TestRangeAttenuation`
- Validates 1/R² relationship
- Tests attenuation coefficient effects
- Verifies edge cases (zero range)

### Requirement 1.5: Noise and Texture
✅ Tested in `test_physics_noise.py`
- Validates Rayleigh/Gamma speckle noise
- Tests procedural texture generation
- Verifies parameter randomization

### Requirement 3.1: Monte Carlo Dropout
✅ Tested in `test_model_uncertainty.py`
- Validates MC Dropout implementation
- Tests multiple forward passes (≥20)
- Verifies mean and variance calculation
- Tests epistemic/aleatoric uncertainty separation

### Requirement 6.5: Error Handling
✅ Tested in `test_integration_pipeline.py::TestEndToEndPipeline`
- Validates graceful error handling
- Tests invalid input handling
- Verifies no silent failures

### Requirement 7.4: Performance Optimization
✅ Tested in `test_integration_pipeline.py::TestPerformanceBenchmarks`
- Benchmarks synthetic generation speed
- Tests inference performance
- Validates batch processing efficiency
- Verifies CPU compatibility

## Test Statistics

- **Total Test Files**: 10
- **Unit Test Files**: 4 (new) + 5 (existing)
- **Integration Test Files**: 1 (new)
- **Test Classes**: 30+
- **Individual Tests**: 100+

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- No GPU required (CPU-only compatible)
- Reasonable execution time (< 5 minutes for full suite)
- Clear pass/fail criteria
- Detailed error messages

## Contributing

When adding new features:
1. Write unit tests for individual components
2. Write integration tests for component interactions
3. Update this README with test coverage information
4. Ensure all tests pass before submitting PR

## Notes

- Tests use small image sizes (64x64, 128x128) for speed
- Performance benchmarks have generous thresholds for CI compatibility
- All tests are deterministic with fixed random seeds
- Tests validate both correctness and performance
