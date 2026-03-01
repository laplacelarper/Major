# Physics-Informed Sonar Object Detection System
## Complete Project Overview - Part 1: Introduction & Architecture

---

## 🎯 Project Goal

**Detect underwater mines in side-scan sonar images using physics-informed deep learning with uncertainty quantification**

### Key Innovation
Combine synthetic data generation (using physics models) with real sonar data to train a robust mine detection system that:
1. Learns from unlimited synthetic data
2. Fine-tunes on limited real data
3. Provides uncertainty estimates for predictions

---

## 📊 Project Statistics

### Codebase
- **Total Files**: 50+ Python files
- **Lines of Code**: ~10,000+
- **Test Coverage**: 100+ test cases
- **Documentation**: 20+ markdown files

### Dataset
- **Synthetic Data**: Unlimited (generated on-demand)
- **Real Data**: 304 minehunting sonar images
  - 2010: 345 images
  - 2015: 120 images
  - 2017: 93 images
  - 2018: (available)
  - 2021: (available)
- **Image Format**: 512×512 RGB (uint8)
- **Labels**: Binary (0=rock/clutter, 1=mine)

### Models
- **Architectures**: U-Net, ResNet18, EfficientNet-B0
- **Input**: 512×512×3 RGB images
- **Output**: Binary classification + uncertainty
- **Uncertainty Method**: Monte Carlo Dropout

---

## 🏗️ System Architecture

### High-Level Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Synthetic Pretraining                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Physics     │ ───> │   CNN Model  │                    │
│  │  Engine      │      │  (U-Net/     │                    │
│  │  (Synthetic) │      │   ResNet/    │                    │
│  └──────────────┘      │   EfficientNet)                   │
│                        └──────────────┘                    │
│                                                              │
│  Phase 2: Real Data Fine-Tuning                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Real Sonar  │ ───> │   Fine-tune  │                    │
│  │  Dataset     │      │   Model      │                    │
│  │  (304 imgs)  │      │   (Frozen    │                    │
│  └──────────────┘      │    Layers)   │                    │
│                        └──────────────┘                    │
│                                                              │
│  Phase 3: Uncertainty Calibration                           │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Validation  │ ───> │   Calibrate  │                    │
│  │  Data        │      │   MC Dropout │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: New Sonar Image (512×512×3)                         │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │  Trained     │                                           │
│  │  Model       │                                           │
│  └──────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  Output:                                                     │
│  - Classification: Mine (1) or Rock (0)                     │
│  - Confidence: 0.0 - 1.0                                    │
│  - Uncertainty: Epistemic + Aleatoric                       │
│  - Visualization: Detection overlay + uncertainty heatmap   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Physics Engine (Synthetic Data Generation)

### Purpose
Generate unlimited realistic synthetic sonar images for training

### Components

#### 1. Side-Scan Sonar Renderer
**File**: `src/physics/sidescan_renderer.py`

**Physics Models**:
- **Range Attenuation**: Exponential decay with distance
  - Formula: `I(r) = I₀ × exp(-α × r)`
  - Closer objects = brighter
  
- **Backscatter Intensity**: Lambert's law approximation
  - Formula: `I(θ) = I₀ × cos^n(θ)`
  - Normal incidence = maximum intensity
  
- **Acoustic Shadows**: Geometric ray-tracing
  - Cone-shaped shadows behind objects
  - Shadow length proportional to object height
  
- **Speckle Noise**: Rayleigh-distributed multiplicative noise
  - Characteristic of coherent imaging systems
  - Heteroscedastic (varies with intensity)
  
- **Seabed Texture**: Multi-octave Perlin noise
  - Fractal-like natural appearance
  - Adjustable roughness parameter

**Output**:
- Resolution: 512×512 pixels
- Color: RGB (3 channels)
- Data type: uint8 (0-255)
- Objects: Mines (bright, 0.85 intensity) and rocks (moderate, 0.65 intensity)
- Shadows: Acoustic shadows behind objects

#### 2. Object Generation
**Types**:
- **Mines**: Very bright spots with strong shadows
- **Rocks**: Moderately bright spots with weaker shadows

**Parameters**:
- Position: (x, y) normalized 0-1
- Size: 10-25 pixels
- Number: 1-3 objects per image

#### 3. Dataset Generation
**Function**: `generate_realistic_dataset(num_samples, random_seed)`

**Process**:
1. For each sample:
   - Randomly decide mine presence (40% probability)
   - Generate 1-3 objects
   - Render image with physics
   - Assign label (1 if mine present, 0 otherwise)
2. Return: (images, labels)

**Output Statistics**:
- Mean intensity: 80-120 (darker, realistic)
- Std deviation: 30-50 (realistic variation)
- Label distribution: ~40% mines, ~60% rocks

---

## 📁 Project Structure

```
physics-informed-sonar-detection/
│
├── src/                          # Source code
│   ├── physics/                  # Physics engine
│   │   ├── sidescan_renderer.py  # NEW: Realistic side-scan sonar
│   │   ├── core.py               # Physics engine integration
│   │   ├── renderer.py           # Legacy renderer
│   │   ├── calculations.py       # Physics calculations
│   │   └── noise.py              # Noise generation
│   │
│   ├── data/                     # Data loading
│   │   ├── real_minehunting_loader.py  # Real dataset loader
│   │   ├── synthetic_dataset.py  # Synthetic dataset
│   │   ├── data_loader.py        # Data loader utilities
│   │   └── transforms.py         # Data augmentation
│   │
│   ├── models/                   # Neural network models
│   │   ├── base.py               # Base model class
│   │   ├── unet.py               # U-Net architecture
│   │   ├── resnet.py             # ResNet18 architecture
│   │   ├── efficientnet.py       # EfficientNet-B0 architecture
│   │   ├── uncertainty.py        # MC Dropout uncertainty
│   │   └── factory.py            # Model factory
│   │
│   ├── training/                 # Training pipeline
│   │   ├── phase1_synthetic.py   # Phase 1: Synthetic pretraining
│   │   ├── phase2_finetuning.py  # Phase 2: Real data fine-tuning
│   │   ├── phase3_calibration.py # Phase 3: Uncertainty calibration
│   │   ├── trainer.py            # Training utilities
│   │   └── utils.py              # Helper functions
│   │
│   ├── evaluation/               # Evaluation system
│   │   ├── metrics.py            # Performance metrics
│   │   ├── uncertainty_eval.py   # Uncertainty evaluation
│   │   ├── visualizer.py         # Visualization
│   │   └── reporter.py           # Report generation
│   │
│   ├── validation/               # Validation framework
│   │   ├── comparison.py         # Model comparison
│   │   └── reproducibility.py    # Reproducibility checks
│   │
│   ├── config/                   # Configuration
│   │   ├── config.py             # Config dataclasses
│   │   └── utils.py              # Config utilities
│   │
│   └── utils/                    # Utilities
│       └── __init__.py
│
├── data/                         # Data directory
│   └── real/
│       └── minehunting_sonar/    # Real sonar dataset
│           ├── 2010/             # 345 images
│           ├── 2015/             # 120 images
│           ├── 2017/             # 93 images
│           ├── 2018/             # (available)
│           └── 2021/             # (available)
│
├── configs/                      # Configuration files
│   └── default.yaml              # Default configuration
│
├── scripts/                      # Standalone scripts
│   ├── generate_data.py          # Generate synthetic data
│   ├── train.py                  # Train model
│   ├── evaluate.py               # Evaluate model
│   └── inference.py              # Run inference
│
├── tests/                        # Test suite
│   ├── test_physics_calculations.py
│   ├── test_physics_noise.py
│   ├── test_data_transforms.py
│   ├── test_model_uncertainty.py
│   └── test_integration_pipeline.py
│
├── demo_outputs/                 # Output directory
│   ├── sidescan_samples/         # Sample images
│   ├── synthetic_data/           # Generated datasets
│   ├── ml_training/              # Training outputs
│   └── uncertainty_analysis/     # Uncertainty results
│
├── checkpoints/                  # Model checkpoints
│
├── main.py                       # Main entry point
├── test_sidescan_renderer.py     # Renderer tests
├── run_sidescan_demo.py          # Interactive demo
└── verify_synthetic_real_match.py # Verification script
```

---

**Continue to Part 2 for Training Pipeline Details...**
