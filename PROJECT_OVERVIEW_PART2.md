# Physics-Informed Sonar Object Detection System
## Complete Project Overview - Part 2: Training Pipeline

---

## 🎓 Three-Phase Training Pipeline

### Overview
The system uses a novel three-phase training approach to maximize performance with limited real data:

```
Phase 1: Synthetic Pretraining (70% of training time)
    ↓
Phase 2: Real Data Fine-Tuning (20% of training time)
    ↓
Phase 3: Uncertainty Calibration (10% of training time)
    ↓
Final Model (Ready for deployment)
```

---

## 📚 Phase 1: Synthetic Pretraining

### Purpose
Train the model on unlimited synthetic data to learn basic sonar image features

### File
`src/training/phase1_synthetic.py`

### Process

#### Step 1: Generate Synthetic Data
```python
from src.physics.core import PhysicsEngine

engine = PhysicsEngine(use_realistic_renderer=True)
images, labels, _ = engine.generate_dataset(
    num_samples=10000,  # Generate 10,000 synthetic images
    save_to_disk=True
)
```

**Generated Data**:
- 10,000 synthetic sonar images
- 512×512×3 RGB format
- ~40% mines, ~60% rocks
- Realistic physics-based appearance

#### Step 2: Data Augmentation
```python
from src.data.transforms import get_augmentation_pipeline

augmentation = get_augmentation_pipeline(phase='train')
```

**Augmentations Applied**:
- Random rotation (±30°)
- Random horizontal/vertical flip
- Random brightness/contrast adjustment
- Noise injection (speckle, Gaussian)
- Elastic deformation

#### Step 3: Model Training
```python
from src.models.factory import create_model

model = create_model(
    architecture='unet',  # or 'resnet18', 'efficientnet-b0'
    num_classes=2,
    input_channels=3
)
```

**Training Configuration**:
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Batch Size**: 16
- **Epochs**: 50
- **Loss Function**: Binary Cross-Entropy
- **Early Stopping**: Patience 10 epochs

**Training Loop**:
```
For each epoch:
    For each batch:
        1. Load synthetic images + labels
        2. Apply augmentation
        3. Forward pass through model
        4. Calculate loss
        5. Backward pass (compute gradients)
        6. Update weights
    
    Validate on validation set
    Save checkpoint if best model
    Check early stopping
```

#### Step 4: Outputs
- **Trained Model**: Checkpoint saved to `checkpoints/phase1_best.pth`
- **Training Curves**: Loss and accuracy plots
- **Validation Metrics**: Accuracy, precision, recall, F1-score

**Expected Performance**:
- Training Accuracy: ~90%
- Validation Accuracy: ~85%
- Overfitting: Minimal (due to augmentation)

---

## 🎯 Phase 2: Real Data Fine-Tuning

### Purpose
Adapt the model to real sonar data characteristics using transfer learning

### File
`src/training/phase2_finetuning.py`

### Process

#### Step 1: Load Real Data
```python
from src.data.real_minehunting_loader import MinehuntingSonarDataset

real_dataset = MinehuntingSonarDataset(
    data_dir='data/real/minehunting_sonar',
    years=[2010, 2015, 2017, 2018, 2021]
)
```

**Real Dataset**:
- 304 total images
- Train: 70% (~213 images)
- Validation: 15% (~46 images)
- Test: 15% (~45 images)

#### Step 2: Load Pretrained Model
```python
model = load_checkpoint('checkpoints/phase1_best.pth')
```

#### Step 3: Freeze Early Layers
```python
# Freeze early convolutional layers
for param in model.encoder[:3].parameters():
    param.requires_grad = False
```

**Rationale**: Early layers learn generic features (edges, textures) that transfer well. Only fine-tune later layers for domain-specific features.

#### Step 4: Fine-Tuning
**Training Configuration**:
- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (10× lower than Phase 1)
- **Batch Size**: 8 (smaller due to limited data)
- **Epochs**: 30
- **Loss Function**: Binary Cross-Entropy
- **Early Stopping**: Patience 5 epochs

**Training Loop**:
```
For each epoch:
    For each batch:
        1. Load real images + labels
        2. Apply light augmentation
        3. Forward pass through model
        4. Calculate loss
        5. Backward pass (only unfrozen layers)
        6. Update weights
    
    Validate on real validation set
    Save checkpoint if best model
    Check early stopping
```

#### Step 5: Outputs
- **Fine-Tuned Model**: Checkpoint saved to `checkpoints/phase2_best.pth`
- **Training Curves**: Loss and accuracy plots
- **Validation Metrics**: Accuracy, precision, recall, F1-score on real data

**Expected Performance**:
- Real Data Accuracy: ~80% (up from ~70% without fine-tuning)
- Transfer Learning Gain: +10%
- Generalization: Good (validated on held-out test set)

---

## 🎲 Phase 3: Uncertainty Calibration

### Purpose
Calibrate uncertainty estimates to be reliable and well-calibrated

### File
`src/training/phase3_calibration.py`

### Process

#### Step 1: Enable Monte Carlo Dropout
```python
from src.models.uncertainty import UncertaintyEstimator

uncertainty_estimator = UncertaintyEstimator(
    model=model,
    num_samples=20,  # Number of forward passes
    dropout_rate=0.2
)
```

**Monte Carlo Dropout**:
- Keep dropout active during inference
- Run multiple forward passes (20×)
- Calculate mean prediction and variance
- Variance = uncertainty estimate

#### Step 2: Calibration
```python
# Run inference with uncertainty
predictions, uncertainties = uncertainty_estimator.predict_with_uncertainty(
    validation_data
)

# Calculate calibration metrics
ece = calculate_expected_calibration_error(predictions, uncertainties, labels)
```

**Calibration Metrics**:
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Reliability Diagram**: Plots confidence vs accuracy
- **Uncertainty-Error Correlation**: Higher uncertainty = higher error rate

#### Step 3: Temperature Scaling (Optional)
```python
# Adjust confidence scores to be well-calibrated
temperature = find_optimal_temperature(predictions, labels)
calibrated_predictions = predictions / temperature
```

#### Step 4: Outputs
- **Calibrated Model**: Checkpoint saved to `checkpoints/phase3_best.pth`
- **Calibration Curves**: Reliability diagrams
- **Uncertainty Metrics**: ECE, correlation coefficients

**Expected Performance**:
- ECE: <0.10 (well-calibrated)
- Uncertainty-Error Correlation: >0.6 (reliable)
- High Confidence = High Accuracy

---

## 📊 Training Outputs & Metrics

### Checkpoints
Saved to `checkpoints/` directory:
- `phase1_best.pth` - Best Phase 1 model
- `phase2_best.pth` - Best Phase 2 model
- `phase3_best.pth` - Final calibrated model

### Training Curves
Saved to `demo_outputs/ml_training/`:
- `phase1_loss.png` - Training/validation loss
- `phase1_accuracy.png` - Training/validation accuracy
- `phase2_loss.png` - Fine-tuning loss
- `phase2_accuracy.png` - Fine-tuning accuracy

### Metrics Reports
Saved to `demo_outputs/ml_training/`:
- `phase1_metrics.json` - Phase 1 performance
- `phase2_metrics.json` - Phase 2 performance
- `phase3_metrics.json` - Phase 3 calibration

**Metrics Included**:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Expected Calibration Error (ECE)

---

## 🚀 Running the Training Pipeline

### Option 1: Full Pipeline (All Phases)
```bash
python main.py --mode full_pipeline
```

**What it does**:
1. Generates 10,000 synthetic images
2. Trains Phase 1 (synthetic pretraining)
3. Fine-tunes Phase 2 (real data)
4. Calibrates Phase 3 (uncertainty)
5. Evaluates final model
6. Generates reports

**Duration**: ~2-4 hours (depending on hardware)

### Option 2: Individual Phases
```bash
# Phase 1 only
python main.py --mode train --phase 1

# Phase 2 only (requires Phase 1 checkpoint)
python main.py --mode train --phase 2

# Phase 3 only (requires Phase 2 checkpoint)
python main.py --mode train --phase 3
```

### Option 3: Synthetic-Only Training
```bash
python main.py --mode full_pipeline --synthetic_only
```

**What it does**:
- Skips Phase 2 (no real data fine-tuning)
- Useful for testing without real dataset

---

**Continue to Part 3 for Evaluation & Inference...**
