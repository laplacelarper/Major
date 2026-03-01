# Physics-Informed Sonar Object Detection System
## Complete Project Overview - Part 3: Evaluation & Inference

---

## 📈 Evaluation System

### Purpose
Comprehensive evaluation of model performance on real sonar data

### File
`src/evaluation/metrics.py`, `src/evaluation/uncertainty_eval.py`

---

## 🎯 Performance Metrics

### Classification Metrics

#### 1. Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Definition**: Percentage of correct predictions
- **Expected**: 75-85% on real test data
- **Interpretation**: Overall correctness

#### 2. Precision
```
Precision = TP / (TP + FP)
```
- **Definition**: Of predicted mines, how many are actually mines?
- **Expected**: 70-80%
- **Interpretation**: False alarm rate (lower FP = higher precision)

#### 3. Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- **Definition**: Of actual mines, how many did we detect?
- **Expected**: 75-85%
- **Interpretation**: Detection rate (lower FN = higher recall)

#### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Definition**: Harmonic mean of precision and recall
- **Expected**: 0.72-0.82
- **Interpretation**: Balanced performance metric

#### 5. Confusion Matrix
```
                Predicted
              Mine    Rock
Actual Mine    TP      FN
       Rock    FP      TN
```

**Example**:
```
                Predicted
              Mine    Rock
Actual Mine    38      7     (Recall: 84%)
       Rock    5       40    (Precision: 88%)
```

### Uncertainty Metrics

#### 1. Expected Calibration Error (ECE)
```
ECE = Σ |accuracy(bin) - confidence(bin)| × weight(bin)
```
- **Definition**: Measures calibration quality
- **Expected**: <0.10 (well-calibrated)
- **Interpretation**: Lower = better calibration

#### 2. Uncertainty-Error Correlation
```
Correlation(uncertainty, error)
```
- **Definition**: Do high uncertainty predictions have higher error rates?
- **Expected**: >0.6 (strong correlation)
- **Interpretation**: Higher = more reliable uncertainty

#### 3. Epistemic vs Aleatoric Uncertainty
- **Epistemic**: Model uncertainty (reducible with more data)
- **Aleatoric**: Data uncertainty (irreducible noise)
- **Ratio**: Helps understand error sources

---

## 🔍 Evaluation Process

### Step 1: Load Test Data
```python
from src.data.real_minehunting_loader import MinehuntingSonarDataset

test_dataset = MinehuntingSonarDataset(
    data_dir='data/real/minehunting_sonar',
    split='test'  # 15% of data (~45 images)
)
```

### Step 2: Run Inference
```python
from src.models.uncertainty import UncertaintyEstimator

# Load trained model
model = load_checkpoint('checkpoints/phase3_best.pth')

# Create uncertainty estimator
estimator = UncertaintyEstimator(model, num_samples=20)

# Run inference
predictions = []
uncertainties = []
for image, label in test_dataset:
    pred, unc = estimator.predict_with_uncertainty(image)
    predictions.append(pred)
    uncertainties.append(unc)
```

### Step 3: Calculate Metrics
```python
from src.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(
    predictions=predictions,
    labels=test_labels,
    uncertainties=uncertainties
)
```

**Output**:
```json
{
    "accuracy": 0.82,
    "precision": 0.78,
    "recall": 0.84,
    "f1_score": 0.81,
    "confusion_matrix": [[40, 5], [7, 38]],
    "roc_auc": 0.88,
    "ece": 0.08,
    "uncertainty_correlation": 0.65
}
```

### Step 4: Generate Visualizations
```python
from src.evaluation.visualizer import create_evaluation_report

create_evaluation_report(
    predictions=predictions,
    labels=test_labels,
    uncertainties=uncertainties,
    output_dir='demo_outputs/evaluation/'
)
```

**Generated Files**:
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve
- `calibration_curve.png` - Reliability diagram
- `uncertainty_distribution.png` - Uncertainty histogram
- `detection_examples.png` - Sample predictions with uncertainty

---

## 🎨 Visualization Outputs

### 1. Detection Overlay
Shows predictions on test images:
```
Original Image + Bounding Box + Confidence + Uncertainty
```

**Example**:
```
┌─────────────────────────────────┐
│  [Sonar Image]                  │
│                                  │
│     ┌──────┐  ← Mine            │
│     │ 0.92 │    (92% confidence)│
│     └──────┘    (Low uncertainty)│
│                                  │
│  ┌────┐  ← Rock                 │
│  │0.15│    (15% confidence)     │
│  └────┘    (High uncertainty)   │
└─────────────────────────────────┘
```

### 2. Uncertainty Heatmap
Shows spatial uncertainty distribution:
```
Blue = Low uncertainty (confident)
Red = High uncertainty (uncertain)
```

### 3. Calibration Curve
Shows confidence vs accuracy:
```
Perfect Calibration: y = x
Overconfident: Above diagonal
Underconfident: Below diagonal
```

### 4. Confusion Matrix
Shows classification performance:
```
        Predicted
        Mine  Rock
Actual
Mine     38    7
Rock      5   40
```

---

## 🚀 Inference Pipeline

### Purpose
Deploy trained model for real-world mine detection

### File
`scripts/inference.py`

---

## 🔮 Running Inference

### Option 1: Single Image
```bash
python scripts/inference.py \
    --image path/to/sonar_image.jpg \
    --checkpoint checkpoints/phase3_best.pth \
    --output demo_outputs/inference/
```

**Output**:
```
Prediction: Mine (Label: 1)
Confidence: 0.92
Uncertainty: 0.08 (Low)
Epistemic: 0.05
Aleatoric: 0.03

Saved:
  - detection_overlay.png
  - uncertainty_heatmap.png
  - prediction_report.json
```

### Option 2: Batch Processing
```bash
python scripts/inference.py \
    --input_dir data/real/minehunting_sonar/2021/ \
    --checkpoint checkpoints/phase3_best.pth \
    --output demo_outputs/batch_inference/
```

**Output**:
- Processes all images in directory
- Generates predictions for each
- Creates summary report

### Option 3: Programmatic API
```python
from src.models.uncertainty import UncertaintyEstimator
from PIL import Image
import numpy as np

# Load model
model = load_checkpoint('checkpoints/phase3_best.pth')
estimator = UncertaintyEstimator(model, num_samples=20)

# Load image
image = Image.open('sonar_image.jpg')
image_array = np.array(image)

# Run inference
prediction, uncertainty = estimator.predict_with_uncertainty(image_array)

# Interpret results
if prediction > 0.5:
    print(f"Mine detected! Confidence: {prediction:.2f}")
    print(f"Uncertainty: {uncertainty:.2f}")
else:
    print(f"Rock/Clutter. Confidence: {1-prediction:.2f}")
    print(f"Uncertainty: {uncertainty:.2f}")
```

---

## 📊 Inference Outputs

### 1. Prediction Report (JSON)
```json
{
    "image_path": "sonar_image.jpg",
    "prediction": {
        "class": "mine",
        "label": 1,
        "confidence": 0.92,
        "probability": 0.92
    },
    "uncertainty": {
        "total": 0.08,
        "epistemic": 0.05,
        "aleatoric": 0.03
    },
    "metadata": {
        "model": "phase3_best.pth",
        "timestamp": "2026-03-01T12:00:00",
        "num_mc_samples": 20
    }
}
```

### 2. Detection Overlay (PNG)
Visual representation with:
- Original sonar image
- Bounding box (if applicable)
- Confidence score
- Uncertainty indicator

### 3. Uncertainty Heatmap (PNG)
Spatial uncertainty distribution:
- Blue regions: Low uncertainty (confident)
- Red regions: High uncertainty (uncertain)

---

## 🎯 Decision Thresholds

### Confidence Threshold
```python
if confidence > 0.7:
    decision = "High confidence mine"
elif confidence > 0.5:
    decision = "Possible mine (investigate)"
else:
    decision = "Likely rock/clutter"
```

### Uncertainty Threshold
```python
if uncertainty < 0.1:
    reliability = "High reliability"
elif uncertainty < 0.2:
    reliability = "Moderate reliability"
else:
    reliability = "Low reliability (manual review)"
```

### Combined Decision
```python
if confidence > 0.7 and uncertainty < 0.1:
    action = "Confirmed mine - High priority"
elif confidence > 0.5 and uncertainty < 0.2:
    action = "Probable mine - Investigate"
elif uncertainty > 0.3:
    action = "Uncertain - Manual review required"
else:
    action = "Likely safe - Low priority"
```

---

**Continue to Part 4 for Critical Fix & Summary...**
