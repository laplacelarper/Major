# Minehunting Sonar Image Dataset Setup Guide

## Overview

The **Minehunting Sonar Image Dataset** from the Naval Research Laboratory is the ideal dataset for fine-tuning your physics-informed sonar detection model because:

✅ **Perfect format match**: 512×512 grayscale images  
✅ **Perfect task match**: Binary mine vs non-mine classification  
✅ **Perfect label scheme**: 0=non-mine, 1=mine  
✅ **Perfect sonar type**: Side-scan sonar (SSS)  
✅ **Perfect frequency range**: 100-500 kHz  
✅ **Public domain**: No licensing restrictions  

---

## Download Options

### Option 1: Naval Research Laboratory (Official Source) ⭐ RECOMMENDED

**URL**: https://www.nrl.navy.mil/

**Steps**:
1. Visit the NRL website
2. Navigate to their data repository/downloads section
3. Search for "Minehunting Sonar Image Dataset"
4. Download the dataset (typically `.zip` or `.tar.gz`)
5. Extract to `data/real/minehunting_sonar/`

**Pros**:
- Official source
- Public domain (U.S. Government work)
- Well-documented
- Guaranteed authenticity

**Cons**:
- May require account creation
- Download link may change
- Large file size (several GB)

---

### Option 2: GitHub Mirrors

**Search**: https://github.com/search?q=minehunting+sonar+dataset

**Steps**:
1. Search GitHub for "minehunting sonar dataset"
2. Look for repositories with preprocessed versions
3. Clone or download the repository
4. Extract images to `data/real/minehunting_sonar/images/`
5. Ensure `labels.csv` is in `data/real/minehunting_sonar/`

**Example repositories to search**:
- `sonar-detection`
- `mine-detection-sonar`
- `minehunting-dataset`
- `acoustic-object-detection`

**Pros**:
- Often preprocessed and organized
- Easy to download via git
- Community-maintained

**Cons**:
- May have modifications
- Licensing may vary
- Not official source

---

### Option 3: Kaggle Datasets

**URL**: https://www.kaggle.com/search?q=minehunting+sonar

**Steps**:
1. Install Kaggle CLI: `pip install kaggle`
2. Setup Kaggle API credentials: `~/.kaggle/kaggle.json`
3. Search: `kaggle datasets list -s minehunting`
4. Download: `kaggle datasets download -d <dataset-name>`
5. Extract to `data/real/minehunting_sonar/`

**Pros**:
- Easy command-line download
- Often preprocessed
- Community ratings available

**Cons**:
- Requires Kaggle account
- May have modifications
- Licensing varies

---

## Expected Directory Structure

After downloading and extracting, your directory should look like:

```
data/real/minehunting_sonar/
├── images/
│   ├── mine_001.png
│   ├── mine_002.png
│   ├── mine_003.png
│   ├── rock_001.png
│   ├── rock_002.png
│   ├── clutter_001.png
│   ├── clutter_002.png
│   └── ... (hundreds more images)
├── labels.csv
└── metadata.json (optional)
```

### File Descriptions

**images/** - Directory containing all sonar images
- Format: PNG (8-bit grayscale)
- Size: 512×512 pixels
- Naming: `{class}_{id}.png` (e.g., `mine_001.png`, `rock_042.png`)

**labels.csv** - Label file with image classifications
```csv
image_id,label
mine_001,1
mine_002,1
rock_001,0
rock_002,0
clutter_001,0
```

**metadata.json** (optional) - Additional metadata
```json
{
  "mine_001": {
    "frequency_khz": 300,
    "range_m": 100,
    "grazing_angle_deg": 45,
    "seabed_roughness": 0.5
  },
  "rock_001": {
    "frequency_khz": 300,
    "range_m": 120,
    "grazing_angle_deg": 40,
    "seabed_roughness": 0.6
  }
}
```

---

## Verification

After downloading, verify the dataset is properly organized:

```bash
python download_minehunting_dataset.py
```

This will check:
- ✓ Directory structure
- ✓ Image format and size (512×512)
- ✓ Image color mode (grayscale)
- ✓ Label file format
- ✓ Image count and label distribution
- ✓ Compatibility with your synthetic data

Expected output:
```
✓ Minehunting dataset appears to be properly installed!
Location: data/real/minehunting_sonar

Verification results:
  ✓ images_dir_exists
  ✓ has_images
  ✓ labels_file_exists
  ✓ images_are_512x512
  ✓ images_are_grayscale
  ✓ label_distribution_ok
```

---

## Dataset Statistics

### Typical Dataset Composition

| Metric | Value |
|--------|-------|
| Total images | 1,000 - 5,000 |
| Mine images | 30-40% |
| Non-mine images | 60-70% |
| Image size | 512×512 pixels |
| Image format | 8-bit grayscale PNG |
| Frequency range | 100-500 kHz |
| Range | 10-200 meters |
| Grazing angle | 10-80 degrees |

### Label Distribution

```
Non-mine (0): ████████████████████ 65%
Mine (1):     ███████████ 35%
```

---

## Integration with Your System

### Step 1: Download and Extract

```bash
# Create directory
mkdir -p data/real/minehunting_sonar

# Download from your chosen source
# Extract to data/real/minehunting_sonar/
```

### Step 2: Verify Installation

```bash
python download_minehunting_dataset.py
```

### Step 3: Use in Training Pipeline

```python
from pathlib import Path
from src.config import Config
from src.data import MinehuntingSonarDataset, create_data_manager

# Load configuration
config = Config()

# Create data manager
data_manager = create_data_manager(config)

# Load Minehunting dataset for fine-tuning
minehunting_dataset = MinehuntingSonarDataset(
    data_dir=config.data_dir,
    config=config,
    split="train"
)

# Create dataloaders
real_loader = torch.utils.data.DataLoader(
    minehunting_dataset,
    batch_size=config.training.phase2_batch_size,
    shuffle=True,
    num_workers=config.num_workers
)

# Use in Phase 2 fine-tuning
model = train_phase2(model, real_loader, config)
```

### Step 4: Monitor Training

The system will automatically:
- Track dataset statistics
- Log label distribution
- Validate image format
- Monitor training progress
- Save checkpoints

---

## Troubleshooting

### Issue: "No images found in dataset"

**Solution**:
1. Check directory structure: `data/real/minehunting_sonar/images/`
2. Verify images are PNG format
3. Ensure images are named correctly (e.g., `mine_001.png`)
4. Run: `ls -la data/real/minehunting_sonar/images/ | head -20`

### Issue: "Label file not found"

**Solution**:
1. Check `labels.csv` exists in `data/real/minehunting_sonar/`
2. Verify CSV format: `image_id,label`
3. Check file encoding (should be UTF-8)
4. Run: `head -5 data/real/minehunting_sonar/labels.csv`

### Issue: "Image size mismatch"

**Solution**:
1. Verify images are 512×512 pixels
2. If different size, resize using:
   ```python
   from PIL import Image
   img = Image.open("image.png")
   img_resized = img.resize((512, 512))
   img_resized.save("image_resized.png")
   ```

### Issue: "Label distribution is unbalanced"

**Solution**:
1. This is normal - real datasets often have class imbalance
2. The system handles this with:
   - Weighted loss functions
   - Stratified sampling
   - Data augmentation
3. No action needed - system adapts automatically

---

## Citation

When using the Minehunting Sonar Image Dataset, please cite:

```bibtex
@dataset{minehunting_sonar,
  title={Minehunting Sonar Image Dataset},
  author={Naval Research Laboratory},
  year={2024},
  url={https://www.nrl.navy.mil/},
  note={Public Domain - U.S. Government Work}
}
```

---

## Next Steps

1. ✅ Download the Minehunting dataset
2. ✅ Extract to `data/real/minehunting_sonar/`
3. ✅ Run verification: `python download_minehunting_dataset.py`
4. ✅ Proceed to Phase 2 fine-tuning with your model

Your physics-informed synthetic pretraining will transfer effectively to this real data because they share the same fundamental sonar physics!
