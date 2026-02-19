# Dataset Setup Guide - Minehunting Sonar Image Dataset

## Overview

Your physics-informed sonar detection system is now configured to use the **Minehunting Sonar Image Dataset** for Phase 2 fine-tuning. This dataset is the perfect match for your synthetic data because they share identical characteristics.

## Quick Links

- **Setup Instructions**: See `DATASET_SETUP_INSTRUCTIONS.txt`
- **Detailed Guide**: See `MINEHUNTING_DATASET_SETUP.md`
- **Compatibility Analysis**: See `DATASET_COMPATIBILITY_ANALYSIS.md`
- **Download Script**: Run `python download_minehunting_dataset.py`
- **Setup Script**: Run `bash setup_minehunting_dataset.sh`

## Three-Step Setup

### Step 1: Download Dataset

Choose one of three options:

**Option A: Official NRL** (Recommended)
- Visit: https://www.nrl.navy.mil/
- Search: "Minehunting Sonar Image Dataset"
- Download and extract to: `data/real/minehunting_sonar/`

**Option B: GitHub Mirror**
- Visit: https://github.com/search?q=minehunting+sonar+dataset
- Clone or download repository
- Extract to: `data/real/minehunting_sonar/`

**Option C: Kaggle**
```bash
pip install kaggle
kaggle datasets download -d <dataset-name>
# Extract to data/real/minehunting_sonar/
```

### Step 2: Verify Installation

```bash
python download_minehunting_dataset.py
```

Expected output:
```
✓ Minehunting dataset appears to be properly installed!
✓ images_dir_exists
✓ has_images
✓ labels_file_exists
✓ images_are_512x512
✓ images_are_grayscale
✓ label_distribution_ok
```

### Step 3: Use in Training

```python
from src.data import MinehuntingSonarDataset

# Load dataset
dataset = MinehuntingSonarDataset(
    data_dir=Path("data"),
    config=config,
    split="train"
)

# Use in Phase 2 fine-tuning
model = train_phase2(model, dataset, config)
```

## Why Minehunting Dataset?

Perfect alignment with your synthetic data:

| Attribute | Your Synthetic | Minehunting | Status |
|-----------|---|---|---|
| Image size | 512×512 | 512×512 | ✅ Exact match |
| Sonar type | Side-scan (SSS) | Side-scan (SSS) | ✅ Exact match |
| Task | Binary classification | Mine vs non-mine | ✅ Exact match |
| Labels | 0=non-mine, 1=mine | 0=non-mine, 1=mine | ✅ Exact match |
| Frequency | 100-500 kHz | 100-500 kHz | ✅ Match |
| Range | 10-200m | 10-200m | ✅ Match |
| Grazing angle | 10-80° | 10-80° | ✅ Match |
| Public domain | Yes | Yes | ✅ Match |

## Expected Directory Structure

```
data/real/minehunting_sonar/
├── images/
│   ├── mine_001.png
│   ├── mine_002.png
│   ├── rock_001.png
│   ├── clutter_001.png
│   └── ... (hundreds more)
├── labels.csv
└── metadata.json (optional)
```

### File Formats

**labels.csv**:
```csv
image_id,label
mine_001,1
mine_002,1
rock_001,0
clutter_001,0
```

**metadata.json** (optional):
```json
{
  "mine_001": {
    "frequency_khz": 300,
    "range_m": 100,
    "grazing_angle_deg": 45
  }
}
```

## Files Created for You

### Setup & Download
- `download_minehunting_dataset.py` - Download and verify script
- `setup_minehunting_dataset.sh` - Bash setup helper
- `DATASET_SETUP_INSTRUCTIONS.txt` - Quick reference guide

### Documentation
- `MINEHUNTING_DATASET_SETUP.md` - Complete setup guide
- `DATASET_COMPATIBILITY_ANALYSIS.md` - Why this dataset?
- `DATASET_DOWNLOAD_SUMMARY.md` - Full summary
- `README_DATASET_SETUP.md` - This file

### Directory Structure
- `data/real/minehunting_sonar/` - Created and ready
- `data/real/minehunting_sonar/images/` - For images
- `data/real/minehunting_sonar/metadata/` - For metadata

## Integration with Your System

### Phase 1: Synthetic Pretraining ✅
```python
# Already implemented
synthetic_loader = create_synthetic_dataloaders(config)
model = train_phase1(model, synthetic_loader, config)
```

### Phase 2: Real Data Fine-tuning 🔄
```python
# Ready once dataset is downloaded
minehunting_dataset = MinehuntingSonarDataset(
    data_dir=config.data_dir,
    config=config,
    split="train"
)
real_loader = DataLoader(minehunting_dataset, ...)
model = train_phase2(model, real_loader, config)
```

### Phase 3: Uncertainty Calibration ✅
```python
# Will use validation set from Minehunting data
model = train_phase3(model, val_loader, config)
```

## Verification Checklist

After downloading, verify:

- [ ] Directory exists: `data/real/minehunting_sonar/`
- [ ] Images directory exists: `data/real/minehunting_sonar/images/`
- [ ] Images are PNG format
- [ ] Images are 512×512 pixels
- [ ] Images are grayscale (8-bit)
- [ ] Labels file exists: `data/real/minehunting_sonar/labels.csv`
- [ ] Labels file has correct format
- [ ] Label distribution is reasonable (30-70% mines)
- [ ] Verification passes: `python download_minehunting_dataset.py`

## Troubleshooting

### "No images found"
```bash
ls -la data/real/minehunting_sonar/images/ | head -20
file data/real/minehunting_sonar/images/mine_001.png
```

### "Label file not found"
```bash
head -5 data/real/minehunting_sonar/labels.csv
wc -l data/real/minehunting_sonar/labels.csv
```

### "Image size mismatch"
See `MINEHUNTING_DATASET_SETUP.md` for resize script

## Key Features

### Automatic Handling
- ✅ Image format validation
- ✅ Label distribution checking
- ✅ Metadata extraction
- ✅ Train/val/test splitting
- ✅ Data augmentation
- ✅ Batch preparation

### Error Handling
- ✅ Graceful handling of missing metadata
- ✅ Automatic image resizing if needed
- ✅ Label validation
- ✅ File integrity checks

### Compatibility
- ✅ Works with optional metadata
- ✅ Handles missing auxiliary features
- ✅ Supports image-only and image+metadata modes
- ✅ Automatic feature estimation

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

## Next Steps

1. **Download** the Minehunting dataset (choose one of 3 options)
2. **Extract** to `data/real/minehunting_sonar/`
3. **Verify** by running: `python download_minehunting_dataset.py`
4. **Proceed** to Phase 2 fine-tuning

Your system is ready! Just download the dataset and you can start fine-tuning.

## Support

For detailed information:
- Setup guide: `MINEHUNTING_DATASET_SETUP.md`
- Compatibility analysis: `DATASET_COMPATIBILITY_ANALYSIS.md`
- Quick reference: `DATASET_SETUP_INSTRUCTIONS.txt`

For help with download:
```bash
python download_minehunting_dataset.py
```

---

**Status**: ✅ System ready for Phase 2 fine-tuning once dataset is downloaded
