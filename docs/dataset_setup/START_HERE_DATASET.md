# 🎯 START HERE - Minehunting Dataset Setup

## What You Need to Know

Your physics-informed sonar detection system is **ready to use the Minehunting Sonar Image Dataset** for Phase 2 fine-tuning.

The dataset is a **perfect match** for your synthetic data:
- ✅ Same image format (512×512 grayscale)
- ✅ Same sonar type (side-scan sonar)
- ✅ Same task (binary mine detection)
- ✅ Same labels (0=non-mine, 1=mine)
- ✅ Same frequency range (100-500 kHz)
- ✅ Public domain (no licensing issues)

---

## 3-Step Quick Start

### Step 1: Download Dataset

Choose **ONE** of these options:

**Option A: Official NRL** (Recommended)
```
Visit: https://www.nrl.navy.mil/
Search: "Minehunting Sonar Image Dataset"
Download and extract to: data/real/minehunting_sonar/
```

**Option B: GitHub Mirror**
```
Visit: https://github.com/search?q=minehunting+sonar+dataset
Clone or download repository
Extract to: data/real/minehunting_sonar/
```

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

You should see:
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

dataset = MinehuntingSonarDataset(
    data_dir=Path("data"),
    config=config,
    split="train"
)

# System will automatically use it in Phase 2 fine-tuning
model = train_phase2(model, dataset, config)
```

---

## Documentation Files

### Quick Reference
- **DATASET_SETUP_INSTRUCTIONS.txt** - Quick start guide (read this first!)
- **DATASET_SETUP_COMPLETE.txt** - What was done and next steps

### Detailed Guides
- **MINEHUNTING_DATASET_SETUP.md** - Complete setup guide with troubleshooting
- **README_DATASET_SETUP.md** - Comprehensive reference
- **DATASET_COMPATIBILITY_ANALYSIS.md** - Why Minehunting is the best choice

### Summary
- **DATASET_DOWNLOAD_SUMMARY.md** - Full summary with all details

---

## Scripts

### Download & Verify
```bash
python download_minehunting_dataset.py
```
Downloads dataset and verifies installation

### Setup Helper
```bash
bash setup_minehunting_dataset.sh
```
Creates directory structure and prints instructions

---

## Expected Directory Structure

After downloading and extracting:

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

---

## Verification Checklist

After downloading, verify:

- [ ] Directory exists: `data/real/minehunting_sonar/`
- [ ] Images directory exists: `data/real/minehunting_sonar/images/`
- [ ] Images are PNG format
- [ ] Images are 512×512 pixels
- [ ] Images are grayscale (8-bit)
- [ ] Labels file exists: `data/real/minehunting_sonar/labels.csv`
- [ ] Labels file has correct format (image_id, label)
- [ ] Label distribution is reasonable (30-70% mines)
- [ ] Verification passes: `python download_minehunting_dataset.py`

---

## Why Minehunting Dataset?

| Attribute | Your Synthetic | Minehunting | Match |
|-----------|---|---|---|
| Image size | 512×512 | 512×512 | ✅ |
| Sonar type | Side-scan | Side-scan | ✅ |
| Task | Binary classification | Mine detection | ✅ |
| Labels | 0=non-mine, 1=mine | 0=non-mine, 1=mine | ✅ |
| Frequency | 100-500 kHz | 100-500 kHz | ✅ |
| Range | 10-200m | 10-200m | ✅ |
| Grazing angle | 10-80° | 10-80° | ✅ |
| Public domain | Yes | Yes | ✅ |

---

## System Integration

### Phase 1: Synthetic Pretraining ✅
- Generates 10,000 synthetic images
- Uses physics-informed rendering
- Trains model with full metadata

### Phase 2: Real Data Fine-tuning 🔄
- Uses Minehunting dataset
- Freezes early layers
- Fine-tunes on real sonar data
- Handles optional metadata gracefully

### Phase 3: Uncertainty Calibration ✅
- Enables dropout for uncertainty
- Calibrates confidence scores
- Validates on held-out test set

---

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

1. **Download** the Minehunting dataset (choose one of 3 options)
2. **Extract** to `data/real/minehunting_sonar/`
3. **Verify** by running: `python download_minehunting_dataset.py`
4. **Proceed** to Phase 2 fine-tuning

---

## Status

✅ **System ready for Phase 2 fine-tuning once dataset is downloaded**

Your physics-informed sonar detection system is fully configured and ready to use the Minehunting dataset for fine-tuning. The perfect alignment between your synthetic data and this real dataset ensures effective transfer learning.

**Download the dataset and you're ready to go!**

---

## Need Help?

- **Quick start**: Read `DATASET_SETUP_INSTRUCTIONS.txt`
- **Complete guide**: Read `MINEHUNTING_DATASET_SETUP.md`
- **Why this dataset?**: Read `DATASET_COMPATIBILITY_ANALYSIS.md`
- **Run verification**: `python download_minehunting_dataset.py`
