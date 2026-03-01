# Pipeline Integration Fixes Summary

## Issues Fixed

### 1. Missing Training Functions
**Problem**: `main.py` was trying to import `train_phase1`, `train_phase2`, and `train_phase3` functions that didn't exist.

**Solution**: Added convenience wrapper functions to each phase training module:
- `src/training/phase1_synthetic.py` - Added `train_phase1()` function
- `src/training/phase2_finetuning.py` - Added `train_phase2()` function  
- `src/training/phase3_calibration.py` - Added `train_phase3()` function

These functions handle:
- Model creation using `create_model_from_config()`
- Data loader setup using `SonarDataManager`
- Trainer initialization
- Training execution

### 2. Incorrect Model Factory Import
**Problem**: Training functions were trying to import `create_model` which doesn't exist.

**Solution**: Changed to use `create_model_from_config()` which is the correct function in `src/models/factory.py`.

### 3. Incorrect Data Loader API
**Problem**: Training functions were calling `create_dataloaders()` as a standalone function with `synthetic_only` parameter.

**Solution**: Updated to use the correct API:
```python
data_manager = create_data_manager(config)
dataloaders = data_manager.create_dataloaders(
    data_dir=config.data_dir,
    phase='phase1',
    use_real_data=False,  # Instead of synthetic_only=True
    create_combined=False
)
train_loader = dataloaders['train']
val_loader = dataloaders['val']
```

### 4. Configuration Adjustments for Testing
**Changes made to `configs/default.yaml`**:
- Reduced `synthetic_dataset_size` from 10000 to 100 for quick testing
- Reduced training epochs:
  - `phase1_epochs`: 100 → 2
  - `phase2_epochs`: 50 → 2
  - `phase3_epochs`: 20 → 2

## Current Status

✅ All pipeline components load successfully:
- Config loading works
- Model creation works (UNet)
- Data manager initialization works
- Data loaders creation works (70 train, 15 val samples)

## Next Steps

1. Run full pipeline with `--synthetic_only` flag to test Phase 1 and Phase 3
2. Monitor for any runtime errors during training
3. Check checkpoint saving/loading
4. Verify evaluation metrics generation

## Test Command

```bash
python3 main.py --mode full_pipeline --synthetic_only --skip_data_generation
```

This will:
- Skip data generation (use existing 100 synthetic images)
- Train Phase 1 (2 epochs on synthetic data)
- Skip Phase 2 (real data fine-tuning)
- Train Phase 3 (2 epochs for uncertainty calibration)
- Run evaluation

## Files Modified

1. `src/training/phase1_synthetic.py` - Added `train_phase1()` wrapper
2. `src/training/phase2_finetuning.py` - Added `train_phase2()` wrapper
3. `src/training/phase3_calibration.py` - Added `train_phase3()` wrapper
4. `configs/default.yaml` - Reduced dataset size and epochs for testing
5. `src/physics/calculations.py` - Fixed intensity calculations (from previous fix)

## Known Warnings

- "Could not extract label from sonar_XXXXX" - Expected, labels are in JSON files not filenames
- These warnings don't affect training as labels are properly loaded from JSON metadata
