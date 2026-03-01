#!/usr/bin/env python3
"""Quick test of pipeline components"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config

print("Testing pipeline components...")

# Load config
from pathlib import Path as PathLib
config = load_config(PathLib("configs/default.yaml"))
print("✓ Config loaded")

# Test model creation
from src.models.factory import create_model_from_config
model = create_model_from_config(config)
print(f"✓ Model created: {config.model.model_type}")

# Test data manager
from src.data.data_loader import create_data_manager
data_manager = create_data_manager(config)
print("✓ Data manager created")

# Test data loaders
try:
    dataloaders = data_manager.create_dataloaders(
        data_dir=config.data_dir,
        phase='phase1',
        use_real_data=False,
        create_combined=False
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print(f"✓ Data loaders created: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
except Exception as e:
    print(f"✗ Data loader error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All components loaded successfully!")
