#!/usr/bin/env python3
"""Quick test of training loop"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.training.phase1_synthetic import train_phase1
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Testing training loop...")

# Load config
config = load_config(Path("configs/default.yaml"))
print("✓ Config loaded")

# Try to run one epoch of training
try:
    results = train_phase1(config, logger)
    print(f"✓ Training completed: {results}")
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback
    traceback.print_exc()
