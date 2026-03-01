#!/usr/bin/env python3
"""Analyze the real dataset structure"""

import pandas as pd
import sys
from pathlib import Path

def analyze_datasets():
    print("\n" + "="*70)
    print("  REAL DATASET ANALYSIS")
    print("="*70)
    
    # Dataset 1: sonar.csv (UCI Sonar dataset)
    print("\n📊 Dataset 1: sonar.csv (Feature-based)")
    print("-"*70)
    try:
        df_sonar = pd.read_csv('data/real/minehunting_sonar/sonar.csv', header=None)
        print(f"✓ Loaded successfully")
        print(f"  Shape: {df_sonar.shape}")
        print(f"  Samples: {df_sonar.shape[0]}")
        print(f"  Features: {df_sonar.shape[1] - 1} (+ 1 label column)")
        print(f"\n  Label distribution:")
        labels = df_sonar.iloc[:, -1].value_counts()
        for label, count in labels.items():
            label_name = "Mine" if label == 'M' else "Rock"
            print(f"    {label} ({label_name}): {count} samples ({count/len(df_sonar)*100:.1f}%)")
        
        print(f"\n  Data type: Numerical features (sonar returns)")
        print(f"  Format: 60 features per sample")
        print(f"  Use case: Traditional ML (not image-based)")
        
    except Exception as e:
        print(f"✗ Error loading: {e}")
    
    # Dataset 2: sonar_labels.csv (Image metadata)
    print("\n📊 Dataset 2: sonar_labels.csv (Image metadata)")
    print("-"*70)
    try:
        df_labels = pd.read_csv('data/real/minehunting_sonar/sonar_labels.csv')
        print(f"✓ Loaded successfully")
        print(f"  Total entries: {len(df_labels)}")
        print(f"  Image size: {df_labels['image_width'].iloc[0]}x{df_labels['image_height'].iloc[0]}")
        print(f"  Image type: {df_labels['image_type'].iloc[0]}")
        
        print(f"\n  Label distribution:")
        labels = df_labels['label'].value_counts().sort_index()
        for label, count in labels.items():
            label_name = "Non-Mine (NOMBO)" if label == 0 else "Mine (MILCO)"
            print(f"    {label} ({label_name}): {count} images ({count/len(df_labels)*100:.1f}%)")
        
        # Check if images exist
        images_dir = Path('data/real/minehunting_sonar/images')
        if images_dir.exists():
            image_files = list(images_dir.glob('*.png'))
            print(f"\n  Images directory: {images_dir}")
            print(f"  Images found: {len(image_files)}")
            if len(image_files) == 0:
                print(f"  ⚠️  WARNING: No image files found!")
                print(f"     Expected {len(df_labels)} images")
        else:
            print(f"\n  ⚠️  WARNING: Images directory not found!")
        
    except Exception as e:
        print(f"✗ Error loading: {e}")
    
    # Recommendations
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    print("\n🎯 Option 1: Use sonar.csv (Feature-based)")
    print("   • 208 samples with 60 numerical features")
    print("   • Can train traditional ML models (SVM, Random Forest)")
    print("   • NOT suitable for CNN/image-based models")
    print("   • Quick to train, good for testing")
    
    print("\n🎯 Option 2: Use synthetic data only")
    print("   • Generate unlimited synthetic sonar images")
    print("   • Train CNN models (U-Net, ResNet, EfficientNet)")
    print("   • No real data needed")
    print("   • Recommended for your physics-informed approach")
    
    print("\n🎯 Option 3: Get real sonar images")
    print("   • sonar_labels.csv has 1170 image references")
    print("   • But actual image files are missing")
    print("   • Would need to download/obtain the images")
    print("   • Then can use for Phase 2 fine-tuning")
    
    print("\n" + "="*70)
    print("  CURRENT STATUS")
    print("="*70)
    
    print("\n✅ Available:")
    print("   • sonar.csv: 208 feature-based samples")
    print("   • sonar_labels.csv: 1170 image metadata entries")
    print("   • Synthetic data generation: Unlimited")
    
    print("\n⚠️  Missing:")
    print("   • Actual sonar image files (*.png)")
    print("   • Expected location: data/real/minehunting_sonar/images/")
    
    print("\n💡 Recommended Action:")
    print("   Use synthetic-only training:")
    print("   $ python main.py --mode full_pipeline --synthetic_only")
    
    print("\n")

if __name__ == "__main__":
    analyze_datasets()
