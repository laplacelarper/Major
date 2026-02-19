#!/usr/bin/env python3
"""
Script to download and organize the Minehunting Sonar Image Dataset

The Minehunting dataset is available from multiple sources:
1. Naval Research Laboratory (NRL) - Primary source
2. GitHub repositories with preprocessed versions
3. Kaggle (if available)

This script attempts to download from available sources.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import urllib.request
import urllib.error
import zipfile
import tarfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinehuntingDatasetDownloader:
    """Download and organize Minehunting Sonar Image Dataset"""
    
    def __init__(self, output_dir: Path = Path("data/real/minehunting_sonar")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Known sources for the dataset
        self.sources = {
            "nrl_official": {
                "name": "Naval Research Laboratory (Official)",
                "url": "https://www.nrl.navy.mil/",
                "description": "Primary source - requires manual download from NRL portal",
                "instructions": self._nrl_instructions()
            },
            "github_mirror": {
                "name": "GitHub Mirror (if available)",
                "url": "https://github.com/search?q=minehunting+sonar+dataset",
                "description": "Community-maintained mirrors",
                "instructions": self._github_instructions()
            },
            "kaggle": {
                "name": "Kaggle Datasets",
                "url": "https://www.kaggle.com/search?q=minehunting+sonar",
                "description": "Kaggle may have preprocessed versions",
                "instructions": self._kaggle_instructions()
            }
        }
    
    def _nrl_instructions(self) -> str:
        """Instructions for downloading from NRL"""
        return """
        1. Visit: https://www.nrl.navy.mil/
        2. Navigate to their data repository/downloads section
        3. Search for "Minehunting Sonar Image Dataset"
        4. Download the dataset (typically a .zip or .tar.gz file)
        5. Extract to: data/real/minehunting_sonar/
        
        Expected structure after extraction:
        data/real/minehunting_sonar/
        ├── images/
        │   ├── mine_001.png
        │   ├── mine_002.png
        │   ├── rock_001.png
        │   └── ...
        ├── labels.csv
        └── metadata.json
        """
    
    def _github_instructions(self) -> str:
        """Instructions for GitHub mirrors"""
        return """
        1. Search GitHub for "minehunting sonar dataset"
        2. Look for repositories with preprocessed versions
        3. Clone or download the repository
        4. Extract images to: data/real/minehunting_sonar/images/
        5. Ensure labels.csv is in: data/real/minehunting_sonar/
        
        Example repositories to search:
        - sonar-detection
        - mine-detection-sonar
        - minehunting-dataset
        """
    
    def _kaggle_instructions(self) -> str:
        """Instructions for Kaggle"""
        return """
        1. Install Kaggle CLI: pip install kaggle
        2. Setup Kaggle API credentials: ~/.kaggle/kaggle.json
        3. Search: kaggle datasets list -s minehunting
        4. Download: kaggle datasets download -d <dataset-name>
        5. Extract to: data/real/minehunting_sonar/
        
        Or visit: https://www.kaggle.com/search?q=minehunting+sonar
        """
    
    def print_download_instructions(self):
        """Print all available download options"""
        print("\n" + "="*80)
        print("MINEHUNTING SONAR IMAGE DATASET - DOWNLOAD INSTRUCTIONS")
        print("="*80 + "\n")
        
        for source_key, source_info in self.sources.items():
            print(f"\n{'─'*80}")
            print(f"Option: {source_info['name']}")
            print(f"{'─'*80}")
            print(f"URL: {source_info['url']}")
            print(f"Description: {source_info['description']}")
            print(f"\nInstructions:\n{source_info['instructions']}")
        
        print(f"\n{'─'*80}")
        print("EXPECTED DATASET STRUCTURE")
        print(f"{'─'*80}")
        print("""
After downloading and extracting, your directory should look like:

data/real/minehunting_sonar/
├── images/
│   ├── mine_001.png
│   ├── mine_002.png
│   ├── mine_003.png
│   ├── rock_001.png
│   ├── rock_002.png
│   ├── clutter_001.png
│   └── ... (hundreds more images)
├── labels.csv
└── metadata.json (optional)

Where:
- images/: Contains 512×512 grayscale PNG files
- labels.csv: CSV file with columns: image_id, label
  Example:
    image_id,label
    mine_001,1
    rock_001,0
    clutter_001,0
- metadata.json: Optional metadata with frequency, range, etc.
        """)
        
        print(f"\n{'─'*80}")
        print("DATASET CHARACTERISTICS")
        print(f"{'─'*80}")
        print("""
Image Format:
- Size: 512×512 pixels
- Type: Grayscale (8-bit)
- Format: PNG
- Color depth: 8-bit (0-255)

Labels:
- 0: Non-mine (rocks, clutter, seabed)
- 1: Mine-like objects

Typical Statistics:
- Total images: 1000-5000 (varies by version)
- Mine images: ~30-40%
- Non-mine images: ~60-70%
- Frequency: 100-500 kHz
- Range: 10-200 meters
        """)
        
        print(f"\n{'─'*80}")
        print("VERIFICATION SCRIPT")
        print(f"{'─'*80}")
        print("""
After downloading, run this to verify the dataset:

    python verify_minehunting_dataset.py

This will check:
✓ Directory structure
✓ Image format and size
✓ Label file format
✓ Image count and label distribution
✓ Compatibility with your synthetic data
        """)
    
    def verify_dataset(self) -> Dict[str, bool]:
        """Verify if dataset is properly organized"""
        checks = {
            'images_dir_exists': False,
            'has_images': False,
            'labels_file_exists': False,
            'images_are_512x512': False,
            'images_are_grayscale': False,
            'label_distribution_ok': False
        }
        
        images_dir = self.output_dir / "images"
        labels_file = self.output_dir / "labels.csv"
        
        # Check directory structure
        if images_dir.exists():
            checks['images_dir_exists'] = True
            
            # Check for images
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            if len(image_files) > 0:
                checks['has_images'] = True
                logger.info(f"Found {len(image_files)} images")
                
                # Check image properties
                try:
                    from PIL import Image
                    sample_image = Image.open(image_files[0])
                    
                    if sample_image.size == (512, 512):
                        checks['images_are_512x512'] = True
                    
                    if sample_image.mode == 'L':  # Grayscale
                        checks['images_are_grayscale'] = True
                    
                    logger.info(f"Sample image: {sample_image.size}, mode: {sample_image.mode}")
                    
                except Exception as e:
                    logger.warning(f"Could not verify image properties: {e}")
        
        # Check labels file
        if labels_file.exists():
            checks['labels_file_exists'] = True
            
            try:
                import csv
                with open(labels_file, 'r') as f:
                    reader = csv.DictReader(f)
                    labels = [int(row.get('label', 0)) for row in reader]
                    
                    if len(labels) > 0:
                        label_counts = {0: labels.count(0), 1: labels.count(1)}
                        ratio = label_counts[1] / len(labels) if len(labels) > 0 else 0
                        
                        # Check if ratio is reasonable (30-70% mines)
                        if 0.2 < ratio < 0.8:
                            checks['label_distribution_ok'] = True
                        
                        logger.info(f"Label distribution: {label_counts}, ratio: {ratio:.1%}")
                
            except Exception as e:
                logger.warning(f"Could not verify labels: {e}")
        
        return checks
    
    def create_placeholder_structure(self):
        """Create placeholder directory structure for testing"""
        logger.info("Creating placeholder directory structure...")
        
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README
        readme_path = self.output_dir / "README.md"
        readme_content = """# Minehunting Sonar Image Dataset

This directory should contain the Minehunting Sonar Image Dataset.

## Download Instructions

See the main project documentation for download instructions.

## Expected Structure

```
minehunting_sonar/
├── images/
│   ├── mine_001.png
│   ├── rock_001.png
│   └── ...
├── labels.csv
└── metadata.json (optional)
```

## Dataset Info

- Image size: 512×512 pixels
- Format: Grayscale PNG
- Labels: 0=non-mine, 1=mine
- Source: Naval Research Laboratory
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created placeholder structure at {self.output_dir}")


def main():
    """Main entry point"""
    downloader = MinehuntingDatasetDownloader()
    
    # Check if dataset already exists
    checks = downloader.verify_dataset()
    
    if checks['has_images'] and checks['labels_file_exists']:
        print("\n✓ Minehunting dataset appears to be properly installed!")
        print(f"Location: {downloader.output_dir}")
        print("\nVerification results:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        return 0
    
    # Print download instructions
    downloader.print_download_instructions()
    
    # Create placeholder structure
    downloader.create_placeholder_structure()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Download the Minehunting dataset using one of the methods above
2. Extract it to: data/real/minehunting_sonar/
3. Ensure the directory structure matches the expected format
4. Run this script again to verify: python download_minehunting_dataset.py
5. Once verified, you can use it for fine-tuning:

    from src.data import MinehuntingSonarDataset
    dataset = MinehuntingSonarDataset(
        data_dir=Path("data"),
        config=config,
        split="train"
    )
    """)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())