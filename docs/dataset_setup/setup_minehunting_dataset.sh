#!/bin/bash
# Setup script for Minehunting Sonar Image Dataset

set -e

echo "=========================================="
echo "Minehunting Dataset Setup"
echo "=========================================="
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/real/minehunting_sonar/images
mkdir -p data/real/minehunting_sonar/metadata

echo "✓ Created directories"
echo ""

# Check if dataset already exists
if [ -f "data/real/minehunting_sonar/labels.csv" ]; then
    echo "✓ Dataset appears to be already installed"
    echo "  Location: data/real/minehunting_sonar/"
    echo ""
    echo "Verifying dataset..."
    python download_minehunting_dataset.py
    exit 0
fi

# Print instructions
echo "=========================================="
echo "DATASET DOWNLOAD INSTRUCTIONS"
echo "=========================================="
echo ""
echo "The Minehunting Sonar Image Dataset needs to be downloaded manually."
echo ""
echo "Choose one of these options:"
echo ""
echo "Option 1: Naval Research Laboratory (Official)"
echo "  URL: https://www.nrl.navy.mil/"
echo "  - Visit the NRL website"
echo "  - Search for 'Minehunting Sonar Image Dataset'"
echo "  - Download and extract to: data/real/minehunting_sonar/"
echo ""
echo "Option 2: GitHub Mirrors"
echo "  URL: https://github.com/search?q=minehunting+sonar+dataset"
echo "  - Search for minehunting sonar repositories"
echo "  - Clone or download"
echo "  - Extract images to: data/real/minehunting_sonar/images/"
echo ""
echo "Option 3: Kaggle"
echo "  URL: https://www.kaggle.com/search?q=minehunting+sonar"
echo "  - Install: pip install kaggle"
echo "  - Setup credentials: ~/.kaggle/kaggle.json"
echo "  - Download and extract"
echo ""
echo "=========================================="
echo "EXPECTED STRUCTURE"
echo "=========================================="
echo ""
echo "After downloading, your directory should look like:"
echo ""
echo "data/real/minehunting_sonar/"
echo "├── images/"
echo "│   ├── mine_001.png"
echo "│   ├── mine_002.png"
echo "│   ├── rock_001.png"
echo "│   └── ..."
echo "├── labels.csv"
echo "└── metadata.json (optional)"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Download the dataset using one of the options above"
echo "2. Extract to: data/real/minehunting_sonar/"
echo "3. Run this script again to verify: bash setup_minehunting_dataset.sh"
echo "4. Once verified, you can use it for fine-tuning"
echo ""
echo "For more details, see: MINEHUNTING_DATASET_SETUP.md"
echo ""
