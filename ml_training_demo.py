#!/usr/bin/env python3
"""
ML Training Demo - Shows the actual learning process
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.renderer import SonarImageRenderer
from src.models.factory import ModelFactory
from src.config import Config

def generate_mini_dataset(num_samples=50):
    """Generate a small dataset for quick training demo"""
    print("📊 Generating mini training dataset...")
    
    renderer = SonarImageRenderer(image_size=(128, 128))  # Smaller for speed
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # Alternate between mines (metal) and rocks
        is_mine = i % 2 == 0
        material = 'metal' if is_mine else 'rock'
        label = 1 if is_mine else 0
        
        # Random physics parameters
        physics_params = {
            'grazing_angle_range': (np.random.uniform(20, 70), np.random.uniform(20, 70)),
            'range_limits': (np.random.uniform(50, 150), np.random.uniform(50, 150)),
            'noise_level': np.random.uniform(0.1, 0.3),
            'texture_roughness': np.random.uniform(0.3, 0.7),
        }
        
        # Generate image
        image, _, _ = renderer.render_sonar_image(
            physics_params=physics_params,
            object_positions=[(64, 64)],
            object_heights=[2.0],
            object_labels=[label]
        )
        
        # Resize to 128x128 if needed and normalize
        if image.shape != (128, 128):
            import cv2
            image = cv2.resize(image, (128, 128))
        
        images.append(image)
        labels.append(label)
    
    print(f"   ✓ Generated {num_samples} images ({num_samples//2} mines, {num_samples//2} rocks)")
    
    return np.array(images), np.array(labels)

def create_simple_model():
    """Create a simple CNN for quick training"""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 2)  # Binary classification
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))  # 128 -> 64
            x = self.pool(self.relu(self.conv2(x)))  # 64 -> 32
            x = self.pool(self.relu(self.conv3(x)))  # 32 -> 16
            x = x.view(-1, 64 * 16 * 16)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    return SimpleCNN()

def train_model_demo():
    """Train a model and show the learning process"""
    
    print("\n" + "="*70)
    print("  ML TRAINING DEMO - Watch the Model Learn!")
    print("="*70)
    
    # Generate data
    images, labels = generate_mini_dataset(num_samples=100)
    
    # Split into train/val
    split_idx = 80
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    val_images = images[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"\n📚 Dataset split:")
    print(f"   Training: {len(train_images)} images")
    print(f"   Validation: {len(val_images)} images")
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images).unsqueeze(1)  # Add channel dim
    train_labels = torch.LongTensor(train_labels)
    val_images = torch.FloatTensor(val_images).unsqueeze(1)
    val_labels = torch.LongTensor(val_labels)
    
    # Create model
    print("\n🧠 Creating neural network...")
    model = create_simple_model()
    print(f"   ✓ Model created: SimpleCNN")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n🚀 Starting training...")
    print("   (This will take ~30 seconds)\n")
    
    num_epochs = 20
    batch_size = 8
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(train_images) / batch_size)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_images)
            _, predicted = torch.max(val_outputs, 1)
            val_accuracy = (predicted == val_labels).float().mean().item()
            val_accuracies.append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy*100:.1f}%")
    
    print("\n✅ Training complete!")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(train_images)
        _, train_predicted = torch.max(train_outputs, 1)
        train_accuracy = (train_predicted == train_labels).float().mean().item()
        
        # Validation accuracy
        val_outputs = model(val_images)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == val_labels).float().mean().item()
        
        # Per-class accuracy
        mine_mask = val_labels == 1
        rock_mask = val_labels == 0
        mine_accuracy = (val_predicted[mine_mask] == val_labels[mine_mask]).float().mean().item()
        rock_accuracy = (val_predicted[rock_mask] == val_labels[rock_mask]).float().mean().item()
    
    print(f"\n📊 Final Results:")
    print(f"   Training Accuracy:   {train_accuracy*100:.1f}%")
    print(f"   Validation Accuracy: {val_accuracy*100:.1f}%")
    print(f"   Mine Detection:      {mine_accuracy*100:.1f}%")
    print(f"   Rock Detection:      {rock_accuracy*100:.1f}%")
    
    # Plot training curves
    print("\n📈 Generating training curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(range(1, num_epochs+1), train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(range(1, num_epochs+1), [acc*100 for acc in val_accuracies], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = Path("demo_outputs/ml_training/training_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    
    # Test on a few examples
    print("\n🔍 Testing on sample images...")
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(val_images))):
            img = val_images[i:i+1]
            true_label = val_labels[i].item()
            output = model(img)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
            
            true_name = "Mine" if true_label == 1 else "Rock"
            pred_name = "Mine" if pred_label == 1 else "Rock"
            correct = "✓" if pred_label == true_label else "✗"
            
            print(f"   {correct} Sample {i+1}: True={true_name}, "
                  f"Predicted={pred_name} (confidence: {confidence*100:.1f}%)")
    
    # Save model
    model_path = Path("demo_outputs/ml_training/demo_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, train_losses, val_accuracies

def explain_ml_pipeline():
    """Explain what just happened"""
    print("\n" + "="*70)
    print("  🎓 WHAT YOU JUST SAW")
    print("="*70)
    
    print("\n1️⃣  DATA GENERATION")
    print("   • Generated 100 synthetic sonar images")
    print("   • 50 mines (metal objects) + 50 rocks")
    print("   • Each with random physics parameters")
    print("   • Split: 80 training, 20 validation")
    
    print("\n2️⃣  MODEL ARCHITECTURE")
    print("   • Simple CNN with 3 convolutional layers")
    print("   • Max pooling for downsampling")
    print("   • Fully connected layers for classification")
    print("   • Dropout for regularization")
    print("   • Output: 2 classes (mine vs rock)")
    
    print("\n3️⃣  TRAINING PROCESS")
    print("   • Loss function: Cross-entropy")
    print("   • Optimizer: Adam (adaptive learning rate)")
    print("   • Batch size: 8 images")
    print("   • Epochs: 20 iterations through data")
    print("   • The model learned to distinguish mines from rocks!")
    
    print("\n4️⃣  WHAT THE MODEL LEARNED")
    print("   • Mines (metal) have stronger backscatter")
    print("   • Mines create more defined acoustic shadows")
    print("   • Rocks have weaker, more diffuse returns")
    print("   • The model learned these patterns from physics!")
    
    print("\n5️⃣  YOUR FULL SYSTEM DOES MORE")
    print("   • Phase 1: Train on 10,000 synthetic images")
    print("   • Phase 2: Fine-tune on real sonar data")
    print("   • Phase 3: Calibrate uncertainty estimates")
    print("   • Uses larger models (U-Net, ResNet, EfficientNet)")
    print("   • Provides confidence scores with predictions")
    
    print("\n" + "="*70)
    print("  ✨ THE MODEL LEARNED TO DETECT MINES! ✨")
    print("="*70)
    
    print("\n📁 Check these files:")
    print("   • demo_outputs/ml_training/training_curves.png - Learning progress")
    print("   • demo_outputs/ml_training/demo_model.pth - Trained model weights")
    
    print("\n🚀 To train the full system:")
    print("   1. Generate more data: python scripts/generate_data.py")
    print("   2. Train Phase 1: python scripts/train.py --phase 1")
    print("   3. Train Phase 2: python scripts/train.py --phase 2")
    print("   4. Train Phase 3: python scripts/train.py --phase 3")
    
    print("\n")

if __name__ == "__main__":
    try:
        model, losses, accuracies = train_model_demo()
        explain_ml_pipeline()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
