#!/usr/bin/env python3
"""
Uncertainty Estimation Demo - Shows Monte Carlo Dropout in action
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.renderer import SonarImageRenderer

class MCDropoutCNN(nn.Module):
    """CNN with Monte Carlo Dropout for uncertainty estimation"""
    def __init__(self, dropout_rate=0.3):
        super(MCDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)  # Dropout stays active!
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)  # Dropout stays active!
        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, num_samples=20):
        """
        Perform multiple forward passes with dropout enabled
        to estimate prediction uncertainty
        """
        self.train()  # Keep dropout active!
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self(x)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # Shape: (num_samples, batch, classes)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty

def demonstrate_uncertainty():
    """Show how uncertainty estimation works"""
    
    print("\n" + "="*70)
    print("  🎲 UNCERTAINTY ESTIMATION DEMO")
    print("  (Monte Carlo Dropout)")
    print("="*70)
    
    print("\n📖 What is Uncertainty Estimation?")
    print("   When a model makes a prediction, it should also tell you")
    print("   how confident it is. This is crucial for safety-critical")
    print("   applications like mine detection!")
    
    print("\n🎯 Monte Carlo Dropout Method:")
    print("   1. Keep dropout layers active during inference")
    print("   2. Run the same image through the model 20 times")
    print("   3. Each time, different neurons are randomly dropped")
    print("   4. Calculate mean prediction and variance")
    print("   5. High variance = high uncertainty")
    
    # Create model
    print("\n🧠 Creating model with MC Dropout...")
    model = MCDropoutCNN(dropout_rate=0.3)
    print("   ✓ Model created with 30% dropout rate")
    
    # Generate test images
    print("\n📸 Generating test scenarios...")
    renderer = SonarImageRenderer(image_size=(128, 128))
    
    scenarios = [
        {
            "name": "Clear Mine (High Confidence Expected)",
            "material": "metal",
            "noise": 0.05,
            "grazing": 45.0
        },
        {
            "name": "Noisy Mine (Medium Confidence Expected)",
            "material": "metal",
            "noise": 0.4,
            "grazing": 45.0
        },
        {
            "name": "Ambiguous Object (Low Confidence Expected)",
            "material": "rock",
            "noise": 0.3,
            "grazing": 70.0
        }
    ]
    
    test_images = []
    for scenario in scenarios:
        physics_params = {
            'grazing_angle_range': (scenario['grazing'], scenario['grazing']),
            'range_limits': (100.0, 100.0),
            'noise_level': scenario['noise'],
        }
        
        label = 1 if scenario['material'] == 'metal' else 0
        image, _, _ = renderer.render_sonar_image(
            physics_params=physics_params,
            object_positions=[(64, 64)],
            object_heights=[2.0],
            object_labels=[label]
        )
        
        if image.shape != (128, 128):
            import cv2
            image = cv2.resize(image, (128, 128))
        
        test_images.append(image)
    
    test_images = torch.FloatTensor(np.array(test_images)).unsqueeze(1)
    
    print("   ✓ Generated 3 test scenarios")
    
    # Perform uncertainty estimation
    print("\n🔮 Running Monte Carlo Dropout (20 forward passes)...")
    mean_preds, uncertainties = model.predict_with_uncertainty(test_images, num_samples=20)
    
    print("\n📊 Results:\n")
    
    for i, scenario in enumerate(scenarios):
        mine_prob = mean_preds[i, 1] * 100
        rock_prob = mean_preds[i, 0] * 100
        mine_uncertainty = uncertainties[i, 1] * 100
        rock_uncertainty = uncertainties[i, 0] * 100
        
        prediction = "Mine" if mine_prob > rock_prob else "Rock"
        confidence = max(mine_prob, rock_prob)
        uncertainty = mine_uncertainty if mine_prob > rock_prob else rock_uncertainty
        
        print(f"   {i+1}. {scenario['name']}")
        print(f"      Prediction: {prediction}")
        print(f"      Confidence: {confidence:.1f}%")
        print(f"      Uncertainty: ±{uncertainty:.1f}%")
        print(f"      Mine probability: {mine_prob:.1f}% (±{mine_uncertainty:.1f}%)")
        print(f"      Rock probability: {rock_prob:.1f}% (±{rock_uncertainty:.1f}%)")
        
        if uncertainty < 5:
            print(f"      → High confidence prediction ✓")
        elif uncertainty < 15:
            print(f"      → Medium confidence prediction ⚠️")
        else:
            print(f"      → Low confidence - needs human review ⚠️⚠️")
        print()
    
    # Visualize
    print("📈 Creating uncertainty visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (scenario, ax) in enumerate(zip(scenarios, axes)):
        mine_prob = mean_preds[i, 1] * 100
        rock_prob = mean_preds[i, 0] * 100
        mine_unc = uncertainties[i, 1] * 100
        rock_unc = uncertainties[i, 0] * 100
        
        # Bar plot with error bars
        categories = ['Rock', 'Mine']
        probs = [rock_prob, mine_prob]
        uncs = [rock_unc, mine_unc]
        
        bars = ax.bar(categories, probs, yerr=uncs, capsize=10, 
                     color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylim([0, 105])
        ax.set_ylabel('Probability (%)')
        ax.set_title(scenario['name'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, prob, unc in zip(bars, probs, uncs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + unc + 2,
                   f'{prob:.1f}%\n±{unc:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path("demo_outputs/ml_training/uncertainty_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    
    return mean_preds, uncertainties

def explain_uncertainty():
    """Explain what uncertainty means"""
    print("\n" + "="*70)
    print("  🎓 WHY UNCERTAINTY MATTERS")
    print("="*70)
    
    print("\n💡 Real-World Scenario:")
    print("   Imagine you're scanning the ocean floor for mines...")
    
    print("\n   Scenario A: Clear Mine Detection")
    print("   • Model says: 95% mine (±2%)")
    print("   • High confidence, low uncertainty")
    print("   → Safe to flag as potential mine ✓")
    
    print("\n   Scenario B: Noisy Image")
    print("   • Model says: 60% mine (±25%)")
    print("   • Low confidence, high uncertainty")
    print("   → Needs human expert review ⚠️")
    
    print("\n   Scenario C: Ambiguous Object")
    print("   • Model says: 52% rock (±30%)")
    print("   • Very uncertain")
    print("   → Definitely needs closer inspection ⚠️⚠️")
    
    print("\n🎯 Benefits of Uncertainty Estimation:")
    print("   ✓ Know when to trust the model")
    print("   ✓ Know when to ask for human review")
    print("   ✓ Reduce false alarms")
    print("   ✓ Increase safety in critical applications")
    print("   ✓ Better decision-making")
    
    print("\n🔬 How Your System Does It:")
    print("   • Uses Monte Carlo Dropout")
    print("   • Runs 20 forward passes per image")
    print("   • Calculates mean and variance")
    print("   • Provides calibrated confidence scores")
    print("   • Phase 3 of training calibrates these scores")
    
    print("\n" + "="*70)
    print("  ✨ UNCERTAINTY = KNOWING WHAT YOU DON'T KNOW ✨")
    print("="*70)
    print("\n")

if __name__ == "__main__":
    try:
        mean_preds, uncertainties = demonstrate_uncertainty()
        explain_uncertainty()
        
        print("📁 Generated files:")
        print("   • demo_outputs/ml_training/uncertainty_demo.png")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
