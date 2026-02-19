"""Generate demonstration data for project review"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import json

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def create_demo_directories():
    """Create directories for demo outputs"""
    demo_dir = Path("demo_outputs")
    demo_dir.mkdir(exist_ok=True)
    
    (demo_dir / "synthetic_data").mkdir(exist_ok=True)
    (demo_dir / "uncertainty_analysis").mkdir(exist_ok=True)
    
    return demo_dir

def generate_synthetic_samples():
    """Generate synthetic sonar data samples"""
    print("🔧 Generating synthetic sonar data samples...")
    
    from physics.renderer import SonarImageRenderer
    
    renderer = SonarImageRenderer((512, 512))
    
    demo_dir = create_demo_directories()
    samples_info = []
    
    # Generate 5 sample images
    for i in range(5):
        # Generate random physics parameters
        physics_params = {
            'range_m': np.random.uniform(20, 150),
            'grazing_angle_deg': np.random.uniform(15, 75),
            'frequency_khz': np.random.uniform(100, 400),
            'beam_width_deg': np.random.uniform(1, 4),
            'cosine_exponent': np.random.uniform(2, 6),
            'base_intensity': np.random.uniform(0.4, 0.8),
            'noise_level': np.random.uniform(0.1, 0.3)
        }
        
        # Render image
        image = renderer.render_sonar_image(physics_params)
        
        # Save image
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title(f'Synthetic Sonar Image {i+1}\nRange: {physics_params["range_m"]:.1f}m, Angle: {physics_params["grazing_angle_deg"]:.1f}°')
        plt.colorbar(label='Intensity')
        plt.axis('off')
        
        image_path = demo_dir / "synthetic_data" / f"sonar_sample_{i+1}.png"
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store metadata
        sample_info = {
            "image_id": i+1,
            "filename": f"sonar_sample_{i+1}.png",
            "physics_parameters": physics_params
        }
        samples_info.append(sample_info)
        print(f"  ✓ Generated sample {i+1}/5")
    
    # Save metadata
    with open(demo_dir / "synthetic_data" / "metadata.json", 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    print(f"✅ Synthetic data saved to: {demo_dir / 'synthetic_data'}")
    return samples_info

def demonstrate_uncertainty():
    """Demonstrate uncertainty estimation"""
    print("\n🧠 Demonstrating uncertainty estimation...")
    
    from models.unet import UNet
    from models.uncertainty import UncertaintyEstimator
    from physics.renderer import SonarImageRenderer
    
    # Create model
    model = UNet(num_classes=2, dropout_rate=0.2, use_physics_metadata=False)
    uncertainty_estimator = UncertaintyEstimator(model, num_samples=5)
    
    # Generate test image
    renderer = SonarImageRenderer((512, 512))
    physics_params = {
        'range_m': 75.0,
        'grazing_angle_deg': 45.0,
        'frequency_khz': 250.0,
        'beam_width_deg': 2.5,
        'cosine_exponent': 4.0,
        'base_intensity': 0.6,
        'noise_level': 0.2
    }
    
    image = renderer.render_sonar_image(physics_params)
    
    # Convert to tensor
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    # Get uncertainty
    mean_preds, uncertainty = uncertainty_estimator.predict_with_uncertainty(x)
    detailed = uncertainty_estimator.predict_with_detailed_uncertainty(x)
    
    demo_dir = Path("demo_outputs")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Synthetic Sonar Image')
    axes[0].axis('off')
    
    # Uncertainty info
    axes[1].axis('off')
    uncertainty_text = f"""
UNCERTAINTY ANALYSIS

Total Uncertainty: {uncertainty.item():.4f}
Epistemic (Model): {detailed['epistemic_uncertainty'].item():.4f}
Aleatoric (Data): {detailed['aleatoric_uncertainty'].item():.4f}

Prediction Probabilities:
• Class 0 (Rock): {detailed['probabilities'][0,0].item():.3f}
• Class 1 (Mine): {detailed['probabilities'][0,1].item():.3f}

Physics Parameters:
• Range: {physics_params['range_m']:.1f} meters
• Grazing Angle: {physics_params['grazing_angle_deg']:.1f}°
• Frequency: {physics_params['frequency_khz']:.1f} kHz
• Noise Level: {physics_params['noise_level']:.3f}

Monte Carlo Samples: 5
Dropout Rate: 0.2
    """
    axes[1].text(0.1, 0.5, uncertainty_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(demo_dir / "uncertainty_analysis" / "uncertainty_demo.png", 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        "total_uncertainty": float(uncertainty.item()),
        "epistemic_uncertainty": float(detailed['epistemic_uncertainty'].item()),
        "aleatoric_uncertainty": float(detailed['aleatoric_uncertainty'].item()),
        "prediction_probabilities": detailed['probabilities'].numpy().tolist(),
        "physics_parameters": physics_params
    }
    
    with open(demo_dir / "uncertainty_analysis" / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Uncertainty analysis saved to: {demo_dir / 'uncertainty_analysis'}")
    return results

if __name__ == "__main__":
    print("🚀 Generating Demo Data for Project Review")
    print("=" * 50)
    
    try:
        samples = generate_synthetic_samples()
        uncertainty_results = demonstrate_uncertainty()
        
        print("\n" + "=" * 50)
        print("✅ DEMO DATA GENERATED SUCCESSFULLY!")
        print("=" * 50)
        print("\n📁 Files created in demo_outputs/:")
        print("   📂 synthetic_data/")
        print("      • 5 synthetic sonar images (PNG)")
        print("      • metadata.json (physics parameters)")
        print("   📂 uncertainty_analysis/")
        print("      • uncertainty_demo.png (visualization)")
        print("      • results.json (uncertainty metrics)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()