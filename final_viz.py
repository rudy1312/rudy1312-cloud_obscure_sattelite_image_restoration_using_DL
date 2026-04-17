import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from src.dataset import SatelliteDataset
from src.model import MultiModalUNet
from src.metrics import calculate_psnr, calculate_ssim

# ==========================================
# CONFIGURATION
# ==========================================
# MODEL_PATH = "final_model_refined.pth"
MODEL_PATH = "best_model.pth"
OUTPUT_FILE = "poster_hero_image.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you know specific indices for good examples of these categories, put them here.
# Otherwise, we pick random ones.
# Example: manual_indices = [12, 450, 899]
MANUAL_INDICES = None 

# Labels for the rows
ROW_LABELS = ["Urban / City", "Rural / Landscape", "Water / Coastal"]

# ==========================================

def load_model(path):
    print(f"Loading model from {path}...")
    model = MultiModalUNet(n_channels=5, n_classes=3).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model not found at {path}")
    model.eval()
    return model

def make_hero_image():
    # 1. Load Data
    data_dir = os.path.join(os.getcwd(), 'data')
    # Use thicker clouds for visualization as requested!
    dataset = SatelliteDataset(data_dir, cloud_min=0.8, cloud_max=1.0)
    print(f"Dataset size: {len(dataset)} (Using Thicker Clouds: 0.8-1.0)")
    
    # 2. Select Samples
    if MANUAL_INDICES and len(MANUAL_INDICES) == 3:
        indices = MANUAL_INDICES
    else:
        # Pick 3 random distinct samples
        # Ideally we would scan for variety, but random is our best bet without labels.
        # We assume the dataset is shuffled or diverse enough.
        indices = random.sample(range(len(dataset)), 3)
    
    print(f"Selected indices: {indices}")

    # 3. Load Model
    model = load_model(MODEL_PATH)
    
    # 4. Setup Plot
    # 3 Rows, 4 Columns
    # Columns: [Cloudy Input | SAR Hint | Refined Prediction | Ground Truth]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    
    # Column Headers
    cols = ["Cloudy Optical (Input)", "SAR (Input)", "Refined Prediction (Ours)", "Ground Truth (Target)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)

    # 5. Process Each Sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        sar = sample['sar'].unsqueeze(0).to(DEVICE)
        cloudy = sample['cloudy_optical'].unsqueeze(0).to(DEVICE)
        truth = sample['ground_truth'].unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            prediction = model(sar, cloudy)
            prediction = torch.clamp(prediction, 0, 1)
            
        # Metrics
        psnr = calculate_psnr(prediction, truth).item()
        ssim = calculate_ssim(prediction, truth).item()
        
        # Prepare Images for Plotting
        # SAR visualization: Add dummy channel to make it RGB-like (H, W, 3)
        sar_np = sar[0].cpu().permute(1, 2, 0).numpy() #(H, W, 2)
        sar_vis = np.zeros((sar_np.shape[0], sar_np.shape[1], 3))
        sar_vis[:,:,0] = sar_np[:,:,0] # R = VV
        sar_vis[:,:,1] = sar_np[:,:,1] if sar_np.shape[2] > 1 else sar_np[:,:,0] # G = VH
        sar_vis[:,:,2] = sar_np[:,:,1] if sar_np.shape[2] > 1 else sar_np[:,:,0] # B = VH
        # Normalize for display if needed, but tensor is already 0-1
        
        cloudy_np = cloudy[0].cpu().permute(1, 2, 0).numpy()
        pred_np = prediction[0].cpu().permute(1, 2, 0).numpy()
        truth_np = truth[0].cpu().permute(1, 2, 0).numpy()
        
        # Plotting
        row_axes = axes[i]
        
        # -- Cloudy --
        row_axes[0].imshow(cloudy_np)
        row_axes[0].set_ylabel(ROW_LABELS[i], fontsize=12, fontweight='bold', labelpad=10)
        
        # -- SAR --
        row_axes[1].imshow(sar_vis)
        
        # -- Prediction --
        row_axes[2].imshow(pred_np)
        # Add Metrics Overlay text to the bottom of the Prediction image
        # Using text box properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        text_str = f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}"
        row_axes[2].text(0.05, 0.05, text_str, transform=row_axes[2].transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props, color='black')
        
        # -- Ground Truth --
        row_axes[3].imshow(truth_np)
        
        # Clean axes
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)

    # 6. Save
    print(f"Saving high-res poster image to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done!")

if __name__ == "__main__":
    make_hero_image()
