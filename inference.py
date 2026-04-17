import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from src.dataset import SatelliteDataset
from src.model import MultiModalUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path="best_model.pth"):
    model = MultiModalUNet(n_channels=5, n_classes=3).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded: {path}")
    else:
        print(f"Model not found at {path}")
        return None
    model.eval()
    return model

def predict_random_sample():
    # Setup Dataset (Reuse the class for easy loading/augmenting)
    data_dir = os.path.join(os.getcwd(), 'data')
    ds = SatelliteDataset(data_dir)
    
    # Pick random index
    idx = random.randint(0, len(ds)-1)
    sample = ds[idx]
    
    sar = sample['sar'].unsqueeze(0).to(DEVICE)       # Add batch dim -> [1, 2, H, W]
    cloudy = sample['cloudy_optical'].unsqueeze(0).to(DEVICE) # [1, 3, H, W]
    truth = sample['ground_truth'].unsqueeze(0).to(DEVICE)
    filename = sample['filename']
    
    print(f"Running inference on: {filename}")
    
    # Run Model
    model = load_model()
    if model is None: return

    with torch.no_grad():
        restored = model(sar, cloudy)
        restored = torch.clamp(restored, 0, 1)

    # Prepare for Plotting (Tensor -> Numpy)
    sar_np = sar[0].cpu().permute(1, 2, 0).numpy()
    cloudy_np = cloudy[0].cpu().permute(1, 2, 0).numpy()
    restored_np = restored[0].cpu().permute(1, 2, 0).numpy()
    truth_np = truth[0].cpu().permute(1, 2, 0).numpy()

    # Visualize SAR (pad 3rd channel)
    sar_vis = np.concatenate([sar_np, np.zeros((sar_np.shape[0], sar_np.shape[1], 1))], axis=2)

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(sar_vis)
    ax[0].set_title("Input 1: SAR (Radar)")
    ax[0].axis('off')
    
    ax[1].imshow(cloudy_np)
    ax[1].set_title("Input 2: Cloudy Optical")
    ax[1].axis('off')
    
    ax[2].imshow(restored_np)
    ax[2].set_title("Result: Restored Image")
    ax[2].axis('off')
    
    ax[3].imshow(truth_np)
    ax[3].set_title("Ground Truth (Target)")
    ax[3].axis('off')

    plt.suptitle(f"Restoration Result: {filename}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_random_sample()
