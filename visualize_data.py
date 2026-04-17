import torch
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dataset import SatelliteDataset

def visualize_dataset():
    data_dir = os.path.join(os.getcwd(), 'data')
    print(f"Looking for data in: {data_dir}")
    
    dataset = SatelliteDataset(data_dir)
    print(f"Found {len(dataset)} images.")
    
    if len(dataset) == 0:
        print("No images found! Check data directory.")
        return

    # Get a few samples
    indices = [0, 10, 20]
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        if idx >= len(dataset):
             break
        sample = dataset[idx]
        sar = sample['sar']
        cloudy = sample['cloudy_optical']
        ground_truth = sample['ground_truth']
        filename = sample['filename']
        
        # Helper to show tensor
        def show_tensor(ax, t, title):
            # (C, H, W) -> (H, W, C)
            img = t.permute(1, 2, 0).numpy()
            
            # If SAR is 2 channel (VV, VH), visualize just the first channel or average
            if img.shape[2] == 2:
                # Pad to 3 for RGB visualization using 0 for B
                import numpy as np
                zeros = np.zeros((img.shape[0], img.shape[1], 1))
                img = np.concatenate([img, zeros], axis=2)
            
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            
        show_tensor(axes[i][0], sar, f"SAR (Input) {filename}")
        show_tensor(axes[i][1], cloudy, "Cloudy Optical (Input)")
        show_tensor(axes[i][2], ground_truth, "Clear Optical (Target)")

    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print("Saved sample_visualization.png")

if __name__ == "__main__":
    visualize_dataset()
