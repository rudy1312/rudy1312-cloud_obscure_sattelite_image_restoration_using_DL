import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None, cloud_min=0.3, cloud_max=0.7):
        """
        Args:
            root_dir (string): Directory with all the 512x256 side-by-side images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            cloud_min (float): Minimum opacity of clouds (0.0 to 1.0).
            cloud_max (float): Maximum opacity of clouds (0.0 to 1.0).
        """
        self.root_dir = root_dir
        self.cloud_min = cloud_min
        self.cloud_max = cloud_max
        # Assuming images are directly in root_dir or we walk to find them.
        # For now, let's assume a flat structure or check subfolders.
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_files.append(os.path.join(root, file))
        
        self.transform = transform
        
        # Base transform to tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def add_synthetic_clouds(self, img_tensor):
        """
        Adds synthetic clouds to an optical image tensor (C, H, W).
        Basic implementation using random noise/blobs.
        """
        c, h, w = img_tensor.shape
        # Create a cloud mask
        # 1. Perlin noise or simple Gaussian blobs could work. 
        # For a simple "fast" Day 1 approach, let's use resized noise.
        
        # Generate low-res noise and upsample to simulate "blobs"
        cloud_structure = torch.rand(1, h // 16, w // 16)
        cloud_mask = torch.nn.functional.interpolate(
            cloud_structure.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Threshold to create gaps
        cloud_mask = (cloud_mask - 0.4).clamp(0, 1) # Shift and clamp
        cloud_mask = cloud_mask / cloud_mask.max() if cloud_mask.max() > 0 else cloud_mask
        
        # Random opacity for this image or per blob
        # The user requested alpha transparency between 0.3 and 0.7 for each blob.
        # Our simple implementation generates one big "blob structure".
        # We can simulate "per blob" variance by multiplying the mask by random noise.
        
        # Base opacity map
        opacity_map = torch.rand(1, h // 16, w // 16) * (self.cloud_max - self.cloud_min) + self.cloud_min
        opacity_mask = torch.nn.functional.interpolate(
            opacity_map.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Apply opacity only where there are clouds
        final_cloud_layer = cloud_mask * opacity_mask
        
        # Apply to image: Cloud adds white to the image, obscuring the ground
        # Standard composition: Result = Image * (1 - Alpha) + CloudColor * Alpha
        # Here Alpha is final_cloud_layer
        
        cloudy_img = img_tensor * (1 - final_cloud_layer) + final_cloud_layer
        
        return cloudy_img.clamp(0, 1)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            
            # Slicing: Left is SAR, Right is Optical
            # Expected side-by-side 512x256 (Width x Height seems wrong in standard HxW notation, usually 512 wide 256 high)
            # Standard: (W=512, H=256) -> SAR left half, Optical right half.
            
            sar_img = img.crop((0, 0, w // 2, h))
            optical_img = img.crop((w // 2, 0, w, h))
            
            # Convert to Tensor (scales to 0-1)
            sar_tensor = self.to_tensor(sar_img)
            optical_tensor = self.to_tensor(optical_img)
            
            # SAR is often grayscale, but if saved as RGB it has 3 channels. 
            # Multi-modal U-Net expects 2 SAR channels (VV, VH) according to prompt.
            # But our dataset is just "SAR Image". 
            # If the SAR image is grayscale (1 channel) or RGB (3 identical channels), we need to adapt.
            # Prompt says: "2 SAR channels (VV, VH polarization)".
            # If the dataset provides visual interpretations of SAR, it might be 1 or 3 channels.
            # We will adhere to the prompt's architecture request of 2 channels if possible, 
            # possibly by taking the first 2 channels of the SAR RGB or duplicating if mono.
            # Let's check SAR shape.
            
            if sar_tensor.shape[0] == 3:
                 # Take first 2 channels to fake VV/VH if meaningful, or just use R and G
                 sar_tensor = sar_tensor[:2, :, :]
            elif sar_tensor.shape[0] == 1:
                # Duplicate to make 2 channels
                sar_tensor = torch.cat([sar_tensor, sar_tensor], dim=0)
            
            # Generate Cloudy Optical
            cloudy_optical_tensor = self.add_synthetic_clouds(optical_tensor)
            
            # Apply extra transforms if any (usually resizing or normalization)
            # We already have 0-1 tensors. 
            # U-Net input expects concatenation of Optical(3) + SAR(2) = 5 channels.
            # But the dataset returns triplets for training.
            
            return {
                'sar': sar_tensor,              # [2, 256, 256]
                'cloudy_optical': cloudy_optical_tensor, # [3, 256, 256]
                'ground_truth': optical_tensor, # [3, 256, 256]
                'filename': os.path.basename(img_path)
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # return None or handle gracefully, for now rely on robust loop or crash
            raise e
