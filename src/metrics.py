import torch
import torch.nn.functional as F
from src.loss import SSIM

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    img1, img2: Tensors [B, C, H, W] in range [0, 1]
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculate SSIM using the implementation in src.loss.
    """
    ssim_module = SSIM()
    if img1.is_cuda:
        ssim_module = ssim_module.cuda()
    
    return ssim_module(img1, img2)
