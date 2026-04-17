import torch
from torch.utils.data import DataLoader
from src.dataset import SatelliteDataset
from src.model import MultiModalUNet
from src.metrics import calculate_psnr, calculate_ssim
from tqdm import tqdm
import os
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model_path, dataset, device):
    print(f"\nEvaluating: {model_path} ...")
    
    model = MultiModalUNet(n_channels=5, n_classes=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Skipping {model_path}: File not found.")
        return [], []

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {os.path.basename(model_path)}"):
            sar = batch['sar'].to(device)
            cloudy = batch['cloudy_optical'].to(device)
            target = batch['ground_truth'].to(device)
            
            output = model(sar, cloudy)
            output = torch.clamp(output, 0, 1)
            
            psnr_scores.append(calculate_psnr(output, target).item())
            ssim_scores.append(calculate_ssim(output, target).item())
            
            # Evaluate on 5000 images as requested previously
            if len(psnr_scores) >= 5000: break 

    return psnr_scores, ssim_scores

def generate_report(model_name, psnr_scores, ssim_scores):
    if not psnr_scores:
        return f"No results for {model_name}"
        
    avg_psnr = np.mean(psnr_scores)
    min_psnr = np.min(psnr_scores)
    max_psnr = np.max(psnr_scores)
    
    avg_ssim = np.mean(ssim_scores)
    min_ssim = np.min(ssim_scores)
    max_ssim = np.max(ssim_scores)
    
    report = f"""
    Evaluation Matrix (Quantitative Metrics) - {model_name}
    ========================================
    Total Images Evaluated: {len(psnr_scores)}
    
    Metric      | Mean   | Min    | Max
    ----------- | ------ | ------ | ------
    PSNR (dB)   | {avg_psnr:.2f}  | {min_psnr:.2f}  | {max_psnr:.2f}
    SSIM (0-1)  | {avg_ssim:.4f} | {min_ssim:.4f} | {max_ssim:.4f}
    
    Interpretation:
    - PSNR > 20dB is generally acceptable for noisy images. > 28dB is good.
    - SSIM > 0.7 indicates recognizable structure. > 0.9 is excellent.
    """
    return report

def main():
    print(f"Device: {DEVICE}")
    data_dir = os.path.join(os.getcwd(), 'data')
    dataset = SatelliteDataset(data_dir)
    print(f"Dataset Size: {len(dataset)} images (Evaluating on subset of 5000)")
    
    # Only test best_model.pth
    model_file = "best_model.pth"
    full_report = ""
    
    psnr, ssim = evaluate_model(model_file, dataset, DEVICE)
    report = generate_report(model_file, psnr, ssim)
    print(report)
    full_report += report + "\n"
    
    with open("evaluation_results.txt", "w") as f:
        f.write(full_report)
    print("Saved results to evaluation_results.txt")

if __name__ == "__main__":
    main()
