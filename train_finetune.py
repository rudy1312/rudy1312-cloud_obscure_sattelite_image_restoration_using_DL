import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import SatelliteDataset
from src.model import MultiModalUNet
from src.loss import TotalLoss

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-5 # Very low for fine-tuning
EPOCHS = 15
NUM_WORKERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
OUTPUT_DIR = "outputs_finetune"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    mean_loss = []
    
    model.train()
    
    for batch_idx, batch in enumerate(loop):
        sar = batch['sar'].to(DEVICE)
        cloudy = batch['cloudy_optical'].to(DEVICE)
        targets = batch['ground_truth'].to(DEVICE)
        
        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(sar, cloudy)
            loss, l1, ssim_loss, perceptual = loss_fn(predictions, targets)
            
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar
        mean_loss.append(loss.item())
        loop.set_postfix(loss=loss.item(), l1=l1.item(), vgg=perceptual.item())
        
    return sum(mean_loss) / len(mean_loss)

def val_fn(loader, model, loss_fn, epoch):
    model.eval()
    loop = tqdm(loader, leave=True, desc="Validation")
    mean_loss = []
    
    with torch.no_grad():
        for batch in loop:
            sar = batch['sar'].to(DEVICE)
            cloudy = batch['cloudy_optical'].to(DEVICE)
            targets = batch['ground_truth'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                predictions = model(sar, cloudy)
                loss, l1, ssim_loss, perceptual = loss_fn(predictions, targets)
            
            mean_loss.append(loss.item())
            
            # Save the first batch for visualization
            if loop.n == 0:
                save_validation_samples(sar, cloudy, targets, predictions, epoch)
                
    return sum(mean_loss) / len(mean_loss)

def save_validation_samples(sar, cloudy, targets, predictions, epoch):
    idx = 0
    sar_img = sar[idx].cpu().permute(1, 2, 0).numpy()
    cloudy_img = cloudy[idx].cpu().permute(1, 2, 0).numpy()
    target_img = targets[idx].cpu().permute(1, 2, 0).numpy()
    pred_img = predictions[idx].float().cpu().permute(1, 2, 0).numpy()
    
    # Handle SAR visualization
    import numpy as np
    zeros = np.zeros((sar_img.shape[0], sar_img.shape[1], 1))
    sar_vis = np.concatenate([sar_img, zeros], axis=2)
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(sar_vis)
    ax[0].set_title("SAR")
    ax[1].imshow(cloudy_img)
    ax[1].set_title("Cloudy")
    ax[2].imshow(pred_img)
    ax[2].set_title(f"Refined (Ep {epoch})")
    ax[3].imshow(target_img)
    ax[3].set_title("Ground Truth")
    
    for a in ax: a.axis('off')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/val_epoch_{epoch}.png")
    plt.close()

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_dir = os.path.join(os.getcwd(), 'data')
    # Uses the updated SatelliteDataset with variable cloud transparency
    dataset = SatelliteDataset(data_dir)
    
    val_size = int(len(dataset) * 0.1) 
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Load Model with Attention
    model = MultiModalUNet(n_channels=5, n_classes=3).to(DEVICE)
    
    # Load Pre-trained weights
    if os.path.exists("best_model.pth"):
        print("Loading weights from best_model.pth...")
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("Warning: best_model.pth not found. Starting from scratch (not recommended for fine-tuning).")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TotalLoss includes PerceptualLoss
    loss_fn = TotalLoss().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"Fine-tuning Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = val_fn(val_loader, model, loss_fn, epoch)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "final_model_refined.pth")
            print("Saved final_model_refined.pth!")

if __name__ == "__main__":
    main()
