import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import SatelliteDataset
from src.model import MultiModalUNet
from src.dataset import SatelliteDataset
from src.model import MultiModalUNet
from src.loss import TotalLoss

# Hyperparameters
BATCH_SIZE = 16  # Adjust based on VRAM (RTX 3060 12GB can handle 4-8 easily with 256x256)
LEARNING_RATE = 0.0002
EPOCHS = 10 
NUM_WORKERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

def train_fn(loader, model,optimizer, loss_fn, scaler):
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
        loop.set_postfix(loss=loss.item(), l1=l1.item(), ssim_loss=ssim_loss.item(), vgg=perceptual.item())
        
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
    # Visualize the first image in the batch
    # sar: (B, 2, H, W) -> take first 2 channels
    # cloudy, targets, preds: (B, 3, H, W)
    
    idx = 0
    sar_img = sar[idx].cpu().permute(1, 2, 0).numpy()
    cloudy_img = cloudy[idx].cpu().permute(1, 2, 0).numpy()
    target_img = targets[idx].cpu().permute(1, 2, 0).numpy()
    pred_img = predictions[idx].float().cpu().permute(1, 2, 0).numpy()
    
    # Handle SAR visualization (pad 3rd channel)
    import numpy as np
    zeros = np.zeros((sar_img.shape[0], sar_img.shape[1], 1))
    sar_vis = np.concatenate([sar_img, zeros], axis=2)
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(sar_vis)
    ax[0].set_title("SAR")
    ax[1].imshow(cloudy_img)
    ax[1].set_title("Cloudy")
    ax[2].imshow(pred_img)
    ax[2].set_title(f"Prediction (Ep {epoch})")
    ax[3].imshow(target_img)
    ax[3].set_title("Ground Truth")
    
    for a in ax: a.axis('off')
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/val_epoch_{epoch}.png")
    plt.close()

def main():
    print(f"Device: {DEVICE}")
    
    data_dir = os.path.join(os.getcwd(), 'data')
    dataset = SatelliteDataset(data_dir)
    
    # Split
    val_size = int(len(dataset) * 0.1) # 10% validation
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    model = MultiModalUNet(n_channels=5, n_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = TotalLoss().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = val_fn(val_loader, model, loss_fn, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")

if __name__ == "__main__":
    main()
