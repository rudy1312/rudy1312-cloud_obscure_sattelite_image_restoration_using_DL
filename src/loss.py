import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in torch.arange(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load VGG16 features
        vgg = models.vgg16(pretrained=True)
        # Extract up to relu3_3 (index 16 in features which is the ReLU after the 3rd block of convs)
        # Structure: 
        # Block 1: Conv(0), Relu(1), Conv(2), Relu(3), MaxPool(4)
        # Block 2: Conv(5), Relu(6), Conv(7), Relu(8), MaxPool(9)
        # Block 3: Conv(10), Relu(11), Conv(12), Relu(13), Conv(14), Relu(15) -> This is relu3_3
        # So we take slice [:16]
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.features(x)

class TotalLoss(nn.Module):
    def __init__(self, l1_weight=0.5, ssim_weight=0.2, perceptual_weight=0.3):
        super(TotalLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1 = nn.L1Loss()
        self.ssim = SSIM()
        self.perceptual = PerceptualLoss()

    def forward(self, output, target):
        # L1 Loss
        l1_loss = self.l1(output, target)
        
        # SSIM Loss (1 - SSIM)
        ssim_val = self.ssim(output, target)
        ssim_loss = 1 - ssim_val
        
        # Perceptual Loss (MSE of features)
        # We assume input is RGB [0,1]. VGG expects roughly that range (actually normalized, 
        # but for simple perceptual loss widely used without strict normalization if domains match)
        out_feat = self.perceptual(output)
        target_feat = self.perceptual(target)
        perceptual_loss = F.mse_loss(out_feat, target_feat)
        
        total = (self.l1_weight * l1_loss) + (self.ssim_weight * ssim_loss) + (self.perceptual_weight * perceptual_loss)
        
        return total, l1_loss, ssim_loss, perceptual_loss
