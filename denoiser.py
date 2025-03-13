import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from .oklab import RGB_to_Oklab, Oklab_to_RGB

@torch.jit.script
def hyper_laplacian_loss(x: torch.Tensor, alpha: float = 0.9, eps: float = 1e-8) -> torch.Tensor:
    """Hyper-laplacian loss with exponent alpha (0 < alpha <= 1)"""
    # x shape: [batch, height, width, channels]
    diff_h = torch.abs(x[..., 1:, :, :] - x[..., :-1, :, :])
    diff_w = torch.abs(x[..., :, 1:, :] - x[..., :, :-1, :])
    
    if alpha == 1.0:
        return (diff_h.mean() + diff_w.mean()) / 2
    else:
        return (((diff_h + eps) ** alpha).mean() + ((diff_w + eps) ** alpha).mean()) / 2

def denoise(image, iterations, learning_rate, smoothing_strength,
            hl_alpha, l_weight, ab_weight):
    if torch.cuda.is_available() and 'cpu' in str(image.device):
        print("Moving input tensor from CPU to GPU...")
        image = image.to('cuda')
    
    device = image.device
    print(f"Running on device: {device}")
    
    with torch.inference_mode(False):
        original_oklab = RGB_to_Oklab(image.to(device))
        
        # Clone and detach for optimization
        denoised_oklab = original_oklab.detach().clone().requires_grad_(True)

        # Use fused AdamW for faster optimization
        optimizer = AdamW([denoised_oklab], 
                         lr=learning_rate, 
                         betas=(0.9, 0.999),
                         fused=True if 'cuda' in device.type else False)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=iterations, 
                                    eta_min=learning_rate*0.001)
       
        for i in range(iterations):
            optimizer.zero_grad()

            # Vectorized channel processing
            L = denoised_oklab[..., 0:1]  # Lightness channel
            ab = (denoised_oklab[..., 1:3] + 0.4) / 0.8  # Combined a+b channels
            
            # Calculate losses
            data_loss = F.mse_loss(denoised_oklab, original_oklab)
            hl_L = hyper_laplacian_loss(L, alpha=hl_alpha) * l_weight
            hl_ab = hyper_laplacian_loss(ab, alpha=hl_alpha) * ab_weight * 2  # 2x for both channels
            total_hl = hl_L + hl_ab
            loss = (1 - smoothing_strength) * data_loss + smoothing_strength * total_hl

            # Scaled backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                print(f"Iter {i:4d}/{iterations} | Loss: {loss.item():.5f} | "
                      f"Data: {data_loss.item():.5f} | HL: {total_hl.item():.5f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Convert back to RGB and clamp
        denoised_rgb = Oklab_to_RGB(denoised_oklab)
        return (torch.clamp(denoised_rgb, 0, 1),)
