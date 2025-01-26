📌 **Oklab Perceptual TV Denoiser**  
*A ComfyUI node implementing edge-preserving image denoising in perceptually uniform Oklab color space using total variation regularization with hyper-Laplacian constraints.*

**Key Features:**
- 🎨 **Oklab Color Space**: Maintains perceptual uniformity for natural color relationships
- 🏔️ **Edge-Preserving TV**: Reduces noise while preserving sharp edges and textures
- ⚡ **Hyper-Laplacian Constraints**: Handles heavy-tailed noise distributions better than L2/L1

**Parameters:**
- **`iterations`** (Default: 3000, Range: 100-5000)  
  Maximum optimization steps

- **`learning_rate`** (Default: 0.0004, Range: 0.00001-0.001)  
  Initial step size for gradient descent

- **`smoothing_strength`** (Default: 0.05, Range: 0.0-1.0)  
  Trade-off between noise removal and detail preservation (0 = original image)

- **`hl_alpha`** (Default: 0.8, Range: 0.1-1.0)  
  Hyper-Laplacian exponent (lower = sharper transitions / sharpening)

- **`l_weight`** (Default: 0.34, Range: 0.0-1.0)  
  Lightness channel regularization strength

- **`ab_weight`** (Default: 0.33, Range: 0.0-1.0)  
  Chroma channels regularization strength

**Tip:** Start with defaults and adjust `smoothing_strength` first. Use lower `hl_alpha` (0.6-0.7) to counteract high smoothing strength color bleeding. Balance `l_weight`/`ab_weight` based on whether noise appears more in luminance or color channels.
