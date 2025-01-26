import torch

def RGB_to_Oklab(rgb):
    batch_size, height, width, channels = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)

    # Convert sRGB to linear RGB
    mask = rgb_flat <= 0.04045
    rgb_linear = torch.where(mask, rgb_flat / 12.92, ((rgb_flat + 0.055) / 1.055) ** 2.4)

    # Linear RGB to LMS
    M1 = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], device=rgb.device, dtype=rgb.dtype)

    lms = torch.matmul(rgb_linear, M1.T)

    # LMS to Oklab using cube root
    lms_cbrt = torch.sign(lms) * torch.abs(lms) ** (1.0 / 3.0)

    M2 = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], device=rgb.device, dtype=rgb.dtype)

    oklab = torch.matmul(lms_cbrt, M2.T)

    # Reshape back to original dimensions
    oklab = oklab.reshape(batch_size, height, width, 3)
    return oklab

def Oklab_to_RGB(oklab):
    batch_size, height, width, channels = oklab.shape
    oklab_flat = oklab.reshape(-1, 3)

    M2_inv = torch.tensor([
        [1.0000000000, 0.3963377774, 0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480]
    ], device=oklab.device, dtype=oklab.dtype)

    lms_cbrt = torch.matmul(oklab_flat, M2_inv.T)

    # Cube operation
    lms = torch.sign(lms_cbrt) * torch.abs(lms_cbrt) ** 3

    M1_inv = torch.tensor([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010]
    ], device=oklab.device, dtype=oklab.dtype)

    rgb_linear = torch.matmul(lms, M1_inv.T)

    # Convert back to sRGB
    mask = rgb_linear <= 0.0031308
    rgb_nonlinear = torch.where(mask, rgb_linear * 12.92, 1.055 * torch.abs(rgb_linear) ** (1 / 2.4) - 0.055)

    # Reshape back
    rgb = rgb_nonlinear.reshape(batch_size, height, width, 3)
    rgb = torch.clamp(rgb, 0, 1)

    return rgb