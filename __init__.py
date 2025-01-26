from .oklab import RGB_to_Oklab, Oklab_to_RGB
from .denoiser import denoise
import torch

class OklabDenoising:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 3000,
                    "min": 100,
                    "max": 5000,
                    "step": 100
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0004,
                    "min": 0.000001,
                    "max": 0.001,
                    "step": 0.000001,
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "hl_alpha": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
                "l_weight": ("FLOAT", {
                    "default": 0.34,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "ab_weight": ("FLOAT", {
                    "default": 0.33,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/processing"

    def denoise(self, image, iterations, learning_rate, smoothing_strength, hl_alpha, l_weight, ab_weight):
        denoised_image = denoise(image, iterations, learning_rate, smoothing_strength, hl_alpha, l_weight, ab_weight)
        return denoised_image

NODE_CLASS_MAPPINGS = {
    "OklabDenoising": OklabDenoising
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OklabDenoising": "Oklab color space denoising"
}