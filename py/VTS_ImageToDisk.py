

import logging
from comfy import model_management
import torch
import torch.nn.functional as F
import sys
import os

# Add the py directory to sys.path to allow imports
import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import save_images


# class VTSImageUpscaleWithModel:
class VTSImageToDisk:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "image", "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": "./output", "multiline": False}),
                "format": ("STRING", {"default": "png", "choices": ["png", "webp"]}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_to_disk"

    CATEGORY = "image/postprocessing"

    def save_to_disk(self, image: torch.Tensor, prefix: str = "image", start_sequence: int = 0, output_dir: str = "./tmp/image_cache", format: str = "png", passthrough: bool = False) -> tuple:
        if passthrough:
            return (image,)
        
        saved_paths = save_images(image, prefix, start_sequence, output_dir, format)
        print(f"Saved {len(saved_paths)} images to {output_dir}")

        newImageData = {
            "prefix": prefix,
            "start_sequence": start_sequence,
            "output_dir": output_dir,
            "format": format,
            "shape": image.shape,
            "ndim": image.ndim,
            "dtype": image.dtype,
        }
        
        return (newImageData,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Image To Disk": VTSImageToDisk
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image To Disk": "Image To Disk VTS"
}
