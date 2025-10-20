

import logging
from comfy import model_management
import torch
import torch.nn.functional as F

# Add the py directory to sys.path to allow imports
import sys
import os
import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import save_images, DiskImage, vtsImageTypes


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
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 4, "min": 1}),
                "compression_level": ("INT", {"default": 4, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_to_disk"

    CATEGORY = "image/postprocessing"

    def save_to_disk(self, passthrough: bool = False, **kwargs) -> tuple:
        if passthrough:
            return (kwargs.get("image"),)

        quality = kwargs.get("quality", 95)
        if quality > 100:
            kwargs["quality"] = None

        saved_paths = save_images(
            **kwargs
        )

        print(f"Saved {len(saved_paths)} images to {kwargs.get('output_dir', './output')}")

        newImageData = DiskImage(
            number_of_images=len(saved_paths),
            **kwargs  # Now DiskImage will ignore num_workers
        )

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
