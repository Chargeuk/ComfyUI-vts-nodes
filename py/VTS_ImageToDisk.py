

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

from vtsUtils import save_images, DiskImage, vtsImageTypes, get_default_image_input_types, deep_merge, ensure_image_defaults


# class VTSImageUpscaleWithModel:
class VTSImageToDisk:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        defaults = get_default_image_input_types()
        input_types =  {
            "required": {
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
            }
        }
        result = deep_merge(defaults, input_types)
        return result

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_to_disk"

    CATEGORY = "image/postprocessing"

    def save_to_disk(self, passthrough: bool = False, **kwargs) -> tuple:
        if passthrough:
            return (kwargs.get("image"),)

        kwargs = ensure_image_defaults(kwargs)
        
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
