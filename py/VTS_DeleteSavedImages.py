

import logging
from comfy import model_management
import torch
import torch.nn.functional as F
import json

# Add the py directory to sys.path to allow imports
import sys
import os
import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import delete_all_saved_images

# wildcard trick is taken from pythongossss's
class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")

# class VTSImageUpscaleWithModel:
class VTSDeleteSavedImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        result = {
            "required": {
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, do nothing"}),
                "dry_run": ("BOOLEAN", {"default": False, "tooltip": "When true, do nothing"}),
            },
            "optional": {
                "any_input": (any, {}),
                "any_input2": (any, {}),
                "any_input3": (any, {}),
                "any_input4": (any, {}),
                "any_input5": (any, {}),
                "any_input6": (any, {}),
                "any_input7": (any, {}),
                "any_input8": (any, {}),
                "any_input9": (any, {}),
            }
        }
        return result

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "stats",
    )
    
    FUNCTION = "delete_images"

    CATEGORY = "image/postprocessing"

    def delete_images(self, passthrough: bool = False, dry_run: bool = False, **kwargs) -> tuple:
        return_value = "passthrough"
        if passthrough:
            return (return_value,)
        
        stats = delete_all_saved_images(dry_run)
        # stats is a dictionary, that we need to turn into a string
        return_value = json.dumps(stats, indent=2)

        return (return_value,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Delete Saved Images": VTSDeleteSavedImages
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Delete Saved Images": "Delete Saved Images VTS"
}
