import comfy
import comfy.utils
import torch

# Add the py directory to sys.path to allow imports
import sys
import os

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, transform_and_save_images, get_default_image_input_types, deep_merge, ensure_image_defaults

def image_alpha_fix(destination, source):
    if destination.shape[-1] < source.shape[-1]:
        source = source[...,:destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        destination = torch.nn.functional.pad(destination, (0, 1))
        destination[..., -1] = 1.0
    return destination, source

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

MAX_RESOLUTION = 16384

class VTS_Image_Composite_Masked:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]
    scale_types = ["small", "large", "max"]

    @classmethod
    def INPUT_TYPES(s):
        defaults = get_default_image_input_types()
        input_types = {
            "required": {
                "image": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
        result = deep_merge(defaults, input_types)
        return result
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "VTS"


    def composite(self, source, x, y, resize_source, mask = None, **kwargs):

        def transform_fn(batch_tensor):
            batch_tensor, updated_source = image_alpha_fix(batch_tensor, source)
            batch_tensor = batch_tensor.movedim(-1, 1)
            output = composite(batch_tensor, updated_source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
            return output

        result = transform_and_save_images(
            transform_fn=transform_fn,
            **kwargs
        )

        return (result,)
        

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Image Composite Masked": VTS_Image_Composite_Masked
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image Composite Masked": "VTS Image Composite Masked"
}