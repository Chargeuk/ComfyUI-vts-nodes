import os
import sys

import torch

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage


class VTS_Image_To_Float32:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "VTS/image"
    DESCRIPTION = "Materializes tensor or DiskImage input and returns a float32 image tensor."

    def convert(self, image):
        if isinstance(image, DiskImage):
            image = image.materialize()
        elif not isinstance(image, torch.Tensor):
            raise TypeError("Image To Float32 VTS expected a tensor or DiskImage.")

        return (image.to(dtype=torch.float32),)


NODE_CLASS_MAPPINGS = {
    "VTS Image To Float32": VTS_Image_To_Float32,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image To Float32": "Image To Float32 VTS",
}
