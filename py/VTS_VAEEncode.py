import os
import sys

import torch
from comfy import model_management
from nodes import VAEEncode as NativeVAEEncode

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage


class VTS_VAEEncode(NativeVAEEncode):
    CATEGORY = "VTS/latent"

    def encode(self, vae, pixels):
        materialized_pixels = None

        if isinstance(pixels, DiskImage):
            materialized_pixels = pixels.materialize()
            pixels = materialized_pixels
        elif not isinstance(pixels, torch.Tensor):
            raise TypeError("VTS VAE Encode expected pixels to be a tensor or DiskImage.")

        try:
            return super().encode(vae, pixels)
        finally:
            if materialized_pixels is not None:
                del materialized_pixels
                model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "VTS VAE Encode": VTS_VAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS VAE Encode": "VAE Encode VTS",
}
