import os
import sys

import torch
from comfy import model_management
from nodes import VAEEncodeTiled as NativeVAEEncodeTiled

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage


class VTS_VAEEncodeTiled(NativeVAEEncodeTiled):
    CATEGORY = "VTS/latent"

    def encode(self, vae, pixels, tile_size, overlap, temporal_size=64, temporal_overlap=8):
        materialized_pixels = None

        if isinstance(pixels, DiskImage):
            materialized_pixels = pixels.materialize()
            pixels = materialized_pixels
        elif not isinstance(pixels, torch.Tensor):
            raise TypeError("VTS VAE Encode (Tiled) expected pixels to be a tensor or DiskImage.")

        try:
            return super().encode(
                vae,
                pixels,
                tile_size,
                overlap,
                temporal_size=temporal_size,
                temporal_overlap=temporal_overlap,
            )
        finally:
            if materialized_pixels is not None:
                del materialized_pixels
                model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "VTS VAE Encode Tiled": VTS_VAEEncodeTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS VAE Encode Tiled": "VAE Encode VTS (Tiled)",
}
