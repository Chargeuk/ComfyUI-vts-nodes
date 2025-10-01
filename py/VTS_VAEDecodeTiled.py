

import logging
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor
from comfy import model_management
import torch
import comfy
import comfy.utils
import folder_paths 


class VTS_VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ),
                             "vae": ("VAE", ),
                             "tile_size_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                             "tile_size_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "_for_testing"

    def decode(self, vae, samples, tile_size_x=512, tile_size_y=512, overlap=64, temporal_size=64, temporal_overlap=8):
        if tile_size_x < overlap * 4:
            overlap = tile_size_x // 4
        if tile_size_y < overlap * 4:
            overlap = tile_size_y // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples["samples"], tile_x=tile_size_x // compression, tile_y=tile_size_y // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS VAE Decode Tiled": VTS_VAEDecodeTiled
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS VAE Decode Tiled": "VTS VAE Decode Tiled"
}
