import os
import sys

from comfy import model_management

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, ensure_image_output_defaults, get_default_image_output_types, save_images, deep_merge


class VTS_VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "tile_size_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                "tile_size_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time.", "advanced": True}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap.", "advanced": True}),
            }
        }
        return deep_merge(input_types, get_default_image_output_types(prefix="vae_decode"))

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "VTS/latent"

    def decode(self, vae, samples, tile_size_x=512, tile_size_y=512, overlap=64, temporal_size=64, temporal_overlap=8, **kwargs):
        kwargs = ensure_image_output_defaults(kwargs)

        if tile_size_x < overlap * 4:
            overlap = tile_size_x // 4
        if tile_size_y < overlap * 4:
            overlap = min(overlap, tile_size_y // 4)
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
        images = vae.decode_tiled(
            samples["samples"],
            tile_x=tile_size_x // compression,
            tile_y=tile_size_y // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        if kwargs["return_type"] == "Tensor":
            return (images,)

        saved_paths = save_images(
            image=images,
            prefix=kwargs["prefix"],
            start_sequence=kwargs["start_sequence"],
            output_dir=kwargs["output_dir"],
            format=kwargs["format"],
            num_workers=kwargs["num_workers"],
            compression_level=kwargs["compression_level"],
            quality=kwargs["quality"],
        )

        disk_image = DiskImage(
            prefix=kwargs["prefix"],
            start_sequence=kwargs["start_sequence"],
            number_of_images=len(saved_paths),
            output_dir=kwargs["output_dir"],
            format=kwargs["format"],
            image=images,
            compression_level=kwargs["compression_level"],
            quality=kwargs["quality"],
        )

        del images
        model_management.soft_empty_cache()
        return (disk_image,)


NODE_CLASS_MAPPINGS = {
    "VTS VAE Decode Tiled": VTS_VAEDecodeTiled
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS VAE Decode Tiled": "VAE Decode VTS (Tiled)"
}
