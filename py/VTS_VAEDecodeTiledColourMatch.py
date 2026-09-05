import os
import sys

from comfy import model_management

import_dir = os.path.dirname(__file__)
if import_dir not in sys.path:
    sys.path.append(import_dir)

from VTS_VAEDecodeTiled import VTS_VAEDecodeTiled
from vtsUtils import DiskImage, ensure_image_output_defaults, resolve_list_mapped_output_identity, save_images
from vts_color_correction import METHODS, MODES, correct_images


class VTS_VAEDecodeTiledColourMatch(VTS_VAEDecodeTiled):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["optional"] = {
            "color_ref": ("IMAGE", {"tooltip": "Tensor or DiskImage reference. One image is reused; fixed mode samples a reference sequence. Other modes pair frames, repeating the last reference if shorter."}),
            "color_match_method": (METHODS, {"default": "reinhard_lab_gpu"}),
            "color_match_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "white_balance_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Match reference colour balance while retaining the current mean luminance."}),
            "brightness_method": (["gamma", "exposure"], {"default": "gamma"}),
            "brightness_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Match median brightness to the reference after colour and white-balance correction."}),
            "contrast_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Match 1st/99th luminance percentiles to the reference after brightness correction."}),
            "calculation_mode": (MODES, {"default": "fixed_per_clip", "tooltip": "Fixed: one transform fitted from up to 16 frames across the clip. Per frame: fit every frame. Smoothed: blend correction tables over time, without blending video frames."}),
            "smoothing": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smoothed mode only. Higher values change more slowly; 0 equals per-frame, 1 holds the first correction."}),
            "overall_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blend the complete correction with the original decode. Zero bypasses correction."}),
            "analysis_size": ("INT", {"default": 128, "min": 16, "max": 512, "step": 16, "advanced": True, "tooltip": "Maximum analysis thumbnail dimension. Reference is resized to match analysis frames; output resolution is unchanged."}),
            "lut_resolution": ([17, 33, 65], {"default": 33, "advanced": True, "tooltip": "RGB lookup-table resolution. Higher values approximate the fitted transforms more accurately, especially near black."}),
        }
        return inputs

    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Tiled VAE decode with optional reference colour correction. Uses weighted "
        "colour matching, reference white balance, brightness and contrast, in that "
        "order. Corrects before saving. No reference or zero overall weight gives "
        "the original decode. GPU Lab follows KJNodes' Lab statistics approach; "
        "reference tone controls adapt Donut-style operations.")

    def decode(self, vae, samples, tile_size_x=512, tile_size_y=512, overlap=64,
               temporal_size=64, temporal_overlap=8, color_ref=None,
               color_match_method="reinhard_lab_gpu", color_match_weight=0.5,
               white_balance_weight=0.0, brightness_method="gamma",
               brightness_weight=0.0, contrast_weight=0.0,
               calculation_mode="fixed_per_clip", smoothing=0.9, overall_weight=1.0,
               analysis_size=128, lut_resolution=33, **kwargs):
        kwargs = ensure_image_output_defaults(kwargs)
        decode_args = dict(tile_size_x=tile_size_x, tile_size_y=tile_size_y,
                           overlap=overlap, temporal_size=temporal_size,
                           temporal_overlap=temporal_overlap)
        if color_ref is None or overall_weight == 0 or not any(
                (color_match_weight, white_balance_weight, brightness_weight, contrast_weight)):
            return super().decode(vae, samples, **decode_args, **kwargs)

        tensor_options = dict(kwargs, return_type="Tensor")
        images, = super().decode(vae, samples, **decode_args, **tensor_options)
        corrected = correct_images(
            images, color_ref, method=color_match_method, color_weight=color_match_weight,
            white_weight=white_balance_weight, brightness_weight=brightness_weight,
            contrast_weight=contrast_weight, brightness_method=brightness_method,
            mode=calculation_mode, smoothing=smoothing, overall_weight=overall_weight,
            analysis_size=analysis_size, lut_resolution=int(lut_resolution))
        # The decoded buffer belongs to this execution; avoid a second full clip.
        for index, frame in enumerate(corrected):
            images[index].copy_(frame)

        if kwargs["return_type"] == "Tensor":
            return (images,)
        prefix, start = resolve_list_mapped_output_identity(kwargs["prefix"], kwargs["start_sequence"])
        save_images(image=images, prefix=prefix, start_sequence=start,
                    output_dir=kwargs["output_dir"], format=kwargs["format"],
                    num_workers=kwargs["num_workers"], compression_level=kwargs["compression_level"],
                    quality=kwargs["quality"])
        result = DiskImage(prefix=prefix, start_sequence=start, number_of_images=len(images),
                           output_dir=kwargs["output_dir"], format=kwargs["format"], image=images,
                           compression_level=kwargs["compression_level"], quality=kwargs["quality"])
        del images
        model_management.soft_empty_cache()
        return (result,)


NODE_CLASS_MAPPINGS = {"VTS VAE Decode Tiled Colour Match": VTS_VAEDecodeTiledColourMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS VAE Decode Tiled Colour Match": "VAE Decode VTS (Tiled + Colour Match)"}
