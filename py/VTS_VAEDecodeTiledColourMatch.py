import os
import sys

import torch

from comfy import model_management

import_dir = os.path.dirname(__file__)
if import_dir not in sys.path:
    sys.path.append(import_dir)

from VTS_VAEDecodeTiled import VTS_VAEDecodeTiled
from VTS_MiniMaxH3MotionContext import _context_length, _pixel_frames, _resize, _steps_for_frames
from VTS_MerserkTemporalEnhance import VTSMerserkTemporalEnhance
from vtsUtils import DiskImage, ensure_image_output_defaults, resolve_list_mapped_output_identity, save_images
from vts_color_correction import METHODS, MODES, correct_images


def _merserk_inputs():
    # Reuse the standalone node's processing controls and tooltips. Output storage
    # belongs to the decoder so context encoding precedes any lossy disk save.
    excluded = {"images", "return_type", "output_dir", "format", "compression_level",
                "quality", "prefix", "start_sequence"}
    return {name: spec for group in VTSMerserkTemporalEnhance.INPUT_TYPES().values()
            for name, spec in group.items() if name not in excluded}




_vts_utils = os.path.join(os.path.dirname(__file__), 'vtsUtils')
if _vts_utils not in sys.path:
    sys.path.append(_vts_utils)
from vts_latent_nodes import disk_latent_node

@disk_latent_node(inputs=('samples',), outputs=(), prefix='VAE Decode VTS (Tiled + Colour Match)')
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
        inputs["optional"].update({
            "encode_corrected_context": ("BOOLEAN", {"default": False, "tooltip": "H3 only: encode a tail for the next loop, before disk compression. Use Merserk For Context selects whether enhancement affects this tail. Frames automatically return to the source latent resolution. Adds VAE encoding work; off returns no replacement context."}),
            "context_length": (["22", "5", "39", "56"], {"default": "22", "tooltip": "Must match Prepare Loop Context. Only this many tail frames are encoded; short clips use the largest valid H3 length available."}),
        })
        inputs["optional"].update({
            "use_merserk_for_context": ("BOOLEAN", {"default": False, "tooltip": "Requires Encode Corrected Context. Off encodes colour-corrected frames before Merserk. On uses the Merserk result when Enable Merserk is on, automatically resized to the original latent dimensions. The main image output always includes enabled Merserk processing."}),
            "context_frame_selection": (["Original", "Interpolated"], {"default": "Original", "tooltip": "Only used when the context includes Merserk and frame interpolation is enabled. Original selects enhanced source frames at the original cadence. Interpolated selects consecutive frames from the denser output tail. H3 still conditions at its fixed 24 FPS: this is not higher-FPS conditioning and can slow apparent continuation motion or affect audio alignment. Context Length is unchanged."}),
            "enable_merserk": ("BOOLEAN", {"default": False, "tooltip": "Run Merserk Temporal Enhance after the full sequence is decoded and colour corrected. Scaling, neural rendering and interpolation have separate switches below. Off preserves the existing decoder behavior. Uses lossless 8-bit PNG transport; final files use this decoder's output settings."}),
        })
        inputs["optional"].update({"merserk_" + name: spec for name, spec in _merserk_inputs().items()})
        return inputs

    RETURN_TYPES = ("IMAGE", "VTS_H3_VIDEO_CONTEXT")
    RETURN_NAMES = ("image", "corrected_video_context")
    DESCRIPTION = (
        "Tiled VAE decode with optional reference colour correction. Uses weighted "
        "colour matching, reference white balance, brightness and contrast, in that "
        "order. Corrects before saving. No reference or zero overall weight bypasses "
        "colour correction. GPU Lab follows KJNodes' Lab statistics approach; "
        "reference tone controls adapt Donut-style operations. Optionally encode "
        "the corrected H3 tail for Prepare Loop Context; original audio stays unchanged. "
        "Optional Merserk scaling, temporal enhancement and interpolation run after "
        "colour correction. Context can include or exclude Merserk, with automatic "
        "resizing back to the original latent resolution before encoding.")

    def decode(self, vae, samples, tile_size_x=512, tile_size_y=512, overlap=64,
               temporal_size=64, temporal_overlap=8, color_ref=None,
               color_match_method="reinhard_lab_gpu", color_match_weight=0.5,
               white_balance_weight=0.0, brightness_method="gamma",
               brightness_weight=0.0, contrast_weight=0.0,
               calculation_mode="fixed_per_clip", smoothing=0.9, overall_weight=1.0,
               analysis_size=128, lut_resolution=33, encode_corrected_context=False,
               context_length="22", enable_merserk=False, use_merserk_for_context=False,
               context_frame_selection="Original", **kwargs):
        merserk_options = {name: kwargs.pop("merserk_" + name)
                           for name in _merserk_inputs() if "merserk_" + name in kwargs}
        kwargs = ensure_image_output_defaults(kwargs)
        decode_args = dict(tile_size_x=tile_size_x, tile_size_y=tile_size_y,
                           overlap=overlap, temporal_size=temporal_size,
                           temporal_overlap=temporal_overlap)
        bypass_correction = color_ref is None or overall_weight == 0 or not any(
            (color_match_weight, white_balance_weight, brightness_weight, contrast_weight))
        if bypass_correction and not encode_corrected_context and not enable_merserk:
            return (*super().decode(vae, samples, **decode_args, **kwargs), None)

        if encode_corrected_context:
            if str(context_length) not in ("5", "22", "39", "56"):
                raise ValueError("Corrected H3 context_length must be 5, 22, 39 or 56.")
            video = samples["samples"]
            if video.is_nested:
                video = video.unbind()[0]
            if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
                raise ValueError("Corrected context encoding requires one H3 video latent [1,24,T,H,W].")
            source_frames = _pixel_frames(int(video.shape[2]))
            frame_count = _context_length(int(context_length), source_frames)

        include_merserk_context = encode_corrected_context and enable_merserk and use_merserk_for_context
        if include_merserk_context and context_frame_selection not in ("Original", "Interpolated"):
            raise ValueError("Context frame selection must be Original or Interpolated.")

        tensor_options = dict(kwargs, return_type="Tensor")
        images, = super().decode(vae, samples, **decode_args, **tensor_options)
        if not bypass_correction:
            corrected = correct_images(
                images, color_ref, method=color_match_method, color_weight=color_match_weight,
                white_weight=white_balance_weight, brightness_weight=brightness_weight,
                contrast_weight=contrast_weight, brightness_method=brightness_method,
                mode=calculation_mode, smoothing=smoothing, overall_weight=overall_weight,
                analysis_size=analysis_size, lut_resolution=int(lut_resolution))
            # The decoded buffer belongs to this execution; avoid a second full clip.
            for index, frame in enumerate(corrected):
                images[index].copy_(frame)

        context = None
        context_images = None
        if encode_corrected_context:
            if images.shape != (source_frames, video.shape[3] * 16, video.shape[4] * 16, 3):
                raise ValueError("Corrected H3 frames must match the source latent's frame count and resolution.")
            if not include_merserk_context:
                # Own just the needed tail so a remote result can release the decode.
                context_images = images[-frame_count:].clone()

        if enable_merserk:
            model_management.throw_exception_if_processing_interrupted()
            images, = VTSMerserkTemporalEnhance().enhance(images, return_type="Tensor", **merserk_options)

        if encode_corrected_context:
            if include_merserk_context:
                stride = (merserk_options.get("interpolation_multiplier", 2)
                          if merserk_options.get("enable_frame_interpolation", False)
                          and context_frame_selection == "Original" else 1)
                context_images = images[::stride][-frame_count:].clone()
                if context_images.shape[1:3] != (video.shape[3] * 16, video.shape[4] * 16):
                    context_images = _resize(context_images, video.shape[4] * 16, video.shape[3] * 16)
            model_management.throw_exception_if_processing_interrupted()
            encoded = vae.encode(context_images)
            expected = (1, 24, _steps_for_frames(frame_count), *video.shape[3:])
            if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
                raise ValueError("Corrected H3 VAE encoding must produce latent shape %s." % (expected,))
            context = {"video": encoded.detach().clone(), "frame_count": frame_count,
                       "source_frames": source_frames}
            del encoded, video, context_images

        if kwargs["return_type"] == "Tensor":
            return images, context
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
        return result, context


NODE_CLASS_MAPPINGS = {"VTS VAE Decode Tiled Colour Match": VTS_VAEDecodeTiledColourMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS VAE Decode Tiled Colour Match": "VAE Decode VTS (Tiled + Colour Match)"}
