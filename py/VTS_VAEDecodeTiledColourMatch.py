import logging
import os
import sys
import weakref

import folder_paths

import torch

from comfy import model_management

import_dir = os.path.dirname(__file__)
if import_dir not in sys.path:
    sys.path.append(import_dir)

from VTS_VAEDecodeTiled import VTS_VAEDecodeTiled
from VTS_MiniMaxH3MotionContext import FRAME_PER_TOKEN, _context_length, _pixel_frames, _resize, _steps_for_frames
from VTS_MerserkTemporalEnhance import VTSMerserkTemporalEnhance
from vtsUtils import DiskImage, ensure_image_output_defaults, resolve_list_mapped_output_identity, save_images
from vts_color_correction import METHODS, MODES, correct_images


LOG = logging.getLogger(__name__)

CONTEXT_SOURCES = ["None", "Provided context samples", "Decoded frames (before corrections+Merserk)",
                   "Colour-corrected frames", "Merserk — original frames", "Merserk — interpolated frames"]


def _context_video(samples):
    video = samples.get("samples") if isinstance(samples, dict) else None
    if getattr(video, "is_nested", False):
        video = video.unbind()[0]
    if (not isinstance(video, torch.Tensor) or video.ndim != 5
            or video.shape[0] != 1 or video.shape[1] != 24 or any(size == 0 for size in video.shape)):
        raise ValueError("Context requires one H3 video latent [1,24,T,H,W], optionally inside H3 AV samples.")
    return video


def _fast_decoder_class():
    # Resolve the registered node lazily: importing another package would create
    # a second decoder cache, and normal VTS workflows do not need this package.
    import nodes
    decoder = nodes.NODE_CLASS_MAPPINGS.get("OpenH3RegenFastVAEDecode")
    if decoder is None:
        raise RuntimeError("Fast VAE requires the installed ComfyUI-Open-H3-Regenerate Fast VAE Decode node. "
                           "Enable that package or select fast_vae_name=None.")
    return decoder


def _offload_fast_model(model):
    model.to(torch.device("cpu"))
    model_management.soft_empty_cache()


class _FastDecodeVAE:
    def __init__(self, vae, samples, decoder_class, model, name, keep_loaded):
        self.vae, self.samples = vae, samples
        self.decoder_class, self.model = decoder_class, model
        self.name, self.keep_loaded = name, keep_loaded

    def temporal_compression_decode(self):
        return self.vae.temporal_compression_decode()

    def spacial_compression_decode(self):
        return self.vae.spacial_compression_decode()

    def decode_tiled(self, video, **tiles):
        vae, model = self.vae, self.model
        used_fallback = False

        class TiledFallback(self.decoder_class):
            def _native(self, unused_vae, samples, why):
                nonlocal used_fallback
                used_fallback = True
                # The upstream fallback uses vae.decode and can retain GPU weights.
                # Release those weights first, then honor VTS spatial/temporal tiles.
                _offload_fast_model(model)
                LOG.warning("VTS fast VAE: %s; using connected VAE with VTS tile settings", why)
                return (vae.decode_tiled(video, **tiles),)

        decoder = TiledFallback()
        LOG.info("VTS fast VAE: %s (keep_loaded=%s)", self.name, self.keep_loaded)
        try:
            result = decoder.decode(self.samples, self.name, self.keep_loaded, vae=vae)
        except torch.OutOfMemoryError:
            if used_fallback:
                raise
            # Upstream allocates its output/normalization buffers before its OOM
            # handler. Leave the exception scope so those buffers can be released.
            result = None
        finally:
            if not self.keep_loaded:
                _offload_fast_model(model)
        if result is None:
            result = decoder._native(vae, self.samples, "out of VRAM")
        return result[0]


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
from vts_h3_context import context_controls, store_context

@disk_latent_node(inputs=('samples', 'context_samples'), outputs=(), prefix='VAE Decode VTS (Tiled + Colour Match)')
class VTS_VAEDecodeTiledColourMatch(VTS_VAEDecodeTiled):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["optional"] = {
            "fast_vae_name": (["None", *folder_paths.get_filename_list("vae")], {"default": "None", "tooltip": "None keeps the original tiled VAE decode. Select installed fast H3 decoder weights (for example vae_fast/fast_h3_vae_noncausal_24l.safetensors) to use Open-H3-Regenerate windowed decoding without extra tiling. H3 video/AV latents only; insufficient VRAM falls back to the connected VAE with the tile settings above. Other VAE weights are not compatible."}),
            "fast_vae_keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Fast decoder only. Off keeps cached weights on CPU between decodes; on retains them on GPU for faster reuse (about 6.5 GB). Switching fast weights or selecting None releases this node's previous fast GPU weights. Memory fallback always releases fast GPU weights before tiled decoding."}),
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
            "context_samples": ("LATENT", {"tooltip": "H3 Tensor or DiskLatent, including nested AV. Provided context samples copies its latent tail without VAE conversion. Generated frame sources use only its spatial dimensions; otherwise they use samples dimensions."}),
            "context_source": (CONTEXT_SOURCES, {"default": "None", "tooltip": "Choose the next loop's video context independently of the image output. Generated frame tails use the connected original VAE to encode at context_samples resolution (or samples resolution). Both Merserk sources require Enable Merserk. Interpolated also requires frame interpolation: H3 still conditions at fixed 24 FPS, so denser frames can slow continuation motion or affect audio alignment."}),
            "context_length": (["22", "5", "39", "56"], {"default": "22", "tooltip": "Must match Prepare Loop Context. Copy or encode only the requested tail; short clips use the largest valid H3 length available. Provided latent tails must begin on the native H3 temporal cycle."}),
            "enable_merserk": ("BOOLEAN", {"default": False, "tooltip": "Run Merserk Temporal Enhance after the full sequence is decoded and colour corrected. Scaling, neural rendering and interpolation have separate switches below. Off preserves the existing decoder behavior. Uses lossless 8-bit PNG transport; final files use this decoder's output settings."}),
        })
        # Hidden tuple inputs keep legacy API constants flowing through Comfy's
        # get_input_data without recreating obsolete frontend controls.
        inputs["hidden"] = {
            "encode_corrected_context": ("BOOLEAN",),
            "use_merserk_for_context": ("BOOLEAN",),
            "context_frame_selection": (["Original", "Interpolated"],),
        }
        inputs["optional"].update({"merserk_" + name: spec for name, spec in _merserk_inputs().items()})
        inputs['optional'].update(context_controls('VAE Decode VTS (Tiled + Colour Match)'))
        return inputs

    RETURN_TYPES = ("IMAGE", "VTS_H3_VIDEO_CONTEXT")
    RETURN_NAMES = ("image", "corrected_video_context")
    OUTPUT_TOOLTIPS = ("Decoded images, using the independent image output settings.",
                       "Corrected H3 video latent tail and frame metadata. Tensor or DiskLatent according to context_return_type; None when Context Source is None.")
    DESCRIPTION = (
        "Tiled VAE decode with optional reference colour correction. Uses weighted "
        "colour matching, reference white balance, brightness and contrast, in that "
        "order. Corrects before saving. No reference or zero overall weight bypasses "
        "colour correction. GPU Lab follows KJNodes' Lab statistics approach; "
        "reference tone controls adapt Donut-style operations. Optionally encode "
        "an H3 frame tail or copy provided context samples for Prepare Loop Context; "
        "original audio stays unchanged. Optional Merserk scaling, temporal enhancement "
        "and interpolation run after colour correction. Context Source selects the "
        "processing stage; context_samples optionally sets the encoding resolution.")

    def _decode_vae(self, vae, samples, name, keep_loaded):
        previous_ref = getattr(self, "_fast_model_ref", None)
        previous = previous_ref() if previous_ref is not None else None
        if name == "None":
            if previous is not None:
                _offload_fast_model(previous)
            self._fast_model_ref = None
            self._fast_model_key = None
            return vae

        video = samples["samples"]
        if getattr(video, "is_nested", False):
            parts = video.unbind()
            video = parts[0] if parts else None
        if (not isinstance(video, torch.Tensor) or video.ndim != 5
                or video.shape[1] != 24 or any(size == 0 for size in video.shape)):
            raise ValueError("Fast VAE requires an H3 video latent [B,24,T,H,W] or an H3 AV latent containing it.")

        decoder_class = _fast_decoder_class()
        path = folder_paths.get_full_path_or_raise("vae", name)
        key = (path, os.path.getmtime(path))
        if previous is not None and key != self._fast_model_key:
            _offload_fast_model(previous)
        # Use the external node's existing path/mtime cache, never a second loader.
        module = sys.modules[decoder_class.__module__]
        model = module._fast_vae(path)
        self._fast_model_ref = weakref.ref(model)
        self._fast_model_key = key
        return _FastDecodeVAE(vae, samples, decoder_class, model, name, keep_loaded)

    def decode(self, vae, samples, tile_size_x=512, tile_size_y=512, overlap=64,
               temporal_size=64, temporal_overlap=8, color_ref=None,
               color_match_method="reinhard_lab_gpu", color_match_weight=0.5,
               white_balance_weight=0.0, brightness_method="gamma",
               brightness_weight=0.0, contrast_weight=0.0,
               calculation_mode="fixed_per_clip", smoothing=0.9, overall_weight=1.0,
               analysis_size=128, lut_resolution=33, encode_corrected_context=False,
               context_length="22", enable_merserk=False, use_merserk_for_context=False,
               context_frame_selection="Original", context_return_type="Tensor",
               context_output_dir="./tmp/disklatents", context_prefix="VAE_Decode_VTS_(Tiled_+_Colour_Match)",
               context_start_sequence=0, context_compression_level=3,
               fast_vae_name="None", fast_vae_keep_loaded=False, context_samples=None,
               context_source=None, **kwargs):
        merserk_options = {name: kwargs.pop("merserk_" + name)
                           for name in _merserk_inputs() if "merserk_" + name in kwargs}
        kwargs = ensure_image_output_defaults(kwargs)
        decode_args = dict(tile_size_x=tile_size_x, tile_size_y=tile_size_y,
                           overlap=overlap, temporal_size=temporal_size,
                           temporal_overlap=temporal_overlap)
        decode_vae = self._decode_vae(vae, samples, fast_vae_name, fast_vae_keep_loaded)
        # Explicit new sources win; missing sources retain old saved API behavior.
        legacy_context = context_source is None
        if legacy_context:
            context_source = "None"
            if encode_corrected_context:
                context_source = "Colour-corrected frames"
                if enable_merserk and use_merserk_for_context:
                    if context_frame_selection not in ("Original", "Interpolated"):
                        raise ValueError("Context frame selection must be Original or Interpolated.")
                    context_source = ("Merserk — interpolated frames" if context_frame_selection == "Interpolated"
                                      and merserk_options.get("enable_frame_interpolation", False)
                                      else "Merserk — original frames")
        if context_source not in CONTEXT_SOURCES:
            raise ValueError("Unknown context_source: %s." % context_source)
        has_context = context_source != "None"
        provided_context = context_source == "Provided context samples"
        merserk_context = context_source.startswith("Merserk —")
        if merserk_context and not enable_merserk:
            raise ValueError("Merserk context sources require enable_merserk.")
        if context_source == "Merserk — interpolated frames" and not merserk_options.get("enable_frame_interpolation", False):
            raise ValueError("Merserk interpolated context requires merserk_enable_frame_interpolation.")
        bypass_correction = color_ref is None or overall_weight == 0 or not any(
            (color_match_weight, white_balance_weight, brightness_weight, contrast_weight))
        if bypass_correction and not has_context and not enable_merserk:
            return (*super().decode(decode_vae, samples, **decode_args, **kwargs), None)

        context = None
        context_images = None
        if has_context:
            if str(context_length) not in ("5", "22", "39", "56"):
                raise ValueError("H3 context_length must be 5, 22, 39 or 56.")
            if provided_context and context_samples is None:
                raise ValueError("Provided context samples requires context_samples.")
            video = _context_video(context_samples if provided_context else samples)
            target_video = _context_video(context_samples) if context_samples is not None else video
            source_frames = _pixel_frames(int(video.shape[2]))
            frame_count = _context_length(int(context_length), source_frames)
            if provided_context:
                steps = _steps_for_frames(frame_count)
                start = int(video.shape[2]) - steps
                # A shifted latent cycle would claim a different number of frames
                # after the next H3 decode; never silently relabel that tail.
                if start % len(FRAME_PER_TOKEN) != 0:
                    raise ValueError("Provided context samples latent tail must start on the native H3 temporal cycle.")
                context = {"video": video[:, :, start:].detach().clone(),
                           "frame_count": frame_count, "source_frames": source_frames}

        tensor_options = dict(kwargs, return_type="Tensor")
        images, = super().decode(decode_vae, samples, **decode_args, **tensor_options)
        if has_context and not provided_context:
            if images.shape != (source_frames, video.shape[3] * 16, video.shape[4] * 16, 3):
                raise ValueError("H3 frames must match the source latent's frame count and resolution.")
            if context_source == "Decoded frames (before corrections+Merserk)":
                # Correction mutates the decoded buffer, so own the chosen tail first.
                context_images = images[-frame_count:].clone()
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

        if context_source == "Colour-corrected frames":
            context_images = images[-frame_count:].clone()

        if enable_merserk:
            model_management.throw_exception_if_processing_interrupted()
            images, = VTSMerserkTemporalEnhance().enhance(images, return_type="Tensor", **merserk_options)

        if has_context and not provided_context:
            if merserk_context:
                stride = (merserk_options.get("interpolation_multiplier", 2)
                          if merserk_options.get("enable_frame_interpolation", False)
                          and context_source == "Merserk — original frames" else 1)
                context_images = images[::stride][-frame_count:].clone()
            if len(context_images) != frame_count:
                raise ValueError("Selected context source has fewer frames than the requested H3 tail.")
            # Select the temporal tail before resizing to avoid resizing a full clip.
            if context_images.shape[1:3] != (target_video.shape[3] * 16, target_video.shape[4] * 16):
                context_images = _resize(context_images, target_video.shape[4] * 16, target_video.shape[3] * 16)
            model_management.throw_exception_if_processing_interrupted()
            encoded = vae.encode(context_images)
            expected = (1, 24, _steps_for_frames(frame_count), *target_video.shape[3:])
            if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
                raise ValueError("Corrected H3 VAE encoding must produce latent shape %s." % (expected,))
            context = {"video": encoded.detach().clone(), "frame_count": frame_count,
                       "source_frames": source_frames}
            del encoded, video, target_video, context_images

        context = store_context(context, context_return_type, context_output_dir, context_prefix,
                                context_start_sequence, context_compression_level)

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
