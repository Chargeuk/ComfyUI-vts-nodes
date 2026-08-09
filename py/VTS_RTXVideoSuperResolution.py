import os
import sys

import torch
from comfy import model_management
import nodes as core_nodes

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, resolve_list_mapped_output_identity, save_images, vtsImageTypes, vtsReturnTypes


class VTS_RTXVideoSuperResolution:
    RESIZE_TYPES = ["scale by multiplier", "target dimensions"]
    VSR_QUALITIES = ["LOW", "MEDIUM", "HIGH", "ULTRA"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_type": (cls.RESIZE_TYPES,),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "vsr_quality": (cls.VSR_QUALITIES, {"default": "ULTRA"}),
                "return_type": (vtsReturnTypes, {"default": "Input", "tooltip": "Return the same type as the input images, force DiskImage output, or force Tensor output."}),
                "batch_size": ("INT", {"default": 4, "min": 1, "tooltip": "Number of input frames processed before DiskImage output is saved and released."}),
                "prefix": ("STRING", {"default": "rtx_vsr", "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for PNG and 0-6 for WebP)."}),
                "output_quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "File quality (1-100), or 101 for lossless WebP. This is separate from VSR quality."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = "VTS/image/upscaling"
    DESCRIPTION = "RTX Video Super Resolution with batched DiskImage input and output support."

    def _resolve_return_type(self, return_type, images):
        if return_type == "Input":
            return "Tensor" if isinstance(images, torch.Tensor) else "DiskImage"
        if return_type == "Input or DiskImage":
            return "DiskImage"
        return return_type

    def _get_rtx_node(self):
        node_cls = core_nodes.NODE_CLASS_MAPPINGS.get("RTXVideoSuperResolution")
        if node_cls is None:
            raise ImportError("VTS RTX Video Super Resolution requires the NVIDIA RTX Video Super Resolution custom node.")
        return node_cls()

    def _resize_config(self, resize_type, scale, width, height):
        if resize_type == "scale by multiplier":
            return {"resize_type": resize_type, "scale": scale}
        return {"resize_type": resize_type, "width": width, "height": height}

    def _run_rtx(self, rtx_node, images, resize_config, vsr_quality):
        result = rtx_node.execute(images=images, resize_type=resize_config, quality=vsr_quality)
        if hasattr(result, "args"):
            result = result.args
        if not isinstance(result, (tuple, list)) or not result or not isinstance(result[0], torch.Tensor):
            raise RuntimeError("RTX Video Super Resolution did not return an image tensor.")
        return result[0]

    def _output_overlaps_input(self, images, prefix, start_sequence, output_dir, format):
        if not isinstance(images, DiskImage) or images.number_of_images == 0:
            return False
        if images.prefix != prefix or images.format != format:
            return False
        if os.path.abspath(images.output_dir) != os.path.abspath(output_dir):
            return False

        input_start = images.start_sequence
        input_end = input_start + images.number_of_images - 1
        output_end = start_sequence + images.number_of_images - 1
        return not (output_end < input_start or start_sequence > input_end)

    def _upscale_to_tensor(self, rtx_node, images, resize_config, vsr_quality):
        materialized = images.materialize() if isinstance(images, DiskImage) else images
        try:
            return self._run_rtx(rtx_node, materialized, resize_config, vsr_quality)
        finally:
            if isinstance(images, DiskImage):
                del materialized

    def _upscale_to_disk(self, rtx_node, images, resize_config, vsr_quality, batch_size, prefix, start_sequence, output_dir, format, num_workers, compression_level, output_quality):
        prefix, start_sequence = resolve_list_mapped_output_identity(prefix, start_sequence)
        if self._output_overlaps_input(images, prefix, start_sequence, output_dir, format):
            raise ValueError("VTS RTX Video Super Resolution: output path overlaps the input DiskImage sequence.")

        number_of_images = images.shape[0] if isinstance(images, torch.Tensor) else images.number_of_images
        output_sequence = start_sequence
        output_shape = None
        normalized_quality = None if output_quality > 100 else output_quality

        for batch_start in range(0, number_of_images, batch_size):
            batch_count = min(batch_size, number_of_images - batch_start)
            if isinstance(images, torch.Tensor):
                input_batch = images[batch_start:batch_start + batch_count]
            else:
                input_batch = images.materialize(start=batch_start, count=batch_count)

            upscaled = None
            try:
                upscaled = self._run_rtx(rtx_node, input_batch, resize_config, vsr_quality)
                if output_shape is None:
                    output_shape = tuple(upscaled.shape[1:])
                save_images(
                    image=upscaled,
                    prefix=prefix,
                    start_sequence=output_sequence,
                    output_dir=output_dir,
                    format=format,
                    num_workers=num_workers,
                    compression_level=compression_level,
                    quality=normalized_quality,
                )
                output_sequence += upscaled.shape[0]
            finally:
                del input_batch
                if upscaled is not None:
                    del upscaled
                model_management.soft_empty_cache()

        result = DiskImage(
            prefix=prefix,
            start_sequence=start_sequence,
            number_of_images=output_sequence - start_sequence,
            output_dir=output_dir,
            format=format,
            image=None,
            compression_level=compression_level,
            quality=normalized_quality,
        )
        if output_shape is not None:
            result.shape = (result.number_of_images,) + output_shape
            result.ndim = len(result.shape)
        else:
            result.shape = (0,) + tuple(images.shape[1:])
            result.ndim = len(result.shape)
        result.dtype = torch.float32
        return result

    def upscale(self, images, resize_type, scale, width, height, vsr_quality, return_type, batch_size, prefix, start_sequence, output_dir, format, num_workers, compression_level, output_quality):
        resolved_return_type = self._resolve_return_type(return_type, images)
        resize_config = self._resize_config(resize_type, scale, width, height)
        rtx_node = self._get_rtx_node()

        if resolved_return_type == "Tensor":
            result = self._upscale_to_tensor(rtx_node, images, resize_config, vsr_quality)
        else:
            result = self._upscale_to_disk(
                rtx_node,
                images,
                resize_config,
                vsr_quality,
                batch_size,
                prefix,
                start_sequence,
                output_dir,
                format,
                num_workers,
                compression_level,
                output_quality,
            )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VTS RTX Video Super Resolution": VTS_RTXVideoSuperResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS RTX Video Super Resolution": "VTS RTX Video Super Resolution",
}
