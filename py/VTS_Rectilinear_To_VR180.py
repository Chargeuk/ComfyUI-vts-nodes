"""ComfyUI nodes for the staged VTS half-equirectangular projection core."""

from __future__ import annotations

import json
import os
import sys

import torch


_UTILS = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _UTILS not in sys.path:
    sys.path.append(_UTILS)

from vr180_projection import rectilinear_to_vr180, vr180_to_rectilinear


def _bchw(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a ComfyUI IMAGE tensor [B,H,W,C]")
    if images.shape[-1] < 3:
        raise ValueError("images must contain at least RGB channels")
    return images.movedim(-1, 1).contiguous()


def _bhwc(images: torch.Tensor) -> torch.Tensor:
    return images.movedim(1, -1).contiguous()


class VTSRectilinearToVR180:
    """Place a rectilinear image in a half-equirectangular VR180 canvas."""

    CATEGORY = "VTS/VR180"
    FUNCTION = "project"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("partial_vr180", "known_mask", "unknown_mask", "metadata")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "output_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "horizontal_fov_degrees": (
                    "FLOAT",
                    {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.1},
                ),
                "yaw_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch_degrees": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "chunk_rows": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 16}),
                "sampling": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            }
        }

    def project(
        self,
        images,
        output_width,
        output_height,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        chunk_rows,
        sampling,
    ):
        result = rectilinear_to_vr180(
            _bchw(images),
            (output_height, output_width),
            horizontal_fov_degrees=horizontal_fov_degrees,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            roll_degrees=roll_degrees,
            unknown_color=(1.0, 0.0, 1.0),
            chunk_rows=chunk_rows,
            mode=sampling,
        )
        metadata = json.dumps(
            {
                "projection": "half_equirectangular_180",
                "coordinate_system": "+X right, +Y up, +Z forward",
                "pixel_centres": True,
                "align_corners": False,
                "unknown_rgb": "#FF00FF",
                "input_size": [int(images.shape[2]), int(images.shape[1])],
                "output_size": [int(output_width), int(output_height)],
                "horizontal_fov_degrees": float(horizontal_fov_degrees),
                "yaw_degrees": float(yaw_degrees),
                "pitch_degrees": float(pitch_degrees),
                "roll_degrees": float(roll_degrees),
                "known_fraction": float(result.known_mask.float().mean().item()),
            },
            sort_keys=True,
        )
        return (
            _bhwc(result.image),
            result.known_mask[:, 0].float(),
            result.unknown_mask[:, 0].float(),
            metadata,
        )


class VTSVR180ToRectilinear:
    """Render a rectilinear view from a half-equirectangular VR180 image."""

    CATEGORY = "VTS/VR180"
    FUNCTION = "project"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("rectilinear", "known_mask", "unknown_mask", "metadata")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "output_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "horizontal_fov_degrees": (
                    "FLOAT",
                    {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.1},
                ),
                "yaw_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch_degrees": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "chunk_rows": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 16}),
                "sampling": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            },
            "optional": {"source_known_mask": ("MASK",)},
        }

    def project(
        self,
        images,
        output_width,
        output_height,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        chunk_rows,
        sampling,
        source_known_mask=None,
    ):
        source_mask = None
        if source_known_mask is not None:
            source_mask = source_known_mask.unsqueeze(1) if source_known_mask.ndim == 3 else source_known_mask
        result = vr180_to_rectilinear(
            _bchw(images),
            (output_height, output_width),
            horizontal_fov_degrees=horizontal_fov_degrees,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            roll_degrees=roll_degrees,
            source_known_mask=source_mask,
            unknown_color=(1.0, 0.0, 1.0),
            chunk_rows=chunk_rows,
            mode=sampling,
        )
        metadata = json.dumps(
            {
                "projection": "rectilinear_from_half_equirectangular_180",
                "coordinate_system": "+X right, +Y up, +Z forward",
                "pixel_centres": True,
                "align_corners": False,
                "unknown_rgb": "#FF00FF",
                "input_size": [int(images.shape[2]), int(images.shape[1])],
                "output_size": [int(output_width), int(output_height)],
                "horizontal_fov_degrees": float(horizontal_fov_degrees),
                "yaw_degrees": float(yaw_degrees),
                "pitch_degrees": float(pitch_degrees),
                "roll_degrees": float(roll_degrees),
                "known_fraction": float(result.known_mask.float().mean().item()),
            },
            sort_keys=True,
        )
        return (
            _bhwc(result.image),
            result.known_mask[:, 0].float(),
            result.unknown_mask[:, 0].float(),
            metadata,
        )


NODE_CLASS_MAPPINGS = {
    "VTSRectilinearToVR180": VTSRectilinearToVR180,
    "VTSVR180ToRectilinear": VTSVR180ToRectilinear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSRectilinearToVR180": "VTS Rectilinear Video To VR180",
    "VTSVR180ToRectilinear": "VTS VR180 To Rectilinear Video",
}

