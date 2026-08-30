"""Deterministic ComfyUI wrappers for rectilinear <-> VR180 projection.

The projection mathematics live in :mod:`vtsUtils.vr180_projection`.  These
thin wrappers add ComfyUI IMAGE/MASK layout handling, bounded frame batching,
and explicit JSON geometry metadata.  No input or output frames are cached.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch


_UTILS = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _UTILS not in sys.path:
    sys.path.append(_UTILS)

from vr180_projection import rectilinear_to_vr180, vr180_to_rectilinear


def _validate_image(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
        raise ValueError(
            "images must be a ComfyUI IMAGE [B,H,W,C] with RGB channels, "
            f"got {tuple(getattr(images, 'shape', ()))}"
        )
    if images.shape[1] < 1 or images.shape[2] < 1:
        raise ValueError("images must have positive height and width")
    if not images.is_floating_point():
        raise TypeError("images must use a floating-point dtype")
    return images


def _validate_parameters(
    output_width: int,
    output_height: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    frame_batch_size: int,
) -> tuple[int, int, int]:
    width, height = int(output_width), int(output_height)
    if width < 1 or height < 1:
        raise ValueError("output_width and output_height must be positive")
    fov = float(horizontal_fov_degrees)
    if not math.isfinite(fov) or not 0.01 <= fov < 180.0:
        raise ValueError("horizontal_fov_degrees must be in [0.01, 180)")
    if not all(
        math.isfinite(float(value))
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ):
        raise ValueError("projection angles must be finite")
    frames = int(frame_batch_size)
    if frames < 1:
        raise ValueError("frame_batch_size must be at least 1")
    return width, height, frames


def _validate_source_mask(mask: torch.Tensor | None, images: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("source_known_mask must be a torch.Tensor")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("source_known_mask must be [B,H,W] or [B,1,H,W]")
    if mask.shape[0] not in (1, images.shape[0]) or tuple(mask.shape[2:]) != tuple(images.shape[1:3]):
        raise ValueError("source_known_mask dimensions must match the source image")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError("source_known_mask must contain finite values")
    return mask


def _mask_batch(mask: torch.Tensor | None, start: int, stop: int) -> torch.Tensor | None:
    if mask is None or mask.shape[0] == 1:
        return mask
    return mask[start:stop]


def _run_frame_batches(
    images_bchw: torch.Tensor,
    *,
    output_size: tuple[int, int],
    frame_batch_size: int,
    projector,
    source_known_mask: torch.Tensor | None = None,
    **projection_kwargs,
):
    """Project bounded frame groups into one required ComfyUI output batch."""

    batch, channels = int(images_bchw.shape[0]), int(images_bchw.shape[1])
    output_height, output_width = output_size
    output = torch.empty(
        (batch, channels, output_height, output_width),
        dtype=images_bchw.dtype,
        device=images_bchw.device,
    )
    known = torch.empty(
        (batch, 1, output_height, output_width),
        dtype=torch.bool,
        device=images_bchw.device,
    )

    for start in range(0, batch, frame_batch_size):
        stop = min(start + frame_batch_size, batch)
        kwargs = dict(projection_kwargs)
        if source_known_mask is not None:
            kwargs["source_known_mask"] = _mask_batch(source_known_mask, start, stop)
        result = projector(images_bchw[start:stop], output_size, **kwargs)
        output[start:stop].copy_(result.image)
        known[start:stop].copy_(result.known_mask)

    return output, known, ~known


def _metadata(
    *,
    direction: str,
    images: torch.Tensor,
    output_width: int,
    output_height: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    chunk_rows: int,
    frame_batch_size: int,
    sampling: str,
    known_mask: torch.Tensor,
) -> str:
    known_counts = [int(value) for value in known_mask[:, 0].sum(dim=(1, 2)).tolist()]
    pixels = int(output_width) * int(output_height)
    return json.dumps(
        {
            "align_corners": False,
            "batch_size": int(images.shape[0]),
            "chunk_rows": int(chunk_rows),
            "coordinate_system": "+X right, +Y up, +Z forward",
            "direction": direction,
            "frame_batch_size": int(frame_batch_size),
            "horizontal_fov_degrees": float(horizontal_fov_degrees),
            "input_size": [int(images.shape[2]), int(images.shape[1])],
            "known_fraction": [value / pixels for value in known_counts],
            "known_pixels": known_counts,
            "output_size": [int(output_width), int(output_height)],
            "pitch_degrees": float(pitch_degrees),
            "pixel_centres": True,
            "projection": "half_equirectangular_180",
            "roll_degrees": float(roll_degrees),
            "sampling": sampling,
            "unknown_rgb": "#FF00FF",
            "yaw_degrees": float(yaw_degrees),
        },
        sort_keys=True,
    )


def project_rectilinear_to_vr180(
    images,
    output_width,
    output_height,
    horizontal_fov_degrees,
    yaw_degrees,
    pitch_degrees,
    roll_degrees,
    chunk_rows,
    frame_batch_size,
    sampling,
):
    images = _validate_image(images)
    width, height, frames = _validate_parameters(
        output_width,
        output_height,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        frame_batch_size,
    )
    bchw = images[..., :3].movedim(-1, 1).contiguous()
    projected, known, unknown = _run_frame_batches(
        bchw,
        output_size=(height, width),
        frame_batch_size=frames,
        projector=rectilinear_to_vr180,
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        unknown_color=(1.0, 0.0, 1.0),
        chunk_rows=chunk_rows,
        mode=sampling,
    )
    metadata = _metadata(
        direction="rectilinear_to_vr180",
        images=images,
        output_width=width,
        output_height=height,
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        chunk_rows=chunk_rows,
        frame_batch_size=frames,
        sampling=sampling,
        known_mask=known,
    )
    return projected.movedim(1, -1).contiguous(), known[:, 0].float(), unknown[:, 0].float(), metadata


def project_vr180_to_rectilinear(
    images,
    output_width,
    output_height,
    horizontal_fov_degrees,
    yaw_degrees,
    pitch_degrees,
    roll_degrees,
    chunk_rows,
    frame_batch_size,
    sampling,
    source_known_mask=None,
):
    images = _validate_image(images)
    width, height, frames = _validate_parameters(
        output_width,
        output_height,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        frame_batch_size,
    )
    source_known_mask = _validate_source_mask(source_known_mask, images)
    bchw = images[..., :3].movedim(-1, 1).contiguous()
    projected, known, unknown = _run_frame_batches(
        bchw,
        output_size=(height, width),
        frame_batch_size=frames,
        projector=vr180_to_rectilinear,
        source_known_mask=source_known_mask,
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        unknown_color=(1.0, 0.0, 1.0),
        chunk_rows=chunk_rows,
        mode=sampling,
    )
    metadata = _metadata(
        direction="vr180_to_rectilinear",
        images=images,
        output_width=width,
        output_height=height,
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        chunk_rows=chunk_rows,
        frame_batch_size=frames,
        sampling=sampling,
        known_mask=known,
    )
    return projected.movedim(1, -1).contiguous(), known[:, 0].float(), unknown[:, 0].float(), metadata


class VTSRectilinearToVR180:
    CATEGORY = "VTS/VR180"
    FUNCTION = "project"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("partial_vr180", "known_mask", "unknown_mask", "metadata")
    DESCRIPTION = "Project a rectilinear image/video batch into a partial square VR180 view."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common_inputs("images")}

    def project(self, **kwargs):
        return project_rectilinear_to_vr180(**kwargs)


class VTSVR180ToRectilinear:
    CATEGORY = "VTS/VR180"
    FUNCTION = "project"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("rectilinear", "known_mask", "unknown_mask", "metadata")
    DESCRIPTION = "Render a rectilinear image/video batch from a square VR180 view."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _common_inputs("images"),
            "optional": {"source_known_mask": ("MASK",)},
        }

    def project(self, **kwargs):
        return project_vr180_to_rectilinear(**kwargs)


def _common_inputs(image_name: str) -> dict:
    return {
        image_name: ("IMAGE",),
        "output_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
        "output_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
        "horizontal_fov_degrees": (
            "FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.1}
        ),
        "yaw_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
        "pitch_degrees": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
        "roll_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
        "chunk_rows": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 16}),
        "frame_batch_size": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
        "sampling": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
    }


NODE_CLASS_MAPPINGS = {
    "VTSRectilinearToVR180": VTSRectilinearToVR180,
    "VTSVR180ToRectilinear": VTSVR180ToRectilinear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSRectilinearToVR180": "VTS Rectilinear Video To VR180",
    "VTSVR180ToRectilinear": "VTS VR180 To Rectilinear Video",
}

