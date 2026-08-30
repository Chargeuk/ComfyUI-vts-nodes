"""Marker-safe Ref1/Ref2 preparation nodes for MiniMax H3 Ref2VA."""

from __future__ import annotations

import json
import os
import sys

import torch


_UTILS = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _UTILS not in sys.path:
    sys.path.append(_UTILS)

from ref2va import mark_ref2, validate_comfy_image
from vr180_projection import rectilinear_to_vr180, vr180_to_rectilinear


def _validate_reference_pair(ref1: torch.Tensor, ref2: torch.Tensor) -> None:
    validate_comfy_image(ref1, "ref1")
    validate_comfy_image(ref2, "ref2")
    if tuple(ref1.shape[:3]) != tuple(ref2.shape[:3]):
        raise ValueError(
            "Ref1 and Ref2 must have identical [B,H,W] dimensions: "
            f"{tuple(ref1.shape[:3])} != {tuple(ref2.shape[:3])}"
        )


def _diagnostics(result, *, width: int, height: int, extra=None) -> str:
    data = {
        "reference_order": ["Ref1", "Ref2"],
        "size": [width, height],
        "batch_size": len(result.marker_counts),
        "effective_mask_pixels": list(result.marker_counts),
        "exact_marker_pixels": list(result.marker_counts),
        "escaped_known_ref2_pixels": list(result.escaped_counts),
        "marker_iff_effective_mask": True,
        "marker_rgb": "#FF00FF",
        "escaped_rgb": "#FF00FE",
    }
    if extra:
        data.update(extra)
    return json.dumps(data, sort_keys=True)


def prepare_stereo_references(ref1_left, incomplete_right, effective_mask):
    """Return Ref1 unchanged and a marker-safe Ref2 in fixed order."""

    _validate_reference_pair(ref1_left, incomplete_right)
    result = mark_ref2(incomplete_right, effective_mask)
    height, width = int(incomplete_right.shape[1]), int(incomplete_right.shape[2])
    return (
        ref1_left,
        result.image,
        result.effective_mask.to(dtype=incomplete_right.dtype),
        _diagnostics(result, width=width, height=height, extra={"workflow": "stereo_repair"}),
    )


def _aspect_parts(rectilinear_aspect: str):
    try:
        width, height = (int(value) for value in rectilinear_aspect.split(":"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("rectilinear_aspect must be '16:9' or '4:3'") from None
    if (width, height) not in {(16, 9), (4, 3)}:
        raise ValueError("rectilinear_aspect must be '16:9' or '4:3'")
    return width, height


def _letterbox_size(canvas_height: int, canvas_width: int, aspect_width: int, aspect_height: int):
    if canvas_width * aspect_height <= canvas_height * aspect_width:
        width = canvas_width
        height = max(1, round(canvas_width * aspect_height / aspect_width))
    else:
        height = canvas_height
        width = max(1, round(canvas_height * aspect_width / aspect_height))
    return height, width


def prepare_projection_references(
    left_vr180,
    rectilinear_aspect,
    horizontal_fov_degrees,
    yaw_degrees,
    pitch_degrees,
    roll_degrees,
    chunk_rows,
    sampling,
):
    """Derive a letterboxed rectilinear Ref1 and reprojected VR180 Ref2."""

    validate_comfy_image(left_vr180, "left_vr180")
    canvas_height = int(left_vr180.shape[1])
    canvas_width = int(left_vr180.shape[2])
    aspect_width, aspect_height = _aspect_parts(rectilinear_aspect)
    rect_height, rect_width = _letterbox_size(
        canvas_height, canvas_width, aspect_width, aspect_height
    )
    source = left_vr180[..., :3].movedim(-1, 1).contiguous()
    rect = vr180_to_rectilinear(
        source,
        (rect_height, rect_width),
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        unknown_color=(0.0, 0.0, 0.0),
        chunk_rows=chunk_rows,
        mode=sampling,
    )
    if not bool(rect.known_mask.all()):
        raise ValueError(
            "The requested rectilinear view extends outside the source VR180 hemisphere; "
            "reduce FOV or pose."
        )

    projected = rectilinear_to_vr180(
        rect.image,
        (canvas_height, canvas_width),
        horizontal_fov_degrees=horizontal_fov_degrees,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        roll_degrees=roll_degrees,
        unknown_color=(1.0, 0.0, 1.0),
        chunk_rows=chunk_rows,
        mode=sampling,
    )

    ref1 = left_vr180.new_zeros(left_vr180.shape[0], canvas_height, canvas_width, 3)
    top = (canvas_height - rect_height) // 2
    left = (canvas_width - rect_width) // 2
    ref1[:, top:top + rect_height, left:left + rect_width] = rect.image.movedim(1, -1)

    raw_ref2 = projected.image.movedim(1, -1).contiguous()
    result = mark_ref2(raw_ref2, projected.unknown_mask[:, 0])
    extra = {
        "workflow": "projection_outpainting",
        "projection": "half_equirectangular_180",
        "pixel_centres": True,
        "align_corners": False,
        "rectilinear_aspect": rectilinear_aspect,
        "rectilinear_content_size": [rect_width, rect_height],
        "letterbox": [left, top, canvas_width - rect_width - left, canvas_height - rect_height - top],
        "horizontal_fov_degrees": float(horizontal_fov_degrees),
        "yaw_degrees": float(yaw_degrees),
        "pitch_degrees": float(pitch_degrees),
        "roll_degrees": float(roll_degrees),
        "known_geometry_pixels": [
            int(value) for value in projected.known_mask[:, 0].sum(dim=(1, 2)).tolist()
        ],
    }
    return (
        ref1,
        result.image,
        result.effective_mask.to(dtype=left_vr180.dtype),
        _diagnostics(result, width=canvas_width, height=canvas_height, extra=extra),
    )


class VTSRef2VAStereoReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref1_left": ("IMAGE",),
                "incomplete_right": ("IMAGE",),
                "effective_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("ref1", "ref2", "effective_mask", "diagnostics")
    FUNCTION = "prepare"
    CATEGORY = "VTS/Ref2VA"
    DESCRIPTION = "Prepare fixed-order stereo Ref1/Ref2 images with exact marker safety."

    def prepare(self, ref1_left, incomplete_right, effective_mask):
        return prepare_stereo_references(ref1_left, incomplete_right, effective_mask)


class VTSRef2VAProjectionReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_vr180": ("IMAGE",),
                "rectilinear_aspect": (["16:9", "4:3"], {"default": "16:9"}),
                "horizontal_fov_degrees": (
                    "FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.1}
                ),
                "yaw_degrees": (
                    "FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}
                ),
                "pitch_degrees": (
                    "FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}
                ),
                "roll_degrees": (
                    "FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}
                ),
                "chunk_rows": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 16}),
                "sampling": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("ref1", "ref2", "effective_mask", "diagnostics")
    FUNCTION = "prepare"
    CATEGORY = "VTS/Ref2VA"
    DESCRIPTION = "Create fixed-order letterboxed rectilinear Ref1 and partial VR180 Ref2."

    def prepare(
        self,
        left_vr180,
        rectilinear_aspect,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        chunk_rows,
        sampling,
    ):
        return prepare_projection_references(
            left_vr180,
            rectilinear_aspect,
            horizontal_fov_degrees,
            yaw_degrees,
            pitch_degrees,
            roll_degrees,
            chunk_rows,
            sampling,
        )


NODE_CLASS_MAPPINGS = {
    "VTSRef2VAStereoReferences": VTSRef2VAStereoReferences,
    "VTSRef2VAProjectionReferences": VTSRef2VAProjectionReferences,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSRef2VAStereoReferences": "VTS Ref2VA Stereo References",
    "VTSRef2VAProjectionReferences": "VTS Ref2VA Projection References",
}
