"""ComfyUI node for placing square VR180 sources on a full ERP canvas."""

from __future__ import annotations

import math
import os
import sys
from typing import Iterator, Tuple

import torch


_UTILS = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _UTILS not in sys.path:
    sys.path.append(_UTILS)

from vr180_outpaint import (  # noqa: E402
    PROJECTION_MODES,
    resolve_projection_fov,
    vr180_square_to_full_erp,
)


def _parse_fill_color(value: str) -> Tuple[float, float, float]:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ValueError("fill_color must be #RGB or #RRGGBB")
    try:
        channels = tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    except ValueError as error:
        raise ValueError("fill_color must be #RGB or #RRGGBB") from error
    return channels


def _is_disk_image(value) -> bool:
    return (
        not isinstance(value, torch.Tensor)
        and hasattr(value, "load_images")
        and hasattr(value, "number_of_images")
        and hasattr(value, "start_sequence")
    )


def _source_count(source) -> int:
    if isinstance(source, torch.Tensor):
        if source.ndim != 4:
            raise ValueError(
                "tensor source must be a ComfyUI IMAGE [B,H,W,C], "
                f"got {tuple(source.shape)}"
            )
        return int(source.shape[0])
    if not _is_disk_image(source):
        raise TypeError("source must be a ComfyUI IMAGE tensor or VTS DiskImage")
    count = int(source.number_of_images)
    if count < 1:
        raise ValueError("DiskImage source must contain at least one image")
    return count


def _source_batches(source, batch_size: int) -> Iterator[torch.Tensor]:
    count = _source_count(source)
    if isinstance(source, torch.Tensor):
        for start in range(0, count, batch_size):
            yield source[start : min(start + batch_size, count)]
        return

    for start in range(0, count, batch_size):
        amount = min(batch_size, count - start)
        batch = source.load_images(
            start_sequence=int(source.start_sequence) + start,
            count=amount,
        )
        if not isinstance(batch, torch.Tensor):
            raise TypeError("DiskImage.load_images must return a torch.Tensor")
        yield batch


def _horizontal_extent_from_known_mask(known_mask: torch.Tensor) -> Tuple[int, int]:
    """Return the smallest circular x interval containing all known pixels.

    For the usual non-wrapped projection, ``left_x <= right_x``. If the
    projection crosses the ERP wrap seam, ``left_x > right_x`` and the retained
    interval is ``[left_x, width)`` plus ``[0, right_x]``.
    """

    if known_mask.ndim != 3:
        raise ValueError(
            "known_mask must have shape [frames,height,width], "
            f"got {tuple(known_mask.shape)}"
        )
    width = int(known_mask.shape[2])
    occupied_columns = known_mask.bool().any(dim=(0, 1))
    indices = occupied_columns.nonzero(as_tuple=False).flatten().cpu().tolist()
    if not indices:
        raise ValueError(
            "projection and trims leave no retained pixels at the requested output resolution"
        )
    if len(indices) == width:
        return 0, width - 1

    # Find the largest circular run of unoccupied columns. Its complement is
    # the smallest circular interval containing the projected source.
    best_gap = -1
    best_left = indices[0]
    best_right = indices[-1]
    for position, right_edge in enumerate(indices):
        next_left = indices[(position + 1) % len(indices)]
        if position == len(indices) - 1:
            next_left += width
        gap = next_left - right_edge - 1
        if gap > best_gap:
            best_gap = gap
            best_left = next_left % width
            best_right = right_edge
    return int(best_left), int(best_right)


def project_square_vr180_to_erp(
    source,
    projection_mode,
    output_width,
    output_height,
    custom_horizontal_fov_degrees,
    custom_vertical_fov_degrees,
    yaw_degrees,
    pitch_degrees,
    roll_degrees,
    fill_color,
    chunk_rows,
    frame_batch_size,
    sampling,
    trim_left,
    trim_right,
    trim_top,
    trim_bottom,
):
    """Project a tensor or streamed VTS DiskImage and return tensor outputs."""

    width, height = int(output_width), int(output_height)
    if width < 2 or height < 1 or width != height * 2:
        raise ValueError("output_width must be exactly twice output_height for full ERP")
    trims = {
        "trim_left": int(trim_left),
        "trim_right": int(trim_right),
        "trim_top": int(trim_top),
        "trim_bottom": int(trim_bottom),
    }
    if any(value < 0 for value in trims.values()):
        raise ValueError("trim values must not be negative")
    batch_size = int(frame_batch_size)
    if batch_size < 1:
        raise ValueError("frame_batch_size must be at least 1")
    if not all(
        math.isfinite(float(value))
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ):
        raise ValueError("yaw, pitch and roll must be finite")

    unknown_color = _parse_fill_color(fill_color)
    horizontal_fov, vertical_fov = resolve_projection_fov(
        projection_mode,
        custom_horizontal_fov_degrees,
        custom_vertical_fov_degrees,
    )
    projected_batches = []
    known_batches = []
    unknown_batches = []
    source_size = None

    for batch in _source_batches(source, batch_size):
        if batch.ndim != 4 or batch.shape[-1] < 3:
            raise ValueError(
                "source images must be [B,H,W,C] with RGB channels, "
                f"got {tuple(batch.shape)}"
            )
        if int(batch.shape[1]) != int(batch.shape[2]):
            raise ValueError(
                "source images must be square, "
                f"got {int(batch.shape[2])}x{int(batch.shape[1])}"
            )
        current_source_size = int(batch.shape[1])
        if source_size is None:
            source_size = current_source_size
        elif current_source_size != source_size:
            raise ValueError(
                "all source frames must have the same square dimensions, "
                f"got both {source_size} and {current_source_size}"
            )
        if not batch.is_floating_point():
            raise TypeError("source images must use a floating-point dtype")
        result = vr180_square_to_full_erp(
            batch[..., :3].movedim(-1, 1).contiguous(),
            (height, width),
            projection_mode=projection_mode,
            custom_horizontal_fov_degrees=custom_horizontal_fov_degrees,
            custom_vertical_fov_degrees=custom_vertical_fov_degrees,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            roll_degrees=roll_degrees,
            unknown_color=unknown_color,
            chunk_rows=chunk_rows,
            mode=sampling,
            **trims,
        )
        projected_batches.append(result.image.movedim(1, -1).contiguous())
        known_batches.append(result.known_mask[:, 0].float())
        unknown_batches.append(result.unknown_mask[:, 0].float())

    source_kind = "tensor" if isinstance(source, torch.Tensor) else "DiskImage"
    projected_output = torch.cat(projected_batches, dim=0)
    known_output = torch.cat(known_batches, dim=0)
    outpaint_output = torch.cat(unknown_batches, dim=0)
    projected_left_x, projected_right_x = _horizontal_extent_from_known_mask(
        known_output
    )
    print(
        "[VTS VR180 -> ERP] "
        f"{source_kind}, mode={projection_mode}, "
        f"effective_fov={horizontal_fov:.6f}x{vertical_fov:.6f}, "
        f"source={source_size}x{source_size}, "
        f"retained_source={source_size - trims['trim_left'] - trims['trim_right']}x"
        f"{source_size - trims['trim_top'] - trims['trim_bottom']}, "
        f"trims=L{trims['trim_left']}/R{trims['trim_right']}/"
        f"T{trims['trim_top']}/B{trims['trim_bottom']}, "
        f"output={width}x{height}, "
        f"projected_x=L{projected_left_x}/R{projected_right_x}, "
        f"frames={sum(int(item.shape[0]) for item in projected_batches)}"
    )
    return (
        projected_output,
        known_output,
        outpaint_output,
        projected_left_x,
        projected_right_x,
    )


class VTSVR180SquareToERPOutpaint:
    CATEGORY = "VTS/VR180"
    FUNCTION = "project"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT")
    RETURN_NAMES = (
        "erp_canvas",
        "known_mask",
        "outpaint_mask",
        "projected_left_x",
        "projected_right_x",
    )
    DESCRIPTION = (
        "Place a square half-ERP or equidistant-fisheye VR180 image/video batch on a "
        "full 2:1 equirectangular canvas. Accepts an IMAGE tensor or VTS DiskImage and "
        "always returns in-memory tensors."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "projection_mode": (
                    list(PROJECTION_MODES),
                    {
                        "default": PROJECTION_MODES[0],
                        "tooltip": (
                            "Ideal/custom rectangular half-ERP, the fitted production "
                            "source preset, or ideal/custom circular equidistant fisheye."
                        ),
                    },
                ),
                "output_width": (
                    "INT",
                    {"default": 2048, "min": 32, "max": 16384, "step": 32},
                ),
                "output_height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 8192, "step": 16},
                ),
                "custom_horizontal_fov_degrees": (
                    "FLOAT",
                    {"default": 180.0, "min": 0.01, "max": 359.99, "step": 0.01},
                ),
                "custom_vertical_fov_degrees": (
                    "FLOAT",
                    {"default": 180.0, "min": 0.01, "max": 180.0, "step": 0.01},
                ),
                "yaw_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1},
                ),
                "pitch_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1},
                ),
                "roll_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1},
                ),
                "fill_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "RGB colour used outside the known source footprint.",
                    },
                ),
                "chunk_rows": (
                    "INT",
                    {"default": 256, "min": 0, "max": 8192, "step": 16},
                ),
                "frame_batch_size": (
                    "INT",
                    {"default": 8, "min": 1, "max": 4096, "step": 1},
                ),
                "sampling": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "trim_left": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16383,
                        "step": 1,
                        "tooltip": "Source-image columns cropped before projection; their projected area is black and included in the outpaint mask.",
                    },
                ),
                "trim_right": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16383,
                        "step": 1,
                        "tooltip": "Source-image columns cropped before projection; their projected area is black and included in the outpaint mask.",
                    },
                ),
                "trim_top": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8191,
                        "step": 1,
                        "tooltip": "Source-image rows cropped before projection; their projected area is black and included in the outpaint mask.",
                    },
                ),
                "trim_bottom": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8191,
                        "step": 1,
                        "tooltip": "Source-image rows cropped before projection; their projected area is black and included in the outpaint mask.",
                    },
                ),
            }
        }

    def project(self, **kwargs):
        return project_square_vr180_to_erp(**kwargs)


NODE_CLASS_MAPPINGS = {
    "VTSVR180SquareToERPOutpaint": VTSVR180SquareToERPOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSVR180SquareToERPOutpaint": "VTS VR180 Square To ERP Outpaint Canvas",
}
