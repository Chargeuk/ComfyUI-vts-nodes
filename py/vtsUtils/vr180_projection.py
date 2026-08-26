"""Pixel-centred rectilinear <-> VR180 projection implemented in PyTorch.

Coordinate convention
---------------------
* Camera space is right-handed: +X right, +Y up, +Z forward.
* A VR180 image is half-equirectangular: longitude [-90, +90] degrees is
  stored left-to-right and latitude [+90, -90] degrees top-to-bottom.
* ``yaw``, ``pitch`` and ``roll`` describe the rectilinear camera pose in the
  VR180/world coordinate system.  Positive yaw turns the camera right.
* All rays pass through pixel centres.  Sampling uses ``align_corners=False``.

The LRU caches CPU float32 sampling grids only.  This deliberately avoids
pinning CUDA memory when ComfyUI changes resolution or camera parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_GRID_CACHE_ENTRIES = 4
_GRID_CACHE_MAX_ITEM_BYTES = 64 * 1024 * 1024
_GRID_BUILD_ROWS = 256


@dataclass(frozen=True)
class ProjectionResult:
    """A projected BCHW image and its exact geometric validity masks."""

    image: torch.Tensor
    known_mask: torch.Tensor
    unknown_mask: torch.Tensor


def _validate_image(image: torch.Tensor, name: str = "image") -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(image.shape)}")
    if image.shape[0] < 1 or image.shape[1] < 3 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError(f"{name} must contain at least one RGB image, got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _validate_size(size: Sequence[int], name: str) -> Tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must be (height, width)")
    height, width = int(size[0]), int(size[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} values must be positive, got {(height, width)}")
    return height, width


def _validate_fov(horizontal_fov_degrees: float) -> float:
    fov = float(horizontal_fov_degrees)
    if not math.isfinite(fov) or not 0.01 <= fov < 180.0:
        raise ValueError("horizontal_fov_degrees must be in [0.01, 180)")
    return fov


def _rotation_camera_to_world(
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Rz(roll) @ Rx(pitch) @ Ry(yaw), camera rays to world rays."""

    yaw, pitch, roll = [
        math.radians(float(value))
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=dtype)
    rz = torch.tensor([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    return rz @ rx @ ry


def _rect_to_vr_grid_uncached(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tan_h = math.tan(math.radians(horizontal_fov_degrees) * 0.5)
    tan_v = tan_h * float(input_height) / float(input_width)
    world_to_camera = _rotation_camera_to_world(
        yaw_degrees, pitch_degrees, roll_degrees
    ).transpose(0, 1)

    grid = torch.empty((output_height, output_width, 2), dtype=torch.float32)
    known = torch.empty((output_height, output_width), dtype=torch.bool)
    u = (torch.arange(output_width, dtype=torch.float32) + 0.5) / float(output_width)
    longitude = (u - 0.5) * math.pi

    for row_start in range(0, output_height, _GRID_BUILD_ROWS):
        row_stop = min(row_start + _GRID_BUILD_ROWS, output_height)
        v = (
            torch.arange(row_start, row_stop, dtype=torch.float32) + 0.5
        ) / float(output_height)
        latitude = (0.5 - v) * math.pi
        cos_lat = torch.cos(latitude)[:, None]
        world = torch.stack(
            (
                cos_lat * torch.sin(longitude)[None, :],
                torch.sin(latitude)[:, None].expand(-1, output_width),
                cos_lat * torch.cos(longitude)[None, :],
            ),
            dim=-1,
        )
        camera = world @ world_to_camera.transpose(0, 1)
        x, y, z = camera.unbind(dim=-1)
        safe_z = torch.where(z.abs() > 1.0e-8, z, torch.ones_like(z))
        grid_x = (x / safe_z) / tan_h
        grid_y = -(y / safe_z) / tan_v
        valid = (
            (z > 1.0e-8)
            & (grid_x >= -1.0)
            & (grid_x <= 1.0)
            & (grid_y >= -1.0)
            & (grid_y <= 1.0)
        )
        grid[row_start:row_stop, :, 0] = grid_x
        grid[row_start:row_stop, :, 1] = grid_y
        known[row_start:row_stop] = valid

    return grid.contiguous(), known.contiguous()


@lru_cache(maxsize=_GRID_CACHE_ENTRIES)
def _cached_rect_to_vr_grid(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _rect_to_vr_grid_uncached(
        input_height,
        input_width,
        output_height,
        output_width,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    )


def _rect_to_vr_grid(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    build = (
        _cached_rect_to_vr_grid
        if output_height * output_width * 9 <= _GRID_CACHE_MAX_ITEM_BYTES
        else _rect_to_vr_grid_uncached
    )
    return build(
        input_height,
        input_width,
        output_height,
        output_width,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    )


def _vr_to_rect_grid_uncached(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del input_height, input_width  # The analytical half-ERP map is resolution independent.
    tan_h = math.tan(math.radians(horizontal_fov_degrees) * 0.5)
    tan_v = tan_h * float(output_height) / float(output_width)
    camera_to_world = _rotation_camera_to_world(
        yaw_degrees, pitch_degrees, roll_degrees
    )

    grid = torch.empty((output_height, output_width, 2), dtype=torch.float32)
    known = torch.empty((output_height, output_width), dtype=torch.bool)
    u = (torch.arange(output_width, dtype=torch.float32) + 0.5) / float(output_width)
    camera_x = (u * 2.0 - 1.0) * tan_h

    for row_start in range(0, output_height, _GRID_BUILD_ROWS):
        row_stop = min(row_start + _GRID_BUILD_ROWS, output_height)
        v = (
            torch.arange(row_start, row_stop, dtype=torch.float32) + 0.5
        ) / float(output_height)
        camera_y = -(v * 2.0 - 1.0) * tan_v
        camera = torch.stack(
            (
                camera_x[None, :].expand(row_stop - row_start, -1),
                camera_y[:, None].expand(-1, output_width),
                torch.ones((row_stop - row_start, output_width), dtype=torch.float32),
            ),
            dim=-1,
        )
        camera = F.normalize(camera, dim=-1)
        world = camera @ camera_to_world.transpose(0, 1)
        x, y, z = world.unbind(dim=-1)
        longitude = torch.atan2(x, z)
        latitude = torch.asin(y.clamp(-1.0, 1.0))
        grid_x = 2.0 * longitude / math.pi
        grid_y = -2.0 * latitude / math.pi
        valid = (
            (longitude >= -0.5 * math.pi)
            & (longitude <= 0.5 * math.pi)
            & (latitude >= -0.5 * math.pi)
            & (latitude <= 0.5 * math.pi)
        )
        grid[row_start:row_stop, :, 0] = grid_x
        grid[row_start:row_stop, :, 1] = grid_y
        known[row_start:row_stop] = valid

    return grid.contiguous(), known.contiguous()


@lru_cache(maxsize=_GRID_CACHE_ENTRIES)
def _cached_vr_to_rect_grid(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _vr_to_rect_grid_uncached(
        input_height,
        input_width,
        output_height,
        output_width,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    )


def _vr_to_rect_grid(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    horizontal_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    build = (
        _cached_vr_to_rect_grid
        if output_height * output_width * 9 <= _GRID_CACHE_MAX_ITEM_BYTES
        else _vr_to_rect_grid_uncached
    )
    return build(
        input_height,
        input_width,
        output_height,
        output_width,
        horizontal_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    )


def _normalise_cache_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("projection angles must be finite")
    return round(value, 6)


def _sample_with_grid(
    image: torch.Tensor,
    grid_cpu: torch.Tensor,
    known_cpu: torch.Tensor,
    *,
    chunk_rows: Optional[int],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = image.shape[0]
    output_height = grid_cpu.shape[0]
    rows = output_height if chunk_rows is None or int(chunk_rows) <= 0 else int(chunk_rows)
    rows = max(1, min(rows, output_height))
    sampled_parts = []
    mask_parts = []
    for start in range(0, output_height, rows):
        stop = min(start + rows, output_height)
        grid = grid_cpu[start:stop].to(device=image.device, dtype=image.dtype)
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
        sampled_parts.append(
            F.grid_sample(
                image,
                grid,
                mode=mode,
                padding_mode="border",
                align_corners=False,
            )
        )
        mask_parts.append(
            known_cpu[start:stop]
            .to(device=image.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, 1, -1, -1)
        )
    return torch.cat(sampled_parts, dim=2), torch.cat(mask_parts, dim=2)


def _apply_unknown_color(
    sampled: torch.Tensor,
    known: torch.Tensor,
    unknown_color: Sequence[float],
) -> torch.Tensor:
    if len(unknown_color) != sampled.shape[1]:
        if len(unknown_color) != 3 or sampled.shape[1] < 3:
            raise ValueError(
                "unknown_color must have three values for RGB input or one value per channel"
            )
        values = list(float(value) for value in unknown_color) + [0.0] * (sampled.shape[1] - 3)
    else:
        values = [float(value) for value in unknown_color]
    color = torch.tensor(values, dtype=sampled.dtype, device=sampled.device).view(1, -1, 1, 1)
    return torch.where(known, sampled, color)


def rectilinear_to_vr180(
    image: torch.Tensor,
    output_size: Sequence[int],
    *,
    horizontal_fov_degrees: float = 90.0,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    unknown_color: Sequence[float] = (1.0, 0.0, 1.0),
    chunk_rows: Optional[int] = 256,
    mode: str = "bilinear",
) -> ProjectionResult:
    """Project a BCHW rectilinear image into a partial half-ERP VR180 image."""

    _validate_image(image)
    output_height, output_width = _validate_size(output_size, "output_size")
    fov = _normalise_cache_float(_validate_fov(horizontal_fov_degrees))
    yaw, pitch, roll = [
        _normalise_cache_float(value)
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ]
    if mode not in {"bilinear", "nearest", "bicubic"}:
        raise ValueError("mode must be bilinear, nearest or bicubic")
    grid, known_cpu = _rect_to_vr_grid(
        int(image.shape[2]),
        int(image.shape[3]),
        output_height,
        output_width,
        fov,
        yaw,
        pitch,
        roll,
    )
    sampled, known = _sample_with_grid(
        image, grid, known_cpu, chunk_rows=chunk_rows, mode=mode
    )
    projected = _apply_unknown_color(sampled, known, unknown_color)
    return ProjectionResult(projected, known, ~known)


def vr180_to_rectilinear(
    image: torch.Tensor,
    output_size: Sequence[int],
    *,
    horizontal_fov_degrees: float = 90.0,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    source_known_mask: Optional[torch.Tensor] = None,
    unknown_color: Sequence[float] = (1.0, 0.0, 1.0),
    chunk_rows: Optional[int] = 256,
    mode: str = "bilinear",
) -> ProjectionResult:
    """Render a rectilinear view from a BCHW half-ERP VR180 image."""

    _validate_image(image)
    output_height, output_width = _validate_size(output_size, "output_size")
    fov = _normalise_cache_float(_validate_fov(horizontal_fov_degrees))
    yaw, pitch, roll = [
        _normalise_cache_float(value)
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ]
    if mode not in {"bilinear", "nearest", "bicubic"}:
        raise ValueError("mode must be bilinear, nearest or bicubic")
    grid, geometric_known_cpu = _vr_to_rect_grid(
        int(image.shape[2]),
        int(image.shape[3]),
        output_height,
        output_width,
        fov,
        yaw,
        pitch,
        roll,
    )
    sampled, known = _sample_with_grid(
        image, grid, geometric_known_cpu, chunk_rows=chunk_rows, mode=mode
    )
    if source_known_mask is not None:
        if source_known_mask.ndim == 3:
            source_known_mask = source_known_mask.unsqueeze(1)
        if source_known_mask.ndim != 4 or source_known_mask.shape[1] != 1:
            raise ValueError("source_known_mask must be [B,H,W] or [B,1,H,W]")
        if source_known_mask.shape[0] not in (1, image.shape[0]) or source_known_mask.shape[2:] != image.shape[2:]:
            raise ValueError("source_known_mask dimensions must match the source image")
        source_known_mask = source_known_mask.to(device=image.device, dtype=image.dtype)
        sampled_mask, _ = _sample_with_grid(
            source_known_mask,
            grid,
            geometric_known_cpu,
            chunk_rows=chunk_rows,
            mode="bilinear",
        )
        known = known & (sampled_mask >= 1.0 - 1.0e-6)
    projected = _apply_unknown_color(sampled, known, unknown_color)
    return ProjectionResult(projected, known, ~known)


def clear_projection_cache() -> None:
    _cached_rect_to_vr_grid.cache_clear()
    _cached_vr_to_rect_grid.cache_clear()


def projection_cache_info() -> dict:
    """Return JSON-friendly cache diagnostics for tests and operational logs."""

    forward = _cached_rect_to_vr_grid.cache_info()
    reverse = _cached_vr_to_rect_grid.cache_info()
    return {
        "max_entries_per_direction": _GRID_CACHE_ENTRIES,
        "max_item_bytes": _GRID_CACHE_MAX_ITEM_BYTES,
        "forward": forward._asdict(),
        "reverse": reverse._asdict(),
    }


__all__ = [
    "ProjectionResult",
    "clear_projection_cache",
    "projection_cache_info",
    "rectilinear_to_vr180",
    "vr180_to_rectilinear",
]
