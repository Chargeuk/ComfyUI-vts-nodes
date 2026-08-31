"""Square VR180 source -> full equirectangular outpaint canvas.

Two source projections are supported:

* rectangular half-equirectangular (the common encoded VR180 eye layout), and
* circular equidistant fisheye (angular radius is linear in image radius).

The output is always a full 360x180 equirectangular image. Pixels not supplied
by the source projection receive an explicit fill colour and are reported by
the returned masks. All rays pass through pixel centres and sampling follows
ComfyUI's ``align_corners=False`` convention.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Optional, Sequence, Tuple

import torch

from vr180_projection import (
    ProjectionResult,
    _apply_unknown_color,
    _cacheable_grid,
    _normalise_cache_float,
    _rotation_camera_to_world,
    _sample_with_grid,
    _validate_image,
    _validate_size,
)


HALF_ERP_IDEAL = "half_equirectangular_ideal_180"
HALF_ERP_PRODUCTION = "half_equirectangular_production_calibrated"
HALF_ERP_CUSTOM = "half_equirectangular_custom"
FISHEYE_IDEAL = "equidistant_fisheye_180"
FISHEYE_CUSTOM = "equidistant_fisheye_custom"

PROJECTION_MODES = (
    HALF_ERP_IDEAL,
    HALF_ERP_PRODUCTION,
    HALF_ERP_CUSTOM,
    FISHEYE_IDEAL,
    FISHEYE_CUSTOM,
)

# Effective angular spans fitted against the production source's left/right
# correspondences. These are not a claim about the physical lens.
PRODUCTION_HORIZONTAL_FOV_DEGREES = 202.6113240305987
PRODUCTION_VERTICAL_FOV_DEGREES = 160.9788880865888

_GRID_CACHE_ENTRIES = 4
_GRID_BUILD_ROWS = 256


def resolve_projection_fov(
    projection_mode: str,
    custom_horizontal_fov_degrees: float,
    custom_vertical_fov_degrees: float,
) -> Tuple[float, float]:
    """Resolve a named source projection to effective horizontal/vertical FOV."""

    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"unsupported projection_mode: {projection_mode!r}")
    if projection_mode in (HALF_ERP_IDEAL, FISHEYE_IDEAL):
        horizontal, vertical = 180.0, 180.0
    elif projection_mode == HALF_ERP_PRODUCTION:
        horizontal = PRODUCTION_HORIZONTAL_FOV_DEGREES
        vertical = PRODUCTION_VERTICAL_FOV_DEGREES
    else:
        horizontal = float(custom_horizontal_fov_degrees)
        vertical = float(custom_vertical_fov_degrees)

    if not math.isfinite(horizontal) or not 0.01 <= horizontal < 360.0:
        raise ValueError("horizontal FOV must be finite and in [0.01, 360)")
    if not math.isfinite(vertical) or not 0.01 <= vertical <= 180.0:
        raise ValueError("vertical FOV must be finite and in [0.01, 180]")
    if projection_mode.startswith("equidistant_fisheye") and abs(horizontal - vertical) > 1.0e-6:
        raise ValueError(
            "an equidistant circular fisheye requires equal horizontal and vertical FOV"
        )
    return _normalise_cache_float(horizontal), _normalise_cache_float(vertical)


def _validate_square_source(image: torch.Tensor) -> None:
    _validate_image(image)
    if int(image.shape[2]) != int(image.shape[3]):
        raise ValueError(
            "source must be a square VR180 eye image/video batch, "
            f"got {int(image.shape[3])}x{int(image.shape[2])}"
        )


def _build_grid(
    input_size: int,
    output_height: int,
    output_width: int,
    projection_mode: str,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an output-ERP -> source sampling grid on CPU."""

    camera_to_world = _rotation_camera_to_world(
        yaw_degrees, pitch_degrees, roll_degrees
    )
    grid = torch.empty((output_height, output_width, 2), dtype=torch.float32)
    known = torch.empty((output_height, output_width), dtype=torch.bool)
    u = (torch.arange(output_width, dtype=torch.float32) + 0.5) / float(output_width)
    longitude = (u - 0.5) * (2.0 * math.pi)
    source_limit = 1.0 - 1.0 / float(input_size)
    horizontal_fov = math.radians(horizontal_fov_degrees)
    vertical_fov = math.radians(vertical_fov_degrees)
    fisheye = projection_mode.startswith("equidistant_fisheye")
    theta_max = horizontal_fov * 0.5

    for row_start in range(0, output_height, _GRID_BUILD_ROWS):
        row_stop = min(row_start + _GRID_BUILD_ROWS, output_height)
        v = (torch.arange(row_start, row_stop, dtype=torch.float32) + 0.5) / float(
            output_height
        )
        latitude = (0.5 - v) * math.pi
        cos_latitude = torch.cos(latitude)[:, None]
        world = torch.stack(
            (
                cos_latitude * torch.sin(longitude)[None, :],
                torch.sin(latitude)[:, None].expand(-1, output_width),
                cos_latitude * torch.cos(longitude)[None, :],
            ),
            dim=-1,
        )
        # camera_to_world maps local column vectors to world. With row vectors,
        # world @ camera_to_world therefore gives source-camera coordinates.
        camera = world @ camera_to_world
        x, y, z = camera.unbind(dim=-1)

        if fisheye:
            angular_radius = torch.acos(z.clamp(-1.0, 1.0))
            radial_xy = torch.hypot(x, y)
            safe_radial = torch.where(
                radial_xy > 1.0e-8, radial_xy, torch.ones_like(radial_xy)
            )
            radius_fraction = angular_radius / theta_max
            grid_x = radius_fraction * (x / safe_radial) * source_limit
            grid_y = -radius_fraction * (y / safe_radial) * source_limit
            centre = radial_xy <= 1.0e-8
            grid_x = torch.where(centre, torch.zeros_like(grid_x), grid_x)
            grid_y = torch.where(centre, torch.zeros_like(grid_y), grid_y)
            valid = angular_radius <= theta_max + 1.0e-7
        else:
            source_longitude = torch.atan2(x, z)
            source_latitude = torch.asin(y.clamp(-1.0, 1.0))
            grid_x = 2.0 * source_longitude / horizontal_fov
            grid_y = -2.0 * source_latitude / vertical_fov
            # Pixel-centre coordinates at the exact FOV boundary can differ
            # from ``source_limit`` by a few float32 ULPs after trigonometry.
            # The tolerance only preserves those mathematically valid edge
            # pixels; grid_sample still uses the original coordinates.
            boundary_tolerance = 1.0e-6
            valid = (
                (grid_x >= -source_limit - boundary_tolerance)
                & (grid_x <= source_limit + boundary_tolerance)
                & (grid_y >= -source_limit - boundary_tolerance)
                & (grid_y <= source_limit + boundary_tolerance)
            )

        grid[row_start:row_stop, :, 0] = grid_x
        grid[row_start:row_stop, :, 1] = grid_y
        known[row_start:row_stop] = valid

    return grid.contiguous(), known.contiguous()


@lru_cache(maxsize=_GRID_CACHE_ENTRIES)
def _cached_grid_small(
    input_size: int,
    output_height: int,
    output_width: int,
    projection_mode: str,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _build_grid(
        input_size,
        output_height,
        output_width,
        projection_mode,
        horizontal_fov_degrees,
        vertical_fov_degrees,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    )


def _projection_grid(*arguments) -> Tuple[torch.Tensor, torch.Tensor]:
    output_height, output_width = int(arguments[1]), int(arguments[2])
    if not _cacheable_grid(output_height, output_width):
        return _build_grid(*arguments)
    return _cached_grid_small(*arguments)


def _fisheye_source_support(size: int, *, device: torch.device) -> torch.Tensor:
    """Return the valid circular fisheye footprint at source pixel centres."""

    centre = (float(size) - 1.0) * 0.5
    radius = max(centre, 1.0)
    coordinates = torch.arange(size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    support = ((xx - centre) ** 2 + (yy - centre) ** 2) <= radius**2 + 1.0e-6
    return support.to(torch.float32).unsqueeze(0).unsqueeze(0)


def _source_trim_keep_mask(
    grid: torch.Tensor,
    size: int,
    *,
    trim_left: int,
    trim_right: int,
    trim_top: int,
    trim_bottom: int,
) -> torch.Tensor:
    """Return output samples whose source coordinates survive the crop."""

    # With align_corners=False, source pixel edges span normalized [-1, +1].
    # Moving an edge inward by N source pixels therefore moves it by 2*N/size.
    left_edge = -1.0 + 2.0 * float(trim_left) / float(size)
    right_edge = 1.0 - 2.0 * float(trim_right) / float(size)
    top_edge = -1.0 + 2.0 * float(trim_top) / float(size)
    bottom_edge = 1.0 - 2.0 * float(trim_bottom) / float(size)
    return (
        (grid[..., 0] >= left_edge)
        & (grid[..., 0] <= right_edge)
        & (grid[..., 1] >= top_edge)
        & (grid[..., 1] <= bottom_edge)
    ).contiguous()


def vr180_square_to_full_erp(
    image: torch.Tensor,
    output_size: Sequence[int],
    *,
    projection_mode: str = HALF_ERP_IDEAL,
    custom_horizontal_fov_degrees: float = 180.0,
    custom_vertical_fov_degrees: float = 180.0,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    unknown_color: Sequence[float] = (0.0, 0.0, 0.0),
    chunk_rows: Optional[int] = 256,
    mode: str = "bilinear",
    trim_left: int = 0,
    trim_right: int = 0,
    trim_top: int = 0,
    trim_bottom: int = 0,
) -> ProjectionResult:
    """Place a square VR180 source into a full 2:1 ERP outpaint canvas."""

    _validate_square_source(image)
    output_height, output_width = _validate_size(output_size, "output_size")
    if output_width != output_height * 2:
        raise ValueError(
            "a full equirectangular canvas must be exactly 2:1 "
            f"(got {output_width}x{output_height})"
        )
    if mode not in {"bilinear", "nearest", "bicubic"}:
        raise ValueError("mode must be bilinear, nearest or bicubic")
    horizontal_fov, vertical_fov = resolve_projection_fov(
        projection_mode,
        custom_horizontal_fov_degrees,
        custom_vertical_fov_degrees,
    )
    yaw, pitch, roll = [
        _normalise_cache_float(value)
        for value in (yaw_degrees, pitch_degrees, roll_degrees)
    ]
    size = int(image.shape[2])
    trims = {
        "trim_left": int(trim_left),
        "trim_right": int(trim_right),
        "trim_top": int(trim_top),
        "trim_bottom": int(trim_bottom),
    }
    if any(value < 0 for value in trims.values()):
        raise ValueError("trim values must not be negative")
    if trims["trim_left"] + trims["trim_right"] >= size:
        raise ValueError("left and right trims must leave at least one source column")
    if trims["trim_top"] + trims["trim_bottom"] >= size:
        raise ValueError("top and bottom trims must leave at least one source row")
    grid, geometric_known_cpu = _projection_grid(
        size,
        output_height,
        output_width,
        projection_mode,
        horizontal_fov,
        vertical_fov,
        yaw,
        pitch,
        roll,
    )
    sampled, known = _sample_with_grid(
        image,
        grid,
        geometric_known_cpu,
        chunk_rows=chunk_rows,
        mode=mode,
    )

    if projection_mode.startswith("equidistant_fisheye"):
        # Reject samples whose bilinear support reaches outside the circular
        # source footprint, preventing black square corners bleeding into ERP.
        support = _fisheye_source_support(size, device=image.device)
        sampled_support, _ = _sample_with_grid(
            support,
            grid,
            geometric_known_cpu,
            chunk_rows=chunk_rows,
            mode="bilinear",
        )
        support_known = sampled_support >= 1.0 - 1.0e-6
        known = known & support_known.expand(int(image.shape[0]), -1, -1, -1)

    # ``known`` now represents the complete source footprint, including the
    # circular support check for fisheye input. Cropping is deliberately kept
    # separate so cropped source pixels belong to neither output mask.
    geometric_known = known
    outpaint = ~geometric_known
    output = _apply_unknown_color(sampled, geometric_known, unknown_color)
    if any(trims.values()):
        trim_keep = _source_trim_keep_mask(grid, size, **trims)
        trim_keep = (
            trim_keep.to(device=image.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(int(image.shape[0]), 1, -1, -1)
        )
        known = geometric_known & trim_keep
        deliberately_trimmed = geometric_known & ~trim_keep
        output = output.masked_fill(deliberately_trimmed, 0.0)
    else:
        known = geometric_known
    return ProjectionResult(output, known, outpaint)


def clear_outpaint_projection_cache() -> None:
    _cached_grid_small.cache_clear()


__all__ = [
    "FISHEYE_CUSTOM",
    "FISHEYE_IDEAL",
    "HALF_ERP_CUSTOM",
    "HALF_ERP_IDEAL",
    "HALF_ERP_PRODUCTION",
    "PRODUCTION_HORIZONTAL_FOV_DEGREES",
    "PRODUCTION_VERTICAL_FOV_DEGREES",
    "PROJECTION_MODES",
    "clear_outpaint_projection_cache",
    "resolve_projection_fov",
    "vr180_square_to_full_erp",
]
