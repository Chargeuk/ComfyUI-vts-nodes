"""Exact Ref2VA marker handling for ComfyUI image tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


_MAGENTA = (255, 0, 255)
_ESCAPED_MAGENTA = (255, 0, 254)


@dataclass(frozen=True)
class MarkerResult:
    image: torch.Tensor
    effective_mask: torch.Tensor
    collision_mask: torch.Tensor
    marker_counts: Tuple[int, ...]
    escaped_counts: Tuple[int, ...]


def validate_comfy_image(image: torch.Tensor, name: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] < 3:
        raise ValueError(
            f"{name} must be a ComfyUI IMAGE [B,H,W,C] with RGB channels, "
            f"got {tuple(getattr(image, 'shape', ()))}"
        )
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} dimensions must be positive")
    if not image.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def validate_effective_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("effective_mask must be a torch.Tensor")
    if mask.ndim != 3:
        raise ValueError(
            "effective_mask must be a ComfyUI MASK [B,H,W], got "
            f"{tuple(mask.shape)}"
        )
    if tuple(mask.shape) != tuple(image.shape[:3]):
        raise ValueError(
            "effective_mask dimensions must exactly match Ref2 [B,H,W]: "
            f"{tuple(mask.shape)} != {tuple(image.shape[:3])}"
        )
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError("effective_mask must contain only finite binary values")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("effective_mask must be binary (exactly 0 or 1)")
    return mask.to(device=image.device, dtype=torch.bool)


def quantized_rgb(image: torch.Tensor) -> torch.Tensor:
    """Return the RGB values produced by lossless 8-bit image encoding."""

    return (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def mark_ref2(image: torch.Tensor, effective_mask: torch.Tensor) -> MarkerResult:
    """Escape known magenta collisions, then mark exactly the effective mask."""

    validate_comfy_image(image, "ref2")
    mask = validate_effective_mask(effective_mask, image)
    rgb8 = quantized_rgb(image)
    collision = (
        (rgb8[..., 0] == _MAGENTA[0])
        & (rgb8[..., 1] == _MAGENTA[1])
        & (rgb8[..., 2] == _MAGENTA[2])
        & ~mask
    )

    output = image.clone()
    escaped_blue = output.new_tensor(_ESCAPED_MAGENTA[2] / 255.0)
    output[..., 2] = torch.where(collision, escaped_blue, output[..., 2])
    marker = output.new_tensor(_MAGENTA).div(255.0)
    output[..., :3] = torch.where(mask.unsqueeze(-1), marker, output[..., :3])

    marker_pixels = (quantized_rgb(output) == output.new_tensor(
        _MAGENTA, dtype=torch.uint8
    )).all(dim=-1)
    if not torch.equal(marker_pixels, mask):
        raise RuntimeError("Ref2 marker invariant failed: #FF00FF must equal effective_mask")

    marker_counts = tuple(int(value) for value in marker_pixels.sum(dim=(1, 2)).tolist())
    escaped_counts = tuple(int(value) for value in collision.sum(dim=(1, 2)).tolist())
    return MarkerResult(output, mask, collision, marker_counts, escaped_counts)


__all__ = [
    "MarkerResult",
    "mark_ref2",
    "quantized_rgb",
    "validate_comfy_image",
    "validate_effective_mask",
]
