"""Create a feathered vertical seam mask at a requested pixel column.

The inward-feather behavior is derived from MickMumpitz's MIT-licensed
``PanoSeamMask``. See ``THIRD_PARTY_NOTICES.md`` at the repository root.
"""

from __future__ import annotations

import torch


def positioned_seam_strip_mask(
    width: int,
    height: int,
    seam_width: int,
    feather: int,
    strip_center_x: int,
) -> tuple[torch.Tensor]:
    """Return a [1,H,W] mask whose strip is centred on ``strip_center_x``."""

    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    center = int(strip_center_x)
    if not 0 <= center < width:
        raise ValueError(
            f"strip_center_x must identify a pixel in [0, {width - 1}], got {center}"
        )

    strip = max(min(int(seam_width), width), 1)
    flank = max(int(feather), 0)
    start = center - strip // 2
    stop = start + strip

    # Build the complete conceptual strip first. If its requested centre is
    # close to an image edge, clipping then removes only the off-canvas part;
    # it does not incorrectly create a new feather at the canvas boundary.
    profile = torch.ones(strip, dtype=torch.float32)
    if flank > 0:
        ramp = torch.linspace(0.0, 1.0, flank + 2, dtype=torch.float32)[1:-1]
        count = min(flank, strip // 2)
        if count > 0:
            profile[:count] = ramp[:count]
            profile[strip - count :] = ramp[:count].flip(0)

    visible_start = max(start, 0)
    visible_stop = min(stop, width)
    profile_start = visible_start - start
    profile_stop = profile_start + max(visible_stop - visible_start, 0)

    columns = torch.zeros(width, dtype=torch.float32)
    if visible_stop > visible_start:
        columns[visible_start:visible_stop] = profile[profile_start:profile_stop]
    mask = columns.view(1, 1, width).expand(1, height, width).clone()
    print(
        "[VTS Positioned Seam Strip Mask] "
        f"{width}x{height}, strip={strip}px, center_x={center}, "
        f"requested_range=[{start},{stop}), "
        f"visible_range=[{visible_start},{visible_stop}), feather={flank}px"
    )
    return (mask,)


class VTSPositionedSeamStripMask:
    CATEGORY = "VTS/mask"
    FUNCTION = "create_mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = (
        "Create a vertical seam strip centred at a specific pixel column. "
        "Feathering stays inside the requested strip; off-canvas portions are clipped."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 2048, "min": 256, "max": 8192, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 4096, "step": 16},
                ),
                "seam_width": (
                    "INT",
                    {
                        "default": 96,
                        "min": 16,
                        "max": 2048,
                        "step": 16,
                        "tooltip": "Total width in pixels of the strip to repaint.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Soft flank inside the strip's left and right edges.",
                    },
                ),
                "strip_center_x": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 8191,
                        "step": 1,
                        "tooltip": "Centre pixel column measured from the image's left edge.",
                    },
                ),
            }
        }

    def create_mask(self, width, height, seam_width, feather, strip_center_x):
        return positioned_seam_strip_mask(
            width,
            height,
            seam_width,
            feather,
            strip_center_x,
        )


NODE_CLASS_MAPPINGS = {
    "VTSPositionedSeamStripMask": VTSPositionedSeamStripMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSPositionedSeamStripMask": "VTS Positioned Seam Strip Mask",
}
