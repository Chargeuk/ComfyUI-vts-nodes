"""Choose nearby, divisible dimensions for an aspect ratio and pixel budget."""

from fractions import Fraction
import math

from comfy_api.latest import io


PRIORITIES = ["Aspect ratio", "Megapixels"]


def _neighboring_multiples(squared_dimension, multiple):
    squared_steps = squared_dimension / (multiple * multiple)
    lower = math.isqrt(squared_steps.numerator // squared_steps.denominator)
    upper = lower + (lower * lower * squared_steps.denominator != squared_steps.numerator)
    return {max(1, lower) * multiple, max(1, upper) * multiple}


def resolution_from_megapixels(width, height, megapixels, multiple, priority="Aspect ratio"):
    for name, value in (("width", width), ("height", height), ("multiple", multiple)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)):
        raise ValueError("megapixels must be a positive finite number.")
    try:
        finite_megapixels = math.isfinite(megapixels)
    except OverflowError:
        finite_megapixels = False
    if not finite_megapixels or megapixels <= 0:
        raise ValueError("megapixels must be a positive finite number.")
    if priority not in PRIORITIES:
        raise ValueError("priority must be 'Aspect ratio' or 'Megapixels'.")

    target_pixels = Fraction(megapixels) * 1_000_000
    aspect_ratio = Fraction(width, height)
    # Exact squared dimensions give the same sqrt floor/ceil neighborhood without
    # overflow, underflow, or rounding an exact multiple to the wrong neighbor.
    widths = _neighboring_multiples(target_pixels * aspect_ratio, multiple)
    heights = _neighboring_multiples(target_pixels / aspect_ratio, multiple)

    def score(size):
        candidate_width, candidate_height = size
        area = candidate_width * candidate_height
        aspect_error = abs(Fraction(candidate_width, candidate_height) / aspect_ratio - 1)
        pixel_error = abs(area / target_pixels - 1)
        errors = (aspect_error, pixel_error) if priority == "Aspect ratio" else (pixel_error, aspect_error)
        # Prefer the smaller result if both requested errors tie exactly.
        return (*errors, area, candidate_width, candidate_height)

    result_width, result_height = min(((w, h) for w in widths for h in heights), key=score)
    try:
        actual_megapixels = result_width * result_height / 1_000_000
    except OverflowError:
        raise ValueError("The resulting actual_megapixels is too large for a finite FLOAT.") from None
    return result_width, result_height, actual_megapixels


class VTS_Resolution_From_Megapixels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VTS Resolution From Megapixels",
            display_name="Resolution from Megapixels VTS",
            category="VTS",
            description=("Choose divisible dimensions near the ideal size for the supplied aspect ratio and megapixels. "
                         "Only the floor/ceil neighboring multiples on each axis are compared (at most four sizes), "
                         "with a minimum of one multiple per axis. Ties prefer smaller area, then width and height."),
            inputs=[
                io.Int.Input("width", default=1920, min=1, step=1,
                             tooltip="Positive aspect-ratio width; its pixel count does not set the output size."),
                io.Int.Input("height", default=1080, min=1, step=1,
                             tooltip="Positive aspect-ratio height; width:height defines the desired shape."),
                io.Float.Input("megapixels", default=3.0, min=5e-324, step=0.1, round=False,
                               tooltip="Positive finite target size. 1 megapixel = 1,000,000 pixels. Rounding can change the actual size."),
                io.Int.Input("multiple", default=64, min=1, step=1,
                             tooltip="Both output dimensions are positive multiples of this integer. Compare only floor/ceil neighbors of the ideal dimensions."),
                io.Combo.Input("priority", options=PRIORITIES, default="Aspect ratio",
                               tooltip=("Within the neighboring sizes, Aspect ratio minimizes relative aspect-ratio error first, "
                                        "then relative pixel-count error. Megapixels reverses that order.")),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Chosen output width in pixels, divisible by multiple."),
                io.Int.Output("height", tooltip="Chosen output height in pixels, divisible by multiple."),
                io.Float.Output("actual_megapixels", tooltip="Actual width × height / 1,000,000 after rounding; may differ from the target."),
            ],
        )

    @classmethod
    def execute(cls, width, height, megapixels, multiple, priority="Aspect ratio"):
        return io.NodeOutput(*resolution_from_megapixels(width, height, megapixels, multiple, priority))


NODE_CLASS_MAPPINGS = {"VTS Resolution From Megapixels": VTS_Resolution_From_Megapixels}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS Resolution From Megapixels": "Resolution from Megapixels VTS"}
