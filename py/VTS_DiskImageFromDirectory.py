import os
import sys

import torch
from PIL import Image

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, vtsImageTypes


def _normalise_input_dir(value):
    value = os.path.expandvars(os.path.expanduser(value.strip()))
    if not value:
        raise ValueError("vts_input_dir cannot be empty")
    value = os.path.realpath(value)
    if not os.path.isdir(value):
        raise ValueError(f"vts_input_dir does not exist or is not a directory: {value}")
    return value


def _normalise_prefix(value):
    value = value.strip()
    if not value:
        raise ValueError("vts_prefix cannot be empty")
    if value != os.path.basename(value):
        raise ValueError("vts_prefix must be a filename prefix, not a path")
    return value


def _normalise_format(value):
    value = value.lower().lstrip(".")
    if value not in vtsImageTypes:
        raise ValueError(
            f"Unsupported vts_format {value!r}; expected one of {', '.join(vtsImageTypes)}"
        )
    return value


def _sequence_path(input_dir, prefix, sequence, image_format):
    return os.path.join(input_dir, f"{prefix}_{sequence:06d}.{image_format}")


def _contiguous_count(input_dir, prefix, start_sequence, end_sequence, image_format):
    if end_sequence != -1 and end_sequence < start_sequence:
        return 0

    sequence = start_sequence
    while end_sequence == -1 or sequence <= end_sequence:
        if not os.path.isfile(
            _sequence_path(input_dir, prefix, sequence, image_format)
        ):
            break
        sequence += 1
    return sequence - start_sequence


def _set_shape_metadata(disk_image):
    if disk_image.number_of_images == 0:
        disk_image.shape = (0, 0, 0, 3)
        disk_image.dtype = torch.float32
        disk_image.ndim = 4
        return

    first_path = _sequence_path(
        disk_image.output_dir,
        disk_image.prefix,
        disk_image.start_sequence,
        disk_image.format,
    )
    with Image.open(first_path) as image:
        width, height = image.size
        channels = 4 if image.mode == "RGBA" else 3

    disk_image.shape = (disk_image.number_of_images, height, width, channels)
    disk_image.dtype = torch.float32
    disk_image.ndim = 4


class VTSDiskImageFromDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vts_input_dir": (
                    "STRING",
                    {"default": default_output_dir, "multiline": False},
                ),
                "vts_prefix": (
                    "STRING",
                    {"default": "image", "multiline": False},
                ),
                "vts_start_sequence": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1},
                ),
                "vts_end_sequence": (
                    "INT",
                    {"default": -1, "min": -1, "step": 1},
                ),
                "vts_format": (vtsImageTypes, {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_disk_image"
    CATEGORY = "VTS/image"
    DESCRIPTION = (
        "Return a DiskImage view of an existing prefix_###### image sequence without "
        "copying it. The range stops before the first missing file."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def create_disk_image(
        self,
        vts_input_dir,
        vts_prefix,
        vts_start_sequence,
        vts_end_sequence,
        vts_format,
    ):
        input_dir = _normalise_input_dir(vts_input_dir)
        prefix = _normalise_prefix(vts_prefix)
        image_format = _normalise_format(vts_format)
        count = _contiguous_count(
            input_dir,
            prefix,
            vts_start_sequence,
            vts_end_sequence,
            image_format,
        )
        disk_image = DiskImage(
            prefix=prefix,
            start_sequence=vts_start_sequence,
            number_of_images=count,
            output_dir=input_dir,
            format=image_format,
            image=None,
        )
        _set_shape_metadata(disk_image)
        return (disk_image,)


NODE_CLASS_MAPPINGS = {
    "VTS DiskImage From Directory": VTSDiskImageFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS DiskImage From Directory": "VTS DiskImage From Directory",
}
