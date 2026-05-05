import os
import sys

import torch
from comfy import model_management

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import (
    DiskImage,
    default_output_dir,
    get_default_image_output_types,
    ensure_image_output_defaults,
    deep_merge,
    save_images,
)


class VTS_ImageFromBatch:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "VTS/image batch"

    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {
                "image": ("IMAGE",),
                "start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "length": ("INT", {"default": -1, "min": -1, "step": 1}),
            }
        }
        return deep_merge(input_types, get_default_image_output_types(prefix="image_from_batch"))

    def _is_tensor(self, image):
        return isinstance(image, torch.Tensor)

    def _image_count(self, image):
        return image.shape[0] if self._is_tensor(image) else image.number_of_images

    def _image_shape(self, image):
        return tuple(image.shape) if self._is_tensor(image) else image.shape

    def _resolve_selection(self, image, start, length):
        count = self._image_count(image)
        if count <= 0:
            return 0, 0

        if length < 0:
            length = count
        start = min(start, count - 1)
        length = min(count - start, length)
        return start, length

    def _load_range_tensor(self, image, start, length):
        if length <= 0:
            shape = self._image_shape(image)
            channels = tuple(shape[1:]) if shape is not None else ()
            return torch.empty((0,) + channels, dtype=torch.float32)

        if self._is_tensor(image):
            return image[start:start + length]

        return image.load_images(start_sequence=image.start_sequence + start, count=length)

    def _build_diskimage_view(self, image, start, length):
        source_shape = self._image_shape(image)
        if self._is_tensor(image):
            raise ValueError("DiskImage view can only be built from DiskImage input")

        result = image.clone()
        result.start_sequence = image.start_sequence + start
        result.number_of_images = length
        if source_shape is not None:
            result.shape = (length,) + tuple(source_shape[1:])
            result.ndim = len(result.shape)
        return result

    def _build_saved_diskimage(self, source_image, start, length, output_kwargs):
        source_shape = self._image_shape(source_image)
        result = DiskImage(
            prefix=output_kwargs["prefix"],
            start_sequence=output_kwargs["start_sequence"],
            number_of_images=length,
            output_dir=output_kwargs["output_dir"],
            format=output_kwargs["format"],
            image=None,
            compression_level=output_kwargs["compression_level"],
            quality=output_kwargs["quality"],
        )
        if source_shape is not None:
            result.shape = (length,) + tuple(source_shape[1:])
            result.ndim = len(result.shape)
        if self._is_tensor(source_image):
            result.dtype = source_image.dtype
        else:
            result.dtype = source_image.dtype
        return result

    def execute(self, image, start, length, **kwargs):
        kwargs = ensure_image_output_defaults(kwargs)
        start, length = self._resolve_selection(image, start, length)

        return_type = kwargs["return_type"]
        if return_type == "Input":
            if self._is_tensor(image):
                return (self._load_range_tensor(image, start, length),)
            return (self._build_diskimage_view(image, start, length),)

        if return_type == "Tensor":
            return (self._load_range_tensor(image, start, length),)

        if return_type == "Input or DiskImage" and not self._is_tensor(image):
            return (self._build_diskimage_view(image, start, length),)

        selected = self._load_range_tensor(image, start, length)
        save_images(
            image=selected,
            prefix=kwargs["prefix"],
            start_sequence=kwargs["start_sequence"],
            output_dir=kwargs["output_dir"],
            format=kwargs["format"],
            num_workers=kwargs["num_workers"],
            compression_level=kwargs["compression_level"],
            quality=kwargs["quality"],
        )
        result = self._build_saved_diskimage(image, start, length, kwargs)
        del selected
        model_management.soft_empty_cache()
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VTS Image From Batch": VTS_ImageFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image From Batch": "VTS Image From Batch",
}
