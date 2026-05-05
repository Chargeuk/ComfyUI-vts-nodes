import os
import shutil
import sys

import torch
from comfy import model_management

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, save_images, vtsImageTypes, vtsReturnTypes


class VTS_ImageBatchExtendWithOverlap:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extended_images",)
    FUNCTION = "extend"
    CATEGORY = "VTS/image"
    DESCRIPTION = """
Helper node for video generation extension.
Returns the extended sequence. If no new images are provided, returns the trailing
overlap frames from source_images for use as the starting images of the extension.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_images": ("IMAGE", {"tooltip": "The source images to extend"}),
                "overlap": ("INT", {"default": 13, "min": 1, "max": 4096, "step": 1, "tooltip": "Number of overlapping frames between source and new images"}),
                "overlap_side": (["source", "new_images"], {"default": "source", "tooltip": "Which side to overlap on"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"], {"default": "linear_blend", "tooltip": "Method to use for overlapping frames"}),
                "return_type": (vtsReturnTypes, {"default": "Input", "tooltip": "Return the result as the same type as source_images, force DiskImage output, or force Tensor output."}),
                "batch_size": ("INT", {"default": 20, "min": 1}),
                "prefix": ("STRING", {"default": "extended_images", "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP"}),
            },
            "optional": {
                "new_images": ("IMAGE", {"tooltip": "The new images to extend with"}),
            }
        }

    def _is_tensor(self, image):
        return isinstance(image, torch.Tensor)

    def _image_count(self, image):
        return image.shape[0] if self._is_tensor(image) else image.number_of_images

    def _image_shape(self, image):
        return tuple(image.shape) if self._is_tensor(image) else image.shape

    def _resolve_return_type(self, return_type, source_images):
        if return_type == "Input":
            return "Tensor" if self._is_tensor(source_images) else "DiskImage"
        if return_type == "Input or DiskImage":
            return "DiskImage"
        return return_type

    def _resolve_output_config(self, source_images, return_type, prefix, start_sequence, output_dir, format, compression_level, quality):
        if quality is not None and quality > 100:
            quality = None

        if not self._is_tensor(source_images) and return_type in ("Input", "Input or DiskImage"):
            prefix = source_images.prefix
            start_sequence = source_images.start_sequence
            output_dir = source_images.output_dir
            format = source_images.format
            compression_level = source_images.compression_level
            quality = source_images.quality

        return {
            "prefix": prefix,
            "start_sequence": start_sequence,
            "output_dir": output_dir,
            "format": format,
            "compression_level": compression_level,
            "quality": quality,
        }

    def _source_file_path(self, image, index):
        sequence_num = image.start_sequence + index
        return os.path.join(image.output_dir, f"{image.prefix}_{sequence_num:06d}.{image.format}")

    def _output_file_path(self, output_config, sequence_num):
        return os.path.join(output_config["output_dir"], f"{output_config['prefix']}_{sequence_num:06d}.{output_config['format']}")

    def _load_range_tensor(self, image, start, count):
        if count <= 0:
            return None

        if self._is_tensor(image):
            batch = image[start:start + count]
            if batch.device.type != "cpu":
                batch = batch.cpu()
            return batch

        return image.load_images(start_sequence=image.start_sequence + start, count=count)

    def _append_range_to_pieces(self, image, start, count, batch_size, pieces):
        if count <= 0:
            return

        for offset in range(0, count, batch_size):
            chunk_count = min(batch_size, count - offset)
            chunk = self._load_range_tensor(image, start + offset, chunk_count)
            if chunk is not None:
                pieces.append(chunk)

    def _copy_disk_range(self, image, start, count, output_config, output_sequence):
        if count <= 0:
            return output_sequence

        os.makedirs(output_config["output_dir"], exist_ok=True)
        for i in range(count):
            src_path = self._source_file_path(image, start + i)
            dst_path = self._output_file_path(output_config, output_sequence + i)
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            shutil.copy2(src_path, dst_path)
        return output_sequence + count

    def _save_tensor_range(self, tensor, output_config, output_sequence, num_workers):
        if tensor is None or tensor.shape[0] == 0:
            return output_sequence

        os.makedirs(output_config["output_dir"], exist_ok=True)
        save_images(
            image=tensor,
            prefix=output_config["prefix"],
            start_sequence=output_sequence,
            output_dir=output_config["output_dir"],
            format=output_config["format"],
            num_workers=num_workers,
            compression_level=output_config["compression_level"],
            quality=output_config["quality"],
        )
        return output_sequence + tensor.shape[0]

    def _write_range_to_disk(self, image, start, count, output_config, output_sequence, batch_size, num_workers):
        if count <= 0:
            return output_sequence

        if not self._is_tensor(image) and image.format == output_config["format"]:
            return self._copy_disk_range(image, start, count, output_config, output_sequence)

        for offset in range(0, count, batch_size):
            chunk_count = min(batch_size, count - offset)
            chunk = self._load_range_tensor(image, start + offset, chunk_count)
            output_sequence = self._save_tensor_range(chunk, output_config, output_sequence, num_workers)
            del chunk
            model_management.soft_empty_cache()
        return output_sequence

    def _blend_overlap(self, blend_src, blend_dst, overlap_mode):
        if overlap_mode == "linear_blend":
            alpha = torch.linspace(0, 1, blend_src.shape[0] + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
            alpha = alpha.view(-1, 1, 1, 1)
            return (1 - alpha) * blend_src + alpha * blend_dst

        if overlap_mode == "filmic_crossfade":
            gamma = 2.2
            alpha = torch.linspace(0, 1, blend_src.shape[0] + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
            alpha = alpha.view(-1, 1, 1, 1)
            linear_src = torch.pow(blend_src, gamma)
            linear_dst = torch.pow(blend_dst, gamma)
            blended = (1 - alpha) * linear_src + alpha * linear_dst
            return torch.pow(blended, 1.0 / gamma)

        if overlap_mode == "ease_in_out":
            t = torch.linspace(0, 1, blend_src.shape[0] + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
            eased_t = 3 * t * t - 2 * t * t * t
            eased_t = eased_t.view(-1, 1, 1, 1)
            return (1 - eased_t) * blend_src + eased_t * blend_dst

        if overlap_mode == "perceptual_crossfade":
            import kornia

            alpha = torch.linspace(0, 1, blend_src.shape[0] + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
            src_nchw = blend_src.movedim(-1, 1)
            dst_nchw = blend_dst.movedim(-1, 1)
            lab_src = kornia.color.rgb_to_lab(src_nchw)
            lab_dst = kornia.color.rgb_to_lab(dst_nchw)
            alpha = alpha.view(-1, 1, 1, 1)
            blended_lab = (1 - alpha) * lab_src + alpha * lab_dst
            blended_rgb = kornia.color.lab_to_rgb(blended_lab)
            return blended_rgb.movedim(1, -1)

        raise ValueError(f"Unsupported overlap_mode: {overlap_mode}")

    def _build_result_metadata(self, source_images, output_count, output_config, start_sequence_override=None):
        source_shape = self._image_shape(source_images)
        result = DiskImage(
            prefix=output_config["prefix"],
            start_sequence=output_config["start_sequence"] if start_sequence_override is None else start_sequence_override,
            number_of_images=output_count,
            output_dir=output_config["output_dir"],
            format=output_config["format"],
            image=None,
            compression_level=output_config["compression_level"],
            quality=output_config["quality"],
        )
        result.shape = (output_count,) + tuple(source_shape[1:])
        result.dtype = torch.float32 if source_shape is not None else None
        result.ndim = len(result.shape)
        return result

    def extend(self, source_images, overlap, overlap_side, overlap_mode, return_type="Input", batch_size=20, prefix="extended_images", start_sequence=0, output_dir=default_output_dir, format="png", num_workers=16, compression_level=9, quality=95, new_images=None):
        source_count = self._image_count(source_images)
        if overlap > source_count:
            raise ValueError(f"overlap ({overlap}) cannot be larger than the number of source images ({source_count})")

        resolved_return_type = self._resolve_return_type(return_type, source_images)
        output_config = self._resolve_output_config(
            source_images,
            return_type,
            prefix,
            start_sequence,
            output_dir,
            format,
            compression_level,
            quality,
        )

        if new_images is None:
            if resolved_return_type == "Tensor":
                tail = self._load_range_tensor(source_images, source_count - overlap, overlap)
                return (tail,)

            if not self._is_tensor(source_images) and return_type in ("Input", "Input or DiskImage"):
                return (self._build_result_metadata(source_images, overlap, output_config, start_sequence_override=source_images.start_sequence + source_count - overlap),)

            output_sequence = output_config["start_sequence"]
            output_sequence = self._write_range_to_disk(
                source_images,
                source_count - overlap,
                overlap,
                output_config,
                output_sequence,
                batch_size,
                num_workers,
            )
            model_management.soft_empty_cache()
            return (self._build_result_metadata(source_images, overlap, output_config),)

        new_count = self._image_count(new_images)
        if overlap > new_count:
            raise ValueError(f"overlap ({overlap}) cannot be larger than the number of new images ({new_count})")

        source_shape = self._image_shape(source_images)
        new_shape = self._image_shape(new_images)
        if source_shape[1:] != new_shape[1:]:
            raise ValueError(f"Source and new images must have the same shape: {source_shape[1:]} vs {new_shape[1:]}")

        if overlap_mode == "cut":
            if overlap_side == "source":
                output_count = (source_count - overlap) + new_count
            else:
                output_count = source_count + (new_count - overlap)
        else:
            output_count = source_count + new_count - overlap

        if resolved_return_type == "Tensor":
            pieces = []

            if overlap_mode == "cut":
                if overlap_side == "source":
                    self._append_range_to_pieces(source_images, 0, source_count - overlap, batch_size, pieces)
                    self._append_range_to_pieces(new_images, 0, new_count, batch_size, pieces)
                else:
                    self._append_range_to_pieces(source_images, 0, source_count, batch_size, pieces)
                    self._append_range_to_pieces(new_images, overlap, new_count - overlap, batch_size, pieces)
            else:
                self._append_range_to_pieces(source_images, 0, source_count - overlap, batch_size, pieces)
                source_overlap = self._load_range_tensor(source_images, source_count - overlap, overlap)
                new_overlap = self._load_range_tensor(new_images, 0, overlap)
                if overlap_side == "source":
                    blended = self._blend_overlap(source_overlap, new_overlap, overlap_mode)
                else:
                    blended = self._blend_overlap(new_overlap, source_overlap, overlap_mode)
                del source_overlap
                del new_overlap
                pieces.append(blended)
                self._append_range_to_pieces(new_images, overlap, new_count - overlap, batch_size, pieces)

            if len(pieces) == 0:
                raise RuntimeError("VTS Image Batch Extend With Overlap - no output images were produced")
            result = torch.cat(pieces, dim=0)
            return (result,)

        os.makedirs(output_config["output_dir"], exist_ok=True)
        output_sequence = output_config["start_sequence"]

        if overlap_mode == "cut":
            if overlap_side == "source":
                output_sequence = self._write_range_to_disk(source_images, 0, source_count - overlap, output_config, output_sequence, batch_size, num_workers)
                output_sequence = self._write_range_to_disk(new_images, 0, new_count, output_config, output_sequence, batch_size, num_workers)
            else:
                output_sequence = self._write_range_to_disk(source_images, 0, source_count, output_config, output_sequence, batch_size, num_workers)
                output_sequence = self._write_range_to_disk(new_images, overlap, new_count - overlap, output_config, output_sequence, batch_size, num_workers)
        else:
            source_overlap = self._load_range_tensor(source_images, source_count - overlap, overlap)
            new_overlap = self._load_range_tensor(new_images, 0, overlap)

            output_sequence = self._write_range_to_disk(source_images, 0, source_count - overlap, output_config, output_sequence, batch_size, num_workers)
            if overlap_side == "source":
                blended = self._blend_overlap(source_overlap, new_overlap, overlap_mode)
            else:
                blended = self._blend_overlap(new_overlap, source_overlap, overlap_mode)
            output_sequence = self._save_tensor_range(blended, output_config, output_sequence, num_workers)
            del blended
            del source_overlap
            del new_overlap
            model_management.soft_empty_cache()
            output_sequence = self._write_range_to_disk(new_images, overlap, new_count - overlap, output_config, output_sequence, batch_size, num_workers)

        model_management.soft_empty_cache()
        return (self._build_result_metadata(source_images, output_count, output_config),)


NODE_CLASS_MAPPINGS = {
    "VTS Image Batch Extend With Overlap": VTS_ImageBatchExtendWithOverlap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image Batch Extend With Overlap": "VTS Image Batch Extend With Overlap"
}
