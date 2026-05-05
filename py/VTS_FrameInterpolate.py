import os
import shutil
import sys

import torch
import comfy.utils
from comfy import model_management

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, save_images, vtsImageTypes, vtsReturnTypes


class VTS_FrameInterpolate:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "VTS/video"
    DESCRIPTION = """
VTS wrapper around ComfyUI frame interpolation that supports tensor or DiskImage inputs
and tensor or DiskImage outputs.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
                "return_type": (vtsReturnTypes, {"default": "Input", "tooltip": "Return the same type as the input images, force DiskImage output, or force Tensor output."}),
                "prefix": ("STRING", {"default": "frame_interpolate", "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP"}),
            }
        }

    def _is_tensor(self, image):
        return isinstance(image, torch.Tensor)

    def _image_count(self, image):
        return image.shape[0] if self._is_tensor(image) else image.number_of_images

    def _image_shape(self, image):
        return tuple(image.shape) if self._is_tensor(image) else image.shape

    def _resolve_return_type(self, return_type, images):
        if return_type == "Input":
            return "Tensor" if self._is_tensor(images) else "DiskImage"
        if return_type == "Input or DiskImage":
            return "DiskImage"
        return return_type

    def _resolve_output_config(self, prefix, start_sequence, output_dir, format, compression_level, quality):
        if quality is not None and quality > 100:
            quality = None
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

    def _load_frame(self, images, idx):
        if self._is_tensor(images):
            frame = images[idx:idx + 1]
            if frame.device.type != "cpu":
                frame = frame.cpu()
            return frame
        return images.load_images(start_sequence=images.start_sequence + idx, count=1)

    def _copy_frame_if_possible(self, images, idx, output_config, output_sequence):
        if self._is_tensor(images) or images.format != output_config["format"]:
            return False

        src_path = self._source_file_path(images, idx)
        dst_path = self._output_file_path(output_config, output_sequence)
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return False
        shutil.copy2(src_path, dst_path)
        return True

    def _save_chunk(self, chunk, output_config, output_sequence, num_workers):
        if chunk is None or chunk.shape[0] == 0:
            return output_sequence
        os.makedirs(output_config["output_dir"], exist_ok=True)
        save_images(
            image=chunk,
            prefix=output_config["prefix"],
            start_sequence=output_sequence,
            output_dir=output_config["output_dir"],
            format=output_config["format"],
            num_workers=num_workers,
            compression_level=output_config["compression_level"],
            quality=output_config["quality"],
        )
        return output_sequence + chunk.shape[0]

    def _append_or_save_original_frame(self, images, idx, frame_cpu, resolved_return_type, output_config, output_sequence, num_workers, tensor_chunks):
        if resolved_return_type == "Tensor":
            tensor_chunks.append(frame_cpu)
            return output_sequence

        if self._copy_frame_if_possible(images, idx, output_config, output_sequence):
            return output_sequence + 1

        return self._save_chunk(frame_cpu, output_config, output_sequence, num_workers)

    def _append_or_save_mids(self, mids_cpu, resolved_return_type, output_config, output_sequence, num_workers, tensor_chunks):
        if mids_cpu is None or mids_cpu.shape[0] == 0:
            return output_sequence

        if resolved_return_type == "Tensor":
            tensor_chunks.append(mids_cpu)
            return output_sequence

        new_sequence = self._save_chunk(mids_cpu, output_config, output_sequence, num_workers)
        del mids_cpu
        model_management.soft_empty_cache()
        return new_sequence

    def _load_all_to_tensor(self, images):
        if self._is_tensor(images):
            return images

        count = self._image_count(images)
        if count == 0:
            shape = self._image_shape(images)
            return torch.empty(shape, dtype=torch.float32)
        return images.load_images(start_sequence=images.start_sequence, count=count)

    def _same_output_range_conflicts(self, images, output_config, total_out_frames):
        if self._is_tensor(images):
            return False
        if images.prefix != output_config["prefix"]:
            return False
        if images.output_dir != output_config["output_dir"]:
            return False
        if images.format != output_config["format"]:
            return False

        input_start = images.start_sequence
        input_end = images.start_sequence + images.number_of_images - 1
        output_start = output_config["start_sequence"]
        output_end = output_config["start_sequence"] + total_out_frames - 1
        return not (output_end < input_start or output_start > input_end)

    def _build_result_metadata(self, input_images, total_out_frames, output_config):
        input_shape = self._image_shape(input_images)
        result = DiskImage(
            prefix=output_config["prefix"],
            start_sequence=output_config["start_sequence"],
            number_of_images=total_out_frames,
            output_dir=output_config["output_dir"],
            format=output_config["format"],
            image=None,
            compression_level=output_config["compression_level"],
            quality=output_config["quality"],
        )
        result.shape = (total_out_frames,) + tuple(input_shape[1:])
        result.dtype = torch.float32
        result.ndim = len(result.shape)
        return result

    def _prepare_frame(self, frame_cpu, device, dtype, align):
        frame = frame_cpu.movedim(-1, 1).to(dtype=dtype, device=device)
        if align > 1:
            from comfy.ldm.common_dit import pad_to_patch_size
            frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
        return frame

    def _stream_interpolate(self, interp_model, images, multiplier, resolved_return_type, output_config, num_workers):
        offload_device = model_management.intermediate_device()
        num_frames = self._image_count(images)
        image_shape = self._image_shape(images)
        H, W = image_shape[1], image_shape[2]

        if num_frames < 2 or multiplier < 2:
            if resolved_return_type == "Tensor":
                return self._load_all_to_tensor(images)

            if not self._is_tensor(images) and not self._same_output_range_conflicts(images, output_config, num_frames):
                output_sequence = output_config["start_sequence"]
                for i in range(num_frames):
                    copied = self._copy_frame_if_possible(images, i, output_config, output_sequence)
                    if copied:
                        output_sequence += 1
                    else:
                        frame = self._load_frame(images, i)
                        output_sequence = self._save_chunk(frame, output_config, output_sequence, num_workers)
                        del frame
                        model_management.soft_empty_cache()
                return self._build_result_metadata(images, num_frames, output_config)

            if not self._is_tensor(images):
                if self._same_output_range_conflicts(images, output_config, num_frames):
                    raise ValueError("VTS Frame Interpolate: output path overlaps the input DiskImage sequence. Use a different prefix, directory, format, or start_sequence.")
            tensor = self._load_all_to_tensor(images)
            output_sequence = self._save_chunk(tensor, output_config, output_config["start_sequence"], num_workers)
            del output_sequence
            del tensor
            model_management.soft_empty_cache()
            return self._build_result_metadata(images, num_frames, output_config)

        device = interp_model.load_device
        dtype = interp_model.model_dtype()
        inference_model = interp_model.model
        activation_mem = inference_model.memory_used_forward((1, H, W, image_shape[3]), dtype)
        model_management.load_models_gpu([interp_model], memory_required=activation_mem)
        align = getattr(inference_model, "pad_align", 1)

        total_pairs = num_frames - 1
        num_interp = multiplier - 1
        total_steps = total_pairs * num_interp
        pbar = comfy.utils.ProgressBar(total_steps)

        if resolved_return_type == "DiskImage" and self._same_output_range_conflicts(images, output_config, total_pairs * multiplier + 1):
            raise ValueError("VTS Frame Interpolate: output path overlaps the input DiskImage sequence. Use a different prefix, directory, format, or start_sequence.")

        batch = num_interp
        t_values = [t / multiplier for t in range(1, multiplier)]
        out_dtype = model_management.intermediate_dtype()
        tensor_chunks = [] if resolved_return_type == "Tensor" else None
        output_sequence = output_config["start_sequence"]

        first_frame_cpu = self._load_frame(images, 0)
        sample = self._prepare_frame(first_frame_cpu, device, dtype, align)
        pH, pW = sample.shape[2], sample.shape[3]
        ts_full = torch.tensor(t_values, device=device, dtype=dtype).reshape(num_interp, 1, 1, 1).expand(-1, 1, pH, pW)
        del sample

        multi_fn = getattr(inference_model, "forward_multi_timestep", None)
        feat_cache = {}
        prev_frame_prepared = None
        prev_frame_cpu = first_frame_cpu

        for i in range(total_pairs):
            img0_cpu = prev_frame_cpu
            img1_cpu = self._load_frame(images, i + 1)
            img0_single = prev_frame_prepared if prev_frame_prepared is not None else self._prepare_frame(img0_cpu, device, dtype, align)
            img1_single = self._prepare_frame(img1_cpu, device, dtype, align)
            prev_frame_prepared = img1_single
            prev_frame_cpu = img1_cpu

            feat_cache["img0"] = feat_cache.pop("next") if "next" in feat_cache else inference_model.extract_features(img0_single)
            feat_cache["img1"] = inference_model.extract_features(img1_single)
            feat_cache["next"] = feat_cache["img1"]

            mids_cpu = None
            used_multi = False
            if multi_fn is not None:
                try:
                    mids = multi_fn(img0_single, img1_single, t_values, cache=feat_cache)
                    mids_cpu = mids[:, :, :H, :W].to(device=offload_device, dtype=out_dtype).movedim(1, -1).clamp_(0.0, 1.0)
                    if mids_cpu.device.type != "cpu":
                        mids_cpu = mids_cpu.cpu()
                    pbar.update(num_interp)
                    used_multi = True
                    del mids
                except model_management.OOM_EXCEPTION:
                    model_management.soft_empty_cache()
                    multi_fn = None

            if not used_multi:
                mids_batches = []
                j = 0
                while j < num_interp:
                    b = min(batch, num_interp - j)
                    try:
                        img0 = img0_single.expand(b, -1, -1, -1)
                        img1 = img1_single.expand(b, -1, -1, -1)
                        mids = inference_model(img0, img1, timestep=ts_full[j:j + b], cache=feat_cache)
                        mids_batch = mids[:, :, :H, :W].to(device=offload_device, dtype=out_dtype).movedim(1, -1).clamp_(0.0, 1.0)
                        if mids_batch.device.type != "cpu":
                            mids_batch = mids_batch.cpu()
                        mids_batches.append(mids_batch)
                        pbar.update(b)
                        j += b
                        del mids
                    except model_management.OOM_EXCEPTION:
                        if batch <= 1:
                            raise
                        batch = max(1, batch // 2)
                        model_management.soft_empty_cache()
                mids_cpu = torch.cat(mids_batches, dim=0) if len(mids_batches) > 0 else None

            output_sequence = self._append_or_save_original_frame(
                images,
                i,
                img0_cpu.to(dtype=out_dtype),
                resolved_return_type,
                output_config,
                output_sequence,
                num_workers,
                tensor_chunks,
            )
            output_sequence = self._append_or_save_mids(
                mids_cpu,
                resolved_return_type,
                output_config,
                output_sequence,
                num_workers,
                tensor_chunks,
            )

            del img0_cpu
            if mids_cpu is not None and resolved_return_type == "Tensor":
                del mids_cpu

        output_sequence = self._append_or_save_original_frame(
            images,
            num_frames - 1,
            prev_frame_cpu.to(dtype=out_dtype),
            resolved_return_type,
            output_config,
            output_sequence,
            num_workers,
            tensor_chunks,
        )

        if resolved_return_type == "Tensor":
            if len(tensor_chunks) == 0:
                raise RuntimeError("VTS Frame Interpolate: no output frames were produced")
            result = torch.cat(tensor_chunks, dim=0)
            return result

        model_management.soft_empty_cache()
        total_out_frames = total_pairs * multiplier + 1
        return self._build_result_metadata(images, total_out_frames, output_config)

    def interpolate(self, interp_model, images, multiplier, return_type="Input", prefix="frame_interpolate", start_sequence=0, output_dir=default_output_dir, format="png", num_workers=16, compression_level=9, quality=95):
        resolved_return_type = self._resolve_return_type(return_type, images)
        output_config = self._resolve_output_config(prefix, start_sequence, output_dir, format, compression_level, quality)
        result = self._stream_interpolate(interp_model, images, multiplier, resolved_return_type, output_config, num_workers)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VTS Frame Interpolate": VTS_FrameInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Frame Interpolate": "VTS Frame Interpolate",
}
