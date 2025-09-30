# ruff: noqa: TID252

import math
import sys
import os

import folder_paths
import torch
from comfy import model_management
from typing import TYPE_CHECKING, NamedTuple

from comfy import latent_formats

# Add the py directory to sys.path to allow imports
tae_dir = os.path.join(os.path.dirname(__file__), "taeVid")
if tae_dir not in sys.path:
    sys.path.append(tae_dir)

from taeVid_previewer import VIDEO_FORMATS, VideoModelInfo
from tae_vid import TAEVid


class VTS_TAEVideoNodeBase:
    FUNCTION = "go"
    CATEGORY = "latent"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent_type": (("wan21", "wan22", "hunyuanvideo", "mochi"),),
                "dtype": (("float32", "float16", "bfloat16"), {"default": "float16"}, {"tooltip": "The data type of the input tensor. Use the type that the TVAE was trained with. Most Tiny VAEs are trained using float16, so it's usually the best."}),
                "parallel_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Parallel mode is faster but requires more memory.",
                    },
                ),
                "batch_window_size": (
                    "INT",
                    {
                        "default": 88,
                        "max": 999999999999999999,
                        "min": 12,
                        "step": 4,
                    },
                ),
                # "overlap_frames": (
                #     "INT",
                #     {
                #         "default": 12,
                #         "min": 0,
                #         "step": 4,
                #     },
                # ),
                "tileX": (
                    "INT",
                    {
                        "default": 128,
                        "tooltip": "The tile size in X. Should be 128 for decoding and 160 for encoding to reduce artifacts.",
                        "min": 0,
                        "step": 32,
                    },
                ),
                "tileY": (
                    "INT",
                    {
                        "default": 128,
                        "tooltip": "The tile size in Y. Should be 128 for decoding and 160 for encoding to reduce artifacts.",
                        "min": 0,
                        "step": 32,
                    },
                ),
                "overlapX": (
                    "INT",
                    {
                        "default": 64,
                        "tooltip": "The overlap size in X (the gap between the previous X start and the next). Should be 64 for decoding and 96 for encoding to reduce artifacts.",
                        "min": 0,
                        "step": 32,
                    },
                ),
                "overlapY": (
                    "INT",
                    {
                        "default": 64,
                        "tooltip": "The overlap size in Y (the gap between the previous Y start and the next). Should be 64 for decoding and 96 for encoding to reduce artifacts.",
                        "min": 0,
                        "step": 32,
                    },
                ),
                 "use_tiled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use tiled processing for reduced peak memory.",
                    },
                ),
                "blend_mode": (
                    ("linear", "cosine", "smoothstep", "gaussian"),
                    {
                        "default": "cosine",
                        "tooltip": "Blending mode for tile edges. Cosine provides smoother seams than linear.",
                    },
                ),
                "blend_exp": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Blend exponent. <1.0 softens edges, >1.0 sharpens them.",
                    },
                ),
                "min_border_fraction": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": "Minimum border as fraction of tile size. Use 0.1-0.25 for very small tiles.",
                    },
                ),
                "clamp_mode": (
                    ("tanh", "hard"),
                    {
                        "default": "tanh",
                        "tooltip": "Activation clamping mode. 'tanh' uses smooth tanh(x/3)*3 (original), 'hard' uses simple clamp(-3,3). Hard clamp may reduce color blotches but can affect quality.",
                    },
                ),
            },
        }

    @classmethod
    def get_taevid_model(
        cls,
        latent_type: str,
        dtype: torch.dtype,
        clamp_mode: str = "tanh",
    ) -> tuple[TAEVid, torch.device, torch.dtype, VideoModelInfo]:
        vmi = VIDEO_FORMATS.get(latent_type)
        if vmi is None or vmi.tae_model is None:
            raise ValueError("Bad latent type")
        tae_model_path = folder_paths.get_full_path("vae_approx", vmi.tae_model)
        if tae_model_path is None:
            if latent_type == "wan21":
                model_src = "taew2_1.pth from https://github.com/madebyollin/taehv"
            elif latent_type == "wan22":
                model_src = "taew2_2.pth from https://github.com/madebyollin/taehv"
            elif latent_type == "hunyuanvideo":
                model_src = "taehv.pth from https://github.com/madebyollin/taehv"
            else:
                model_src = "taem1.pth from https://github.com/madebyollin/taem1"
            err_string = f"Missing TAE video model. Download {model_src} and place it in the models/vae_approx directory"
            raise RuntimeError(err_string)
        device = model_management.vae_device()
        # Keep weights in fp32 (training style); runtime autocast will handle fp16
        model = TAEVid(checkpoint_path=tae_model_path, vmi=vmi, device=device, clamp_mode=clamp_mode).to(device=device, dtype=torch.float32)
        return (
            model,
            device,
            dtype,  # requested runtime dtype
            vmi,
        )

    @classmethod
    def get_dtype_from_string(cls, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch.dtype"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)

    @classmethod
    def go(cls, *, latent, latent_type: str, dtype: str, parallel_mode: bool, batch_window_size: int, # overlap_frames: int,
           tileX: int, tileY: int, overlapX: int, overlapY: int, use_tiled: bool, blend_mode: str, blend_exp: float, min_border_fraction: float, clamp_mode: str) -> tuple:
        raise NotImplementedError


class VTS_TAEVideoDecode(VTS_TAEVideoNodeBase):
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "latent"
    DESCRIPTION = "Fast decoding of Wan, Hunyuan and Mochi video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "latent": ("LATENT",),
        }
        return result

    @classmethod
    def go(cls, *, latent: dict, latent_type: str, dtype: str, parallel_mode: bool, batch_window_size: int, # overlap_frames: int,
            tileX: int, tileY: int, overlapX: int, overlapY: int, use_tiled: bool, blend_mode: str, blend_exp: float, min_border_fraction: float, clamp_mode: str) -> tuple:
        # number_of_overlapping_latents = overlap_frames // 4
        # number_of_overlapping_images = overlap_frames
        # number_of_images_to_drop = 1 + overlap_frames
        # original_batch_window_size = batch_window_size
        # original_overlap_frames = overlap_frames
        # batch_window_size = (batch_window_size + 3) // 4
        # overlap_frames = overlap_frames // 4
        # number_of_latents_to_drop = 1 + overlap_frames
        samples = latent["samples"]

        # decode images in batch_window_size sized batches with overlap
        a, b, total_items, c, d = samples.shape
        decoded_images = None  # Initialize as None instead of list
        print(f"Starting decoding of {total_items} latents with batch size {batch_window_size}, tileX {tileX}, tileY {tileY}, overlapX {overlapX}, overlapY {overlapY}")

        decoded_images = cls.execute(latent=latent, latent_type=latent_type, dtype=dtype, parallel_mode=parallel_mode,
                                 tileX=tileX, tileY=tileY, overlapX=overlapX, overlapY=overlapY, use_tiled=use_tiled,
                                 blend_mode=blend_mode, blend_exp=blend_exp, min_border_fraction=min_border_fraction, clamp_mode=clamp_mode,
                                 time_chunk=batch_window_size)
        return (decoded_images,)
        # # Calculate number of decode loops
        # is_last_loop = False
        # loop_idx = 0
        # while not is_last_loop:
        #     if loop_idx == 0:
        #         # First decode: start from beginning, decode batch_window_size latents
        #         start_idx = 0
        #         end_idx = min(batch_window_size, total_items)
        #     else:
        #         # Subsequent decodes: start one latent back from where we left off
        #         start_idx = (loop_idx * (batch_window_size - number_of_latents_to_drop))
        #         end_idx = min(start_idx + batch_window_size, total_items)

        #     is_last_loop = (end_idx >= total_items)
        #     # Extract batch for decoding
        #     batch_samples = samples[:, :, start_idx:end_idx, :, :]
        #     batch_latent = {"samples": batch_samples}

        #     # decode this batch
        #     images = cls.execute(latent=batch_latent, latent_type=latent_type, dtype=dtype, parallel_mode=parallel_mode,
        #                          tileX=tileX, tileY=tileY, overlapX=overlapX, overlapY=overlapY, use_tiled=use_tiled,
        #                          blend_mode=blend_mode, blend_exp=blend_exp, min_border_fraction=min_border_fraction, clamp_mode=clamp_mode)
        #     print(f"Decoded image shape for batch {loop_idx + 1}: {images.shape}")
        #     # Drop the last image except on the final loop

        #     if not is_last_loop and images.shape[0] > 1:
        #         images = images[:-1, :, :, :]  # Drop last image
        #         images_kept = images.shape[0]
        #     else:
        #         images_kept = images.shape[0]

        #     # Extend the tensor instead of appending to list
        #     if decoded_images is None:
        #         decoded_images = images
        #     else:
        #         # the last number_of_overlapping_images of decoded_images should be equal to
        #         # the first number_of_overlapping_images of images
        #         # we need to blend them
        #         if number_of_overlapping_images > 0:
        #             # Get the overlapping regions
        #             new_overlap = images[:number_of_overlapping_images, :, :, :]  # First number_of_overlapping_images from new batch
        #             existing_overlap = decoded_images[-number_of_overlapping_images:, :, :, :]  # Last number_of_overlapping_images from existing

        #             # Create linear blending weights with extended range to avoid extremes
        #             extended_size = number_of_overlapping_images + 2  # Add 2 extra
        #             extended_weights = torch.linspace(1.0, 0.0, extended_size, device=existing_overlap.device)
        #             blend_weights = extended_weights[1:-1]  # Remove first and last elements
        #             # Reshape weights to match tensor dimensions [ motion_sample_size, 1, 1, 1]
        #             blend_weights = blend_weights.view(-1, 1, 1, 1)

        #             # Perform linear blending
        #             blended_overlap = existing_overlap * blend_weights + new_overlap * (1.0 - blend_weights)
        #             blended_overlap_number_of_images = blended_overlap.shape[0]
        #             number_of_generated_images = decoded_images.shape[0]
        #             print(f"Loop {loop_idx}: replacing {number_of_overlapping_images} images with {blended_overlap_number_of_images} images in current total of {number_of_generated_images} images")

        #             # Replace the last number_of_overlapping_images frames in samples with the blended version
        #             decoded_images[-number_of_overlapping_images:, :, :, :] = blended_overlap
        #             number_of_generated_images = decoded_images.shape[0]
        #             print(f"Loop {loop_idx}: replaced {number_of_overlapping_images} images with {blended_overlap_number_of_images} images to give new total of {number_of_generated_images} images")

        #             # Append only the new frames (skip the overlapping region)
        #             new_samples_only = images[number_of_overlapping_images:, :, :, :]
        #         else:
        #             new_samples_only = images

        #         new_samples_images_count = new_samples_only.shape[0]
        #         if new_samples_images_count > 0:  # Only concatenate if there are new frames
        #             decoded_images = torch.cat([decoded_images, new_samples_only], dim=0)

        #     print(f"Encode loop {loop_idx + 1}: Images {start_idx}-{end_idx-1}, encoded {end_idx-start_idx} images, kept {images_kept} images")
        #     loop_idx += 1

        # # No concatenation step needed anymore
        # print(f"Final decoded images shape: {decoded_images.shape}")

        # return (decoded_images,)


    @classmethod
    def execute(cls, *, latent: dict, latent_type: str, dtype: str, parallel_mode: bool, tileX: int, tileY: int,
                overlapX: int, overlapY: int, use_tiled: bool, blend_mode: str, blend_exp: float, min_border_fraction: float, clamp_mode: str, time_chunk: int):
        torch_dtype = cls.get_dtype_from_string(dtype)
        model, device, model_dtype, vmi = cls.get_taevid_model(latent_type, torch_dtype, clamp_mode)
        samples = latent["samples"].detach().to(device=device, dtype=torch_dtype if torch_dtype != torch.float16 else torch.float16, copy=True)
        samples = vmi.latent_format().process_in(samples)

        from contextlib import nullcontext
        dev_type = torch.device(device).type
        use_amp = (torch_dtype == torch.float16 and dev_type in ("cuda", "hip"))
        amp_ctx = torch.autocast(device_type=dev_type, dtype=torch.float16) if use_amp else nullcontext()

        with amp_ctx:
            if use_tiled:
                img = model.decode_tiled(
                    samples.transpose(1, 2),
                    pixel_tile_size_x=tileX,
                    pixel_tile_size_y=tileY,
                    pixel_tile_stride_x=overlapX,
                    pixel_tile_stride_y=overlapY,
                    show_progress=True,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    time_chunk=time_chunk
                )
            else:
                img = model.decode(
                    samples.transpose(1, 2),
                    parallel=parallel_mode,
                    show_progress=True,
                )

        img = (
            img.movedim(2, -1)
               .to(dtype=torch.float32, device="cpu")
        )
        img = img.reshape(-1, *img.shape[-3:])
        return img


class VTS_TAEVideoEncode(VTS_TAEVideoNodeBase):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent"
    DESCRIPTION = "Fast encoding of Wan, Hunyuan and Mochi video latents with the video equivalent of TAESD."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        result = super().INPUT_TYPES()
        result["required"] |= {
            "image": ("IMAGE",),
        }
        return result

    @classmethod
    def go(cls, *, image: torch.Tensor, latent_type: str, dtype: str, parallel_mode: bool, batch_window_size: int, # overlap_frames: int,
            tileX: int, tileY: int, overlapX: int, overlapY: int, use_tiled: bool, blend_mode: str, blend_exp: float, min_border_fraction: float, clamp_mode: str) -> tuple:
        # number_of_overlapping_latents = overlap_frames // 4
        # number_of_images_to_drop = 1 + overlap_frames

        # decode images in batch_window_size sized batches with overlap
        total_items, image_height, image_width, C = image.shape
        encoded_latents = None  # Initialize as None instead of list
        print(f"Starting encoding of {total_items} images with batch size {batch_window_size}, tileX {tileX}, tileY {tileY}, overlapX {overlapX}, overlapY {overlapY}")

        encoded_latents = cls.execute(image=image, latent_type=latent_type, dtype=dtype, parallel_mode=parallel_mode,
                                  tileX=tileX, tileY=tileY, overlapX=overlapX, overlapY=overlapY, use_tiled=use_tiled,
                                  blend_mode=blend_mode, blend_exp=blend_exp, min_border_fraction=min_border_fraction, clamp_mode=clamp_mode,
                                  time_chunk=batch_window_size)
         # No concatenation step needed anymore
        print(f"Final encoded latents shape: {encoded_latents.shape}")

        return ({"samples": encoded_latents},)
        # # Calculate number of decode loops
        # is_last_loop = False
        # loop_idx = 0
        # while not is_last_loop:
        #     if loop_idx == 0:
        #         # First decode: start from beginning, decode batch_window_size latents
        #         start_idx = 0
        #         end_idx = min(batch_window_size, total_items)
        #     else:
        #         # Subsequent decodes: start one latent back from where we left off
        #         start_idx = (loop_idx * (batch_window_size - number_of_images_to_drop))
        #         end_idx = min(start_idx + batch_window_size, total_items)

        #     is_last_loop = (end_idx >= total_items)
        #     # Extract batch for decoding
        #     batch_Images = image[start_idx:end_idx, :, :, :]

        #     # Encode this batch
        #     latent = cls.execute(image=batch_Images, latent_type=latent_type, dtype=dtype, parallel_mode=parallel_mode,
        #                           tileX=tileX, tileY=tileY, overlapX=overlapX, overlapY=overlapY, use_tiled=use_tiled,
        #                           blend_mode=blend_mode, blend_exp=blend_exp, min_border_fraction=min_border_fraction, clamp_mode=clamp_mode)
        #     print(f"Encoded latent shape for batch {loop_idx + 1}: {latent.shape}")
        #     # Drop the last latent except on the final loop
            
        #     if not is_last_loop and latent.shape[2] > 1:
        #         latent = latent[:, :, :-1, :, :]  # Drop last latent
        #         latents_kept = latent.shape[2]
        #     else:
        #         latents_kept = latent.shape[2]

        #     # Extend the tensor instead of appending to list
        #     if encoded_latents is None:
        #         encoded_latents = latent
        #     else:
        #         # the last number_of_overlapping_latents of encoded_latents should be equal to
        #         # the first number_of_overlapping_latents of latent
        #         # we need to blend them
        #         if number_of_overlapping_latents > 0:
        #             # Get the overlapping regions
        #             new_overlap = latent[:, :, :number_of_overlapping_latents, :, :]  # First number_of_overlapping_latents from new batch
        #             existing_overlap = encoded_latents[:, :, -number_of_overlapping_latents:, :, :]  # Last number_of_overlapping_latents from existing

        #             # Create linear blending weights with extended range to avoid extremes
        #             extended_size = number_of_overlapping_latents + 2  # Add 2 extra
        #             extended_weights = torch.linspace(1.0, 0.0, extended_size, device=existing_overlap.device)
        #             blend_weights = extended_weights[1:-1]  # Remove first and last elements
        #             # Reshape weights to match tensor dimensions [1, 1, motion_sample_size, 1, 1]
        #             blend_weights = blend_weights.view(1, 1, -1, 1, 1)
                    
        #             # Perform linear blending
        #             blended_overlap = existing_overlap * blend_weights + new_overlap * (1.0 - blend_weights)
        #             blended_overlap_number_of_latents = blended_overlap.shape[2]
        #             number_of_generated_latents = encoded_latents.shape[2]
        #             print(f"Loop {loop_idx}: replacing {number_of_overlapping_latents} latents with {blended_overlap_number_of_latents} latents in current total of {number_of_generated_latents} latents")

        #             # Replace the last number_of_overlapping_latents frames in samples with the blended version
        #             encoded_latents[:, :, -number_of_overlapping_latents:, :, :] = blended_overlap
        #             number_of_generated_latents = encoded_latents.shape[2]
        #             print(f"Loop {loop_idx}: replaced {number_of_overlapping_latents} latents with {blended_overlap_number_of_latents} latents to give new total of {number_of_generated_latents} latents")

        #             # Append only the new frames (skip the overlapping region)
        #             new_samples_only = latent[:, :, number_of_overlapping_latents:, :, :]
        #         else:
        #             new_samples_only = latent

        #         new_samples_latent_count = new_samples_only.shape[2]
        #         if new_samples_latent_count > 0:  # Only concatenate if there are new frames
        #             encoded_latents = torch.cat([encoded_latents, new_samples_only], dim=2)

        #     print(f"Encode loop {loop_idx + 1}: Images {start_idx}-{end_idx-1}, encoded {end_idx-start_idx} images, kept {latents_kept} latents")
        #     loop_idx += 1

        # # No concatenation step needed anymore
        # print(f"Final encoded latents shape: {encoded_latents.shape}")

        # return ({"samples": encoded_latents},)

    @classmethod
    def execute(cls, *, image: torch.Tensor, latent_type: str, dtype: str, parallel_mode: bool,
                 tileX: int, tileY: int, overlapX: int, overlapY: int, use_tiled: bool, blend_mode: str, blend_exp: float, min_border_fraction: float, clamp_mode: str, time_chunk: int) -> torch.Tensor:
        torch_dtype = cls.get_dtype_from_string(dtype)
        model, device, model_dtype, vmi = cls.get_taevid_model(latent_type, torch_dtype, clamp_mode)
        image = image.detach().to(device=device, dtype=torch_dtype if torch_dtype != torch.float16 else torch.float16, copy=True)
        if image.ndim < 5:
            image = image.unsqueeze(0)
        if image.ndim < 5:
            image = image.unsqueeze(0)
        if image.ndim != 5:
            raise ValueError("Unexpected input image dimensions")
        frames = image.shape[1]
        add_frames = (
            math.ceil(frames / vmi.temporal_compression) * vmi.temporal_compression
            - frames
        )
        if add_frames > 0:
            image = torch.cat(
                (
                    image,
                    image[:, frames - 1 :, ...].expand(
                        image.shape[0],
                        add_frames,
                        *image.shape[2:],
                    ),
                ),
                dim=1,
            )

        from contextlib import nullcontext
        dev_type = torch.device(device).type
        use_amp = (torch_dtype == torch.float16 and dev_type in ("cuda", "hip"))
        amp_ctx = torch.autocast(device_type=dev_type, dtype=torch.float16) if use_amp else nullcontext()

        with amp_ctx:
            if use_tiled:
                latent = model.encode_tiled(
                    image[..., :3].movedim(-1, 2),
                    pixel_tile_size_x=tileX,
                    pixel_tile_size_y=tileY,
                    pixel_tile_stride_x=overlapX,
                    pixel_tile_stride_y=overlapY,
                    show_progress=True,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    time_chunk=time_chunk
                ).transpose(1, 2)
            else:
                latent = model.encode(
                    image[..., :3].movedim(-1, 2),
                    parallel=parallel_mode,
                    show_progress=True,
                ).transpose(1, 2)

        latent = (
            vmi.latent_format()
            .process_out(latent)
            .to(dtype=torch.float32, device="cpu")
        )
        return latent

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS TAE Video Decode": VTS_TAEVideoDecode,
    "VTS TAE Video Encode": VTS_TAEVideoEncode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS TAE Video Decode": "VTS TAE Video Decode",
    "VTS TAE Video Encode": "VTS TAE Video Encode"
}