"""Masked-video conditioning for ComfyUI's native MiniMax H3 pipeline."""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils
import nodes
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from comfy_api.latest import io


_UTILS = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _UTILS not in sys.path:
    sys.path.append(_UTILS)

from vtsUtils import DiskImage


_LOG = logging.getLogger("vts_h3_masked_video")


def _pixel_frames(latent_steps: int) -> int:
    """Return the number of pixel frames represented by H3 video tokens."""

    return sum(
        FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
        for index in range(int(latent_steps))
    )


def _available_frames(images, name: str) -> int:
    if isinstance(images, torch.Tensor):
        if images.ndim not in (3, 4):
            raise ValueError(
                f"{name} must be [T,H,W] or [T,H,W,C], got {tuple(images.shape)}."
            )
        return int(images.shape[0])
    if isinstance(images, DiskImage):
        count = images.number_of_images
        if count is None and images.shape is not None:
            count = images.shape[0]
        if count is None:
            raise ValueError(f"{name} does not report how many frames it contains.")
        return int(count)
    raise TypeError(f"{name} must be a Tensor or VTS DiskImage, got {type(images)!r}.")


def _materialize_head(images, frame_count: int, name: str) -> torch.Tensor:
    available = _available_frames(images, name)
    if available < 1:
        raise ValueError(f"{name} is empty.")
    count = min(int(frame_count), available)
    if isinstance(images, DiskImage):
        images = images.materialize(start=0, count=count)
    else:
        images = images[:count]
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} could not be materialized as a tensor.")
    return images


def _resize_video(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(
            "control_video must be [T,H,W,C] with at least three channels, "
            f"got {tuple(images.shape)}."
        )
    samples = images[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "bilinear", "center"
    )
    return samples.movedim(1, -1)


def _resize_valid_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if mask.ndim == 4:
        if mask.shape[-1] < 1:
            raise ValueError("control_mask has no channels.")
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(
            "control_mask must be [T,H,W] or [T,H,W,C], "
            f"got {tuple(mask.shape)}."
        )
    return F.interpolate(
        mask.unsqueeze(1).float(),
        size=(int(height), int(width)),
        mode="nearest",
    ).squeeze(1)


def _prepare_control(
    control_video,
    control_mask,
    frame_count: int,
    width: int,
    height: int,
    hole_fill: str,
    invert_mask: bool,
):
    """Create the masked RGB target and its white-known validity mask."""

    if int(width) % 32 or int(height) % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32.")
    if hole_fill not in ("black", "gray"):
        raise ValueError("hole_fill must be 'black' or 'gray'.")

    video = _resize_video(
        _materialize_head(control_video, frame_count, "control_video"),
        width,
        height,
    )
    mask = _resize_valid_mask(
        _materialize_head(control_mask, frame_count, "control_mask"),
        width,
        height,
    )
    mask = (mask > 0.5).to(device=video.device, dtype=torch.float32)
    if invert_mask:
        mask = 1.0 - mask

    fill_value = 0.0 if hole_fill == "black" else 0.5
    image = torch.full(
        (int(frame_count), int(height), int(width), 3),
        fill_value,
        device=video.device,
        dtype=video.dtype,
    )
    valid = torch.zeros(
        (int(frame_count), int(height), int(width)),
        device=video.device,
        dtype=torch.float32,
    )
    video_count = min(int(frame_count), int(video.shape[0]))
    mask_count = min(int(frame_count), int(mask.shape[0]))
    image[:video_count] = video[:video_count]
    valid[:mask_count] = mask[:mask_count]

    # Unknown pixels must not leak stale RGB content through the VAE.
    image = torch.where(
        valid.unsqueeze(-1) >= 0.5,
        image,
        torch.as_tensor(fill_value, device=image.device, dtype=image.dtype),
    )
    return image, valid


def _latent_video_mask(valid: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
    """Pool a pixel validity video onto H3's exact temporal/spatial token grid."""

    if target_video.ndim != 5 or target_video.shape[0] != 1:
        raise ValueError(
            "MiniMax H3 target video must be [1,C,T,H,W], "
            f"got {tuple(target_video.shape)}."
        )
    frame_count, pixel_h, pixel_w = valid.shape
    latent_t, latent_h, latent_w = (
        int(target_video.shape[2]),
        int(target_video.shape[3]),
        int(target_video.shape[4]),
    )
    expected_frames = _pixel_frames(latent_t)
    if int(frame_count) != expected_frames:
        raise ValueError(
            f"The H3 target represents {expected_frames} frames, but the mask has "
            f"{frame_count}."
        )
    if pixel_h % latent_h or pixel_w % latent_w:
        raise ValueError(
            "The pixel mask cannot be divided exactly onto the H3 latent grid: "
            f"{pixel_w}x{pixel_h} -> {latent_w}x{latent_h}."
        )

    # H3/ComfyUI noise-mask convention: 0 preserves; 1 generates.  Max pooling
    # is conservative: a latent token is regenerated if any source pixel/frame
    # represented by it is a hole.
    generate = 1.0 - valid
    block_h, block_w = pixel_h // latent_h, pixel_w // latent_w
    generate = generate.reshape(
        frame_count, latent_h, block_h, latent_w, block_w
    ).amax(dim=(2, 4))

    pooled = []
    offset = 0
    for index in range(latent_t):
        owned_frames = int(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)])
        pooled.append(generate[offset:offset + owned_frames].amax(dim=0))
        offset += owned_frames
    if offset != frame_count:
        raise RuntimeError("MiniMax H3's temporal mask grid changed unexpectedly.")

    mask = torch.stack(pooled, dim=0).unsqueeze(0).unsqueeze(0)
    return mask.expand_as(target_video).contiguous()


def _latent_streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError("Native MiniMax H3 did not return a joint AV latent.")
    if len(streams) != 2:
        raise ValueError(
            f"Native MiniMax H3 returned {len(streams)} latent streams; expected video and audio."
        )
    return streams


class VTSMiniMaxH3MaskedVideoConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VTSMiniMaxH3MaskedVideoConditioning",
            display_name="VTS MiniMax H3 Masked-Video Conditioning",
            category="VTS/wrappers/conditioning/minimax",
            description=(
                "MiniMax H3 reference-to-video conditioning with a pixel-aligned "
                "masked control video. White mask pixels are preserved; black "
                "pixels are generated. The audio stream is generated."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Image.Input("control_video", tooltip="Pixel-aligned target frames at 24 fps."),
                io.Image.Input(
                    "control_mask",
                    tooltip="Validity-mask video: white=known/preserve, black=hole/generate.",
                ),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frame count at 24 fps on H3's 17k+5 frame grid.",
                ),
                io.Combo.Input("hole_fill", options=["black", "gray"], default="black"),
                io.Boolean.Input(
                    "invert_mask",
                    default=False,
                    tooltip="Invert white-known/black-hole interpretation.",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Use the same reference-image sizing as native MiniMax H3 Ref2VA.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps."),
                        prefix="ref_video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        control_video,
        control_mask,
        width,
        height,
        length,
        hole_fill="black",
        invert_mask=False,
        ref_image_size="match",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        native = MiniMaxH3ReferenceToVideo.execute(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        native_result = native.result if isinstance(native, io.NodeOutput) else native
        if not native_result or len(native_result) != 2:
            raise RuntimeError("Native MiniMax H3 Reference to Video returned unexpected outputs.")
        conditioning, latent = native_result
        target_video, target_audio = _latent_streams(latent)

        frame_count = _pixel_frames(int(target_video.shape[2]))
        image, valid = _prepare_control(
            control_video,
            control_mask,
            frame_count,
            int(width),
            int(height),
            hole_fill,
            bool(invert_mask),
        )
        encoded_video = vae.encode(image)
        if not isinstance(encoded_video, torch.Tensor) or encoded_video.ndim != 5:
            raise ValueError(
                "The H3 video VAE must return [B,C,T,H,W], got "
                f"{tuple(getattr(encoded_video, 'shape', ())) }."
            )
        if tuple(encoded_video.shape) != tuple(target_video.shape):
            raise ValueError(
                "The encoded control video does not match the requested H3 target: "
                f"{tuple(encoded_video.shape)} != {tuple(target_video.shape)}."
            )

        video_mask = _latent_video_mask(valid, target_video).to(encoded_video.device)
        audio_mask = torch.ones_like(target_audio)
        output = latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor(
            (encoded_video, target_audio)
        )
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        )
        _LOG.info(
            "VTS H3 masked video: %d frames, %dx%d, %.2f%% video tokens generated",
            frame_count,
            int(width),
            int(height),
            100.0 * float(video_mask[:, :1].mean()),
        )
        return io.NodeOutput(conditioning, output)


NODE_CLASS_MAPPINGS = {
    "VTSMiniMaxH3MaskedVideoConditioning": VTSMiniMaxH3MaskedVideoConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSMiniMaxH3MaskedVideoConditioning":
        "VTS MiniMax H3 Masked-Video Conditioning",
}
