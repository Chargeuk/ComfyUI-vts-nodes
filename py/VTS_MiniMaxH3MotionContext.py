import logging
import os
import sys

import torch
import torchaudio

import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage


_LOG = logging.getLogger("vts_h3_motion_context")
_FPS = 24.0
_AUDIO_LATENT_FPS = 40.0
_VIDEO_RUN_GRID = (1, 5, 22, 39, 56)


def _require_native_guides():
    if MiniMaxH3AddGuide is None:
        raise RuntimeError(
            "VTS H3 Motion Context requires a ComfyUI version with the native "
            "Add Guide for MiniMax H3 node. Update ComfyUI and restart.")


def _latent_streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "tensors"):
        streams = list(samples.tensors)
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError(
            "VTS H3 Motion Context expects a MiniMax H3 AV latent containing "
            "video and audio streams.")
    if len(streams) != 2:
        raise ValueError(
            "VTS H3 Motion Context expects exactly two H3 latent streams "
            "(video and audio), got %d." % len(streams))
    return streams


def _video_stream(latent):
    video = _latent_streams(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            "VTS H3 Motion Context expects video latent [B,24,T,H,W], got %s."
            % (tuple(video.shape),))
    return video


def _audio_stream(latent):
    audio = _latent_streams(latent)[1]
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "VTS H3 Motion Context expects audio latent [B,32,2,T], got %s."
            % (tuple(audio.shape),))
    return audio


def _pixel_frames(latent_steps):
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
               for index in range(latent_steps))


def _steps_for_frames(frame_count):
    covered = 0
    for steps in range(1, frame_count + 1):
        covered += FRAME_PER_TOKEN[(steps - 1) % len(FRAME_PER_TOKEN)]
        if covered == frame_count:
            return steps
        if covered > frame_count:
            return None
    return None


def _available_frames(images):
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            raise ValueError(
                "VTS H3 Motion Context expects context_frames [B,H,W,C], got %s."
                % (tuple(images.shape),))
        return int(images.shape[0])
    if isinstance(images, DiskImage):
        count = images.number_of_images
        if count is None and images.shape is not None:
            count = images.shape[0]
        return int(count or 0)
    raise TypeError(
        "VTS H3 Motion Context context_frames must be a Tensor or DiskImage, "
        "got %r." % type(images))


def _context_length(requested, available):
    limit = min(int(requested), int(available))
    valid = [length for length in _VIDEO_RUN_GRID if length <= limit]
    if not valid:
        raise ValueError("VTS H3 Motion Context has no context frames to pin.")
    length = valid[-1]
    if length != int(requested):
        _LOG.warning(
            "VTS H3 Motion Context requested %d frames but %d are available; "
            "pinning the last %d on the H3 video grid.",
            int(requested), int(available), length)
    return length


def _materialize_tail(images, available, length):
    start = available - length
    if isinstance(images, DiskImage):
        images = images.materialize(start=start, count=length)
    else:
        images = images[start:available]
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(
            "VTS H3 Motion Context could not materialize context_frames as "
            "a [B,H,W,C] tensor.")
    return images


def _resize(images, width, height):
    if images.shape[-1] < 3:
        raise ValueError("VTS H3 Motion Context needs RGB context frames.")
    channels_first = images[..., :3].movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        channels_first, width, height, "lanczos", "disabled")
    return resized.movedim(1, -1)


def _latent_video_tail(context_latent, frame_count, target_video):
    video = _video_stream(context_latent)
    if video.shape[1] != target_video.shape[1] or video.shape[3:] != target_video.shape[3:]:
        raise ValueError(
            "VTS H3 Motion Context context_latent is %dx%d but the target is "
            "%dx%d. H3 latents cannot be resized."
            % (video.shape[4] * 16, video.shape[3] * 16,
               target_video.shape[4] * 16, target_video.shape[3] * 16))
    steps = _steps_for_frames(frame_count)
    if steps is None or steps > video.shape[2]:
        raise ValueError(
            "VTS H3 Motion Context cannot slice a %d-frame H3 latent tail."
            % frame_count)
    start = int(video.shape[2]) - steps
    if start % len(FRAME_PER_TOKEN) != 0:
        raise RuntimeError(
            "VTS H3 Motion Context latent tail starts at temporal cycle %d; "
            "refusing a shifted continuation." % (start % len(FRAME_PER_TOKEN)))
    return video[:1, :, start:].clone()


def _encoded_video_tail(vae, context_frames, available, frame_count,
                        width, height):
    frames = _materialize_tail(context_frames, available, frame_count)
    encoded = vae.encode(_resize(frames, width, height))
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
        raise ValueError(
            "VTS H3 Motion Context video VAE returned %s, expected "
            "[B,C,T,H,W]." % (tuple(getattr(encoded, "shape", ())),))
    if encoded.shape[1] != 24 or encoded.shape[3:] != (height // 16, width // 16):
        raise ValueError(
            "VTS H3 Motion Context video VAE returned incompatible shape %s."
            % (tuple(encoded.shape),))
    covered = _pixel_frames(int(encoded.shape[2]))
    if covered != frame_count:
        raise RuntimeError(
            "VTS H3 Motion Context encoded %d frames into %d latent steps "
            "covering %d frames; the H3 temporal grid has changed."
            % (frame_count, int(encoded.shape[2]), covered))
    return encoded[:1]


def _latent_audio_tail(context_latent, frame_count):
    video = _video_stream(context_latent)
    audio = _audio_stream(context_latent)
    total_steps = int(audio.shape[-1])
    source_frames = _pixel_frames(int(video.shape[2]))
    overhang = total_steps - FRAME_RESCALE * source_frames
    if not -0.5 < overhang < 0.5:
        _LOG.warning(
            "VTS H3 Motion Context found %d audio steps for %d video frames; "
            "ignoring the unexpected audio-grid offset.",
            total_steps, source_frames)
        overhang = 0.0
    steps = int(round(frame_count / _FPS * _AUDIO_LATENT_FPS))
    if steps > total_steps:
        _LOG.warning(
            "VTS H3 Motion Context requested %d audio steps but the latent "
            "contains %d; pinning all available audio.", steps, total_steps)
        steps = total_steps
    if steps < 1:
        raise ValueError("VTS H3 Motion Context audio window is empty.")
    return audio[:1, ..., total_steps - steps:].clone(), steps, float(overhang)


def _encoded_audio_tail(audio_vae, audio, frame_count):
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or sample_rate is None:
        raise ValueError(
            "VTS H3 Motion Context expects AUDIO with waveform [B,C,L] and "
            "sample_rate.")
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    sample_rate = int(sample_rate)
    if sample_rate != vae_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)
    wanted = int(round(frame_count / _FPS * vae_rate))
    if waveform.shape[-1] > wanted:
        waveform = waveform[..., -wanted:]
    encoded = audio_vae.encode(waveform[:1].movedim(1, -1))
    if (not isinstance(encoded, torch.Tensor) or encoded.ndim != 4
            or encoded.shape[1] != 32 or encoded.shape[2] != 2):
        raise ValueError(
            "VTS H3 Motion Context audio VAE returned %s, expected "
            "[B,C,2,T]." % (tuple(getattr(encoded, "shape", ())),))
    return encoded[:1], int(encoded.shape[-1]), 0.0


def _merge_guides(conditioning, guides, head_frames):
    output = []
    dropped = []
    for embedding, metadata in conditioning:
        metadata = metadata.copy()
        kept = []
        for guide in metadata.get("minimax_keyframes") or ():
            position = float(guide.get("resolved_frame_index", 0))
            if position < head_frames:
                dropped.append(position)
            else:
                kept.append(dict(guide))
        metadata["minimax_keyframes"] = kept + guides
        output.append([embedding, metadata])
    if dropped:
        _LOG.warning(
            "VTS H3 Motion Context dropped %d existing guide(s) inside the "
            "repeated %d-frame head; later guides were preserved.",
            len(dropped), head_frames)
    return output


class VTS_MiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "context_length": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Frames from the previous clip to repeat at "
                               "the new clip's head."}),
                "audio_context_length": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Previous audio frames to carry. Zero follows "
                               "the video context length."}),
            },
            "optional": {
                "context_frames": ("IMAGE", {
                    "tooltip": "Previous decoded frames as a Tensor or VTS "
                               "DiskImage. Only the required tail is loaded."}),
                "context_latent": ("LATENT", {
                    "tooltip": "Previous H3 sampler latent. Preferred over "
                               "context_frames and reused without VAE loss."}),
                "audio_vae": ("VAE",),
                "context_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "execute"
    CATEGORY = "VTS/wrappers/conditioning/minimax"
    DESCRIPTION = (
        "VTS-native MiniMax H3 motion continuation. Accepts Tensor or "
        "DiskImage frames and uses ComfyUI's native H3 guides, so it does not "
        "compete with SolAttn for PackedLayout ownership.")

    def execute(self, conditioning, vae, latent, context_length,
                audio_context_length=24, context_frames=None,
                context_latent=None, audio_vae=None, context_audio=None):
        _require_native_guides()
        target_video = _video_stream(latent)
        target_frames = _pixel_frames(int(target_video.shape[2]))
        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        if context_latent is not None:
            available = _pixel_frames(int(_video_stream(context_latent).shape[2]))
        elif context_frames is not None:
            available = _available_frames(context_frames)
        else:
            raise ValueError(
                "VTS H3 Motion Context needs context_latent or context_frames.")

        frame_count = _context_length(int(context_length), available)
        if frame_count >= target_frames:
            raise ValueError(
                "VTS H3 Motion Context cannot pin %d frames into a %d-frame "
                "target." % (frame_count, target_frames))

        if context_latent is not None:
            video_guide = _latent_video_tail(
                context_latent, frame_count, target_video)
            video_source = "latent"
        else:
            video_guide = _encoded_video_tail(
                vae, context_frames, available, frame_count, width, height)
            video_source = "frames"

        guides = [{
            "resolved_frame_index": 0,
            "latent": video_guide,
        }]

        audio_steps = 0
        if context_latent is not None or context_audio is not None:
            audio_frames = int(audio_context_length) or frame_count
            if context_latent is not None:
                audio_guide, audio_steps, overhang = _latent_audio_tail(
                    context_latent, audio_frames)
            else:
                if audio_vae is None:
                    raise ValueError(
                        "VTS H3 Motion Context needs audio_vae when "
                        "context_audio is connected.")
                audio_guide, audio_steps, overhang = _encoded_audio_tail(
                    audio_vae, context_audio, audio_frames)

            end_frame = frame_count + overhang / FRAME_RESCALE
            end_frame = round(FRAME_RESCALE * end_frame) / FRAME_RESCALE
            audio_start = end_frame - audio_steps / FRAME_RESCALE
            guides.append({
                "resolved_frame_index": audio_start,
                "audio_latent": audio_guide,
            })

        output = _merge_guides(conditioning, guides, frame_count)
        _LOG.info(
            "VTS H3 Motion Context: %d-frame %s guide, trim %d, audio %s",
            frame_count, video_source, frame_count,
            "%d latent steps" % audio_steps if audio_steps else "off")
        return output, frame_count


NODE_CLASS_MAPPINGS = {
    "VTSWrapper_ComfyUI_H3_Motion_Context_MiniMaxH3MotionContext":
        VTS_MiniMaxH3MotionContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSWrapper_ComfyUI_H3_Motion_Context_MiniMaxH3MotionContext":
        "VTS MiniMax H3 Motion Context (Native)",
}
