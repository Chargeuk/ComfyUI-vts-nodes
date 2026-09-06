import importlib.util
import math
import os
import sys

import torch


# VTS loads sibling files under their absolute paths, without a package name.
_motion_path = os.path.join(os.path.dirname(__file__), "VTS_MiniMaxH3MotionContext.py")
_motion_name = os.path.splitext(_motion_path)[0]
_motion = sys.modules.get(_motion_name)
if _motion is None:
    _spec = importlib.util.spec_from_file_location(_motion_name, _motion_path)
    _motion = importlib.util.module_from_spec(_spec)
    sys.modules[_motion_name] = _motion
    _spec.loader.exec_module(_motion)


class VTS_H3PrepareLoopContext:
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous H3 sampler output. Only its required "
                               "video and audio tails are copied."}),
                "context_length": (["22", "5", "39", "56"], {
                    "default": "22",
                    "tooltip": "Previous video frames to repeat at the next "
                               "clip's head."}),
                "audio_context_length": ("INT", {
                    "default": 24, "min": 0, "max": 240,
                    "tooltip": "Previous audio frames to carry. Zero follows "
                               "the video context length."}),
            },
        }

    RETURN_TYPES = ("VTS_H3_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "execute"
    CATEGORY = "VTS/wrappers/conditioning/minimax"
    DESCRIPTION = (
        "Copy compact H3 video/audio tails for a loop's next iteration. "
        "Preserves native temporal alignment without retaining the full "
        "previous latent or its noise masks.")

    def execute(self, context_latent, context_length="22", audio_context_length=24):
        if str(context_length) not in ("22", "5", "39", "56"):
            raise ValueError("VTS H3 Prepare Loop Context context_length must be 5, 22, 39 or 56.")
        if not isinstance(audio_context_length, int) or not 0 <= audio_context_length <= 240:
            raise ValueError("VTS H3 Prepare Loop Context audio_context_length must be an integer from 0 to 240.")
        video = _motion._video_stream(context_latent)
        available = _motion._pixel_frames(int(video.shape[2]))
        frame_count = _motion._context_length(int(context_length), available)
        video_guide = _motion._latent_video_tail(
            context_latent, frame_count, video).detach()
        audio_frames = int(audio_context_length) or frame_count
        audio_guide, audio_steps, overhang = _motion._latent_audio_tail(
            context_latent, audio_frames)
        if video_guide.numel() == 0 or audio_guide.numel() == 0:
            raise ValueError("VTS H3 Prepare Loop Context received an empty latent.")
        end_frame = frame_count + overhang / _motion.FRAME_RESCALE
        end_frame = round(_motion.FRAME_RESCALE * end_frame) / _motion.FRAME_RESCALE
        return ({
            "video": video_guide,
            "audio": audio_guide.detach(),
            "audio_start": end_frame - audio_steps / _motion.FRAME_RESCALE,
        },)


class VTS_H3ApplyLoopContext:
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context": ("VTS_H3_CONTEXT", {
                    "tooltip": "Compact tails from VTS H3 Prepare Loop Context."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "masked_latent")
    FUNCTION = "execute"
    CATEGORY = "VTS/wrappers/conditioning/minimax"
    DESCRIPTION = (
        "Apply compact H3 loop context as native video/audio guides. Returns "
        "the repeated-frame count and the same optional masked video-prefix "
        "latent as VTS MiniMax H3 Motion Context.")

    def execute(self, conditioning, latent, context):
        _motion._require_native_guides()
        if (not isinstance(context, dict)
                or not isinstance(context.get("video"), torch.Tensor)
                or not isinstance(context.get("audio"), torch.Tensor)
                or not isinstance(context.get("audio_start"), (int, float))
                or not math.isfinite(context["audio_start"])):
            raise ValueError(
                "VTS H3 Apply Loop Context needs context from "
                "VTS H3 Prepare Loop Context.")
        streams = {"samples": (context["video"], context["audio"])}
        video_guide = _motion._video_stream(streams)
        audio_guide = _motion._audio_stream(streams)
        if video_guide.shape[0] != 1 or audio_guide.shape[0] != 1:
            raise ValueError("VTS H3 Apply Loop Context expects one compact video/audio batch.")
        frame_count = _motion._pixel_frames(int(video_guide.shape[2]))
        if video_guide.numel() == 0 or audio_guide.numel() == 0:
            raise ValueError("VTS H3 Apply Loop Context received an empty context.")
        target_frames = _motion._pixel_frames(int(_motion._video_stream(latent).shape[2]))
        if frame_count >= target_frames:
            raise ValueError(
                "VTS H3 Apply Loop Context cannot pin %d frames into a "
                "%d-frame target." % (frame_count, target_frames))
        guides = [
            {"resolved_frame_index": 0, "latent": video_guide},
            {"resolved_frame_index": context["audio_start"],
             "audio_latent": audio_guide},
        ]
        output = _motion._merge_guides(conditioning, guides, frame_count)
        masked_latent = _motion._masked_latent_prefix(latent, video_guide)
        return output, frame_count, masked_latent


NODE_CLASS_MAPPINGS = {
    "VTS_H3PrepareLoopContext": VTS_H3PrepareLoopContext,
    "VTS_H3ApplyLoopContext": VTS_H3ApplyLoopContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_H3PrepareLoopContext": "VTS H3 Prepare Loop Context",
    "VTS_H3ApplyLoopContext": "VTS H3 Apply Loop Context",
}
