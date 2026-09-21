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




_vts_utils = os.path.join(os.path.dirname(__file__), 'vtsUtils')
if _vts_utils not in sys.path:
    sys.path.append(_vts_utils)
from vts_latent_nodes import disk_latent_node
from vts_h3_context import context_controls, context_tensor_info, materialize_context, store_context

@disk_latent_node(inputs=('context_latent',), outputs=(), prefix='VTS H3 Prepare Loop Context')
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
            "optional": {
                "corrected_video_context": ("VTS_H3_VIDEO_CONTEXT", {
                    "tooltip": "Optional in-memory or disk-backed corrected tail from VAE Decode VTS (Tiled + Colour Match). Use the same context_length and source clip. Audio still comes from context_latent."}),
                **context_controls('VTS H3 Prepare Loop Context', has_input=True),
            },
        }

    RETURN_TYPES = ("VTS_H3_CONTEXT",)
    RETURN_NAMES = ("context",)
    OUTPUT_TOOLTIPS = ("Compact H3 video/audio latent tails, either tensors or metadata-only disk references. Timing stays in memory. Apply Loop Context loads disk tensors when needed.",)
    FUNCTION = "execute"
    CATEGORY = "VTS/wrappers/conditioning/minimax"
    DESCRIPTION = (
        "Copy compact H3 video/audio tails for a loop's next iteration. "
        "Preserves native temporal alignment without retaining the full "
        "previous latent or its noise masks.")

    def execute(self, context_latent, context_length="22", audio_context_length=24,
                corrected_video_context=None, context_return_type="Tensor",
                context_output_dir="./tmp/disklatents", context_prefix="VTS_H3_Prepare_Loop_Context",
                context_start_sequence=0, context_compression_level=3,
                context_device_policy="Original", context_device="cpu"):
        if str(context_length) not in ("22", "5", "39", "56"):
            raise ValueError("VTS H3 Prepare Loop Context context_length must be 5, 22, 39 or 56.")
        if not isinstance(audio_context_length, int) or not 0 <= audio_context_length <= 240:
            raise ValueError("VTS H3 Prepare Loop Context audio_context_length must be an integer from 0 to 240.")
        video = _motion._video_stream(context_latent)
        available = _motion._pixel_frames(int(video.shape[2]))
        frame_count = _motion._context_length(int(context_length), available)
        if corrected_video_context is None:
            video_guide = _motion._latent_video_tail(
                context_latent, frame_count, video).detach()
        else:
            replacement = corrected_video_context
            if (not isinstance(replacement, dict)
                    or replacement.get("source_frames") != available
                    or replacement.get("frame_count") != frame_count):
                raise ValueError("Corrected video context must use the same source clip length and context_length as Prepare Loop Context.")
            encoded = context_tensor_info(replacement.get("video"))
            expected = (1, 24, _motion._steps_for_frames(frame_count), *video.shape[3:])
            if encoded is None or tuple(encoded.shape) != expected:
                raise ValueError("Corrected video context must match the source H3 resolution and context length: %s." % (expected,))
            encoded = materialize_context(replacement, context_device_policy, context_device)['video']
            video_guide = encoded.detach().clone()
        audio_frames = int(audio_context_length) or frame_count
        audio_guide, audio_steps, overhang = _motion._latent_audio_tail(
            context_latent, audio_frames)
        if video_guide.numel() == 0 or audio_guide.numel() == 0:
            raise ValueError("VTS H3 Prepare Loop Context received an empty latent.")
        end_frame = frame_count + overhang / _motion.FRAME_RESCALE
        end_frame = round(_motion.FRAME_RESCALE * end_frame) / _motion.FRAME_RESCALE
        context = {
            "video": video_guide,
            "audio": audio_guide.detach(),
            "audio_start": end_frame - audio_steps / _motion.FRAME_RESCALE,
        }
        return (store_context(context, context_return_type, context_output_dir, context_prefix,
                              context_start_sequence, context_compression_level),)


@disk_latent_node(inputs=('latent',), outputs=(2,), prefix='VTS H3 Apply Loop Context')
class VTS_H3ApplyLoopContext:
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "context": ("VTS_H3_CONTEXT", {
                    "tooltip": "Compact video/audio latent tails from VTS H3 Prepare Loop Context. Accepts in-memory, disk-backed, or mixed tensors; disk data loads only when applying the guides."}),
            },
            "optional": context_controls(has_input=True, has_output=False),
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "masked_latent")
    FUNCTION = "execute"
    CATEGORY = "VTS/wrappers/conditioning/minimax"
    DESCRIPTION = (
        "Apply compact H3 loop context as native video/audio guides. Returns "
        "the repeated-frame count and the same optional masked video-prefix "
        "latent as VTS MiniMax H3 Motion Context.")

    def execute(self, conditioning, latent, context, context_device_policy="Original", context_device="cpu"):
        _motion._require_native_guides()
        if (not isinstance(context, dict)
                or context_tensor_info(context.get("video")) is None
                or context_tensor_info(context.get("audio")) is None
                or not isinstance(context.get("audio_start"), (int, float))
                or not math.isfinite(context["audio_start"])):
            raise ValueError(
                "VTS H3 Apply Loop Context needs context from "
                "VTS H3 Prepare Loop Context.")
        # Reject mismatched/empty metadata before reading context tensor payloads.
        video_info, audio_info = (context_tensor_info(context[key]) for key in ('video', 'audio'))
        video_shape = tuple(video_info.shape)
        audio_shape = tuple(audio_info.shape)
        if len(video_shape) == 4:
            video_shape = (1, *video_shape)
        if len(audio_shape) == 3:
            audio_shape = (1, *audio_shape)
        if (len(video_shape) != 5 or video_shape[1] != 24
                or len(audio_shape) != 4 or audio_shape[1:3] != (32, 2)):
            raise ValueError('VTS H3 Apply Loop Context expects compact H3 video [1,24,T,H,W] and audio [1,32,2,T].')
        if video_shape[0] != 1 or audio_shape[0] != 1:
            raise ValueError('VTS H3 Apply Loop Context expects one compact video/audio batch.')
        if video_info.numel() == 0 or audio_info.numel() == 0:
            raise ValueError('VTS H3 Apply Loop Context received an empty context.')
        frame_count = _motion._pixel_frames(video_shape[2])
        target_video = _motion._video_stream(latent)
        if video_shape[3:] != tuple(target_video.shape[3:]):
            raise ValueError('VTS H3 Motion Context prefix shape %s does not match target %s.' %
                             (video_shape, tuple(target_video.shape)))
        target_frames = _motion._pixel_frames(int(target_video.shape[2]))
        if frame_count >= target_frames:
            raise ValueError('VTS H3 Apply Loop Context cannot pin %d frames into a %d-frame target.' % (frame_count, target_frames))
        context = materialize_context(context, context_device_policy, context_device)
        streams = {"samples": (context["video"], context["audio"])}
        video_guide = _motion._video_stream(streams)
        audio_guide = _motion._audio_stream(streams)
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
