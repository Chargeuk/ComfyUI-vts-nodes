import logging

import torch


_LOG = logging.getLogger(__name__)


class VTSH3MotionContextAudioTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "trim_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096},
                ),
            },
            "optional": {
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 240.0,
                        "step": 0.001,
                        "tooltip": (
                            "Frame rate used to convert trim_frames into an audio "
                            "duration. It must match the video frame rate."
                        ),
                    },
                ),
                "match_tail": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Trim or zero-pad the H3 audio tail to the nearest whole "
                            "video-frame duration after removing the pinned head."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim"
    CATEGORY = "VTS/audio"
    DESCRIPTION = (
        "Remove the leading pinned-frame duration from H3 audio without requiring "
        "or processing an image batch."
    )

    def trim(self, audio, trim_frames, fps=24.0, match_tail=True):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        total_samples = int(waveform.shape[-1])
        total_frames = int(round(total_samples / sample_rate * float(fps)))
        frames_to_trim = max(0, int(trim_frames))

        if frames_to_trim >= total_frames:
            raise ValueError(
                "vts_h3_audio_trim: asked to trim %d frames from approximately "
                "%d frames of audio" % (frames_to_trim, total_frames)
            )

        cut = int(round(frames_to_trim / float(fps) * sample_rate))
        if cut >= total_samples:
            raise ValueError(
                "vts_h3_audio_trim: trimming %.3fs from %.3fs of audio would "
                "leave nothing. Check that fps matches the clip."
                % (frames_to_trim / float(fps), total_samples / sample_rate)
            )
        waveform = waveform[..., cut:]

        frames_left = total_frames - frames_to_trim
        if match_tail:
            wanted_samples = int(round(frames_left / float(fps) * sample_rate))
            current_samples = int(waveform.shape[-1])
            if current_samples > wanted_samples:
                waveform = waveform[..., :wanted_samples]
                _LOG.info(
                    "vts_h3_audio_trim: tail trimmed %d samples so audio matches "
                    "%d frames",
                    current_samples - wanted_samples,
                    frames_left,
                )
            elif current_samples < wanted_samples:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, wanted_samples - current_samples),
                )
                _LOG.info(
                    "vts_h3_audio_trim: tail padded %d zero samples so audio "
                    "matches %d frames",
                    wanted_samples - current_samples,
                    frames_left,
                )

        return ({"waveform": waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "VTS H3 Motion Context Audio Trim": VTSH3MotionContextAudioTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS H3 Motion Context Audio Trim": "VTS H3 Motion Context Audio Trim",
}
