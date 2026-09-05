import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_H3MotionContextAudioTrim.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vts_h3_motion_context_audio_trim_test_module",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H3MotionContextAudioTrimTests(unittest.TestCase):
    sample_rate = 2400
    fps = 24.0

    def setUp(self):
        self.node = MODULE.VTSH3MotionContextAudioTrim()

    def audio(self, samples):
        waveform = torch.arange(samples, dtype=torch.float32).reshape(1, 1, -1)
        return {"waveform": waveform, "sample_rate": self.sample_rate}

    def test_schema_has_only_audio_media_input_and_output(self):
        inputs = self.node.INPUT_TYPES()

        self.assertEqual(inputs["required"]["audio"][0], "AUDIO")
        self.assertNotIn("images", inputs["required"])
        self.assertEqual(self.node.RETURN_TYPES, ("AUDIO",))
        self.assertIn(
            "VTS H3 Motion Context Audio Trim",
            MODULE.NODE_CLASS_MAPPINGS,
        )

    def test_trims_head_and_long_h3_tail(self):
        source = self.audio(12420)

        result = self.node.trim(source, trim_frames=5, fps=self.fps)[0]

        self.assertEqual(result["waveform"].shape[-1], 11900)
        self.assertEqual(result["waveform"][0, 0, 0].item(), 500)
        self.assertEqual(source["waveform"].shape[-1], 12420)

    def test_trims_head_and_zero_pads_short_h3_tail(self):
        source = self.audio(25980)

        result = self.node.trim(source, trim_frames=5, fps=self.fps)[0]

        self.assertEqual(result["waveform"].shape[-1], 25500)
        self.assertEqual(result["waveform"][0, 0, 0].item(), 500)
        self.assertTrue(torch.equal(
            result["waveform"][..., -20:],
            torch.zeros((1, 1, 20), dtype=torch.float32),
        ))

    def test_match_tail_false_only_removes_leading_duration(self):
        source = self.audio(12420)

        result = self.node.trim(
            source,
            trim_frames=5,
            fps=self.fps,
            match_tail=False,
        )[0]

        self.assertEqual(result["waveform"].shape[-1], 11920)
        self.assertEqual(result["waveform"][0, 0, 0].item(), 500)

    def test_zero_trim_still_normalises_h3_tail(self):
        source = self.audio(12420)

        result = self.node.trim(source, trim_frames=0, fps=self.fps)[0]

        self.assertEqual(result["waveform"].shape[-1], 12400)
        self.assertEqual(result["waveform"][0, 0, 0].item(), 0)

    def test_rejects_trimming_the_whole_clip(self):
        source = self.audio(1200)

        with self.assertRaisesRegex(ValueError, "asked to trim"):
            self.node.trim(source, trim_frames=12, fps=self.fps)


if __name__ == "__main__":
    unittest.main()
