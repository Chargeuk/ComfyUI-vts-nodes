import gc
import importlib.util
import os
import sys
import unittest
import weakref
from pathlib import Path

import torch

from comfy.cli_args import args

args.cpu = True

from comfy.nested_tensor import NestedTensor


NODE_PATH = Path(__file__).parents[1] / "py" / "VTS_H3LoopContext.py"
HELPER_PATH = NODE_PATH.with_name("VTS_MiniMaxH3MotionContext.py")
HELPER_NAME = str(HELPER_PATH.with_suffix(""))
if not HELPER_PATH.exists():
    HELPER_PATH = (Path(os.environ["COMFYUI_ROOT"]) / "custom_nodes" /
                   "ComfyUI-vts-nodes" / "py" / "VTS_MiniMaxH3MotionContext.py")
helper_spec = importlib.util.spec_from_file_location(HELPER_NAME, HELPER_PATH)
MOTION = importlib.util.module_from_spec(helper_spec)
sys.modules[HELPER_NAME] = MOTION
helper_spec.loader.exec_module(MOTION)

spec = importlib.util.spec_from_file_location(str(NODE_PATH.with_suffix("")), NODE_PATH)
MODULE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def av_latent(video_steps=22, audio_steps=None, dtype=torch.float32, batch=1,
              height=2, width=2, requires_grad=False):
    if audio_steps is None:
        audio_steps = round(MOTION.FRAME_RESCALE * MOTION._pixel_frames(video_steps))
    video = torch.arange(batch * 24 * video_steps * height * width, dtype=torch.float32)
    video = (video.reshape(batch, 24, video_steps, height, width) / 100).to(dtype)
    audio = torch.arange(batch * 32 * 2 * audio_steps, dtype=torch.float32)
    audio = (audio.reshape(batch, 32, 2, audio_steps) / 100).to(dtype)
    video.requires_grad_(requires_grad)
    audio.requires_grad_(requires_grad)
    return {"samples": NestedTensor((video, audio))}


def conditioning():
    return [[torch.arange(4).reshape(1, 2, 2), {
        "minimax_keyframes": [
            {"resolved_frame_index": -3, "audio_latent": torch.ones(1, 32, 2, 1)},
            {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 2, 2)},
            {"resolved_frame_index": 65, "latent": torch.ones(1, 24, 1, 2, 2)},
        ],
        "unrelated": "preserved",
    }]]


class H3LoopContextTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, MOTION._LOG, "disabled", MOTION._LOG.disabled)
        MOTION._LOG.disabled = True
        self.prepare = MODULE.VTS_H3PrepareLoopContext()
        self.apply = MODULE.VTS_H3ApplyLoopContext()

    def assert_result_equal(self, expected, actual):
        self.assertEqual(expected[1], actual[1])
        for old, new in zip(expected[0], actual[0]):
            self.assertIs(old[0], new[0])
            self.assertEqual(old[1].keys(), new[1].keys())
            old_guides = old[1]["minimax_keyframes"]
            new_guides = new[1]["minimax_keyframes"]
            self.assertEqual(len(old_guides), len(new_guides))
            for old_guide, new_guide in zip(old_guides, new_guides):
                self.assertEqual(old_guide.keys(), new_guide.keys())
                for key, value in old_guide.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, new_guide[key], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, new_guide[key])
        for key in ("samples", "noise_mask"):
            for old, new in zip(expected[2][key].unbind(), actual[2][key].unbind()):
                torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_exact_parity_with_real_motion_helpers_and_native_layout(self):
        from comfy.ldm.minimax.model import PackedLayout

        for video_steps in (12, 17, 22, 27):
            for frame_count in ("5", "22", "39", "56"):
                for audio_frames in (0, 22, 24, 240):
                    with self.subTest(source=video_steps, frames=frame_count, audio=audio_frames):
                        source = av_latent(video_steps)
                        target = av_latent(27)
                        cond = conditioning()
                        expected = MOTION.VTS_MiniMaxH3MotionContext().execute(
                            cond, None, target, frame_count,
                            audio_context_length=audio_frames, context_latent=source)
                        context, = self.prepare.execute(source, frame_count, audio_frames)
                        actual = self.apply.execute(cond, target, context)
                        self.assert_result_equal(expected, actual)
                        old_layout = PackedLayout(4, 27, 2, 2, 150,
                            keyframes=expected[0][0][1]["minimax_keyframes"])
                        new_layout = PackedLayout(4, 27, 2, 2, 150,
                            keyframes=actual[0][0][1]["minimax_keyframes"])
                        self.assertEqual(old_layout.segments, new_layout.segments)
                        torch.testing.assert_close(old_layout.position_ids,
                                                   new_layout.position_ids, rtol=0, atol=0)

    def test_audio_overhang_survives_compaction(self):
        # Positive/negative third-step rounding must use the full source video,
        # not recalculate alignment from the compact video/audio lengths.
        positive, = self.prepare.execute(av_latent(22, 122), "5", 24)
        negative, = self.prepare.execute(av_latent(17, 93), "5", 24)
        self.assertAlmostEqual(positive["audio_start"], -18.6)
        self.assertAlmostEqual(negative["audio_start"], -19.2)
        self.assertEqual(positive["audio"].shape[-1], 40)
        self.assertEqual(negative["audio"].shape[-1], 40)

    def test_unexpected_audio_offset_and_short_audio_match_existing_node(self):
        for steps in (2, 113):
            with self.subTest(audio_steps=steps):
                source = av_latent(22, steps)
                target = av_latent(27)
                cond = conditioning()
                expected = MOTION.VTS_MiniMaxH3MotionContext().execute(
                    cond, None, target, "22", context_latent=source)
                context, = self.prepare.execute(source)
                self.assert_result_equal(expected, self.apply.execute(cond, target, context))

    def test_context_owns_only_compact_storage_and_releases_source(self):
        source = av_latent(27, batch=2, requires_grad=True)
        video, audio = source["samples"].unbind()
        source["noise_mask"] = NestedTensor((torch.ones_like(video), torch.ones_like(audio)))
        source["extra"] = torch.ones(1000)
        pointers = {video.untyped_storage().data_ptr(), audio.untyped_storage().data_ptr()}
        references = [weakref.ref(video), weakref.ref(audio), weakref.ref(source["extra"])]
        context, = self.prepare.execute(source)
        self.assertEqual(set(context), {"video", "audio", "audio_start"})
        self.assertEqual(context["video"].shape, (1, 24, 7, 2, 2))
        self.assertEqual(context["audio"].shape, (1, 32, 2, 40))
        for key in ("video", "audio"):
            compact = context[key]
            self.assertNotIn(compact.untyped_storage().data_ptr(), pointers)
            self.assertEqual(compact.untyped_storage().nbytes(), compact.numel() * compact.element_size())
            self.assertFalse(compact.requires_grad)
            self.assertIsNone(compact.grad_fn)
        del source, video, audio
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_dtype_existing_masks_metadata_and_inputs_are_preserved(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = av_latent(dtype=dtype)
                target = av_latent(27, dtype=torch.float32, batch=2)
                target["noise_mask"] = NestedTensor((
                    torch.full((1, 1, 27, 1, 1), 0.25),
                    torch.full((1, 1, 1, 150), 0.75)))
                target["batch_index"] = [8, 9]
                cond = conditioning()
                target_copies = [x.clone() for x in target["samples"].unbind()]
                original_guides = list(cond[0][1]["minimax_keyframes"])
                context, = self.prepare.execute(source)
                expected = MOTION.VTS_MiniMaxH3MotionContext().execute(
                    cond, None, target, "22", context_latent=source)
                result = self.apply.execute(cond, target, context)
                self.assert_result_equal(expected, result)
                self.assertEqual(result[2]["batch_index"], [8, 9])
                self.assertEqual(cond[0][1]["minimax_keyframes"], original_guides)
                self.assertEqual(result[0][0][1]["unrelated"], "preserved")
                for copied, original in zip(target_copies, target["samples"].unbind()):
                    self.assertTrue(torch.equal(copied, original))
                for key in ("video", "audio"):
                    self.assertEqual(context[key].dtype, dtype)
                    self.assertEqual(context[key].device.type, "cpu")
                guides = result[0][0][1]["minimax_keyframes"]
                for key, guide in (("latent", guides[-2]), ("audio_latent", guides[-1])):
                    self.assertEqual(guide[key].dtype, dtype)
                    self.assertEqual(guide[key].device.type, "cpu")
                self.assertEqual(result[2]["samples"].tensors[0].dtype, torch.float32)

    def test_unbatched_streams_and_one_frame_source(self):
        source = av_latent(1, 2)
        source["samples"] = tuple(t[0] for t in source["samples"].unbind())
        context, = self.prepare.execute(source)
        result = self.apply.execute([], av_latent(), context)
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], 1)
        self.assertEqual(context["video"].shape[0], 1)

    def test_rejects_shifted_temporal_cycle(self):
        with self.assertRaisesRegex(RuntimeError, "temporal cycle"):
            self.prepare.execute(av_latent(23))

    def test_corrected_context_is_compact_and_preserves_audio_timing(self):
        source = av_latent(22, 122)
        base, = self.prepare.execute(source)
        backing = torch.ones(1, 24, 50, 2, 2, requires_grad=True)
        replacement = {"video": backing[:, :, :7], "frame_count": 22, "source_frames": 73}
        corrected, = self.prepare.execute(source, corrected_video_context=replacement)
        torch.testing.assert_close(corrected["video"], replacement["video"])
        torch.testing.assert_close(corrected["audio"], base["audio"], rtol=0, atol=0)
        self.assertEqual(corrected["audio_start"], base["audio_start"])
        self.assertFalse(corrected["video"].requires_grad)
        self.assertEqual(corrected["video"].untyped_storage().nbytes(), 1 * 24 * 7 * 2 * 2 * 4)
        self.assertNotEqual(corrected["video"].untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())

    def test_rejects_mismatched_corrected_context(self):
        source = av_latent()
        good = {"video": torch.ones(1, 24, 7, 2, 2), "frame_count": 22, "source_frames": 73}
        for replacement in ({}, dict(good, source_frames=56), dict(good, frame_count=5),
                            dict(good, video=torch.ones(1, 24, 7, 4, 4)),
                            dict(good, video=torch.ones(1, 24, 8, 2, 2)),
                            dict(good, video=None)):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "Corrected video context"):
                self.prepare.execute(source, corrected_video_context=replacement)

    def test_rejects_invalid_api_lengths(self):
        source = av_latent()
        for length in (None, "bad", "1", "7", "0", "999", -22, 22.5):
            with self.subTest(video_length=length), self.assertRaisesRegex(ValueError, "context_length"):
                self.prepare.execute(source, length)
        for length in (-1, 241, 22.5, None, "22"):
            with self.subTest(audio_length=length), self.assertRaisesRegex(ValueError, "audio_context_length"):
                self.prepare.execute(source, audio_context_length=length)

    def test_rejects_empty_or_unsupported_source(self):
        for source in (None, {}, {"samples": torch.zeros(1)},
                       {"samples": (torch.zeros(1),)}, av_latent(0, 0),
                       av_latent(22, 0), av_latent(22, batch=0)):
            with self.subTest(source_type=type(source)), self.assertRaises(ValueError):
                self.prepare.execute(source)

    def test_rejects_bad_compact_context_or_target(self):
        good, = self.prepare.execute(av_latent())
        for context in (None, {}, av_latent(), dict(good, audio_start=float("nan")),
                        dict(good, video=torch.empty(1, 24, 0, 2, 2)),
                        dict(good, audio=torch.empty(1, 32, 2, 0)),
                        dict(good, video=torch.empty(1, 16, 7, 2, 2))):
            with self.subTest(context_type=type(context)), self.assertRaises(ValueError):
                self.apply.execute([], av_latent(), context)
        with self.assertRaisesRegex(ValueError, "cannot pin"):
            self.apply.execute([], av_latent(7), good)
        with self.assertRaisesRegex(ValueError, "does not match target"):
            self.apply.execute([], av_latent(height=4), good)
        with self.assertRaisesRegex(ValueError, "does not match target"):
            self.apply.execute([], av_latent(width=4), good)
        for field in ("video", "audio"):
            invalid = dict(good, **{field: torch.cat([good[field], good[field]], dim=0)})
            with self.subTest(batch_field=field), self.assertRaisesRegex(ValueError, "one compact"):
                self.apply.execute([], av_latent(batch=3), invalid)
        target = av_latent()
        target["noise_mask"] = (torch.zeros(9), torch.zeros(11))
        with self.assertRaisesRegex(ValueError, "noise-mask shapes"):
            self.apply.execute([], target, good)

    def test_additive_schema_and_absolute_path_helper_loading(self):
        self.assertIs(MODULE._motion, MOTION)
        self.assertEqual(set(MODULE.NODE_CLASS_MAPPINGS),
                         {"VTS_H3PrepareLoopContext", "VTS_H3ApplyLoopContext"})
        self.assertEqual(self.prepare.RETURN_TYPES, ("VTS_H3_CONTEXT",))
        self.assertEqual(self.apply.RETURN_TYPES, ("CONDITIONING", "INT", "LATENT"))
        self.assertNotIn("vae", self.prepare.INPUT_TYPES()["required"])
        self.assertEqual(self.apply.INPUT_TYPES()["required"]["context"][0], "VTS_H3_CONTEXT")
        self.assertIn("VTSWrapper_ComfyUI_H3_Motion_Context_MiniMaxH3MotionContext",
                      MOTION.NODE_CLASS_MAPPINGS)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
