import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from comfy.cli_args import args
from comfy.nested_tensor import NestedTensor

args.cpu = True


MODULE_PATH = (Path(__file__).parents[1] / "py" /
               "VTS_MiniMaxH3MotionContext.py")
SPEC = importlib.util.spec_from_file_location(
    "vts_h3_motion_context_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeVideoVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, images):
        self.inputs.append(images.clone())
        steps = MODULE._steps_for_frames(int(images.shape[0]))
        return torch.full(
            (1, 24, steps, images.shape[1] // 16, images.shape[2] // 16),
            float(images.mean()))


def av_latent(video_steps=12, audio_steps=65):
    video = torch.arange(
        24 * video_steps * 2 * 2, dtype=torch.float32
    ).reshape(1, 24, video_steps, 2, 2)
    audio = torch.arange(
        32 * 2 * audio_steps, dtype=torch.float32
    ).reshape(1, 32, 2, audio_steps)
    return {"samples": NestedTensor((video, audio))}


def conditioning():
    first = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 2, 2)}
    last = {"resolved_frame_index": 38, "latent": torch.ones(1, 24, 1, 2, 2)}
    return [[torch.zeros(1, 1, 1), {"minimax_keyframes": [first, last]}]]


class MotionContextTests(unittest.TestCase):
    def test_tensor_and_disk_image_load_the_same_tail(self):
        frames = torch.linspace(0, 1, 30 * 32 * 32 * 3).reshape(30, 32, 32, 3)
        target = av_latent()

        tensor_vae = FakeVideoVAE()
        tensor_output, tensor_trim, tensor_masked = (
            MODULE.VTS_MiniMaxH3MotionContext().execute(
                conditioning(), tensor_vae, target, "22",
                context_frames=frames,
            )
        )

        disk_image = MODULE.DiskImage(
            prefix="context",
            start_sequence=0,
            number_of_images=30,
            output_dir="unused",
            format="png",
            image=frames,
        )
        materialized = []

        def materialize(start=0, count=None):
            materialized.append((start, count))
            return frames[start:start + count]

        disk_image.materialize = materialize
        disk_vae = FakeVideoVAE()
        disk_output, disk_trim, disk_masked = (
            MODULE.VTS_MiniMaxH3MotionContext().execute(
                conditioning(), disk_vae, target, "22",
                context_frames=disk_image,
            )
        )

        self.assertEqual(tensor_trim, 22)
        self.assertEqual(disk_trim, 22)
        self.assertEqual(materialized, [(8, 22)])
        self.assertTrue(torch.equal(tensor_vae.inputs[0], disk_vae.inputs[0]))

        tensor_guides = tensor_output[0][1]["minimax_keyframes"]
        disk_guides = disk_output[0][1]["minimax_keyframes"]
        self.assertEqual([guide["resolved_frame_index"] for guide in tensor_guides], [38, 0])
        self.assertEqual([guide["resolved_frame_index"] for guide in disk_guides], [38, 0])
        self.assertTrue(torch.equal(tensor_guides[-1]["latent"],
                                    disk_guides[-1]["latent"]))
        tensor_video, tensor_audio = tensor_masked["samples"].unbind()
        disk_video, disk_audio = disk_masked["samples"].unbind()
        tensor_video_mask, tensor_audio_mask = tensor_masked["noise_mask"].unbind()
        disk_video_mask, disk_audio_mask = disk_masked["noise_mask"].unbind()
        self.assertTrue(torch.equal(tensor_video[:, :, :7],
                                    tensor_guides[-1]["latent"]))
        self.assertTrue(torch.equal(tensor_video, disk_video))
        self.assertTrue(torch.equal(tensor_audio, disk_audio))
        self.assertTrue(torch.equal(tensor_video_mask, disk_video_mask))
        self.assertTrue(torch.equal(tensor_audio_mask, disk_audio_mask))
        self.assertTrue(torch.equal(tensor_video_mask[:, :, :7],
                                    torch.zeros_like(tensor_video_mask[:, :, :7])))
        self.assertTrue(torch.equal(tensor_video_mask[:, :, 7:],
                                    torch.ones_like(tensor_video_mask[:, :, 7:])))
        self.assertTrue(torch.equal(tensor_audio_mask,
                                    torch.ones_like(tensor_audio_mask)))

    def test_latent_context_does_not_touch_disk_image_or_layout_owner(self):
        from comfy.ldm.minimax import model as minimax_model

        target = av_latent()
        previous = av_latent()
        frames = torch.zeros(39, 32, 32, 3)
        disk_image = MODULE.DiskImage(
            prefix="unused",
            start_sequence=0,
            number_of_images=39,
            output_dir="unused",
            format="png",
            image=frames,
        )

        def fail_materialize(start=0, count=None):
            raise AssertionError("context_frames was loaded despite context_latent")

        disk_image.materialize = fail_materialize
        original_init = minimax_model.PackedLayout.__init__

        def solattn_observer(self, *observer_args, **observer_kwargs):
            return original_init(self, *observer_args, **observer_kwargs)

        solattn_observer.__module__ = "ComfyUI-SolAttn_triton._morton_h3"
        minimax_model.PackedLayout.__init__ = solattn_observer
        try:
            output, trim, masked_latent = (
                MODULE.VTS_MiniMaxH3MotionContext().execute(
                    conditioning(), FakeVideoVAE(), target, "22",
                    audio_context_length=24,
                    context_frames=disk_image,
                    context_latent=previous,
                )
            )
            self.assertIs(minimax_model.PackedLayout.__init__, solattn_observer)
            guides = output[0][1]["minimax_keyframes"]
            self.assertEqual(trim, 22)
            self.assertEqual([guide["resolved_frame_index"] for guide in guides[:2]], [38, 0])
            self.assertEqual(guides[1]["latent"].shape, (1, 24, 7, 2, 2))
            self.assertAlmostEqual(guides[2]["resolved_frame_index"], -1.8)
            self.assertEqual(guides[2]["audio_latent"].shape, (1, 32, 2, 40))

            masked_video, masked_audio = masked_latent["samples"].unbind()
            video_mask, audio_mask = masked_latent["noise_mask"].unbind()
            previous_video = previous["samples"].unbind()[0]
            target_audio = target["samples"].unbind()[1]
            self.assertTrue(torch.equal(masked_video[:, :, :7],
                                        previous_video[:, :, -7:]))
            self.assertTrue(torch.equal(masked_audio, target_audio))
            self.assertTrue(torch.equal(video_mask[:, :, :7],
                                        torch.zeros_like(video_mask[:, :, :7])))
            self.assertTrue(torch.equal(video_mask[:, :, 7:],
                                        torch.ones_like(video_mask[:, :, 7:])))
            self.assertTrue(torch.equal(audio_mask, torch.ones_like(audio_mask)))

            layout = minimax_model.PackedLayout(
                4, 12, 2, 2, 65, keyframes=guides)
            self.assertTrue(any(kind == "cond" for _, _, kind in layout.segments))
            self.assertTrue(any(kind == "cond_audio" for _, _, kind in layout.segments))
            self.assertIs(minimax_model.PackedLayout.__init__, solattn_observer)
        finally:
            minimax_model.PackedLayout.__init__ = original_init

    def test_existing_wrapper_id_and_non_image_outputs_are_preserved(self):
        node_id = "VTSWrapper_ComfyUI_H3_Motion_Context_MiniMaxH3MotionContext"
        self.assertIs(MODULE.NODE_CLASS_MAPPINGS[node_id],
                      MODULE.VTS_MiniMaxH3MotionContext)
        self.assertEqual(MODULE.VTS_MiniMaxH3MotionContext.RETURN_TYPES,
                         ("CONDITIONING", "INT", "LATENT"))
        self.assertEqual(MODULE.VTS_MiniMaxH3MotionContext.RETURN_NAMES[:2],
                         ("conditioning", "trim_frames"))
        required = MODULE.VTS_MiniMaxH3MotionContext.INPUT_TYPES()["required"]
        self.assertNotIn("vts_return_type", required)


if __name__ == "__main__":
    unittest.main()
