import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from comfy.cli_args import args
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io


args.cpu = True

MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_MiniMaxH3MaskedVideoConditioning.py"
)
SPEC = importlib.util.spec_from_file_location("vts_h3_masked_video_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, images):
        self.inputs.append(images.clone())
        frame_count = int(images.shape[0])
        steps = next(
            count
            for count in range(1, frame_count + 1)
            if MODULE._pixel_frames(count) == frame_count
        )
        return torch.full(
            (1, 24, steps, images.shape[1] // 16, images.shape[2] // 16),
            float(images.mean()),
        )


class FakeClip:
    def tokenize(self, prompt, minimax_ref_items=None):
        return {"prompt": prompt, "refs": minimax_ref_items}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 1), {"tokens": tokens}]]


def empty_av(video_steps=2, audio_steps=8):
    video = torch.zeros(1, 24, video_steps, 2, 2)
    audio = torch.zeros(1, 32, 2, audio_steps)
    return {"samples": NestedTensor((video, audio))}


class FakeReferenceNode:
    calls = []

    @classmethod
    def execute(cls, **kwargs):
        cls.calls.append(kwargs)
        return io.NodeOutput("conditioning", empty_av())


class MiniMaxH3MaskedVideoTests(unittest.TestCase):
    def setUp(self):
        self.original_reference_node = MODULE.MiniMaxH3ReferenceToVideo
        MODULE.MiniMaxH3ReferenceToVideo = FakeReferenceNode
        FakeReferenceNode.calls.clear()

    def tearDown(self):
        MODULE.MiniMaxH3ReferenceToVideo = self.original_reference_node

    def test_known_pixels_are_preserved_and_holes_are_generated(self):
        valid = torch.ones(5, 32, 32)
        valid[1:, :16, :16] = 0
        target = torch.zeros(1, 24, 2, 2, 2)

        mask = MODULE._latent_video_mask(valid, target)

        self.assertEqual(tuple(mask.shape), tuple(target.shape))
        self.assertTrue(bool((mask[:, :, 0] == 0).all()))
        self.assertTrue(bool((mask[:, :, 1, 0, 0] == 1).all()))
        self.assertTrue(bool((mask[:, :, 1, 1, 1] == 0).all()))

    def test_missing_frames_are_filled_and_generated(self):
        video = torch.ones(2, 16, 16, 3)
        valid = torch.ones(2, 16, 16, 3)

        image, mask = MODULE._prepare_control(
            video, valid, 5, 32, 32, "gray", False
        )

        self.assertEqual(tuple(image.shape), (5, 32, 32, 3))
        self.assertTrue(bool((mask[:2] == 1).all()))
        self.assertTrue(bool((mask[2:] == 0).all()))
        self.assertTrue(bool((image[2:] == 0.5).all()))

    def test_invert_mask_matches_wan_user_facing_option(self):
        video = torch.ones(5, 32, 32, 3)
        valid = torch.ones(5, 32, 32, 3)

        image, mask = MODULE._prepare_control(
            video, valid, 5, 32, 32, "black", True
        )

        self.assertTrue(bool((mask == 0).all()))
        self.assertTrue(bool((image == 0).all()))

    def test_execute_forwards_references_and_builds_joint_masks(self):
        vae = FakeVAE()
        video = torch.ones(5, 32, 32, 3)
        valid = torch.ones(5, 32, 32, 3)
        refs = {"ref_image_0": torch.zeros(1, 32, 32, 3)}

        result = MODULE.VTSMiniMaxH3MaskedVideoConditioning.execute(
            clip="clip",
            vae=vae,
            audio_vae="audio-vae",
            prompt="<Picture 1> walks forward",
            control_video=video,
            control_mask=valid,
            width=32,
            height=32,
            length=5,
            ref_images=refs,
        )

        self.assertEqual(result[0], "conditioning")
        latent = result[1]
        video_latent, audio_latent = latent["samples"].unbind()
        video_mask, audio_mask = latent["noise_mask"].unbind()
        self.assertEqual(tuple(video_latent.shape), (1, 24, 2, 2, 2))
        self.assertEqual(tuple(audio_latent.shape), (1, 32, 2, 8))
        self.assertTrue(bool((video_mask == 0).all()))
        self.assertTrue(bool((audio_mask == 1).all()))
        self.assertIs(FakeReferenceNode.calls[0]["ref_images"], refs)

    def test_execute_delegates_to_the_real_native_h3_reference_node(self):
        MODULE.MiniMaxH3ReferenceToVideo = self.original_reference_node
        result = MODULE.VTSMiniMaxH3MaskedVideoConditioning.execute(
            clip=FakeClip(),
            vae=FakeVAE(),
            audio_vae="unused-without-audio-references",
            prompt="A camera moves through the repaired scene",
            control_video=torch.ones(5, 32, 32, 3),
            control_mask=torch.ones(5, 32, 32, 3),
            width=32,
            height=32,
            length=5,
        )

        self.assertEqual(result[0][0][1]["tokens"]["prompt"],
                         "A camera moves through the repaired scene")
        video, audio = result[1]["samples"].unbind()
        self.assertEqual(tuple(video.shape), (1, 24, 2, 2, 2))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, 8))

    def test_node_uses_native_autogrow_reference_inputs(self):
        schema = MODULE.VTSMiniMaxH3MaskedVideoConditioning.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        self.assertIn("ref_images", inputs)
        self.assertIn("ref_videos", inputs)
        self.assertIn("ref_video_audios", inputs)
        self.assertIn("ref_audios", inputs)
        self.assertEqual(len(schema.outputs), 2)
        self.assertEqual(schema.outputs[0].display_name, "positive")
        self.assertEqual(schema.outputs[1].display_name, "latent")


if __name__ == "__main__":
    unittest.main()
