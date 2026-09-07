import importlib.util
import gc
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from comfy.cli_args import args
args.cpu = True

NODE_PATH = Path(__file__).parents[1] / "py" / "VTS_VAEDecodeTiledColourMatch.py"
SPEC = importlib.util.spec_from_file_location("vts_colour_decode_test", NODE_PATH)
NODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE)
import vts_color_correction as CORE
from comfy.nested_tensor import NestedTensor
from comfy_execution.utils import CurrentNodeContext


def frames(count=4):
    generator = torch.Generator().manual_seed(72)
    return torch.rand(count, 24, 32, 3, generator=generator) * 0.6 + 0.2


def correct(images, ref, **kwargs):
    return torch.stack(list(CORE.correct_images(images, ref, **kwargs)))


class FakeVAE:
    def __init__(self, image):
        self.image = image
        self.calls = []

    def temporal_compression_decode(self):
        return 4

    def spacial_compression_decode(self):
        return 8

    def decode_tiled(self, latent, **kwargs):
        self.calls.append((latent, kwargs))
        return self.image.clone()


class ColourCorrectionTests(unittest.TestCase):
    def test_bypass_is_exact_and_does_not_load_reference(self):
        image = frames()
        self.assertTrue(torch.equal(correct(image, None), image))
        self.assertTrue(torch.equal(correct(image, object(), overall_weight=0), image))
        self.assertTrue(torch.equal(correct(image, object(), color_weight=0), image))

    def test_all_methods_modes_and_different_reference_sizes(self):
        image = frames(3)
        ref = torch.nn.functional.interpolate(
            (image[:1] * 0.8).permute(0, 3, 1, 2), (18, 20)).permute(0, 2, 3, 1)
        for method in CORE.METHODS:
            for mode in CORE.MODES:
                with self.subTest(method=method, mode=mode):
                    result = correct(image, ref, method=method, mode=mode,
                                     white_weight=0.1, brightness_weight=0.2, contrast_weight=0.1)
                    self.assertEqual(result.shape, image.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertGreaterEqual(result.min().item(), 0)
                    self.assertLessEqual(result.max().item(), 1)

    def test_fixed_mapping_does_not_change_with_frame_position(self):
        image = frames()
        image[3] = image[0]
        output = correct(image, image[:1] * 0.8)
        self.assertTrue(torch.equal(output[0], output[3]))

    def test_fitted_cpu_transforms_match_library_on_analysis_pixels(self):
        from color_matcher import ColorMatcher
        source = frames(1)[0]
        reference = source * torch.tensor([0.8, 0.9, 0.7]) + 0.04
        for method in CORE.METHODS[:-1]:
            with self.subTest(method=method):
                fitted = CORE._cpu_color_transform(source, reference, method)(source)
                expected = ColorMatcher(method=method).transfer(
                    src=source.numpy().copy(), ref=reference.numpy().copy(), method=method)
                self.assertTrue(torch.allclose(fitted, torch.from_numpy(expected).float(), atol=2e-5))

    def test_gpu_lab_lookup_approximates_direct_kj_formula(self):
        source = frames(1)[0]
        reference = source * torch.tensor([0.8, 0.9, 0.7]) + 0.04
        direct = CORE._lab_transform(source, reference)(source)
        actual = correct(source[None], reference[None], color_weight=1, lut_resolution=65)[0]
        self.assertLess((actual - direct).abs().mean().item(), 0.001)

    def test_fixed_mode_analyzes_generated_frames_beyond_matching_head(self):
        image = frames(1).repeat(4, 1, 1, 1)
        ref = image[:1].clone()
        image[1:] *= 0.6
        output = correct(image, ref, color_weight=0, brightness_weight=1,
                         brightness_method="exposure")
        self.assertLess((output[1:] - ref).abs().mean(), (image[1:] - ref).abs().mean())

    def test_smoothing_endpoints_and_reexecution(self):
        image = frames()
        image[1:] *= 0.6
        ref = frames(1)
        per_frame = correct(image, ref, mode="per_frame")
        no_smoothing = correct(image, ref, mode="smoothed_over_time", smoothing=0)
        self.assertTrue(torch.equal(per_frame, no_smoothing))
        held = correct(image, ref, mode="smoothed_over_time", smoothing=1)
        first_lut = CORE._fit_lut(image[0], ref[0], "reinhard_lab_gpu", 0.5, 0, 0, 0, "gamma", 33)
        expected = torch.stack([CORE._apply_lut(frame, first_lut) for frame in image])
        self.assertTrue(torch.allclose(held, expected, atol=1e-6))
        self.assertTrue(torch.equal(held, correct(image, ref, mode="smoothed_over_time", smoothing=1)))

    def test_overall_blend_and_input_preservation(self):
        image, ref = frames(), frames(1) * 0.7
        original, reference = image.clone(), ref.clone()
        full = correct(image, ref, overall_weight=1)
        half = correct(image, ref, overall_weight=0.5)
        self.assertTrue(torch.allclose(half, (image + full) / 2, atol=1e-6))
        self.assertTrue(torch.equal(image, original))
        self.assertTrue(torch.equal(ref, reference))

    def test_brightness_modes_reduce_reference_error(self):
        ref = frames(1)
        for method in ("gamma", "exposure"):
            image = ref.pow(1.4) if method == "gamma" else ref * 0.7
            output = correct(image, ref, color_weight=0, brightness_weight=1,
                             brightness_method=method, mode="per_frame", lut_resolution=65)
            self.assertLess((output - ref).abs().mean(), 0.005)

    def test_white_balance_and_contrast(self):
        image = frames(1)
        tinted = image * torch.tensor([1.1, 0.9, 0.8])
        balanced = correct(tinted, image, color_weight=0, white_weight=1)
        mean_ref = image.mean((0, 1, 2))
        mean_out = balanced.mean((0, 1, 2))
        self.assertTrue(torch.allclose(mean_out / mean_out.sum(), mean_ref / mean_ref.sum(), atol=0.002))
        flat = image * 0.6 + 0.2
        contrasted = correct(flat, image, color_weight=0, contrast_weight=1)
        self.assertLess((contrasted - image).abs().mean(), 0.005)

    def test_flat_black_and_white_are_finite(self):
        for value in (0.0, 0.5, 1.0):
            for method in CORE.METHODS:
                image = torch.full((1, 16, 16, 3), value)
                out = correct(image, frames(1), method=method, white_weight=1,
                              brightness_weight=1, contrast_weight=1)
                self.assertTrue(torch.isfinite(out).all(), method)

    def test_alpha_preserved(self):
        rgba = torch.cat([frames(2), frames(2)[..., :1]], dim=-1)
        output = correct(rgba, frames(1) * 0.8)
        self.assertTrue(torch.equal(output[..., 3], rgba[..., 3]))

    def test_disk_reference_reads_one_frame_and_matches_tensor(self):
        image, ref = frames(4), frames(2) * 0.8
        disk = CORE.DiskImage("ref", 17, 2, "unused", "png", ref)
        calls = []
        def materialize(start=0, count=None):
            calls.append((start, count))
            return ref[start:start + count]
        disk.materialize = materialize
        for mode in CORE.MODES:
            calls.clear()
            expected = correct(image, ref, mode=mode)
            actual = correct(image, disk, mode=mode)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(count == 1 for _, count in calls))
        self.assertEqual(calls, [(0, 1), (1, 1), (1, 1), (1, 1)])

    def test_invalid_reference_is_reported(self):
        with self.assertRaisesRegex(ValueError, "no images"):
            correct(frames(), torch.zeros(0, 16, 16, 3))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            correct(frames(), torch.full((1, 16, 16, 3), float("nan")))

    def test_gpu_matches_cpu_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        image, ref = frames(), frames(1) * 0.8
        cpu = correct(image, ref)
        with patch.object(CORE.model_management, "get_torch_device", return_value=torch.device("cuda")):
            gpu = correct(image, ref)
        self.assertTrue(torch.allclose(cpu, gpu, atol=2e-4))


class DecodeIntegrationTests(unittest.TestCase):
    def test_both_decoder_schemas_allow_zero_overlap(self):
        for node in (NODE.VTS_VAEDecodeTiled, NODE.VTS_VAEDecodeTiledColourMatch):
            with self.subTest(node=node.__name__):
                required = node.INPUT_TYPES()["required"]
                self.assertEqual(required["overlap"][1]["min"], 0)
                self.assertEqual(required["temporal_overlap"][1]["min"], 0)
                self.assertEqual(required["overlap"][1]["default"], 64)
                self.assertEqual(required["temporal_overlap"][1]["default"], 8)

    def test_zero_overlap_forwarded_by_base_and_colour_decode(self):
        latent = {"samples": torch.zeros(1, 4, 4, 4, 4)}
        for node, extra in (
            (NODE.VTS_VAEDecodeTiled(), {}),
            (NODE.VTS_VAEDecodeTiledColourMatch(), {}),
            (NODE.VTS_VAEDecodeTiledColourMatch(), {"color_ref": frames(1) * 0.8}),
        ):
            with self.subTest(node=type(node).__name__, correction=bool(extra)):
                vae = FakeVAE(frames())
                result = node.decode(vae, latent, overlap=0, temporal_overlap=0,
                                     return_type="Tensor", **extra)
                self.assertEqual(vae.calls[-1][1]["overlap"], 0)
                self.assertEqual(vae.calls[-1][1]["overlap_t"], 0)
                self.assertEqual(result[0].shape, vae.image.shape)
                self.assertTrue(torch.isfinite(result[0]).all())

    def test_positive_overlap_conversion_and_image_vae_unchanged(self):
        node = NODE.VTS_VAEDecodeTiled()
        latent = {"samples": torch.zeros(1, 4, 4, 4, 4)}
        for requested, expected in ((1, 1), (4, 1), (8, 2)):
            vae = FakeVAE(frames())
            node.decode(vae, latent, temporal_overlap=requested, return_type="Tensor")
            self.assertEqual(vae.calls[-1][1]["overlap"], 8)
            self.assertEqual(vae.calls[-1][1]["overlap_t"], expected)
        vae = FakeVAE(frames())
        with patch.object(vae, "temporal_compression_decode", return_value=None):
            node.decode(vae, latent, overlap=0, temporal_overlap=0, return_type="Tensor")
        self.assertIsNone(vae.calls[-1][1]["tile_t"])
        self.assertIsNone(vae.calls[-1][1]["overlap_t"])
        self.assertEqual(vae.calls[-1][1]["overlap"], 0)

    def test_schema_preserves_decoder_inputs(self):
        base = NODE.VTS_VAEDecodeTiled.INPUT_TYPES()
        new = NODE.VTS_VAEDecodeTiledColourMatch.INPUT_TYPES()
        self.assertEqual(base["required"], new["required"])
        self.assertIn("color_ref", new["optional"])
        self.assertEqual(NODE.VTS_VAEDecodeTiledColourMatch.RETURN_TYPES,
                         ("IMAGE", "VTS_H3_VIDEO_CONTEXT"))
        self.assertFalse(new["optional"]["encode_corrected_context"][1]["default"])

    def test_nested_latent_decode_matches_base_and_forwards_tiles(self):
        video, audio = torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 40)
        latent = {"samples": NestedTensor((video, audio))}
        vae = FakeVAE(frames())
        node = NODE.VTS_VAEDecodeTiledColourMatch()
        expected, = NODE.VTS_VAEDecodeTiled().decode(vae, latent, return_type="Tensor")
        output, context = node.decode(vae, latent, color_ref=None, return_type="Tensor")
        self.assertIsNone(context)
        self.assertTrue(torch.equal(output, expected))
        self.assertIs(vae.calls[-1][0], video)
        self.assertEqual(vae.calls[0][1], vae.calls[-1][1])

    def test_disk_output_roundtrip_numbering_and_list_suffix(self):
        image, reference = frames(2), frames(1) * 0.8
        latent = {"samples": torch.zeros(1, 4, 4, 4)}
        node = NODE.VTS_VAEDecodeTiledColourMatch()
        expected, context = node.decode(FakeVAE(image), latent, color_ref=reference, return_type="Tensor")
        self.assertIsNone(context)
        with tempfile.TemporaryDirectory() as directory:
            with CurrentNodeContext("test", "decode", 3):
                disk, context = node.decode(FakeVAE(image), latent, color_ref=reference,
                                    return_type="DiskImage", output_dir=directory,
                                    prefix="clip", start_sequence=12, format="png", num_workers=1)
            self.assertEqual(disk.prefix, "clip_list_000003")
            self.assertEqual(disk.start_sequence, 12)
            self.assertEqual(disk.number_of_images, 2)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["clip_list_000003_000012.png", "clip_list_000003_000013.png"])
            self.assertTrue(torch.allclose(disk.materialize(), expected, atol=1 / 255 + 1e-6))
            self.assertIsNone(context)


class CorrectedContextTests(unittest.TestCase):
    def setUp(self):
        self.node = NODE.VTS_VAEDecodeTiledColourMatch()
        self.image = torch.rand(39, 32, 32, 3, generator=torch.Generator().manual_seed(4))
        self.source = {"samples": NestedTensor((torch.zeros(1, 24, 12, 2, 2),
                                                 torch.zeros(1, 32, 2, 65)))}

    def encoder(self, frames):
        # A view deliberately tests that the returned context owns its storage.
        self.encoded_input = frames.clone()
        self.input_reference = weakref.ref(frames)
        steps = NODE._steps_for_frames(len(frames))
        return frames.reshape(-1)[:24 * steps * 4].reshape(1, 24, steps, 2, 2)

    def test_encodes_corrected_tail_only_for_each_length(self):
        for requested, count in (("5", 5), ("22", 22), ("39", 39), ("56", 39)):
            vae = FakeVAE(self.image)
            vae.encode = self.encoder
            output, context = self.node.decode(
                vae, self.source, color_ref=self.image[:1] * 0.7,
                return_type="Tensor", encode_corrected_context=True, context_length=requested)
            torch.testing.assert_close(self.encoded_input, output[-count:], rtol=0, atol=0)
            self.assertFalse(torch.equal(self.encoded_input, self.image[-count:]))
            self.assertEqual(context["frame_count"], count)
            self.assertEqual(context["source_frames"], 39)
            encoded = context["video"]
            self.assertNotEqual(encoded.untyped_storage().data_ptr(), output.untyped_storage().data_ptr())
            self.assertEqual(encoded.untyped_storage().nbytes(), encoded.numel() * encoded.element_size())
            self.assertFalse(encoded.requires_grad)
            self.assertEqual(len(vae.calls), 1)
            del output
            gc.collect()
            self.assertIsNone(self.input_reference())

    def test_bypass_still_encodes_when_enabled(self):
        for settings in ({"color_ref": None}, {"color_ref": object(), "overall_weight": 0},
                         {"color_ref": object(), "color_match_weight": 0}):
            vae = FakeVAE(self.image)
            vae.encode = self.encoder
            output, context = self.node.decode(vae, self.source, return_type="Tensor",
                encode_corrected_context=True, **settings)
            torch.testing.assert_close(output, self.image, rtol=0, atol=0)
            torch.testing.assert_close(self.encoded_input, self.image[-22:], rtol=0, atol=0)
            self.assertEqual(context["frame_count"], 22)

    def test_full_56_frame_tail_and_single_frame_clip(self):
        for source_steps, count in ((22, 56), (1, 1)):
            source_frames = NODE._pixel_frames(source_steps)
            image = self.image[:1].repeat(source_frames, 1, 1, 1)
            source = {"samples": torch.zeros(1, 24, source_steps, 2, 2)}
            vae = FakeVAE(image)
            vae.encode = self.encoder
            _, context = self.node.decode(vae, source, return_type="Tensor",
                encode_corrected_context=True, context_length="56")
            self.assertEqual(len(self.encoded_input), count)
            self.assertEqual(context["frame_count"], count)
            self.assertEqual(context["video"].shape[2], NODE._steps_for_frames(count))

    def test_disk_output_does_not_read_saved_images_for_encoding(self):
        vae = FakeVAE(self.image)
        vae.encode = self.encoder
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(NODE.DiskImage, "materialize", side_effect=AssertionError("disk read")):
                disk, context = self.node.decode(vae, self.source, return_type="DiskImage",
                    output_dir=directory, prefix="context", format="png", num_workers=1,
                    encode_corrected_context=True)
            self.assertEqual(disk.number_of_images, 39)
            self.assertEqual(len(list(Path(directory).glob('*.png'))), 39)
            self.assertEqual(context["frame_count"], 22)

    def test_rejects_wrong_model_batch_geometry_and_encoded_shape(self):
        for video in (torch.zeros(1, 4, 2, 2), torch.zeros(2, 24, 12, 2, 2)):
            with self.assertRaisesRegex(ValueError, "one H3"):
                self.node.decode(FakeVAE(self.image), {"samples": video}, encode_corrected_context=True)
        for image in (self.image[:-1], self.image[:, :16]):
            with self.assertRaisesRegex(ValueError, "frame count and resolution"):
                self.node.decode(FakeVAE(image), self.source, encode_corrected_context=True)
        vae = FakeVAE(self.image)
        vae.encode = lambda frames: torch.zeros(1, 24, 8, 2, 2)
        with self.assertRaisesRegex(ValueError, "latent shape"):
            self.node.decode(vae, self.source, encode_corrected_context=True)
        with self.assertRaisesRegex(ValueError, "context_length"):
            self.node.decode(vae, self.source, encode_corrected_context=True, context_length="9")

    def test_corrected_tail_is_used_in_guide_and_mask_without_changing_audio(self):
        from VTS_H3LoopContext import VTS_H3PrepareLoopContext, VTS_H3ApplyLoopContext
        vae = FakeVAE(self.image)
        vae.encode = self.encoder
        _, replacement = self.node.decode(vae, self.source, return_type="Tensor",
            encode_corrected_context=True, color_ref=self.image[:1] * 0.8)
        prepare = VTS_H3PrepareLoopContext()
        original, = prepare.execute(self.source)
        corrected, = prepare.execute(self.source, corrected_video_context=replacement)
        torch.testing.assert_close(corrected["audio"], original["audio"], rtol=0, atol=0)
        self.assertEqual(corrected["audio_start"], original["audio_start"])
        output, trim, masked = VTS_H3ApplyLoopContext().execute(
            [[torch.zeros(1), {}]], self.source, corrected)
        self.assertEqual(trim, 22)
        torch.testing.assert_close(output[0][1]["minimax_keyframes"][-2]["latent"], replacement["video"])
        torch.testing.assert_close(masked["samples"].tensors[0][:, :, :7], replacement["video"])


if __name__ == "__main__":
    unittest.main()
