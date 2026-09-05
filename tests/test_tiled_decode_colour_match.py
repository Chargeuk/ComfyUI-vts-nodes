import importlib.util
import sys
import tempfile
import unittest
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
    def test_schema_preserves_decoder_inputs(self):
        base = NODE.VTS_VAEDecodeTiled.INPUT_TYPES()
        new = NODE.VTS_VAEDecodeTiledColourMatch.INPUT_TYPES()
        self.assertEqual(base["required"], new["required"])
        self.assertIn("color_ref", new["optional"])
        self.assertEqual(NODE.VTS_VAEDecodeTiledColourMatch.RETURN_TYPES, ("IMAGE",))

    def test_nested_latent_decode_matches_base_and_forwards_tiles(self):
        video, audio = torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 40)
        latent = {"samples": NestedTensor((video, audio))}
        vae = FakeVAE(frames())
        node = NODE.VTS_VAEDecodeTiledColourMatch()
        expected, = NODE.VTS_VAEDecodeTiled().decode(vae, latent, return_type="Tensor")
        output, = node.decode(vae, latent, color_ref=None, return_type="Tensor")
        self.assertTrue(torch.equal(output, expected))
        self.assertIs(vae.calls[-1][0], video)
        self.assertEqual(vae.calls[0][1], vae.calls[-1][1])

    def test_disk_output_roundtrip_numbering_and_list_suffix(self):
        image, reference = frames(2), frames(1) * 0.8
        latent = {"samples": torch.zeros(1, 4, 4, 4)}
        node = NODE.VTS_VAEDecodeTiledColourMatch()
        expected, = node.decode(FakeVAE(image), latent, color_ref=reference, return_type="Tensor")
        with tempfile.TemporaryDirectory() as directory:
            with CurrentNodeContext("test", "decode", 3):
                disk, = node.decode(FakeVAE(image), latent, color_ref=reference,
                                    return_type="DiskImage", output_dir=directory,
                                    prefix="clip", start_sequence=12, format="png", num_workers=1)
            self.assertEqual(disk.prefix, "clip_list_000003")
            self.assertEqual(disk.start_sequence, 12)
            self.assertEqual(disk.number_of_images, 2)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["clip_list_000003_000012.png", "clip_list_000003_000013.png"])
            self.assertTrue(torch.allclose(disk.materialize(), expected, atol=1 / 255 + 1e-6))


if __name__ == "__main__":
    unittest.main()
