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



class MerserkDecodeTests(unittest.TestCase):
    def setUp(self):
        self.node = NODE.VTS_VAEDecodeTiledColourMatch()
        # Five H3 source frames, with a different constant value per frame.
        self.image = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 32, 32, 3).clone() / 10
        self.source = {"samples": torch.zeros(1, 24, 2, 2, 2)}
        self.vae = FakeVAE(self.image)
        self.vae.encode = self.encoder
        self.events = []

    def encoder(self, images):
        self.events.append("encode")
        self.encoded_input = images.clone()
        self.input_reference = weakref.ref(images)
        steps = NODE._steps_for_frames(len(images))
        return images.reshape(-1)[:24 * steps * 4].reshape(1, 24, steps, 2, 2)

    def enhance(self, images, **options):
        self.events.append("enhance")
        self.sent = images.clone()
        self.options = options
        result = images * 0.5 + 0.25
        if options.get("enable_frame_interpolation", False) and len(result) > 1:
            multiplier = options.get("interpolation_multiplier", 2)
            expanded = []
            for first, second in zip(result[:-1], result[1:]):
                expanded.extend(torch.lerp(first, second, i / multiplier) for i in range(multiplier))
            result = torch.stack([*expanded, result[-1]])
        # Deliberately change both dimensions/aspect ratio to exercise context resize.
        self.enhanced = result.repeat_interleave(2, dim=1).repeat_interleave(3, dim=2)
        return (self.enhanced,)

    def test_schema_reuses_all_processing_controls_and_no_disk_or_outer_controls(self):
        schema = self.node.INPUT_TYPES()["optional"]
        excluded = {"images", "return_type", "output_dir", "format", "compression_level",
                    "quality", "prefix", "start_sequence"}
        expected = {"merserk_" + name: spec
                    for group in NODE.VTSMerserkTemporalEnhance.INPUT_TYPES().values()
                    for name, spec in group.items() if name not in excluded}
        actual = {name: spec for name, spec in schema.items() if name.startswith("merserk_")}
        self.assertEqual(actual, expected)
        self.assertFalse(schema["enable_merserk"][1]["default"])
        self.assertFalse(schema["use_merserk_for_context"][1]["default"])
        self.assertEqual(schema["context_frame_selection"][1]["default"], "Original")
        self.assertNotIn("merserk_iterations", actual)

    def test_master_off_preserves_images_and_context_and_ignores_merserk_settings(self):
        with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=AssertionError("remote call")):
            for encode in (False, True):
                output, context = self.node.decode(self.vae, self.source, return_type="Tensor",
                    encode_corrected_context=encode, use_merserk_for_context=True,
                    context_frame_selection="ignored", merserk_enable_frame_interpolation=True,
                    merserk_interpolation_multiplier=999, merserk_server_url="invalid")
                torch.testing.assert_close(output, self.image, rtol=0, atol=0)
                if encode:
                    torch.testing.assert_close(self.encoded_input, self.image, rtol=0, atol=0)
                else:
                    self.assertIsNone(context)

    def test_colour_correction_precedes_merserk_and_all_controls_are_forwarded(self):
        ref = torch.full_like(self.image[:1], 0.3)
        corrected, _ = self.node.decode(self.vae, self.source, color_ref=ref, return_type="Tensor")
        options = {name: spec[1].get("default", spec[0][0] if isinstance(spec[0], list) else None)
                   for name, spec in NODE._merserk_inputs().items()}
        options.update(server_url="http://test:7865", nr_passes=3, skin_structure_strength=1.5,
                       automatic_mask=True, enable_frame_interpolation=True, interpolation_multiplier=4)
        with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=self.enhance):
            output, context = self.node.decode(self.vae, self.source, color_ref=ref,
                enable_merserk=True, return_type="Tensor",
                **{"merserk_" + name: value for name, value in options.items()})
        torch.testing.assert_close(self.sent, corrected, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.sent, self.image))
        self.assertEqual(self.options, dict(options, return_type="Tensor"))
        self.assertEqual(tuple(output.shape), (17, 64, 96, 3))
        self.assertIsNone(context)

    def test_context_can_exclude_merserk_even_when_output_is_scaled_and_interpolated(self):
        with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=self.enhance):
            output, context = self.node.decode(self.vae, self.source, return_type="Tensor",
                enable_merserk=True, encode_corrected_context=True,
                merserk_enable_frame_interpolation=True, merserk_interpolation_multiplier=4,
                context_frame_selection="Interpolated")
        torch.testing.assert_close(self.encoded_input, self.image, rtol=0, atol=0)
        self.assertEqual(tuple(output.shape), (17, 64, 96, 3))
        self.assertEqual(context["source_frames"], 5)
        self.assertEqual(context["frame_count"], 5)
        self.assertEqual(self.events, ["enhance", "encode"])

    def test_context_original_vs_interpolated_tail_and_automatic_resize(self):
        for multiplier in (2, 3, 4, 8):
            for selection in ("Original", "Interpolated"):
                with self.subTest(multiplier=multiplier, selection=selection):
                    with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=self.enhance):
                        output, context = self.node.decode(self.vae, self.source, return_type="Tensor",
                            enable_merserk=True, encode_corrected_context=True, use_merserk_for_context=True,
                            context_frame_selection=selection, merserk_enable_frame_interpolation=True,
                            merserk_interpolation_multiplier=multiplier)
                    stride = multiplier if selection == "Original" else 1
                    # Each mock frame is spatially constant, so resizing must retain its value.
                    values = output[::stride][-5:, :1, :1]
                    torch.testing.assert_close(self.encoded_input, values.expand(5, 32, 32, 3), atol=1/255, rtol=0)
                    self.assertEqual(context["video"].shape, (1, 24, 2, 2, 2))
                    self.assertEqual(context["source_frames"], 5)
                    self.assertEqual(len(output), 4 * multiplier + 1)
                    gc.collect()
                    self.assertIsNone(self.input_reference())
                    encoded = context["video"]
                    self.assertEqual(encoded.untyped_storage().nbytes(), encoded.numel() * encoded.element_size())

    def test_single_frame_and_disabled_interpolation(self):
        for count, steps in ((1, 1), (5, 2)):
            for interpolate in (False, True):
                for selection in ("Original", "Interpolated"):
                    with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=self.enhance):
                        output, context = self.node.decode(FakeVAEWithEncode(self.image[:count], self.encoder),
                            {"samples": torch.zeros(1, 24, steps, 2, 2)}, return_type="Tensor",
                            enable_merserk=True, encode_corrected_context=True, use_merserk_for_context=True,
                            context_frame_selection=selection, merserk_enable_frame_interpolation=interpolate)
                    self.assertEqual(len(self.encoded_input), count)
                    self.assertEqual(context["frame_count"], count)
                    if not interpolate or count == 1:
                        torch.testing.assert_close(self.encoded_input, self.image[:count] * 0.5 + 0.25, atol=1/255, rtol=0)

    def test_jpeg_is_saved_only_after_context_encode_without_readback(self):
        real_save = NODE.save_images
        def save(**kwargs):
            self.events.append("save")
            return real_save(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=self.enhance), \
                 patch.object(NODE.DiskImage, "materialize", side_effect=AssertionError("disk read")), \
                 patch.object(NODE, "save_images", side_effect=save), \
                 CurrentNodeContext("test", "decode", 2):
                disk, context = self.node.decode(self.vae, self.source, return_type="DiskImage",
                    enable_merserk=True, encode_corrected_context=True, use_merserk_for_context=True,
                    merserk_enable_frame_interpolation=True, output_dir=directory,
                    prefix="enhanced", start_sequence=12, format="jpg", quality=20, num_workers=1)
            self.assertEqual(self.events, ["enhance", "encode", "save"])
            self.assertEqual(len(list(Path(directory).glob("*.jpg"))), 9)
            self.assertTrue((Path(directory) / "enhanced_list_000002_000020.jpg").exists())
            self.assertEqual(disk.number_of_images, 9)
            torch.testing.assert_close(self.encoded_input, self.image * 0.5 + 0.25, atol=1/255, rtol=0)
            self.assertEqual(context["frame_count"], 5)

    def test_failures_and_cancellation_propagate_without_saving_or_encoding(self):
        for error in (RuntimeError("server unavailable"), NODE.model_management.InterruptProcessingException()):
            with patch.object(NODE.VTSMerserkTemporalEnhance, "enhance", side_effect=error), \
                 patch.object(NODE, "save_images") as save:
                with self.assertRaises(type(error)):
                    self.node.decode(self.vae, self.source, enable_merserk=True,
                        encode_corrected_context=True, use_merserk_for_context=True, return_type="DiskImage")
                save.assert_not_called()
                self.assertEqual(self.events, [])

    def test_real_client_local_resize_and_bypass_need_no_server(self):
        with patch("VTS_MerserkTemporalEnhance.connect", side_effect=AssertionError("network")):
            output, context = self.node.decode(self.vae, self.source, enable_merserk=True,
                merserk_enable_scaling=False, merserk_enable_neural_rendering=False, return_type="Tensor")
            torch.testing.assert_close(output, self.image, rtol=0, atol=0)
            output, context = self.node.decode(self.vae, self.source, enable_merserk=True,
                merserk_enable_neural_rendering=False, merserk_sizing_mode="Multiplier",
                merserk_upscaling_factor=0.5, return_type="Tensor", encode_corrected_context=True,
                use_merserk_for_context=True)
        self.assertEqual(output.shape, (5, 16, 16, 3))
        self.assertEqual(self.encoded_input.shape, (5, 32, 32, 3))
        self.assertEqual(context["video"].shape, (1, 24, 2, 2, 2))


class FakeVAEWithEncode(FakeVAE):
    def __init__(self, image, encoder):
        super().__init__(image)
        self.encode = encoder

if __name__ == "__main__":
    unittest.main()
