from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "py"
UTILS_DIR = PY_DIR / "vtsUtils"
sys.path.insert(0, str(UTILS_DIR))
sys.path.insert(0, str(PY_DIR))

import ref2va
import vr180_projection as projection


def load_node_module():
    spec = importlib.util.spec_from_file_location("VTS_Ref2VA_test", PY_DIR / "VTS_Ref2VA.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_node_module()


def synthetic_image(batch=1, height=64, width=64, device="cpu"):
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack((xx, yy, 0.25 * xx + 0.75 * yy), dim=-1)
    return image.unsqueeze(0).repeat(batch, 1, 1, 1)


def marker_pixels(image):
    rgb = ref2va.quantized_rgb(image)
    expected = torch.tensor([255, 0, 255], dtype=torch.uint8, device=image.device)
    return (rgb == expected).all(dim=-1)


class Ref2VANodeTests(unittest.TestCase):
    def test_node_registration_import_and_fixed_output_semantics(self):
        self.assertEqual(
            set(nodes.NODE_CLASS_MAPPINGS),
            {"VTSRef2VAStereoReferences", "VTSRef2VAProjectionReferences"},
        )
        stereo = nodes.VTSRef2VAStereoReferences
        self.assertEqual(stereo.RETURN_NAMES[:2], ("ref1", "ref2"))
        self.assertEqual(stereo.INPUT_TYPES()["required"]["effective_mask"][0], "MASK")

    def test_stereo_ref1_identity_order_and_marker_safety(self):
        ref1 = synthetic_image(batch=2)
        target = synthetic_image(batch=2).flip(2)
        target_before = target.clone()
        right = synthetic_image(batch=2)
        right[0, 2, 3, :3] = torch.tensor([1.0, 0.0, 1.0])
        right[1, 4, 5, :3] = torch.tensor([1.0, 0.0, 254.0 / 255.0])
        right_before = right.clone()
        mask = torch.zeros((2, 64, 64), dtype=torch.float32)
        mask[0, 7, 11] = 1.0
        mask[1, 13, 17] = 1.0

        out_ref1, out_ref2, out_mask, diagnostic_text = nodes.prepare_stereo_references(
            ref1, right, mask
        )
        self.assertIs(out_ref1, ref1)
        self.assertTrue(torch.equal(out_ref1, ref1))
        self.assertTrue(torch.equal(target, target_before))
        self.assertTrue(torch.equal(right, right_before))
        self.assertTrue(torch.equal(marker_pixels(out_ref2), mask.bool()))
        self.assertTrue(torch.equal(out_mask.bool(), mask.bool()))
        self.assertEqual(
            ref2va.quantized_rgb(out_ref2)[0, 2, 3].tolist(), [255, 0, 254]
        )
        self.assertEqual(
            ref2va.quantized_rgb(out_ref2)[1, 4, 5].tolist(), [255, 0, 254]
        )
        untouched = ~mask.bool()
        untouched[0, 2, 3] = False
        self.assertTrue(torch.equal(out_ref2[untouched], right[untouched]))
        diagnostics = json.loads(diagnostic_text)
        self.assertEqual(diagnostics["reference_order"], ["Ref1", "Ref2"])
        self.assertEqual(diagnostics["escaped_known_ref2_pixels"], [1, 0])
        self.assertEqual(diagnostics["exact_marker_pixels"], [1, 1])

    def test_empty_full_and_one_pixel_masks(self):
        image = synthetic_image(height=64, width=64)
        for mask in (
            torch.zeros((1, 64, 64)),
            torch.ones((1, 64, 64)),
            torch.nn.functional.pad(torch.ones((1, 1, 1)), (31, 32, 31, 32)),
        ):
            result = ref2va.mark_ref2(image, mask)
            self.assertTrue(torch.equal(marker_pixels(result.image), mask.bool()))
            self.assertEqual(result.marker_counts, (int(mask.sum()),))

    def test_invalid_size_and_nonbinary_mask_leave_no_artifacts(self):
        image = synthetic_image()
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "dimensions"):
                ref2va.mark_ref2(image, torch.zeros((1, 63, 64)))
            with self.assertRaisesRegex(ValueError, "binary"):
                ref2va.mark_ref2(image, torch.full((1, 64, 64), 0.5))
            self.assertEqual(list(Path(path).iterdir()), [])

    def test_projection_matches_shared_core_and_letterboxes_true_aspect(self):
        target = synthetic_image(batch=2, height=256, width=256)
        target_before = target.clone()
        output = nodes.prepare_projection_references(
            target, "16:9", 90.0, 0.0, 0.0, 0.0, 37, "bilinear"
        )
        ref1, ref2, mask, diagnostic_text = output
        source = target.movedim(-1, 1).contiguous()
        direct_rect = projection.vr180_to_rectilinear(
            source, (144, 256), horizontal_fov_degrees=90.0,
            unknown_color=(0.0, 0.0, 0.0), chunk_rows=37,
        )
        direct_ref2 = projection.rectilinear_to_vr180(
            direct_rect.image, (256, 256), horizontal_fov_degrees=90.0,
            chunk_rows=37,
        )
        self.assertTrue(torch.equal(target, target_before))
        self.assertTrue(torch.equal(ref1[:, 56:200], direct_rect.image.movedim(1, -1)))
        self.assertEqual(int(torch.count_nonzero(ref1[:, :56])), 0)
        self.assertEqual(int(torch.count_nonzero(ref1[:, 200:])), 0)
        self.assertTrue(torch.equal(ref2, direct_ref2.image.movedim(1, -1)))
        self.assertTrue(torch.equal(mask.bool(), direct_ref2.unknown_mask[:, 0]))
        self.assertTrue(torch.equal(marker_pixels(ref2), mask.bool()))
        diagnostics = json.loads(diagnostic_text)
        self.assertEqual(diagnostics["rectilinear_content_size"], [256, 144])
        self.assertEqual(diagnostics["letterbox"], [0, 56, 0, 56])

    def test_projection_rejects_view_outside_vr180_without_mutation(self):
        target = synthetic_image()
        before = target.clone()
        with self.assertRaisesRegex(ValueError, "outside"):
            nodes.prepare_projection_references(
                target, "16:9", 120.0, 80.0, 0.0, 0.0, 16, "bilinear"
            )
        self.assertTrue(torch.equal(target, before))

    def test_1024_and_2048_synthetic_frame_batches(self):
        for size in (1024, 2048):
            target = torch.zeros((2, size, size, 3), dtype=torch.float32)
            ref1, ref2, mask, diagnostics = nodes.prepare_projection_references(
                target, "4:3", 90.0, 0.0, 0.0, 0.0, 256, "bilinear"
            )
            self.assertEqual(ref1.shape, target.shape)
            self.assertEqual(ref2.shape, target.shape)
            self.assertEqual(mask.shape, target.shape[:3])
            self.assertTrue(torch.equal(marker_pixels(ref2), mask.bool()))
            self.assertEqual(json.loads(diagnostics)["batch_size"], 2)
            del target, ref1, ref2, mask

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu_and_preserves_device(self):
        cpu_image = synthetic_image(batch=2)
        mask = torch.zeros((2, 64, 64))
        mask[:, 3, 5] = 1.0
        cpu = nodes.prepare_stereo_references(cpu_image, cpu_image, mask)
        gpu = nodes.prepare_stereo_references(
            cpu_image.cuda(), cpu_image.cuda(), mask.cuda()
        )
        self.assertTrue(gpu[0].is_cuda and gpu[1].is_cuda and gpu[2].is_cuda)
        self.assertTrue(torch.equal(cpu[1], gpu[1].cpu()))
        self.assertTrue(torch.equal(cpu[2], gpu[2].cpu()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
