from pathlib import Path
import importlib.util
import json
from unittest import mock
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "py"
UTILS_DIR = PY_DIR / "vtsUtils"
sys.path.insert(0, str(UTILS_DIR))

import vr180_projection as projection


def load_nodes():
    spec = importlib.util.spec_from_file_location(
        "VTS_Rectilinear_To_VR180_test", PY_DIR / "VTS_Rectilinear_To_VR180.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_nodes()


def smooth_image(batch=1, height=65, width=97, device="cpu"):
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack((xx, yy, 0.25 * xx + 0.75 * yy), dim=-1)
    return image.unsqueeze(0).repeat(batch, 1, 1, 1)


class VR180NodeTests(unittest.TestCase):
    def test_registration_and_explicit_interface(self):
        self.assertEqual(
            set(nodes.NODE_CLASS_MAPPINGS),
            {"VTSRectilinearToVR180", "VTSVR180ToRectilinear"},
        )
        inputs = nodes.VTSRectilinearToVR180.INPUT_TYPES()["required"]
        for name in (
            "output_width", "output_height", "horizontal_fov_degrees",
            "yaw_degrees", "pitch_degrees", "roll_degrees",
            "chunk_rows", "frame_batch_size", "sampling",
        ):
            self.assertIn(name, inputs)

    def test_forward_matches_shared_core_and_marks_exact_magenta(self):
        images = smooth_image(batch=3)
        before = images.clone()
        output, known, unknown, metadata = nodes.project_rectilinear_to_vr180(
            images, 128, 96, 90.0, 12.0, -4.0, 3.0, 17, 2, "bilinear"
        )
        direct = projection.rectilinear_to_vr180(
            images.movedim(-1, 1), (96, 128), horizontal_fov_degrees=90.0,
            yaw_degrees=12.0, pitch_degrees=-4.0, roll_degrees=3.0,
            chunk_rows=17, mode="bilinear",
        )
        self.assertTrue(torch.equal(images, before))
        self.assertTrue(torch.equal(output, direct.image.movedim(1, -1)))
        self.assertTrue(torch.equal(known.bool(), direct.known_mask[:, 0]))
        self.assertTrue(torch.equal(unknown.bool(), direct.unknown_mask[:, 0]))
        marker = torch.tensor([1.0, 0.0, 1.0])
        self.assertTrue(torch.equal(output[unknown.bool()], marker.expand_as(output)[unknown.bool()]))
        parsed = json.loads(metadata)
        self.assertEqual(parsed["direction"], "rectilinear_to_vr180")
        self.assertEqual(parsed["frame_batch_size"], 2)
        self.assertEqual(parsed["output_size"], [128, 96])

    def test_frame_batch_size_bounds_each_projection_call(self):
        images = smooth_image(batch=5)
        original = nodes.rectilinear_to_vr180
        seen = []

        def observe(batch, *args, **kwargs):
            seen.append(int(batch.shape[0]))
            return original(batch, *args, **kwargs)

        with mock.patch.object(nodes, "rectilinear_to_vr180", side_effect=observe):
            nodes.project_rectilinear_to_vr180(
                images, 64, 64, 80.0, 0.0, 0.0, 0.0, 11, 2, "nearest"
            )
        self.assertEqual(seen, [2, 2, 1])

    def test_reverse_round_trip_and_broadcast_source_mask(self):
        source = smooth_image(batch=3, height=81, width=121)
        vr, known, _, _ = nodes.project_rectilinear_to_vr180(
            source, 160, 96, 90.0, 20.0, 0.0, 0.0, 13, 2, "bilinear"
        )
        reverse, reverse_known, _, metadata = nodes.project_vr180_to_rectilinear(
            vr, 121, 81, 90.0, 20.0, 0.0, 0.0, 13, 2, "bilinear", known[:1]
        )
        valid = reverse_known.bool().unsqueeze(-1).expand_as(reverse)
        self.assertGreater(float(valid.float().mean()), 0.90)
        self.assertLess(float(torch.mean(torch.abs(reverse[valid] - source[valid]))), 0.02)
        self.assertEqual(json.loads(metadata)["direction"], "vr180_to_rectilinear")

    def test_invalid_inputs_fail_without_mutation(self):
        image = smooth_image()
        before = image.clone()
        args = (image, 64, 64, 90.0, 0.0, 0.0, 0.0, 16, 1, "bilinear")
        with self.assertRaisesRegex(ValueError, "frame_batch_size"):
            nodes.project_rectilinear_to_vr180(*args[:-2], 0, args[-1])
        with self.assertRaisesRegex(ValueError, "finite"):
            nodes.project_rectilinear_to_vr180(
                image, 64, 64, 90.0, float("nan"), 0.0, 0.0, 16, 1, "bilinear"
            )
        with self.assertRaisesRegex(ValueError, "source_known_mask"):
            nodes.project_vr180_to_rectilinear(*args, source_known_mask=torch.ones(1, 63, 64))
        self.assertTrue(torch.equal(image, before))

    def test_1024_and_2048_batches_and_no_retained_images(self):
        instance = nodes.VTSRectilinearToVR180()
        for size in (1024, 2048):
            image = torch.zeros((2, size, size, 3), dtype=torch.float32)
            output, known, unknown, _ = instance.project(
                images=image,
                output_width=size,
                output_height=size,
                horizontal_fov_degrees=90.0,
                yaw_degrees=0.0,
                pitch_degrees=0.0,
                roll_degrees=0.0,
                chunk_rows=256,
                frame_batch_size=1,
                sampling="nearest",
            )
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(torch.equal(known.bool(), ~unknown.bool()))
            del image, output, known, unknown
        self.assertEqual(vars(instance), {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_preserves_device_and_matches_cpu(self):
        image = smooth_image(batch=2)
        arguments = (128, 96, 90.0, 5.0, 0.0, 0.0, 17, 1, "bilinear")
        cpu = nodes.project_rectilinear_to_vr180(image, *arguments)
        gpu = nodes.project_rectilinear_to_vr180(image.cuda(), *arguments)
        self.assertTrue(gpu[0].is_cuda and gpu[1].is_cuda and gpu[2].is_cuda)
        self.assertTrue(torch.equal(cpu[1], gpu[1].cpu()))
        self.assertTrue(torch.allclose(cpu[0], gpu[0].cpu(), atol=2e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)

