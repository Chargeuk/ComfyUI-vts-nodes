from pathlib import Path
import importlib.util
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "py"
UTILS_DIR = PY_DIR / "vtsUtils"
sys.path.insert(0, str(UTILS_DIR))

import vr180_outpaint as projection


def load_nodes():
    spec = importlib.util.spec_from_file_location(
        "VTS_VR180_To_ERP_Outpaint_test",
        PY_DIR / "VTS_VR180_To_ERP_Outpaint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_nodes()


def smooth_square(batch=1, size=64, device="cpu"):
    y = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / size
    x = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) / size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack((xx, yy, 0.25 * xx + 0.75 * yy), dim=-1)
    return image.unsqueeze(0).repeat(batch, 1, 1, 1)


class FakeDiskImage:
    def __init__(self, images, start_sequence=7):
        self.images = images
        self.number_of_images = int(images.shape[0])
        self.start_sequence = int(start_sequence)
        self.shape = images.shape
        self.calls = []

    def load_images(self, start_sequence, count):
        self.calls.append((int(start_sequence), int(count)))
        start = int(start_sequence) - self.start_sequence
        return self.images[start : start + int(count)]


class VR180OutpaintCoreTests(unittest.TestCase):
    def test_ideal_half_erp_is_exact_central_half(self):
        source = smooth_square(size=64).movedim(-1, 1)
        result = projection.vr180_square_to_full_erp(
            source,
            (64, 128),
            projection_mode=projection.HALF_ERP_IDEAL,
            unknown_color=(0.0, 1.0, 0.0),
            mode="bilinear",
        )
        self.assertTrue(result.known_mask[:, :, :, 32:96].all())
        self.assertFalse(result.known_mask[:, :, :, :32].any())
        self.assertFalse(result.known_mask[:, :, :, 96:].any())
        self.assertTrue(torch.allclose(result.image[:, :, :, 32:96], source, atol=2.0e-6))
        unknown = result.unknown_mask.expand_as(result.image)
        green = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).expand_as(result.image)
        self.assertTrue(torch.equal(result.image[unknown], green[unknown]))

    def test_production_preset_is_wider_and_does_not_claim_poles(self):
        source = smooth_square(size=128).movedim(-1, 1)
        ideal = projection.vr180_square_to_full_erp(
            source, (128, 256), projection_mode=projection.HALF_ERP_IDEAL
        )
        fitted = projection.vr180_square_to_full_erp(
            source, (128, 256), projection_mode=projection.HALF_ERP_PRODUCTION
        )
        equator = 64
        self.assertGreater(
            int(fitted.known_mask[0, 0, equator].sum()),
            int(ideal.known_mask[0, 0, equator].sum()),
        )
        self.assertFalse(bool(fitted.known_mask[0, 0, 0, 128]))
        self.assertTrue(bool(ideal.known_mask[0, 0, 0, 128]))

    def test_equidistant_fisheye_uses_circle_and_marks_back_unknown(self):
        source = smooth_square(size=96).movedim(-1, 1)
        result = projection.vr180_square_to_full_erp(
            source,
            (64, 128),
            projection_mode=projection.FISHEYE_IDEAL,
            unknown_color=(1.0, 0.0, 1.0),
        )
        self.assertTrue(bool(result.known_mask[0, 0, 32, 64]))
        self.assertFalse(bool(result.known_mask[0, 0, 32, 0]))
        self.assertTrue(torch.equal(result.unknown_mask, ~result.known_mask))
        fraction = float(result.known_mask.float().mean())
        self.assertGreater(fraction, 0.45)
        self.assertLess(fraction, 0.51)

    def test_custom_fisheye_requires_equal_axes(self):
        with self.assertRaisesRegex(ValueError, "equal horizontal and vertical"):
            projection.vr180_square_to_full_erp(
                smooth_square(size=32).movedim(-1, 1),
                (32, 64),
                projection_mode=projection.FISHEYE_CUSTOM,
                custom_horizontal_fov_degrees=180.0,
                custom_vertical_fov_degrees=170.0,
            )

    def test_invalid_source_and_canvas_fail_early(self):
        with self.assertRaisesRegex(ValueError, "square"):
            projection.vr180_square_to_full_erp(
                torch.zeros((1, 3, 32, 48)), (32, 64)
            )
        with self.assertRaisesRegex(ValueError, "2:1"):
            projection.vr180_square_to_full_erp(
                torch.zeros((1, 3, 32, 32)), (32, 65)
            )


class VR180OutpaintNodeTests(unittest.TestCase):
    def arguments(self, source):
        return dict(
            source=source,
            projection_mode=projection.HALF_ERP_IDEAL,
            output_width=64,
            output_height=32,
            custom_horizontal_fov_degrees=180.0,
            custom_vertical_fov_degrees=180.0,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            fill_color="#00ff00",
            chunk_rows=11,
            frame_batch_size=2,
            sampling="bilinear",
        )

    def test_registration_and_tensor_outputs(self):
        self.assertEqual(
            nodes.NODE_CLASS_MAPPINGS["VTSVR180SquareToERPOutpaint"],
            nodes.VTSVR180SquareToERPOutpaint,
        )
        inputs = nodes.VTSVR180SquareToERPOutpaint.INPUT_TYPES()["required"]
        self.assertIn("projection_mode", inputs)
        self.assertIn(projection.HALF_ERP_PRODUCTION, inputs["projection_mode"][0])
        output, known, unknown = nodes.project_square_vr180_to_erp(
            **self.arguments(smooth_square(batch=3, size=32))
        )
        self.assertEqual(output.shape, (3, 32, 64, 3))
        self.assertEqual(known.shape, (3, 32, 64))
        self.assertTrue(torch.equal(known.bool(), ~unknown.bool()))

    def test_disk_image_is_loaded_in_bounded_batches(self):
        disk = FakeDiskImage(smooth_square(batch=5, size=32), start_sequence=7)
        output, known, unknown = nodes.project_square_vr180_to_erp(
            **self.arguments(disk)
        )
        self.assertEqual(output.shape, (5, 32, 64, 3))
        self.assertEqual(known.shape, (5, 32, 64))
        self.assertEqual(unknown.shape, (5, 32, 64))
        self.assertEqual(disk.calls, [(7, 2), (9, 2), (11, 1)])

    def test_invalid_fill_colour_is_rejected(self):
        arguments = self.arguments(smooth_square(size=32))
        arguments["fill_color"] = "green"
        with self.assertRaisesRegex(ValueError, "fill_color"):
            nodes.project_square_vr180_to_erp(**arguments)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_tensor_stays_on_cuda(self):
        output, known, unknown = nodes.project_square_vr180_to_erp(
            **self.arguments(smooth_square(batch=2, size=32, device="cuda"))
        )
        self.assertTrue(output.is_cuda and known.is_cuda and unknown.is_cuda)


if __name__ == "__main__":
    unittest.main(verbosity=2)

