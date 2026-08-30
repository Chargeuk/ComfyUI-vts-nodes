"""End-to-end geometry contract tests for the portable VR180 workflows.

These tests deliberately exercise the public ComfyUI wrappers rather than
private grid builders.  They are small enough for routine CPU execution, but
cover the 16:9 and 4:3 source shapes used by the projection-LoRA dataset.
"""

from pathlib import Path
import importlib.util
import json
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "py"
UTILS_DIR = PY_DIR / "vtsUtils"
sys.path.insert(0, str(UTILS_DIR))


def load_nodes():
    spec = importlib.util.spec_from_file_location(
        "VTS_Rectilinear_To_VR180_contract_test",
        PY_DIR / "VTS_Rectilinear_To_VR180.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_nodes()


def smooth_image(height, width, batch=2):
    """Return a deterministic smooth NHWC image with no marker-magenta pixels."""

    y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack((0.8 * xx, 0.8 * yy, 0.2 + 0.3 * xx + 0.3 * yy), dim=-1)
    return image.unsqueeze(0).repeat(batch, 1, 1, 1)


class VR180RoundTripContractTests(unittest.TestCase):
    def _assert_round_trip(self, height, width, fov, yaw, pitch, roll):
        source = smooth_image(height, width)
        arguments = (192, 192, fov, yaw, pitch, roll, 31, 1, "bilinear")
        partial, forward_known, forward_unknown, forward_metadata = (
            nodes.project_rectilinear_to_vr180(source, *arguments)
        )
        reconstructed, reverse_known, reverse_unknown, reverse_metadata = (
            nodes.project_vr180_to_rectilinear(
                partial,
                width,
                height,
                fov,
                yaw,
                pitch,
                roll,
                29,
                1,
                "bilinear",
                forward_known,
            )
        )

        self.assertTrue(torch.equal(forward_known.bool(), ~forward_unknown.bool()))
        self.assertTrue(torch.equal(reverse_known.bool(), ~reverse_unknown.bool()))
        valid = reverse_known.bool().unsqueeze(-1).expand_as(reconstructed)
        self.assertGreater(float(valid.float().mean()), 0.88)
        self.assertLess(
            float(torch.mean(torch.abs(reconstructed[valid] - source[valid]))),
            0.025,
        )

        forward = json.loads(forward_metadata)
        reverse = json.loads(reverse_metadata)
        for name in (
            "horizontal_fov_degrees",
            "yaw_degrees",
            "pitch_degrees",
            "roll_degrees",
            "projection",
            "pixel_centres",
            "align_corners",
        ):
            self.assertEqual(forward[name], reverse[name])
        self.assertEqual(forward["input_size"], [width, height])
        self.assertEqual(forward["output_size"], [192, 192])
        self.assertEqual(reverse["input_size"], [192, 192])
        self.assertEqual(reverse["output_size"], [width, height])
        self.assertEqual(
            forward["known_pixels"],
            [int(value) for value in forward_known.sum(dim=(1, 2)).tolist()],
        )
        self.assertEqual(
            reverse["known_pixels"],
            [int(value) for value in reverse_known.sum(dim=(1, 2)).tolist()],
        )

    def test_16_by_9_round_trip_with_pose(self):
        self._assert_round_trip(72, 128, 82.0, 13.0, -5.0, 4.0)

    def test_4_by_3_round_trip_with_opposite_pose(self):
        self._assert_round_trip(96, 128, 88.0, -11.0, 6.0, -3.0)

    def test_unknown_marker_is_confined_to_unknown_geometry(self):
        source = smooth_image(72, 128, batch=3)
        partial, known, unknown, _ = nodes.project_rectilinear_to_vr180(
            source, 192, 192, 78.0, 9.0, 2.0, 0.0, 37, 2, "nearest"
        )
        marker = torch.tensor([1.0, 0.0, 1.0])
        marker_pixels = torch.all(partial == marker, dim=-1)
        self.assertTrue(torch.equal(marker_pixels, unknown.bool()))
        self.assertFalse(bool(marker_pixels[known.bool()].any()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
