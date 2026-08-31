from pathlib import Path
import importlib.util
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "py" / "VTS_Positioned_Seam_Strip_Mask.py"


def load_node_module():
    spec = importlib.util.spec_from_file_location(
        "VTS_Positioned_Seam_Strip_Mask_test",
        NODE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


node_module = load_node_module()


class PositionedSeamStripMaskTests(unittest.TestCase):
    def test_registration_and_schema(self):
        self.assertEqual(
            node_module.NODE_CLASS_MAPPINGS["VTSPositionedSeamStripMask"],
            node_module.VTSPositionedSeamStripMask,
        )
        required = node_module.VTSPositionedSeamStripMask.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required),
            ["width", "height", "seam_width", "feather", "strip_center_x"],
        )

    def test_centred_strip_matches_original_inward_feather_shape(self):
        (mask,) = node_module.positioned_seam_strip_mask(
            width=32,
            height=4,
            seam_width=8,
            feather=2,
            strip_center_x=16,
        )
        expected_row = torch.zeros(32)
        expected_row[12:20] = torch.tensor(
            [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]
        )
        self.assertEqual(mask.shape, (1, 4, 32))
        self.assertTrue(torch.allclose(mask[0, 0], expected_row))
        self.assertTrue(torch.equal(mask[0, 0], mask[0, 3]))

    def test_strip_uses_requested_pixel_as_its_centre(self):
        (mask,) = node_module.positioned_seam_strip_mask(
            width=32,
            height=2,
            seam_width=8,
            feather=0,
            strip_center_x=7,
        )
        expected = torch.zeros(32)
        expected[3:11] = 1.0
        self.assertTrue(torch.equal(mask[0, 0], expected))

    def test_edge_clipping_preserves_the_conceptual_strip_profile(self):
        (left_mask,) = node_module.positioned_seam_strip_mask(
            width=16,
            height=1,
            seam_width=8,
            feather=2,
            strip_center_x=0,
        )
        expected_left = torch.zeros(16)
        expected_left[:4] = torch.tensor([1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(torch.allclose(left_mask[0, 0], expected_left))

        (right_mask,) = node_module.positioned_seam_strip_mask(
            width=16,
            height=1,
            seam_width=8,
            feather=2,
            strip_center_x=15,
        )
        expected_right = torch.zeros(16)
        expected_right[11:] = torch.tensor(
            [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0]
        )
        self.assertTrue(torch.allclose(right_mask[0, 0], expected_right))

    def test_invalid_centre_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "identify a pixel"):
            node_module.positioned_seam_strip_mask(32, 16, 8, 2, 32)
        with self.assertRaisesRegex(ValueError, "identify a pixel"):
            node_module.positioned_seam_strip_mask(32, 16, 8, 2, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
