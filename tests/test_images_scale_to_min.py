import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (Path(__file__).parents[1] / "py" /
               "VTS_Images_ScaleToMin.py")
SPEC = importlib.util.spec_from_file_location(
    "vts_images_scale_to_min_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ScaleToMinDimensionTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.VTS_Images_ScaleToMin()

    def dimensions(self, width, height, small, large, divisible=2,
                   scale_type="small"):
        return self.node._calculate_target_dimensions(
            width, height, small, large, divisible, scale_type)

    def test_reversed_limits_are_normalized_for_every_scale_type(self):
        for scale_type in ("small", "large", "max"):
            with self.subTest(scale_type=scale_type):
                normal = self.dimensions(
                    1920, 1080, 1080, 1920, scale_type=scale_type)
                reversed_limits = self.dimensions(
                    1920, 1080, 1920, 1080, scale_type=scale_type)
                self.assertEqual(reversed_limits, normal)

    def test_min_snaps_landscape_just_below_requested_aspect(self):
        self.assertEqual(
            self.dimensions(1918, 1080, 1080, 1920),
            (1920, 1080),
        )

    def test_min_snaps_landscape_just_above_requested_aspect(self):
        self.assertEqual(
            self.dimensions(1922, 1080, 1080, 1920),
            (1920, 1080),
        )

    def test_min_snaps_portrait_near_requested_aspect(self):
        self.assertEqual(
            self.dimensions(1080, 1918, 1080, 1920),
            (1080, 1920),
        )

    def test_min_preserves_a_meaningfully_different_aspect(self):
        self.assertEqual(
            self.dimensions(2048, 1080, 1080, 1920),
            (1920, 1012),
        )

    def test_divisibility_is_applied_after_dimension_selection(self):
        self.assertEqual(
            self.dimensions(2048, 1080, 1079, 1919, divisible=8),
            (1912, 1008),
        )


if __name__ == "__main__":
    unittest.main()
