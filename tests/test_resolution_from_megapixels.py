"""Small CPU checks against the installed ComfyUI V3 schema and sizing contract."""

import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("VTS_COMFY_ROOT", ROOT.parents[1]))
sys.path.insert(0, str(COMFY))
from comfy.cli_args import args
args.cpu = True
from comfy_api.latest import io

SPEC = importlib.util.spec_from_file_location("vts_resolution_test", ROOT / "py/VTS_Resolution_From_Megapixels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NODE = MODULE.VTS_Resolution_From_Megapixels


class ResolutionFromMegapixelsTests(unittest.TestCase):
    def test_dynamic_registration_and_real_v3_schema(self):
        key = "VTS Resolution From Megapixels"
        self.assertIs(MODULE.NODE_CLASS_MAPPINGS[key], NODE)
        self.assertEqual(MODULE.NODE_DISPLAY_NAME_MAPPINGS[key], "Resolution from Megapixels VTS")
        self.assertTrue(issubclass(NODE, io.ComfyNode))
        schema = NODE.GET_SCHEMA()
        self.assertEqual((schema.node_id, schema.display_name), (key, MODULE.NODE_DISPLAY_NAME_MAPPINGS[key]))
        self.assertEqual([item.id for item in schema.inputs], ["width", "height", "megapixels", "multiple", "priority"])
        self.assertEqual([item.get_io_type() for item in schema.inputs], ["INT", "INT", "FLOAT", "INT", "COMBO"])
        self.assertEqual(NODE.RETURN_TYPES, ["INT", "INT", "FLOAT"])
        self.assertEqual(NODE.RETURN_NAMES, ["width", "height", "actual_megapixels"])
        required = NODE.INPUT_TYPES()["required"]
        self.assertEqual(required["priority"][0], "COMBO")
        self.assertEqual(required["priority"][1]["options"], ["Aspect ratio", "Megapixels"])
        self.assertEqual(required["priority"][1]["default"], "Aspect ratio")
        for name in ("width", "height", "multiple"):
            self.assertEqual(required[name][1]["min"], 1)
        self.assertGreater(required["megapixels"][1]["min"], 0)
        self.assertIn("four", schema.description)
        self.assertTrue(all(item.tooltip for item in schema.inputs + schema.outputs))
        self.assertIsInstance(NODE.execute(1, 1, 1.0, 1), io.NodeOutput)

    def test_landscape_priority_examples_and_default(self):
        self.assertEqual(NODE.execute(1920, 1080, 3.0, 64).result, (2368, 1344, 3.182592))
        self.assertEqual(NODE.execute(1920, 1080, 3.0, 64, "Aspect ratio").result, (2368, 1344, 3.182592))
        self.assertEqual(NODE.execute(1920, 1080, 3.0, 64, "Megapixels").result, (2368, 1280, 3.03104))
        # Aspect priority can exceed the pixel budget when a larger neighbor
        # preserves the shape better than the apparently closer size.
        self.assertLess(abs((2368 / 1344) / (16 / 9) - 1), abs((2304 / 1280) / (16 / 9) - 1))

    def test_portrait_and_ratio_scale_do_not_change_target_size(self):
        for priority, expected in (("Aspect ratio", (1344, 2368, 3.182592)),
                                   ("Megapixels", (1280, 2368, 3.03104))):
            with self.subTest(priority=priority):
                self.assertEqual(NODE.execute(1080, 1920, 3.0, 64, priority).result, expected)
                self.assertEqual(NODE.execute(9, 16, 3.0, 64, priority).result, expected)

    def test_square_aspect_priority_stays_square(self):
        self.assertEqual(NODE.execute(1, 1, 1.5, 64).result, (1216, 1216, 1.478656))

    def test_exact_dimensions_and_one_megapixel_units(self):
        for priority in MODULE.PRIORITIES:
            with self.subTest(priority=priority):
                self.assertEqual(NODE.execute(1920, 1080, 2.0736, 8, priority).result, (1920, 1080, 2.0736))
                self.assertEqual(NODE.execute(1, 1, 1.0, 1, priority).result, (1000, 1000, 1.0))

    def test_exact_tie_prefers_smaller_area_deterministically(self):
        # Both square neighbors are equally far from 2.5 MP; exact shape wins
        # before size, and the smaller square resolves the remaining tie.
        for _ in range(3):
            self.assertEqual(NODE.execute(1, 1, 2.5, 1000).result, (1000, 1000, 1.0))
        self.assertEqual(NODE.execute(1, 1, 2.5, 1000, "Megapixels").result, (1000, 2000, 2.0))

    def test_local_neighborhood_prevents_distant_exact_ratio_search(self):
        width, height, _ = NODE.execute(16, 9, 3.0, 64).result
        self.assertIn(width, (2304, 2368))
        self.assertIn(height, (1280, 1344))
        self.assertNotEqual((width, height), (2048, 1152))

    def test_tiny_targets_still_return_one_positive_multiple(self):
        for target in (0.000001, 1e-300, 5e-324):
            for priority in MODULE.PRIORITIES:
                with self.subTest(target=target, priority=priority):
                    self.assertEqual(NODE.execute(16, 9, target, 64, priority).result, (64, 64, 0.004096))

    def test_large_multiple_and_integer_aspect_inputs(self):
        width, height, actual = NODE.execute(1, 1, 5e-324, 10**100).result
        self.assertEqual((width, height), (10**100, 10**100))
        self.assertEqual(actual, 1e194)
        self.assertEqual(NODE.execute(10**400, 10**400, 1.0, 1).result, (1000, 1000, 1.0))

    def test_finite_extremes_do_not_overflow_intermediate_pixel_math(self):
        width, height, actual = NODE.execute(1, 1, 1e308, 64).result
        self.assertEqual(width, height)
        self.assertEqual(width % 64, 0)
        self.assertTrue(math.isfinite(actual))
        self.assertAlmostEqual(actual / 1e308, 1.0)
        width, height, actual = NODE.execute(10**400, 1, 1.0, 1).result
        self.assertGreater(width, 10**200)
        self.assertEqual(height, 1)
        self.assertTrue(math.isfinite(actual))

    def test_outputs_are_scalars_and_divisible_for_both_priorities(self):
        for width, height, target, multiple in ((17, 11, 0.75, 1), (3, 4, 5.0, 7), (21, 9, 1.0, 4096)):
            for priority in MODULE.PRIORITIES:
                with self.subTest(width=width, height=height, multiple=multiple, priority=priority):
                    result = NODE.execute(width, height, target, multiple, priority).result
                    self.assertEqual(tuple(type(value) for value in result), (int, int, float))
                    self.assertGreaterEqual(result[0], multiple)
                    self.assertGreaterEqual(result[1], multiple)
                    self.assertEqual((result[0] % multiple, result[1] % multiple), (0, 0))
                    self.assertEqual(result[2], result[0] * result[1] / 1_000_000)

    def test_invalid_api_inputs_have_clear_errors(self):
        defaults = dict(width=1920, height=1080, megapixels=3.0, multiple=64, priority="Aspect ratio")
        for name in ("width", "height", "multiple"):
            for value in (0, -1, True, False, 1.5, "64", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name + " must be a positive integer"):
                        NODE.execute(**(defaults | {name: value}))
        for value in (0, -1.0, True, False, "3", None, float("nan"), float("inf"), -float("inf"), 10**400):
            with self.subTest(megapixels=value):
                with self.assertRaisesRegex(ValueError, "megapixels must be a positive finite number"):
                    NODE.execute(**(defaults | {"megapixels": value}))
        for value in ("Closest", "aspect ratio", None, True):
            with self.subTest(priority=value):
                with self.assertRaisesRegex(ValueError, "priority must be"):
                    NODE.execute(**(defaults | {"priority": value}))
        with self.assertRaisesRegex(ValueError, "actual_megapixels is too large"):
            NODE.execute(1, 1, 1.0, 10**200)


if __name__ == "__main__":
    unittest.main()
