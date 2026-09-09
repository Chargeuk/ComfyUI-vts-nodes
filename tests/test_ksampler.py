import importlib.util
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
import nodes

SPEC = importlib.util.spec_from_file_location("vts_ksampler_test_module", ROOT / "py/VTS_KSampler.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def conditioning(label):
    return [[torch.zeros(1, 2, 3), {"label": label}]]


class KSamplerTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.VTS_KSamplerAdvanced()
        self.calls = []
        self.parameters = dict(
            model=[object()], add_noise=["enable"], noise_seed=[10], seed_per_image=["fixed"],
            steps=[20], cfg=[7.0], sampler_name=["euler"], scheduler=["normal"],
            positive=[conditioning("positive")], negative=[conditioning("negative")],
            latent_image=[{"samples": torch.zeros(1, 4, 2, 2)}],
            start_at_step=[0], end_at_step=[20], return_with_leftover_noise=["disable"],
        )

    def capture(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ({**args[8], "samples": args[8]["samples"] + args[1]},)

    def run_node(self, **overrides):
        with patch.object(MODULE, "common_ksampler", side_effect=self.capture):
            return self.node.sample(**{**self.parameters, **overrides})[0]

    def test_schema_matches_builtin_advanced_plus_vts_seed_control(self):
        expected = nodes.KSamplerAdvanced.INPUT_TYPES()["required"]
        actual = self.node.INPUT_TYPES()["required"]
        self.assertEqual(set(actual), set(expected) | {"seed_per_image"})
        for name in expected.keys() - {"positive", "negative", "latent_image"}:
            self.assertEqual(actual[name], expected[name])
        self.assertTrue(self.node.INPUT_IS_LIST)
        self.assertEqual(self.node.RETURN_TYPES, ("LATENT",))
        self.assertIs(MODULE.NODE_CLASS_MAPPINGS["VTS KSampler (Advanced)"], type(self.node))

    def test_advanced_controls_match_builtin_for_all_noise_combinations(self):
        for add_noise in ["enable", "disable"]:
            for leftover in ["enable", "disable"]:
                with self.subTest(add_noise=add_noise, leftover=leftover):
                    self.calls.clear()
                    values = {key: value[0] for key, value in self.parameters.items() if key != "seed_per_image"}
                    values.update(add_noise=add_noise, return_with_leftover_noise=leftover, start_at_step=4, end_at_step=12)
                    with patch.object(nodes, "common_ksampler", side_effect=self.capture):
                        expected = nodes.KSamplerAdvanced().sample(**values)[0]
                    expected_args, expected_options = self.calls[-1]
                    actual = self.run_node(add_noise=[add_noise], return_with_leftover_noise=[leftover], start_at_step=[4], end_at_step=[12])
                    actual_args, actual_options = self.calls[-1]
                    self.assertEqual(actual_options, expected_options)
                    self.assertEqual(actual_args[1:6], expected_args[1:6])
                    self.assertTrue(torch.equal(actual["samples"], expected["samples"]))

    def test_all_seed_modes_and_reproducible_randomization(self):
        rng = random.Random(10)
        expectations = {"fixed": [10, 10, 10], "increment": [10, 11, 12], "decrement": [10, 9, 8],
                        "randomize": [rng.randint(0, 0xFFFFFFFFFFFFFFFF) for _ in range(3)]}
        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                self.calls.clear()
                global_state = random.getstate()
                self.run_node(seed_per_image=[mode], latent_image=[{"samples": torch.zeros(3, 4, 2, 2)}])
                self.assertEqual([args[1] for args, _ in self.calls], expected)
                self.assertEqual(random.getstate(), global_state)

    def test_conditioning_lists_repeat_last_and_preserve_masks(self):
        positive = [conditioning(str(i)) for i in range(3)]
        negative = [conditioning(str(i)) for i in range(2)]
        latent = {"samples": torch.arange(32).reshape(2, 4, 2, 2).float(),
                  "noise_mask": torch.stack([torch.zeros(1, 2, 2), torch.ones(1, 2, 2)]),
                  "batch_index": [7, 8], "metadata": "preserved"}
        original = latent["samples"].clone()
        result = self.run_node(positive=positive, negative=negative, latent_image=[latent])
        self.assertEqual(len(self.calls), 3)
        for index, (args, _) in enumerate(self.calls):
            self.assertIs(args[6], positive[index])
            self.assertIs(args[7], negative[min(index, 1)])
            self.assertEqual(args[8]["batch_index"], [7 + min(index, 1)])
        self.assertTrue(torch.equal(result["samples"][2], original[1] + 10))
        self.assertTrue(torch.equal(result["noise_mask"], latent["noise_mask"][[0, 1, 1]]))
        self.assertTrue(torch.equal(latent["samples"], original))
        self.assertEqual(result["metadata"], "preserved")
        self.assertEqual(result["batch_index"], [0, 1, 2])

    def test_longest_input_drives_count_and_video_time_dimension_is_preserved(self):
        for count, negatives in [(2, 4), (5, 1)]:
            with self.subTest(count=count, negatives=negatives):
                self.calls.clear()
                result = self.run_node(
                    latent_image=[{"samples": torch.zeros(count, 4, 3, 2, 2), "noise_mask": torch.ones(1, 1, 3, 2, 2)}],
                    negative=[conditioning(str(i)) for i in range(negatives)],
                )
                self.assertEqual(result["samples"].shape, (max(count, negatives), 4, 3, 2, 2))
                self.assertTrue(all(args[8]["samples"].shape[0] == 1 for args, _ in self.calls))

    def test_legacy_vts_sampler_keeps_denoise_and_seed_behavior(self):
        values = {key: value for key, value in self.parameters.items() if key in MODULE.VTS_KSampler.INPUT_TYPES()["required"]}
        values.update(seed=[10], denoise=[0.65], seed_per_image=["increment"],
                      latent_image=[{"samples": torch.zeros(2, 4, 2, 2)}])
        with patch.object(MODULE, "common_ksampler", side_effect=self.capture):
            output = MODULE.VTS_KSampler().sample(**values)[0]
        self.assertEqual([args[1] for args, _ in self.calls], [10, 11])
        self.assertEqual(self.calls[0][1], dict(denoise=.65, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False))
        self.assertEqual(output["samples"].shape, (2, 4, 2, 2))


if __name__ == "__main__":
    unittest.main()
