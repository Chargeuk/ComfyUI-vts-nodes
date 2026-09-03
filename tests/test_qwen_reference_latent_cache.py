import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_Qwen_Reference_Latent_Cache.py"
)
SPEC = importlib.util.spec_from_file_location("vts_qwen_latent_cache_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeVAE:
    def __init__(self):
        self.encode_calls = []

    def encode(self, image):
        self.encode_calls.append(image.clone())
        return torch.full((1, 16, 128, 128), float(image.mean()))


class FakeClip:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, prompt, images, llama_template):
        self.tokenize_calls.append((prompt, images, llama_template))
        return {"prompt": prompt, "images": images}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 1), {"tokens": tokens}]]


def fake_conditioning_set_values(conditioning, values, append=False):
    result = [[conditioning[0][0], dict(conditioning[0][1])]]
    result[0][1].update(values)
    result[0][1]["append"] = append
    return result


class ReferenceLatentCacheTests(unittest.TestCase):
    def latent(self, value=1.0):
        return {
            "samples": torch.full((1, 16, 8, 8), value),
            "batch_index": [0],
        }

    def test_atomic_disk_round_trip_is_exact_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "first_latent_fisheye180.pt"
            audit = Path(directory) / "audit.jsonl"
            source = self.latent(0.125)
            saved = MODULE.VTSSaveReferenceLatent().save(
                source,
                str(cache),
                True,
                "first",
                "qwen-model",
                "qwen-vae",
                str(audit),
            )
            self.assertTrue(cache.is_file())
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            digest = json.loads(saved[2])["sha256"]

            with patch.object(MODULE.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")):
                loaded = MODULE.VTSLoadReferenceLatent().load(
                    str(cache), "intermediate", digest, "qwen-model", "qwen-vae", str(audit)
                )
            self.assertTrue(torch.equal(source["samples"], loaded[0]["samples"]))
            report = json.loads(loaded[2])
            self.assertTrue(report["loaded_from_disk"])
            self.assertEqual(report["sha256"], digest)
            events = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["save", "load"])

    def test_overwrite_false_protects_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "previous_latent_fisheye180.pt"
            node = MODULE.VTSSaveReferenceLatent()
            node.save(self.latent(1), str(cache), True)
            with self.assertRaises(FileExistsError):
                node.save(self.latent(2), str(cache), False)

    def test_qwen_node_vae_encodes_only_current_image_once(self):
        vae = FakeVAE()
        clip = FakeClip()
        current = torch.full((1, 64, 64, 3), 0.25)
        fixed_image = torch.full((1, 64, 64, 3), 0.5)
        previous_image = torch.full((1, 64, 64, 3), 0.75)
        fixed_latent = self.latent(2)
        previous_latent = self.latent(3)

        with patch.object(MODULE.node_helpers, "conditioning_set_values", fake_conditioning_set_values):
            result = MODULE.VTSQwenImageEditCachedReferences().encode(
                clip,
                vae,
                "repair edge",
                " ",
                current,
                fixed_image,
                fixed_latent,
                previous_image,
                previous_latent,
            )

        self.assertEqual(len(vae.encode_calls), 1)
        self.assertEqual(len(clip.tokenize_calls), 2)
        self.assertEqual(len(clip.tokenize_calls[0][1]), 3)
        refs = result[0][0][1]["reference_latents"]
        self.assertEqual(len(refs), 3)
        self.assertIs(refs[1], fixed_latent["samples"])
        self.assertIs(refs[2], previous_latent["samples"])
        report = json.loads(result[3])
        self.assertEqual(report["current_vae_encode_count"], 1)
        self.assertEqual(report["cached_reference_vae_encode_count"], 0)
        self.assertFalse(report["prompt_conditioning_cached"])

    def test_reference_image_and_latent_must_be_paired(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            MODULE.VTSQwenImageEditCachedReferences().encode(
                FakeClip(),
                FakeVAE(),
                "repair",
                " ",
                torch.zeros(1, 64, 64, 3),
                fixed_reference_image=torch.zeros(1, 64, 64, 3),
            )

    def test_qwen_encode_audit_records_reference_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            with patch.object(MODULE.node_helpers, "conditioning_set_values", fake_conditioning_set_values):
                MODULE.VTSQwenImageEditCachedReferences().encode(
                    FakeClip(),
                    FakeVAE(),
                    "repair",
                    " ",
                    torch.zeros(1, 64, 64, 3),
                    fixed_reference_image=torch.ones(1, 64, 64, 3),
                    fixed_reference_latent=self.latent(2),
                    audit_log_path=str(audit),
                    audit_label="frame_0002",
                )
            event = json.loads(audit.read_text().strip())
            self.assertEqual(event["event"], "qwen_cached_reference_encode")
            self.assertEqual(event["audit_label"], "frame_0002")
            self.assertEqual(event["cached_reference_vae_encode_count"], 0)

    def test_expected_node_mappings_are_exported(self):
        self.assertIn("VTS_Save_Reference_Latent", MODULE.NODE_CLASS_MAPPINGS)
        self.assertIn("VTS_Load_Reference_Latent", MODULE.NODE_CLASS_MAPPINGS)
        self.assertIn("VTS_Qwen_Image_Edit_Cached_References", MODULE.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
