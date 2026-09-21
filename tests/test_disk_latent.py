import gc
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "py" / "vtsUtils"))
from vts_disk_latent import DiskLatent, DiskNestedTensorInfo, DiskTensorInfo, MAGIC, save_latent
from vts_latent_nodes import disk_latent_node


def import_node(filename):
    spec = importlib.util.spec_from_file_location("disk_test_" + filename, ROOT / "py" / (filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@disk_latent_node(inputs=("latent",), outputs=(0,), prefix="Test Native")
class NativeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, latent):
        assert isinstance(latent, dict)
        assert isinstance(latent["samples"], torch.Tensor)
        return ({**latent, "samples": latent["samples"] + 1},)


@disk_latent_node(inputs=("latent",), outputs=(0,), prefix="Test V3")
class V3Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3DiskTest", inputs=[io.Latent.Input("latent")], outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, latent):
        return io.NodeOutput({**latent, "samples": latent["samples"] * 2})


class DiskLatentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def save(self, value, **kwargs):
        return save_latent(value, output_dir=self.temp.name, **kwargs)

    def assertBytesEqual(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))

    def test_exact_values_dtypes_masks_and_structure(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                samples = torch.arange(120).reshape(2, 3, 4, 5).to(dtype).transpose(2, 3)
                latent = {"samples": samples, "noise_mask": torch.ones(2, 1, 8, 8),
                          "batch_index": [2, 4], "extra": (None, True, {"scalar": torch.tensor(7.), "label": "hello"})}
                disk = self.save(latent)
                loaded = disk.materialize()
                self.assertBytesEqual(samples, loaded["samples"])
                self.assertBytesEqual(latent["noise_mask"], loaded["noise_mask"])
                self.assertEqual(loaded["batch_index"], [2, 4])
                self.assertIsInstance(loaded["extra"], tuple)
                self.assertBytesEqual(loaded["extra"][2]["scalar"], torch.tensor(7.))

    def test_special_float_bits_and_empty_tensors(self):
        samples = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"), -0.73, 2.18])
        disk = self.save({"samples": samples, "empty": torch.empty(0, 4)})
        loaded = disk.materialize()
        self.assertBytesEqual(samples, loaded["samples"])
        self.assertEqual(loaded["empty"].shape, (0, 4))

    def test_nested_h3_and_shared_tensor_references(self):
        video = torch.randn(1, 24, 3, 4, 4, dtype=torch.bfloat16)
        audio = torch.randn(1, 8, 20)
        disk = self.save({"samples": NestedTensor([video, audio]), "duplicate": video})
        self.assertIsInstance(disk["samples"], DiskNestedTensorInfo)
        self.assertEqual(disk["samples"].unbind()[1].shape, audio.shape)
        loaded = disk.materialize()
        self.assertIsInstance(loaded["samples"], NestedTensor)
        self.assertBytesEqual(video, loaded["samples"].tensors[0])
        self.assertBytesEqual(audio, loaded["samples"].tensors[1])
        self.assertIs(loaded["duplicate"], loaded["samples"].tensors[0])

    def test_metadata_does_not_decompress_or_retain_tensor(self):
        value = torch.zeros(1, 16, 4, 8)
        reference = weakref.ref(value)
        disk = self.save({"samples": value, "batch_index": [1]})
        del value
        gc.collect()
        self.assertIsNone(reference())
        with mock.patch("vts_disk_latent.zstandard.ZstdDecompressor", side_effect=AssertionError("decompressed")):
            reopened = DiskLatent(disk.path)
            samples = reopened["samples"]
            self.assertIsInstance(samples, DiskTensorInfo)
            self.assertEqual(samples.size(), (1, 16, 4, 8))
            self.assertEqual(samples.numel(), 512)
            self.assertEqual(samples.element_size(), 4)
            self.assertEqual(samples.dim(), 4)
            self.assertEqual(str(samples.device), "cpu")
            self.assertEqual(reopened.tensor_size_bytes, 2048)
            self.assertIsInstance(reopened.copy(), dict)
            self.assertEqual(reopened.clone().path, reopened.path)
            changed = reopened["batch_index"]
            changed.append(99)
            self.assertEqual(reopened["batch_index"], [1])
            with self.assertRaisesRegex(TypeError, "materialize"):
                samples + 1
            with self.assertRaisesRegex(TypeError, "materialize"):
                torch.sin(samples)

    def test_deferred_device_and_dtype(self):
        disk = self.save({"samples": torch.ones(2, dtype=torch.bfloat16)})
        gpu = disk.to("cuda:0", dtype=torch.float32)
        self.assertEqual(str(gpu["samples"].device), "cuda:0")
        self.assertEqual(str(gpu["samples"].original_device), "cpu")
        self.assertEqual(gpu["samples"].dtype, torch.float32)
        self.assertEqual(disk["samples"].dtype, torch.bfloat16)
        self.assertEqual(gpu.materialize(device="cpu")["samples"].dtype, torch.float32)

    def test_unique_names_and_list_mapped_naming(self):
        with mock.patch("vtsUtils.get_executing_context", return_value=mock.Mock(list_index=3)):
            first = self.save({"samples": torch.ones(1)}, prefix="VTS_Test", start_sequence=4)
            second = self.save({"samples": torch.zeros(1)}, prefix="VTS_Test", start_sequence=4)
        self.assertNotEqual(first.path, second.path)
        self.assertIn("VTS_Test_list_000003_000004_", first.path)
        self.assertEqual(first.materialize()["samples"].item(), 1)

    def test_reject_unsupported_values_and_path_prefix(self):
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            self.save({"samples": torch.ones(1), "object": object()})
        for prefix in ("../outside", "/absolute", "a\\b", ""):
            with self.assertRaises(ValueError):
                self.save({"samples": torch.ones(1)}, prefix=prefix)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_atomic_failure_leaves_no_files(self):
        with mock.patch("vts_disk_latent.zstandard.ZstdCompressor", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                self.save({"samples": torch.ones(1)})
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_corruption_and_manifest_replacement_detected(self):
        disk = self.save({"samples": torch.ones(20)})
        original = Path(disk.path).read_bytes()
        Path(disk.path).write_bytes(original[:-3])
        with self.assertRaises(Exception):
            disk.materialize()
        size = struct.unpack("<Q", original[8:16])[0]
        manifest = json.loads(original[16:16 + size])
        manifest["tensors"]["tensor_0"]["device"] = "cuda:0"
        updated = json.dumps(manifest).encode()
        Path(disk.path).write_bytes(MAGIC + struct.pack("<Q", len(updated)) + updated + original[16 + size:])
        with self.assertRaisesRegex(ValueError, "changed"):
            disk.materialize()

    def test_native_controls_old_calls_and_native_inputs(self):
        latent = {"samples": torch.zeros(1)}
        disk = self.save(latent)
        self.assertTrue(torch.equal(NativeNode().execute(latent)[0]["samples"], torch.ones(1)))
        self.assertTrue(torch.equal(NativeNode().execute(disk)[0]["samples"], torch.ones(1)))
        result = NativeNode().execute(disk, latent_return_type="Input", latent_output_dir=self.temp.name)[0]
        self.assertIsInstance(result, DiskLatent)
        self.assertEqual(result.materialize()["samples"].item(), 1)
        schema = NativeNode.INPUT_TYPES()
        self.assertNotIn("latent_return_type", schema["required"])
        self.assertEqual(schema["optional"]["latent_output_dir"][1]["default"], "./tmp/disklatents")

    def test_v3_node_output_and_schema(self):
        disk = self.save({"samples": torch.ones(1)})
        result = V3Node.execute(disk, latent_return_type="DiskLatent", latent_output_dir=self.temp.name)
        self.assertIsInstance(result, io.NodeOutput)
        self.assertIsInstance(result.args[0], DiskLatent)
        self.assertEqual(result.args[0].materialize()["samples"].item(), 2)
        self.assertIn("latent_return_type", [v.id for v in V3Node.define_schema().inputs])

    def test_node_does_not_retain_materialized_or_output_tensors(self):
        references = []

        @disk_latent_node(inputs=("latent",), outputs=(0,), prefix="Lifetime")
        class LifetimeNode(NativeNode):
            def execute(self, latent):
                output = latent["samples"] + 1
                references.extend([weakref.ref(latent["samples"]), weakref.ref(output)])
                return ({"samples": output},)

        disk = self.save({"samples": torch.ones(16)})
        result = LifetimeNode().execute(disk, latent_return_type="DiskLatent", latent_output_dir=self.temp.name)[0]
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(result.materialize()["samples"].mean().item(), 2)

    def test_real_latent_list_and_batch_nodes(self):
        module = import_node("VTS_Latent_List_Batch")
        disk = self.save({"samples": torch.arange(8.).reshape(2, 1, 2, 2), "batch_index": [4, 7]})
        parts = module.VTS_Latent_Batch_To_List().to_list(
            disk, latent_return_type="Input", latent_output_dir=self.temp.name)[0]
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(isinstance(v, DiskLatent) for v in parts))
        result = module.VTS_Latent_List_To_Batch().to_batch(
            parts, latent_return_type=["DiskLatent"], latent_output_dir=[self.temp.name])[0]
        self.assertBytesEqual(result.materialize()["samples"], disk.materialize()["samples"])
        self.assertEqual(result["batch_index"], [4, 7])


if __name__ == "__main__":
    unittest.main()
