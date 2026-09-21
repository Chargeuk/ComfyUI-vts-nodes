import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from comfy_api.latest import io

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "py" / "vtsUtils"))
from vts_disk_latent import DiskLatent, save_latent
from vtsUtils import DiskImage

spec = importlib.util.spec_from_file_location("vts_disk_wrapper_test", ROOT / "py" / "VTS_Generated_Wrappers.py")
wrappers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrappers
with mock.patch("server.PromptServer.instance", None, create=True):
    spec.loader.exec_module(wrappers)


class MixedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "text")
    FUNCTION = "execute"

    def execute(self, latent, image):
        assert type(latent) is dict
        assert isinstance(latent["samples"], torch.Tensor)
        assert isinstance(image, torch.Tensor)
        return {"ui": {"text": ["preserved"]}, "result": ({**latent, "samples": latent["samples"] + 2}, image, "ok")}


class LatentOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("first", "second")
    FUNCTION = "execute"

    def execute(self, latent):
        return latent, {"samples": latent["samples"] * 3}


class DynamicLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DynamicLatentTest", inputs=[
            io.Autogrow.Input("latents", template=io.Autogrow.TemplatePrefix(io.Latent.Input("latent"), prefix="latent", min=1, max=3))
        ], outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, latents):
        assert all(isinstance(v, dict) for v in latents.values())
        return io.NodeOutput({"samples": sum(v["samples"] for v in latents.values())})


class DiskWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.disk = save_latent({"samples": torch.ones(1, 4, 2, 2)}, output_dir=self.temp.name)

    def test_latent_only_nodes_and_multiple_output_names(self):
        spec = wrappers._build_legacy_wrapper_spec("LatentOnly", LatentOnly, {})
        self.assertIsNotNone(spec)
        result = wrappers._execute_wrapped_node(spec, {"latent": self.disk,
            "vts_latent_return_type": "DiskLatent", "vts_latent_output_dir": self.temp.name,
            "vts_latent_prefix": "two"})
        self.assertTrue(all(isinstance(v, DiskLatent) for v in result))
        self.assertIn("two_first_", result[0].path)
        self.assertIn("two_second_", result[1].path)
        self.assertEqual(result[1].materialize()["samples"].mean().item(), 3)

    def test_loader_fingerprint_and_terminal_node_contract(self):
        class FileNode(LatentOnly):
            OUTPUT_NODE = True

            @classmethod
            def IS_CHANGED(cls, latent):
                return latent

            @classmethod
            def VALIDATE_INPUTS(cls, latent):
                return bool(latent)

        spec = wrappers._build_legacy_wrapper_spec("FileNode", FileNode, {})
        wrapper, _ = wrappers._create_wrapper_class(spec)
        self.assertTrue(wrapper.OUTPUT_NODE)
        self.assertEqual(wrapper.IS_CHANGED(latent="changed", vts_latent_return_type="DiskLatent"), "changed")
        self.assertTrue(wrapper.VALIDATE_INPUTS(latent="path"))

    def test_independent_image_latent_controls_and_ui(self):
        spec = wrappers._build_legacy_wrapper_spec("MixedNode", MixedNode, {})
        inputs = wrappers._build_input_types(spec, "VTS Mixed Wrapper")
        self.assertIn("vts_return_type", inputs["required"])
        self.assertIn("vts_latent_return_type", inputs["optional"])
        image = torch.ones(1, 4, 4, 3)
        result = wrappers._execute_wrapped_node(spec, {"latent": self.disk, "image": image,
            "vts_return_type": "Tensor", "vts_latent_return_type": "DiskLatent",
            "vts_latent_output_dir": self.temp.name})
        self.assertEqual(result["ui"], {"text": ["preserved"]})
        self.assertIsInstance(result["result"][0], DiskLatent)
        self.assertIs(result["result"][1], image)
        self.assertEqual(result["result"][2], "ok")
        result = wrappers._execute_wrapped_node(spec, {"latent": self.disk, "image": image,
            "vts_return_type": "DiskImage", "vts_output_dir": self.temp.name, "vts_format": "png",
            "vts_latent_return_type": "Tensor"})
        self.assertIsInstance(result["result"][0], dict)
        self.assertIsInstance(result["result"][1], DiskImage)

    def test_v3_autogrow_latents(self):
        spec = wrappers._build_v3_wrapper_spec("DynamicLatentTest", DynamicLatent, {})
        self.assertIsNotNone(spec)
        wrapper, _ = wrappers._create_wrapper_class(spec)
        schema = wrapper.define_schema()
        self.assertIn("vts_latent_device_policy", [v.id for v in schema.inputs])
        result = wrapper.execute(latents={"latent0": self.disk, "latent1": self.disk},
            vts_latent_return_type="Input", vts_latent_output_dir=self.temp.name)
        self.assertIsInstance(result, io.NodeOutput)
        self.assertEqual(result.args[0].materialize()["samples"].mean().item(), 2)

    def test_generic_image_wrapper_accepts_disk_latent(self):
        spec = importlib.util.spec_from_file_location("vts_generic_disk_test", ROOT / "py" / "VTS_Generic_Image_Wrapper.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Decode:
            def decode(self, samples):
                assert isinstance(samples, dict)
                return (samples["samples"],)

        node_spec = {"image_input_names": set(), "image_input_count": 0,
                     "all_input_names": ["samples"], "class": Decode, "function_name": "decode"}
        result = module._execute_wrapped_node(node_spec, "Tensor", "test", 0, self.temp.name,
            "png", 1, 1, 95, {"wrapped__samples": self.disk})
        self.assertTrue(torch.equal(result[0], torch.ones(1, 4, 2, 2)))


if __name__ == "__main__":
    unittest.main()
