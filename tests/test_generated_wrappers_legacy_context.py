"""Existing wrappers survive extra upstream controls and engine context inputs."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "py" / "VTS_Generated_Wrappers.py"
SPEC = importlib.util.spec_from_file_location("vts_legacy_context_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
with mock.patch("server.PromptServer.instance", None, create=True):
    SPEC.loader.exec_module(MODULE)

class ContextNode:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    calls = []
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)},
                "optional": {f"option_{i}": ("INT", {"default": 0}) for i in range(16)},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}
    def execute(self, images, dynprompt=None, unique_id=None, **options):
        self.calls.append((images, dynprompt, unique_id, options))
        return {"ui": {"text": [unique_id]}, "result": (unique_id,)}

class LegacyContextTests(unittest.TestCase):
    def test_extended_inputs_and_hidden_context_preserve_execution(self):
        spec = MODULE._build_legacy_wrapper_spec("ContextNode", ContextNode, {})
        self.assertIsNotNone(spec)
        inputs = MODULE._build_input_types(spec, "ContextNode")
        self.assertEqual(inputs["hidden"], ContextNode.INPUT_TYPES()["hidden"])
        images, context = object(), object()
        result = MODULE._execute_wrapped_node(spec, {
            "images": images, "dynprompt": context, "unique_id": "42", "option_15": 7})
        self.assertEqual(result, {"ui": {"text": ["42"]}, "result": ("42",)})
        self.assertIs(ContextNode.calls[-1][1], context)
        self.assertEqual(ContextNode.calls[-1][3], {"option_15": 7})
    def test_unknown_hidden_types_stay_excluded(self):
        with mock.patch.object(ContextNode, "INPUT_TYPES", return_value={
                "required": {"images": ("IMAGE",)}, "hidden": {"token": "AUTH_TOKEN"}}):
            self.assertIsNone(MODULE._build_legacy_wrapper_spec("ContextNode", ContextNode, {}))

if __name__ == "__main__":
    unittest.main()
