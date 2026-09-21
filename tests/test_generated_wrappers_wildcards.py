"""Third-party wildcard sockets must not break or select disk adapters."""
import unittest

from test_disk_latent_wrappers import wrappers
from vts_audio_nodes import audio_ports
from vts_tooltips import socket_help


class AlwaysEqualProxy(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class WildcardNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (AlwaysEqualProxy("*"),)}}

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    FUNCTION = "execute"

    def execute(self, anything):
        return (anything,)


class WildcardWrapperTests(unittest.TestCase):
    def test_wildcard_tooltips_are_not_media_tooltips(self):
        for output in (False, True):
            self.assertEqual(socket_help(AlwaysEqualProxy("*"), output, True, True), "")
        self.assertIn("DiskLatent", socket_help(AlwaysEqualProxy("LATENT"), True, True))

    def test_wildcard_only_node_does_not_get_disk_adapters(self):
        self.assertEqual(audio_ports(WildcardNode), ([], []))
        self.assertIsNone(wrappers._build_legacy_wrapper_spec("Wildcard", WildcardNode, {}))

    def test_mixed_node_preserves_wildcard_and_wraps_real_audio(self):
        class MixedNode(WildcardNode):
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"anything": (AlwaysEqualProxy("*"),), "audio": ("AUDIO",)}}

            RETURN_TYPES = (AlwaysEqualProxy("*"), "AUDIO")

            def execute(self, anything, audio):
                return anything, audio

        spec = wrappers._build_legacy_wrapper_spec("Mixed", MixedNode, {})
        self.assertEqual(spec["image_output_indexes"], [])
        self.assertEqual(spec["latent_output_indexes"], [])
        self.assertEqual(spec["image_input_names"], set())
        self.assertEqual(spec["latent_input_names"], ())
        wrapper, _ = wrappers._create_wrapper_class(spec)
        self.assertEqual(audio_ports(wrapper), (["audio"], [1]))
        schema = wrapper.INPUT_TYPES()
        self.assertIsInstance(schema["required"]["anything"][0], AlwaysEqualProxy)
        self.assertIsInstance(wrapper.RETURN_TYPES[0], AlwaysEqualProxy)
        self.assertNotIn("tooltip", schema["required"]["anything"][1] if len(schema["required"]["anything"]) > 1 else {})
        self.assertEqual(wrapper.OUTPUT_TOOLTIPS[0], "")
        self.assertIn("DiskAudio", wrapper.OUTPUT_TOOLTIPS[1])
        anything, audio = object(), {"waveform": object(), "sample_rate": 44100}
        result = getattr(wrapper(), wrapper.FUNCTION)(anything=anything, audio=audio)
        self.assertIs(result[0], anything)
        self.assertIs(result[1], audio)
