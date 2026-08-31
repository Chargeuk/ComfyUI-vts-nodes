import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from comfy.cli_args import args
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo


args.cpu = True

MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_Generated_Wrappers.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vts_generated_wrappers_autogrow_test_module", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
with mock.patch.object(threading.Thread, "start", autospec=True):
    SPEC.loader.exec_module(MODULE)


class FakeAutogrowNode(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FakeAutogrowNode",
            display_name="Fake Autogrow Node",
            category="tests",
            inputs=[
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, ref_images=None, ref_audios=None):
        cls.calls.append((ref_images, ref_audios))
        return io.NodeOutput("conditioning", {"samples": "latent"})


class FakeAutogrowImageOutputNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FakeAutogrowImageOutputNode",
            inputs=[
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=2,
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images=None):
        return io.NodeOutput(images["image_0"], ui={"source": "fake"})


class GeneratedWrapperAutogrowTests(unittest.TestCase):
    def setUp(self):
        FakeAutogrowNode.calls.clear()

    def test_native_h3_reference_node_becomes_eligible(self):
        spec = MODULE._build_v3_wrapper_spec(
            "MiniMaxH3ReferenceToVideo",
            MiniMaxH3ReferenceToVideo,
            {},
        )

        self.assertIsNotNone(spec)
        self.assertEqual(spec["schema_style"], "v3_dynamic")
        self.assertEqual(
            spec["image_input_names"],
            {"ref_images", "ref_videos"},
        )
        self.assertEqual(spec["return_types"], ("CONDITIONING", "LATENT"))

        wrapper, display_name = MODULE._create_wrapper_class(spec)
        schema = wrapper.GET_SCHEMA()
        self.assertEqual(display_name, "VTS MiniMax H3 Reference to Video Wrapper")
        self.assertEqual(
            schema.node_id,
            "VTSWrapper_comfy_extras_MiniMaxH3ReferenceToVideo",
        )
        self.assertEqual(
            wrapper.RELATIVE_PYTHON_MODULE,
            "custom_nodes.ComfyUI-vts-nodes",
        )
        autogrow = {item.id: item for item in schema.inputs if isinstance(item, io.Autogrow.Input)}
        self.assertEqual(autogrow["ref_images"].template.prefix, "ref_image_")
        self.assertEqual(autogrow["ref_images"].template.max, 9)
        self.assertEqual(autogrow["ref_videos"].template.prefix, "ref_video_")
        self.assertEqual(autogrow["ref_video_audios"].template.prefix, "ref_video_audio_")
        self.assertEqual(autogrow["ref_audios"].template.prefix, "ref_audio_")

    def test_native_h3_wrapper_is_emitted_by_registration_pass(self):
        node_id = "VTSWrapper_comfy_extras_MiniMaxH3ReferenceToVideo"
        with mock.patch.dict(
            MODULE.core_nodes.NODE_CLASS_MAPPINGS,
            {"MiniMaxH3ReferenceToVideo": MiniMaxH3ReferenceToVideo},
            clear=True,
        ), mock.patch.dict(
            MODULE.core_nodes.NODE_DISPLAY_NAME_MAPPINGS,
            {"MiniMaxH3ReferenceToVideo": "MiniMax H3 Reference to Video"},
            clear=True,
        ):
            mappings, display_mappings = MODULE._build_generated_mappings()

        self.assertIn(node_id, mappings)
        self.assertEqual(
            display_mappings[node_id],
            "VTS MiniMax H3 Reference to Video Wrapper",
        )
        self.assertEqual(mappings[node_id].GET_SCHEMA().node_id, node_id)

    def test_comfy_loader_relative_module_metadata_is_accepted(self):
        filesystem_module = (
            "/home/d_a_s/code/comfyui/comfy_extras/nodes_minimax_h3"
        )
        relative_module = "comfy_extras.nodes_minimax_h3"
        with mock.patch.object(
            MiniMaxH3ReferenceToVideo,
            "__module__",
            filesystem_module,
        ), mock.patch.object(
            MiniMaxH3ReferenceToVideo,
            "RELATIVE_PYTHON_MODULE",
            relative_module,
            create=True,
        ):
            self.assertTrue(
                MODULE._is_allowed_wrapper_source(
                    "MiniMaxH3ReferenceToVideo",
                    MiniMaxH3ReferenceToVideo,
                )
            )
            spec = MODULE._build_v3_wrapper_spec(
                "MiniMaxH3ReferenceToVideo",
                MiniMaxH3ReferenceToVideo,
                {},
            )
            wrapper, _ = MODULE._create_wrapper_class(spec)

        self.assertEqual(
            wrapper.GET_SCHEMA().node_id,
            "VTSWrapper_comfy_extras_MiniMaxH3ReferenceToVideo",
        )

    def test_nested_disk_images_materialize_without_renaming_keys(self):
        spec = MODULE._build_v3_wrapper_spec(
            "FakeAutogrowNode", FakeAutogrowNode, {}
        )
        wrapper, _ = MODULE._create_wrapper_class(spec)

        materialized = torch.ones(1, 8, 8, 3)
        disk_image = MODULE.DiskImage(
            prefix="unused",
            start_sequence=0,
            number_of_images=1,
            output_dir="unused",
            format="png",
            image=materialized,
        )
        materialize_calls = []

        def materialize():
            materialize_calls.append(True)
            return materialized

        disk_image.materialize = materialize
        tensor_image = torch.zeros(1, 8, 8, 3)
        ref_images = {
            "ref_image_0": disk_image,
            "ref_image_2": tensor_image,
        }
        ref_audios = {"ref_audio_1": {"waveform": "audio"}}

        result = wrapper.execute(ref_images=ref_images, ref_audios=ref_audios)

        self.assertIsInstance(result, io.NodeOutput)
        self.assertEqual(result.args, ("conditioning", {"samples": "latent"}))
        self.assertEqual(materialize_calls, [True])
        received_images, received_audios = FakeAutogrowNode.calls[0]
        self.assertEqual(list(received_images), ["ref_image_0", "ref_image_2"])
        self.assertIs(received_images["ref_image_0"], materialized)
        self.assertIs(received_images["ref_image_2"], tensor_image)
        self.assertIs(received_audios, ref_audios)

    def test_template_names_stays_excluded(self):
        dynamic_input = io.Autogrow.Input(
            "images",
            template=io.Autogrow.TemplateNames(
                input=io.Image.Input("image"),
                names=["first", "second"],
                min=0,
            ),
        )
        self.assertFalse(MODULE._is_safe_v3_input(dynamic_input))

    def test_v3_image_output_controls_and_ui_are_preserved(self):
        spec = MODULE._build_v3_wrapper_spec(
            "FakeAutogrowImageOutputNode",
            FakeAutogrowImageOutputNode,
            {},
        )
        wrapper, _ = MODULE._create_wrapper_class(spec)
        schema = wrapper.GET_SCHEMA()
        input_ids = [item.id for item in schema.inputs]
        self.assertIn("vts_return_type", input_ids)
        self.assertIn("vts_output_dir", input_ids)

        image = torch.zeros(1, 8, 8, 3)
        result = wrapper.execute(
            images={"image_0": image},
            vts_return_type="Tensor",
            vts_prefix="test",
            vts_start_sequence=0,
            vts_output_dir="unused",
            vts_format=MODULE.vtsImageTypes[0],
            vts_num_workers=1,
            vts_compression_level=9,
            vts_quality=95,
        )
        self.assertIsInstance(result, io.NodeOutput)
        self.assertIs(result.args[0], image)
        self.assertEqual(result.ui, {"source": "fake"})


if __name__ == "__main__":
    unittest.main()
