import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_Image_composite_masked.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vts_image_composite_masked_test_module",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def image_frames(values):
    values = torch.tensor(values, dtype=torch.float32)
    return values.reshape(-1, 1, 1, 1).repeat(1, 1, 1, 3)


class ImageCompositeMaskedDiskTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.VTS_Image_Composite_Masked()

    def run_composite(self, destination, source, mask=None, batch_size=2):
        return self.node.composite(
            source=source,
            x=0,
            y=0,
            resize_source=False,
            color_match_method="mkl",
            color_match_strength=0.0,
            mask=mask,
            image=destination,
            return_type="Tensor",
            batch_size=batch_size,
            edit_in_place=False,
        )[0]

    def test_disk_source_is_streamed_and_repeated_across_global_batches(self):
        destination = image_frames([0, 0, 0, 0, 0])
        source_frames = image_frames([0.1, 0.2, 0.3])
        source = MODULE.DiskImage(
            prefix="source",
            start_sequence=0,
            number_of_images=3,
            output_dir="unused",
            format="png",
            image=source_frames,
        )
        materialize_calls = []

        def materialize(start=0, count=None):
            materialize_calls.append((start, count))
            return source_frames[start:start + count]

        source.materialize = materialize

        output = self.run_composite(destination, source)

        expected = image_frames([0.1, 0.2, 0.3, 0.1, 0.2])
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(
            materialize_calls,
            [(0, 2), (2, 1), (0, 1), (1, 1)],
        )

    def test_tensor_source_uses_global_not_per_batch_alignment(self):
        destination = image_frames([0, 0, 0, 0, 0])
        source = image_frames([0.1, 0.2, 0.3])

        output = self.run_composite(destination, source)

        expected = image_frames([0.1, 0.2, 0.3, 0.1, 0.2])
        self.assertTrue(torch.equal(output, expected))

    def test_mask_uses_the_same_global_batch_alignment(self):
        destination = image_frames([0, 0, 0, 0, 0])
        source = image_frames([1.0])
        mask = torch.tensor([1.0, 0.0, 0.5]).reshape(3, 1, 1)

        output = self.run_composite(destination, source, mask=mask)

        expected = image_frames([1.0, 0.0, 0.5, 1.0, 0.0])
        self.assertTrue(torch.equal(output, expected))

    def test_input_or_diskimage_is_normalized_before_shared_helper(self):
        primary = MODULE.DiskImage(
            prefix="primary",
            start_sequence=7,
            number_of_images=1,
            output_dir="primary-output",
            format="webp",
            image=image_frames([0.0]),
            compression_level=4,
            quality=91,
        )
        captured = {}
        sentinel = object()

        def fake_transform_and_save_images(transform_fn, **kwargs):
            captured.update(kwargs)
            return sentinel

        with mock.patch.object(
            MODULE,
            "transform_and_save_images",
            side_effect=fake_transform_and_save_images,
        ):
            result = self.node.composite(
                source=image_frames([1.0]),
                x=0,
                y=0,
                resize_source=False,
                color_match_method="mkl",
                image=primary,
                return_type="Input or DiskImage",
            )

        self.assertIs(result[0], sentinel)
        self.assertEqual(captured["return_type"], "Input")
        self.assertEqual(captured["prefix"], "primary")
        self.assertEqual(captured["start_sequence"], 7)
        self.assertEqual(captured["output_dir"], "primary-output")
        self.assertEqual(captured["format"], "webp")
        self.assertEqual(captured["compression_level"], 4)
        self.assertEqual(captured["quality"], 91)


if __name__ == "__main__":
    unittest.main()
