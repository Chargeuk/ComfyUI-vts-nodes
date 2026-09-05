import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


MODULE_PATH = (
    Path(__file__).parents[1]
    / "py"
    / "VTS_DiskImageFromDirectory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vts_disk_image_from_directory_test_module",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DiskImageFromDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.VTSDiskImageFromDirectory()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_dir = self.temporary_directory.name

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_image(self, sequence, prefix="frame", image_format="png", mode="RGB"):
        path = os.path.join(
            self.input_dir,
            f"{prefix}_{sequence:06d}.{image_format}",
        )
        Image.new(mode, (4, 3)).save(path)
        return path

    def create(self, start, end, prefix="frame", image_format="png"):
        return self.node.create_disk_image(
            vts_input_dir=self.input_dir,
            vts_prefix=prefix,
            vts_start_sequence=start,
            vts_end_sequence=end,
            vts_format=image_format,
        )[0]

    def test_minus_one_stops_before_first_missing_sequence_without_copying(self):
        self.write_image(10)
        self.write_image(11)
        self.write_image(13)
        files_before = sorted(os.listdir(self.input_dir))

        disk_image = self.create(10, -1)

        self.assertEqual(disk_image.start_sequence, 10)
        self.assertEqual(disk_image.number_of_images, 2)
        self.assertEqual(disk_image.output_dir, os.path.realpath(self.input_dir))
        self.assertEqual(disk_image.shape, (2, 3, 4, 3))
        self.assertEqual(disk_image.dtype, torch.float32)
        self.assertEqual(sorted(os.listdir(self.input_dir)), files_before)

    def test_requested_end_is_inclusive(self):
        for sequence in range(20, 25):
            self.write_image(sequence)

        disk_image = self.create(20, 22)

        self.assertEqual(disk_image.number_of_images, 3)
        self.assertTrue(torch.equal(
            disk_image.materialize(),
            torch.zeros((3, 3, 4, 3), dtype=torch.float32),
        ))

    def test_end_above_available_range_clamps_at_first_missing_sequence(self):
        self.write_image(30)
        self.write_image(31)

        disk_image = self.create(30, 100)

        self.assertEqual(disk_image.number_of_images, 2)

    def test_missing_start_returns_empty_disk_image(self):
        self.write_image(41)

        disk_image = self.create(40, -1)

        self.assertEqual(disk_image.start_sequence, 40)
        self.assertEqual(disk_image.number_of_images, 0)
        self.assertEqual(disk_image.shape, (0, 0, 0, 3))
        self.assertEqual(disk_image.dtype, torch.float32)
        self.assertEqual(disk_image.ndim, 4)

    def test_end_before_start_returns_empty_disk_image(self):
        self.write_image(50)

        disk_image = self.create(50, 49)

        self.assertEqual(disk_image.number_of_images, 0)

    def test_rgba_shape_metadata_is_preserved(self):
        self.write_image(60, mode="RGBA")

        disk_image = self.create(60, -1)

        self.assertEqual(disk_image.shape, (1, 3, 4, 4))

    def test_prefix_cannot_escape_input_directory(self):
        with self.assertRaisesRegex(ValueError, "filename prefix"):
            self.create(0, -1, prefix="../frame")


if __name__ == "__main__":
    unittest.main()
