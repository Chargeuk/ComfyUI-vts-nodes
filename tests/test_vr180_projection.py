from pathlib import Path
import sys
import unittest

import torch


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "py"
    / "vtsUtils"
)
sys.path.insert(0, str(MODULE_DIR))
import vr180_projection as projection


def smooth_rect(batch=1, height=129, width=193, device="cpu"):
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.stack((xx, yy, 0.25 * xx + 0.75 * yy), dim=0)
    return image.unsqueeze(0).repeat(batch, 1, 1, 1)


class ProjectionTests(unittest.TestCase):
    def test_forward_shape_masks_and_exact_magenta(self):
        source = smooth_rect(batch=2, height=65, width=97)
        result = projection.rectilinear_to_vr180(source, (96, 128), chunk_rows=17)
        self.assertEqual(result.image.shape, (2, 3, 96, 128))
        self.assertEqual(result.known_mask.shape, (2, 1, 96, 128))
        self.assertEqual(result.known_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(result.unknown_mask, ~result.known_mask))
        unknown = result.unknown_mask.expand_as(result.image)
        expected = torch.tensor([1.0, 0.0, 1.0]).view(1, 3, 1, 1).expand_as(result.image)
        self.assertTrue(torch.equal(result.image[unknown], expected[unknown]))

    def test_pixel_centred_optical_axis_hits_image_centre(self):
        source = smooth_rect(height=101, width=151)
        result = projection.rectilinear_to_vr180(
            source, (101, 151), horizontal_fov_degrees=90.0, chunk_rows=19
        )
        centre = result.image[0, :, 50, 75]
        expected = source[0, :, 50, 75]
        self.assertTrue(result.known_mask[0, 0, 50, 75])
        self.assertTrue(torch.allclose(centre, expected, atol=2.0e-6, rtol=0.0))

    def test_chunking_and_batching_are_numerically_identical(self):
        source = smooth_rect(batch=3, height=81, width=123)
        whole = projection.rectilinear_to_vr180(source, (95, 127), chunk_rows=0)
        chunked = projection.rectilinear_to_vr180(source, (95, 127), chunk_rows=7)
        self.assertTrue(torch.equal(whole.known_mask, chunked.known_mask))
        self.assertTrue(torch.equal(whole.unknown_mask, chunked.unknown_mask))
        self.assertTrue(torch.equal(whole.image, chunked.image))

    def test_yaw_moves_known_region_and_reverse_respects_pose(self):
        source = smooth_rect(height=81, width=121)
        centred = projection.rectilinear_to_vr180(source, (96, 160), yaw_degrees=0.0)
        right = projection.rectilinear_to_vr180(source, (96, 160), yaw_degrees=25.0)
        x0 = centred.known_mask[0, 0].nonzero()[:, 1].float().mean()
        x1 = right.known_mask[0, 0].nonzero()[:, 1].float().mean()
        self.assertGreater(x1, x0)
        reverse = projection.vr180_to_rectilinear(
            right.image,
            (81, 121),
            yaw_degrees=25.0,
            source_known_mask=right.known_mask,
        )
        valid = reverse.known_mask.expand_as(reverse.image)
        # A conservative bilinear mask rejects the thin image-plane rim when
        # any of its four source contributors is unknown.  At this deliberately
        # small test resolution the retained area is still above 90%.
        self.assertGreater(float(valid.float().mean()), 0.90)
        self.assertLess(float(torch.mean(torch.abs(reverse.image[valid] - source[valid]))), 0.02)

    def test_reverse_marks_unknown_source_samples_magenta(self):
        vr = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        vr[:, 1] = 0.5
        source_known = torch.ones((1, 1, 64, 64), dtype=torch.bool)
        source_known[:, :, 24:40, 24:40] = False
        result = projection.vr180_to_rectilinear(
            vr, (48, 48), source_known_mask=source_known, horizontal_fov_degrees=90.0
        )
        self.assertTrue(result.unknown_mask.any())
        expected = torch.tensor([1.0, 0.0, 1.0]).view(1, 3, 1, 1).expand_as(result.image)
        unknown = result.unknown_mask.expand_as(result.image)
        self.assertTrue(torch.equal(result.image[unknown], expected[unknown]))

    def test_grid_cache_is_bounded_and_reports_hits(self):
        projection.clear_projection_cache()
        source = smooth_rect(height=33, width=49)
        for width in range(40, 48):
            projection.rectilinear_to_vr180(source, (32, width), horizontal_fov_degrees=80.0)
        info = projection.projection_cache_info()
        self.assertLessEqual(info["forward"]["currsize"], info["max_entries_per_direction"])
        before = info["forward"]["hits"]
        projection.rectilinear_to_vr180(source, (32, 47), horizontal_fov_degrees=80.0)
        self.assertEqual(projection.projection_cache_info()["forward"]["hits"], before + 1)

    def test_1024_and_2048_sampling_grids_are_scale_consistent(self):
        projection.clear_projection_cache()
        low, low_known = projection._cached_rect_to_vr_grid(
            1024, 1024, 1024, 1024, 90.0, 0.0, 0.0, 0.0
        )
        high, high_known = projection._cached_rect_to_vr_grid(
            2048, 2048, 2048, 2048, 90.0, 0.0, 0.0, 0.0
        )
        high_mean = high.reshape(1024, 2, 1024, 2, 2).mean(dim=(1, 3))
        self.assertLess(float(torch.max(torch.abs(high_mean[low_known] - low[low_known]))), 2.0e-5)
        high_vote = high_known.reshape(1024, 2, 1024, 2).all(dim=(1, 3))
        mismatch = (high_vote != low_known).float().mean()
        self.assertLess(float(mismatch), 0.005)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu_and_preserves_device(self):
        cpu_source = smooth_rect(batch=2, height=73, width=111)
        gpu_source = cpu_source.cuda()
        cpu = projection.rectilinear_to_vr180(cpu_source, (96, 128), chunk_rows=13)
        gpu = projection.rectilinear_to_vr180(gpu_source, (96, 128), chunk_rows=13)
        self.assertTrue(gpu.image.is_cuda and gpu.known_mask.is_cuda)
        self.assertTrue(torch.equal(cpu.known_mask, gpu.known_mask.cpu()))
        self.assertTrue(torch.allclose(cpu.image, gpu.image.cpu(), atol=2.0e-5, rtol=1.0e-5))

    def test_invalid_inputs_fail_early(self):
        with self.assertRaisesRegex(ValueError, "B,C,H,W"):
            projection.rectilinear_to_vr180(torch.zeros(3, 4, 5), (64, 64))
        with self.assertRaisesRegex(ValueError, "180"):
            projection.rectilinear_to_vr180(smooth_rect(), (64, 64), horizontal_fov_degrees=180)
        with self.assertRaisesRegex(ValueError, "source_known_mask"):
            projection.vr180_to_rectilinear(
                smooth_rect(height=64, width=64),
                (32, 32),
                source_known_mask=torch.ones(2, 31, 31),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
