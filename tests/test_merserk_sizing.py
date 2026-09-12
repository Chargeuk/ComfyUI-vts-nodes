import base64
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

NODE_PATH = Path(__file__).parents[1] / 'py' / 'VTS_MerserkEnhance.py'
SPEC = importlib.util.spec_from_file_location('merserk_sizing_tests_node', NODE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Job:
    def __init__(self, value):
        self.value = value
    def result(self, timeout=None):
        return self.value
    def cancel(self):
        pass


class RecordingClient:
    calls = []
    def __init__(self, url, **kwargs):
        assert kwargs['download_files'] is False
    def submit(self, source, parameters, request_id, api_name):
        assert api_name == '/vts_enhance_memory'
        params = json.loads(parameters)
        with Image.open(io.BytesIO(base64.b64decode(source))) as image, io.BytesIO() as buffer:
            self.calls.append((image.copy(), params))
            image.resize((params['target_width'], params['target_height'])).save(buffer, format='PNG')
            return Job(base64.b64encode(buffer.getvalue()).decode('ascii'))
    def close(self):
        pass


class MerserkSizingTests(unittest.TestCase):
    def setUp(self):
        self.node = module.VTSMerserkEnhance()
        self.image = torch.rand(1, 120, 240, 3)
        RecordingClient.calls = []

    def run_node(self, **kwargs):
        options = dict(sizing_mode='Scale to Min', smallMaxSize=180, largeMaxSize=360)
        options.update(kwargs)
        return self.node.enhance(self.image, **options)[0]

    def test_bypass_preserves_exact_tensor_without_server_or_sizing_validation(self):
        with patch.object(module, 'Client', side_effect=AssertionError('network')):
            result = self.run_node(enable_scaling=False, enable_neural_rendering=False,
                                   server_url='offline', smallMaxSize=0, iterations=0)
        self.assertIs(result, self.image)

    def test_local_downscale_is_lanczos_and_works_offline(self):
        with patch.object(module, 'Client', side_effect=AssertionError('network')):
            result = self.run_node(enable_neural_rendering=False, server_url='offline',
                                   smallMaxSize=60, largeMaxSize=120, iterations=0)
        original = Image.fromarray((self.image[0].numpy()*255).round().astype(np.uint8))
        expected = np.array(original.resize((120, 60), Image.Resampling.LANCZOS)) / 255
        np.testing.assert_allclose(result[0].numpy(), expected, atol=1e-7)

    def test_scaling_only_uses_vsr_and_ignores_neural_settings(self):
        with patch.object(module, 'Client', RecordingClient):
            result = self.run_node(enable_neural_rendering=False, iterations=0, vsr_quality='High')
        source, params = RecordingClient.calls[0]
        self.assertEqual(source.size, (240, 120))
        self.assertEqual(params, dict(operation='vsr', vsr_quality=3, target_width=360, target_height=180))
        self.assertEqual(tuple(result.shape), (1, 180, 360, 3))

    def test_neural_only_keeps_original_odd_dimensions(self):
        self.image = torch.rand(1, 121, 241, 4)
        with patch.object(module, 'Client', RecordingClient):
            result = self.run_node(enable_scaling=False, iterations=3)
        params = RecordingClient.calls[0][1]
        self.assertEqual((params['target_width'], params['target_height']), (241, 121))
        self.assertEqual(params['upscaling_factor'], 1)
        self.assertEqual(params['iterations'], 3)
        self.assertEqual(result.shape, self.image.shape)

    def test_downscale_then_neural_sends_small_image_at_one_x(self):
        with patch.object(module, 'Client', RecordingClient):
            self.run_node(smallMaxSize=80, largeMaxSize=160, iterations=2)
        source, params = RecordingClient.calls[0]
        self.assertEqual(source.size, (160, 80))
        self.assertEqual(params['upscaling_factor'], 1)
        self.assertEqual(params['iterations'], 2)

    def test_center_crop_happens_before_direction_choice(self):
        with patch.object(module, 'Client', RecordingClient):
            self.run_node(smallMaxSize=180, largeMaxSize=180, crop='center',
                          scale_type='large', enable_neural_rendering=False)
        source, params = RecordingClient.calls[0]
        self.assertEqual(source.size, (120, 120))
        self.assertEqual(params['operation'], 'vsr')

    def test_mixed_axis_resize_without_crop_is_local(self):
        with patch.object(module, 'Client', side_effect=AssertionError('network')):
            result = self.run_node(smallMaxSize=180, largeMaxSize=180, crop='disabled',
                                   scale_type='large', enable_neural_rendering=False, server_url='offline')
        self.assertEqual(tuple(result.shape), (1, 180, 180, 3))

    def test_no_size_change_does_not_contact_server(self):
        with patch.object(module, 'Client', side_effect=AssertionError('network')):
            result = self.run_node(smallMaxSize=120, largeMaxSize=240, enable_neural_rendering=False)
        self.assertTrue(torch.equal(result, self.image))

    def test_reversed_limits_and_odd_targets(self):
        with patch.object(module, 'Client', RecordingClient):
            result = self.run_node(smallMaxSize=333, largeMaxSize=181, divisible_by=1, scale_type='large')
        self.assertEqual(tuple(result.shape), (1, 181, 333, 3))
        self.assertEqual(RecordingClient.calls[0][1]['upscaling_factor'], 1.5)

    def test_old_api_prompt_uses_multiplier(self):
        with patch.object(module, 'Client', RecordingClient):
            result = self.node.enhance(self.image, upscaling_factor='2')[0]
        self.assertEqual(tuple(result.shape), (1, 240, 480, 3))
        self.assertEqual(RecordingClient.calls[0][1]['upscaling_factor'], 2)

    def test_local_disk_output_saves_only_final_image(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'Client', side_effect=AssertionError('network')):
            result = self.run_node(smallMaxSize=60, largeMaxSize=120, enable_neural_rendering=False,
                                   return_type='DiskImage', output_dir=directory)
            self.assertIsInstance(result, module.DiskImage)
            self.assertEqual(len(list(Path(result.output_dir).glob('*.png'))), 1)
            self.assertEqual(tuple(result[0].shape), (60, 120, 3))
            self.assertIs(self.node.enhance(result, enable_scaling=False, enable_neural_rendering=False,
                                           return_type='DiskImage')[0], result)

    def test_invalid_zero_size_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'zero dimension'):
            self.run_node(smallMaxSize=1, largeMaxSize=2, divisible_by=64)


if __name__ == '__main__':
    unittest.main()
