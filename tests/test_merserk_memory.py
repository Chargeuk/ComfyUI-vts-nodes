import base64
import io
import importlib.util
import json
import tempfile
import unittest
from concurrent.futures import TimeoutError
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

spec = importlib.util.spec_from_file_location('merserk_memory_test_node', Path(__file__).parents[1] / 'py' / 'VTS_MerserkEnhance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Job:
    def __init__(self, result):
        self.value = result
        self.cancelled = False
    def result(self, timeout=None):
        return self.value
    def cancel(self):
        self.cancelled = True


class Client:
    calls = []
    closed = False
    fail_second = False
    def __init__(self, url, **kwargs):
        assert kwargs['download_files'] is False
        self.count = 0
    def submit(self, *args, api_name):
        if api_name == '/vts_cancel':
            self.calls.append((api_name, args, None))
            return Job(True)
        self.count += 1
        if self.fail_second and self.count == 2:
            raise RuntimeError('render failed')
        params = json.loads(args[1])
        with Image.open(io.BytesIO(base64.b64decode(args[0]))) as source, io.BytesIO() as buffer:
            source.resize((params['target_width'], params['target_height'])).save(buffer, format='PNG')
            job = Job(base64.b64encode(buffer.getvalue()).decode('ascii'))
        self.calls.append((api_name, args, job))
        return job
    def close(self):
        Client.closed = True


class MemoryTests(unittest.TestCase):
    def setUp(self):
        Client.calls, Client.closed, Client.fail_second = [], False, False
        self.node = module.VTSMerserkEnhance()
        self.images = torch.rand(2, 96, 128, 4)

    def test_tensor_transfer_never_creates_image_files_or_temporary_directory(self):
        original = Image.Image.save
        def memory_save(image, target, *args, **kwargs):
            self.assertTrue(hasattr(target, 'write'), str(target))
            return original(image, target, *args, **kwargs)
        with patch.object(module, 'Client', Client), patch.object(Image.Image, 'save', memory_save), \
             patch.object(module.tempfile, 'TemporaryDirectory', side_effect=AssertionError('temporary directory')):
            result = self.node.enhance(self.images)[0]
        self.assertEqual(result.shape, self.images.shape)
        self.assertEqual(len(Client.calls), 2)
        self.assertTrue(Client.closed)

    def test_disk_output_still_saves_final_frames(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'Client', Client):
            result = self.node.enhance(self.images, return_type='DiskImage', output_dir=folder)[0]
            self.assertEqual(len(list(Path(result.output_dir).glob('*.png'))), 2)
            self.assertEqual(tuple(result[0].shape), (96, 128, 4))

    def test_partial_disk_output_cleanup(self):
        Client.fail_second = True
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'Client', Client):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                self.node.enhance(self.images, return_type='DiskImage', output_dir=folder)
            self.assertEqual(list(Path(folder).iterdir()), [])
        self.assertTrue(Client.closed)

    def test_timeout_cancels_only_this_request(self):
        with patch.object(module, 'Client', Client), patch.object(module.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                self.node.enhance(self.images, timeout_seconds=1)
        self.assertEqual(Client.calls[0][1][2], Client.calls[1][1][0])
        self.assertTrue(Client.calls[0][2].cancelled)


if __name__ == '__main__':
    unittest.main()
