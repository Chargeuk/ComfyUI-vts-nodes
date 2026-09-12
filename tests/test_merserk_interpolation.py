import io
import importlib.util
import json
import tempfile
import unittest
from queue import Queue, Empty
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location('vts_merserk_interpolation_test', Path(__file__).parents[1] / 'py/VTS_MerserkFrameInterpolate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Socket:
    cut = False
    fail = False
    instances = []
    def __init__(self, *args, **kwargs):
        self.messages = Queue()
        self.uploads = []
        self.closed = False
        self.__class__.instances.append(self)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def close(self):
        self.closed = True
    def message(self, value):
        self.messages.put(json.dumps(value))
    def send(self, value):
        if isinstance(value, str):
            header = json.loads(value)
            if 'version' in header:
                self.setup = header
                self.message(dict(type='ready', output_count=(header['frame_count'] - 1) * header['multiplier'] + 1))
            elif header['type'] == 'frame':
                self.header = header
                self.data = bytearray()
            elif header['type'] == 'end':
                self.message(dict(type='done'))
            return
        self.data.extend(value)
        if len(self.data) != self.header['bytes']:
            return
        self.uploads.append(bytes(self.data))
        index = self.header['index']
        if self.fail and index == 1:
            self.message(dict(type='error', message='GPU failed'))
            return
        for slot in range(1, self.setup['multiplier']) if index else ():
            if self.cut:
                self.message(dict(type='repeat', slot=slot, source_index=index - 1 if slot * 2 <= self.setup['multiplier'] else index))
            else:
                with Image.open(io.BytesIO(self.data)) as image, io.BytesIO() as buffer:
                    image.save(buffer, format='PNG')
                    data = buffer.getvalue()
                self.message(dict(type='generated', slot=slot, bytes=len(data)))
                for offset in range(0, len(data), module.CHUNK_BYTES):
                    self.messages.put(data[offset:offset + module.CHUNK_BYTES])
        self.message(dict(type='frame_done', index=index, scene_cut=self.cut))
    def recv(self, timeout):
        try:
            return self.messages.get(timeout=timeout)
        except Empty:
            raise TimeoutError


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        Socket.instances, Socket.cut, Socket.fail = [], False, False
        self.node = module.VTSMerserkFrameInterpolate()
        self.frames = torch.rand((3, 64, 96, 4))

    def test_all_multipliers_preserve_originals_and_upload_each_source_once(self):
        for multiplier in (2, 3, 4, 8):
            with self.subTest(multiplier=multiplier), patch.object(module, 'connect', Socket):
                output, = self.node.interpolate(self.frames, multiplier=multiplier)
                self.assertEqual(tuple(output.shape), ((3 - 1) * multiplier + 1, 64, 96, 4))
                self.assertTrue(torch.equal(output[::multiplier], self.frames))
                self.assertEqual(len(Socket.instances[-1].uploads), 3)
                self.assertTrue(Socket.instances[-1].closed)

    def test_disk_input_and_lossless_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for index in range(3):
                Image.fromarray((self.frames[index].numpy() * 255).round().astype(np.uint8)).save(folder / f'source_{index + 7:06d}.png')
            disk = module.DiskImage(prefix='source', start_sequence=7, number_of_images=3,
                                   output_dir=directory, format='png', image=self.frames)
            for format in ('png', 'webp'):
                with self.subTest(format=format), patch.object(module, 'connect', Socket):
                    output, = self.node.interpolate(disk, multiplier=3, output_dir=directory,
                                                     prefix='test', start_sequence=11, format=format)
                    self.assertIsInstance(output, module.DiskImage)
                    self.assertEqual(len(output), 7)
                    self.assertEqual(len(list(Path(output.output_dir).glob('*.' + format))), 7)
                    self.assertTrue(torch.equal(output[0], disk[0]))

    def test_cut_repeats_local_frames_without_changing_count(self):
        Socket.cut = True
        with patch.object(module, 'connect', Socket):
            output, = self.node.interpolate(self.frames, multiplier=4)
        self.assertEqual(len(output), 9)
        self.assertTrue(torch.equal(output[1], self.frames[0]))
        self.assertTrue(torch.equal(output[3], self.frames[1]))

    def test_tensor_mode_never_writes_image_files(self):
        original = Image.Image.save
        def save(image, target, *args, **kwargs):
            self.assertTrue(hasattr(target, 'write'))
            return original(image, target, *args, **kwargs)
        with patch.object(module, 'connect', Socket), patch.object(Image.Image, 'save', save):
            self.node.interpolate(self.frames)

    def test_partial_output_cleanup_and_connection_close_on_error(self):
        Socket.fail = True
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'connect', Socket):
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                self.node.interpolate(self.frames, return_type='DiskImage', output_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue(Socket.instances[-1].closed)

    def test_interrupt_closes_stream_and_removes_partial_output(self):
        checks = 0
        def check():
            nonlocal checks
            checks += 1
            if checks > 6:
                raise RuntimeError('interrupted')
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'connect', Socket), \
             patch.object(module.model_management, 'throw_exception_if_processing_interrupted', side_effect=check):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.node.interpolate(self.frames, return_type='DiskImage', output_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue(Socket.instances[-1].closed)

    def test_timeout_check(self):
        with patch.object(module.time, 'monotonic', return_value=2):
            with self.assertRaises(TimeoutError):
                self.node._receive(None, deadline=1)

    def test_single_frame_bypasses_server(self):
        single = self.frames[:1]
        with patch.object(module, 'connect', side_effect=AssertionError('server')):
            self.assertIs(self.node.interpolate(single, server_url='offline')[0], single)

    def test_unsafe_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Prefix'):
            self.node.interpolate(self.frames, return_type='DiskImage', prefix='../escape')


if __name__ == '__main__':
    unittest.main()
