import gc
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch

import test_h3_loop_context as loop
import test_tiled_decode_colour_match as colour
from vts_disk_latent import DiskLatent, save_latent
from vts_h3_context import DiskContextTensor, materialize_context, store_context


class DiskH3ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prepare = loop.MODULE.VTS_H3PrepareLoopContext()
        self.apply = loop.MODULE.VTS_H3ApplyLoopContext()

    def test_disk_context_has_only_tails_and_metadata_does_not_load(self):
        native, = self.prepare.execute(loop.av_latent())
        refs = [weakref.ref(native[key]) for key in ('video', 'audio')]
        disk = store_context(native, 'DiskLatent', self.temp.name)
        expected_bytes = sum(native[key].numel() * native[key].element_size() for key in ('video', 'audio'))
        del native
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIs(disk['video'].payload, disk['audio'].payload)
        self.assertEqual(disk['video'].payload.tensor_size_bytes, expected_bytes)
        with mock.patch.object(DiskLatent, 'materialize', side_effect=AssertionError('loaded')):
            self.assertEqual(disk['video'].shape, (1, 24, 7, 2, 2))
            self.assertEqual(disk['video'].dtype, torch.float32)
            self.assertEqual(disk['audio'].size(1), 32)
            self.assertIsInstance(disk['audio_start'], float)
            self.assertEqual(disk['video'].cpu().device, torch.device('cpu'))

    def test_apply_matches_native_and_loads_shared_file_once(self):
        source, target, cond = loop.av_latent(), loop.av_latent(27), loop.conditioning()
        native, = self.prepare.execute(source)
        disk, = self.prepare.execute(save_latent(source, output_dir=self.temp.name),
                                      context_return_type='DiskLatent', context_output_dir=self.temp.name)
        expected = self.apply.execute(cond, target, native)
        with mock.patch.object(disk['video'].payload, 'materialize', wraps=disk['video'].payload.materialize) as load:
            actual = self.apply.execute(cond, target, disk, context_device_policy='CPU')
        self.assertEqual(load.call_count, 1)
        loop.H3LoopContextTests.assert_result_equal(self, expected, actual)

    def test_mixed_context_and_regular_disk_latent_output(self):
        native, = self.prepare.execute(loop.av_latent())
        disk = store_context(native, 'DiskLatent', self.temp.name)
        for video in (disk['video'], save_latent({'samples': native['video']}, output_dir=self.temp.name)):
            context = dict(native, video=video)
            result = self.apply.execute([], loop.av_latent(27), context,
                context_device_policy='Specified', context_device='cpu',
                latent_return_type='DiskLatent', latent_output_dir=self.temp.name)
            self.assertIsInstance(result[2], DiskLatent)
            restored = materialize_context(context, 'Specified', 'cpu')
            self.assertIs(restored['audio'], native['audio'])
            torch.testing.assert_close(restored['video'], native['video'], rtol=0, atol=0)

    def test_corrected_decoder_context_can_be_disk_and_prepare_can_be_either(self):
        source = loop.av_latent(12)
        images = torch.rand(39, 32, 32, 3)
        vae = colour.FakeVAE(images)
        vae.encode = lambda frames: frames.reshape(-1)[:24 * 7 * 4].reshape(1, 24, 7, 2, 2)
        node = colour.NODE.VTS_VAEDecodeTiledColourMatch()
        output, corrected = node.decode(vae, source, encode_corrected_context=True,
            context_return_type='DiskLatent', context_output_dir=self.temp.name, return_type='Tensor')
        self.assertIsInstance(corrected['video'], DiskContextTensor)
        self.assertEqual(corrected['source_frames'], 39)
        self.assertEqual(corrected['frame_count'], 22)
        self.assertIsInstance(output, torch.Tensor)
        expected = materialize_context(corrected)
        for mode in ('Tensor', 'DiskLatent'):
            context, = self.prepare.execute(source, corrected_video_context=corrected,
                context_return_type=mode, context_output_dir=self.temp.name)
            actual = materialize_context(context)
            torch.testing.assert_close(actual['video'], expected['video'], rtol=0, atol=0)
            self.assertEqual(isinstance(context['video'], DiskContextTensor), mode == 'DiskLatent')

    def test_bad_metadata_fails_before_loading(self):
        context, = self.prepare.execute(loop.av_latent(), context_return_type='DiskLatent',
                                        context_output_dir=self.temp.name)
        with mock.patch.object(DiskLatent, 'materialize', side_effect=AssertionError('loaded')):
            with self.assertRaisesRegex(ValueError, 'cannot pin'):
                self.apply.execute([], loop.av_latent(7), context)
            with self.assertRaisesRegex(ValueError, 'does not match target'):
                self.apply.execute([], loop.av_latent(height=4), context)
            corrected = {'video': context['video'], 'frame_count': 22, 'source_frames': -1}
            with self.assertRaisesRegex(ValueError, 'same source'):
                self.prepare.execute(loop.av_latent(), corrected_video_context=corrected)

    def test_file_integrity_and_deferred_context_dtype(self):
        context, = self.prepare.execute(loop.av_latent(), context_return_type='DiskLatent',
                                        context_output_dir=self.temp.name)
        changed = dict(context, video=context['video'].to(dtype=torch.float64))
        actual = materialize_context(changed)
        self.assertEqual(actual['video'].dtype, torch.float64)
        self.assertEqual(actual['audio'].dtype, torch.float32)
        Path(context['video'].disk_path).unlink()
        with self.assertRaises(FileNotFoundError):
            materialize_context(context)

    def test_disabled_context_creates_no_files_and_controls_have_tooltips(self):
        node = colour.NODE.VTS_VAEDecodeTiledColourMatch()
        _, context = node.decode(colour.FakeVAE(torch.zeros(2, 8, 8, 3)), {'samples': torch.zeros(1, 4, 1, 1)},
                                 context_return_type='DiskLatent', context_output_dir=self.temp.name)
        self.assertIsNone(context)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        for cls in (type(self.prepare), type(self.apply), type(node)):
            schema = cls.INPUT_TYPES()
            for name, spec in schema['optional'].items():
                if name.startswith('context_') and name != 'context_length':
                    self.assertTrue(spec[1].get('tooltip'), name)


if __name__ == '__main__':
    unittest.main()
