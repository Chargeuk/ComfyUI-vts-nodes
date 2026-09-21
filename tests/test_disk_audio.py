import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from comfy_api.latest import io

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'py' / 'vtsUtils'))
from vts_disk_audio import DiskAudio, materialize_audio, save_audio
from vts_audio_nodes import disk_audio_node
from vts_latent_nodes import disk_latent_node
from vts_disk_latent import DiskLatent


def tone(rate=44100, batch=1, channels=2):
    t = torch.arange(rate // 4) / rate
    wave = (.25 * torch.sin(t * 440 * 2 * torch.pi)).expand(batch, channels, -1).clone()
    return {'waveform': wave, 'sample_rate': rate, 'label': 'test'}


class DiskAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def save(self, audio, format='FLAC'):
        return save_audio(audio, 'test', self.temp.name, format)

    def test_exact_preserves_bits_and_extra_tensor_dtypes(self):
        audio = tone(batch=2)
        audio['waveform'] = audio['waveform'].to(torch.float64)
        audio['waveform'][0, 0, :3] = torch.tensor([float('nan'), float('inf'), -0.0])
        audio['mask'] = torch.tensor([1, 0], dtype=torch.int64)
        disk = self.save(audio, 'Exact')
        native = disk.materialize()
        self.assertTrue(torch.equal(native['waveform'].view(torch.int64), audio['waveform'].view(torch.int64)))
        self.assertEqual(native['mask'].dtype, torch.int64)
        self.assertEqual(native['label'], 'test')
        self.assertEqual(list(disk['waveform'].shape), [2, 2, 11025])

    def test_flac_batch_roundtrip_and_metadata_only(self):
        audio = tone(batch=2)
        disk = self.save(audio)
        with mock.patch('vts_disk_audio._run', side_effect=AssertionError('decoded')):
            self.assertEqual(disk['sample_rate'], 44100)
            self.assertEqual(disk['waveform'].size(2), 11025)
            self.assertEqual(disk['waveform'].numel(), 44100)
            self.assertEqual(dict(disk)['label'], 'test')
            clone = DiskAudio(disk.path).to('cpu')
        restored = clone.materialize()['waveform']
        self.assertLessEqual((restored - audio['waveform']).abs().max().item(), 2 ** -23)
        self.assertEqual(restored.dtype, torch.float32)
        self.assertEqual(len(disk.files), 2)

    def test_lossy_shapes_rate_and_duration(self):
        for format in ('Opus', 'MP3'):
            disk = self.save(tone(), format)
            audio = disk.materialize()
            self.assertEqual(list(audio['waveform'].shape), disk.manifest['shape'])
            self.assertEqual(audio['sample_rate'], 48000 if format == 'Opus' else 44100)
            self.assertAlmostEqual(audio['waveform'].shape[-1] / audio['sample_rate'], .25, places=3)

    def test_flac_rejects_clipping_and_removes_partial_files(self):
        audio = tone()
        audio['waveform'][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'Exact'):
            self.save(audio)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_source_slice_and_streamed_transcode(self):
        original = self.save(tone())
        source = DiskAudio.from_file(original.files[0], start=.05, duration=.10)
        self.assertEqual(source['waveform'].shape[-1], 4410)
        with mock.patch.object(DiskAudio, 'materialize', side_effect=AssertionError('full load')):
            encoded = self.save(source, 'Opus')
        self.assertAlmostEqual(encoded['waveform'].shape[-1] / encoded['sample_rate'], .10, places=3)

    def test_file_to_exact_streams_and_preserves_decoded_bits(self):
        source = self.save(tone(batch=2))
        expected = source.materialize()['waveform']
        with mock.patch.object(DiskAudio, 'materialize', side_effect=AssertionError('full load')):
            exact = self.save(source, 'Exact')
        self.assertEqual(exact.manifest['exact_encoding'], 'pcm_f32le_zstd')
        actual = exact.materialize()['waveform']
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_exact_honors_deferred_dtype(self):
        disk = self.save(tone()).to(dtype=torch.float64)
        exact = self.save(disk, 'Exact')
        self.assertEqual(exact['waveform'].dtype, torch.float64)
        self.assertEqual(exact.materialize()['waveform'].dtype, torch.float64)

    def test_native_passthrough_and_disk_memo(self):
        audio = tone()
        self.assertIs(materialize_audio(audio, 'CPU'), audio)
        disk = self.save(audio)
        values = materialize_audio([disk, disk], 'CPU')
        self.assertIs(values[0], values[1])
        with self.assertRaisesRegex(ValueError, 'policy'):
            materialize_audio(disk, 'bad')

    def test_changed_file_fails(self):
        disk = self.save(tone())
        with open(disk.files[0], 'ab') as output:
            output.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            disk.materialize()

    def test_paths_and_unsupported_channels(self):
        with self.assertRaisesRegex(ValueError, 'prefix'):
            save_audio(tone(), '../escape', self.temp.name)
        with self.assertRaisesRegex(ValueError, 'mono/stereo'):
            self.save(tone(channels=3), 'MP3')

    def test_legacy_decorator_old_calls_ui_and_device(self):
        class Gain:
            @classmethod
            def INPUT_TYPES(cls):
                return {'required': {'audio': ('AUDIO',)}}
            RETURN_TYPES = ('AUDIO',)
            FUNCTION = 'run'
            def run(self, audio):
                return {'ui': {'text': ['ok']}, 'result': ({**audio, 'waveform': audio['waveform'] / 2},)}
        disk_audio_node(Gain)
        node = Gain()
        self.assertIsInstance(node.run(tone())['result'][0], dict)
        disk = self.save(tone())
        result = node.run(disk, audio_return_type='Input', audio_output_dir=self.temp.name)
        self.assertEqual(result['ui']['text'], ['ok'])
        self.assertIsInstance(result['result'][0], DiskAudio)
        self.assertEqual(Gain.INPUT_TYPES()['optional']['audio_format'][1]['default'], 'FLAC')

    def test_v3_dynamic_audio_and_generated_eligibility(self):
        class Mix(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id='TestAudioMix', inputs=[io.Autogrow.Input('audio',
                    template=io.Autogrow.TemplatePrefix(io.Audio.Input('clip'), prefix='clip', min=1, max=3))],
                    outputs=[io.Audio.Output()])
            @classmethod
            def execute(cls, audio):
                clips = list(audio.values())
                return io.NodeOutput({**clips[0], 'waveform': sum(c['waveform'] for c in clips)})
        disk_audio_node(Mix, control_prefix='vts_audio_')
        result = Mix.execute({'clip0': self.save(tone())}, vts_audio_return_type='DiskAudio',
                             vts_audio_output_dir=self.temp.name)
        self.assertIsInstance(result.args[0], DiskAudio)
        self.assertTrue(any(x.id == 'vts_audio_format' for x in Mix.define_schema().inputs))

    def test_native_audio_and_latent_controls_compose(self):
        @disk_latent_node(inputs=('latent',), outputs=(1,), prefix='Mixed')
        class Mixed:
            @classmethod
            def INPUT_TYPES(cls):
                return {'required': {'audio': ('AUDIO',), 'latent': ('LATENT',)}}
            RETURN_TYPES = ('AUDIO', 'LATENT')
            FUNCTION = 'run'
            def run(self, audio, latent):
                return audio, latent
        disk_audio_node(Mixed)
        audio, latent = Mixed().run(self.save(tone()), {'samples': torch.zeros(1, 4, 2, 2)},
                                    audio_return_type='DiskAudio', audio_output_dir=self.temp.name,
                                    latent_return_type='DiskLatent', latent_output_dir=self.temp.name)
        self.assertIsInstance(audio, DiskAudio)
        self.assertIsInstance(latent, DiskLatent)


if __name__ == '__main__':
    unittest.main()
