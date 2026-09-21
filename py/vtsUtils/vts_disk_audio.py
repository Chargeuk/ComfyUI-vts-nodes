"""Disk-backed ComfyUI AUDIO. Inspection never decodes a waveform."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import zstandard

from vts_disk_latent import DiskLatent, save_latent
from vtsUtils import resolve_list_mapped_output_identity

DEFAULT_AUDIO_DIR = './tmp/diskaudio'
FORMATS = ('FLAC', 'Exact', 'Opus', 'MP3')
FORMAT_HELP = ('FLAC (default): compressed 24-bit PCM, lossless after float-to-PCM rounding. '
               'Exact: DiskLatent Zstandard compression, preserving tensor bits and dtype. '
               'Opus: compact lossy audio. MP3: lossy audio with broad player support. '
               'FLAC rejects non-finite or out-of-range samples; use Exact to preserve them.')
BITRATE_HELP = ('Total kbps for the clip, not per channel. Auto: Opus 128, MP3 192. '
                'Opus: 32-48 mono speech; 64 higher-quality speech; 96-128 stereo music; '
                '160-192 demanding stereo. MP3: 64-96 mono speech; 128 compact stereo; '
                '192 general music; 256-320 higher-quality stereo. Higher values make larger files. '
                'Ignored for FLAC and Exact. These are starting points, not quality guarantees; '
                'repeated lossy encoding can reduce quality.')
INPUT_HELP = ('Native AUDIO dictionary or DiskAudio. Waveform metadata is available without decoding; '
              'processing loads samples using the audio device policy.')
OUTPUT_HELP = ('AUDIO dictionary or DiskAudio file reference, selected by audio_return_type. '
               'DiskAudio exposes sample rate, waveform shape, dtype and device without loading samples.')


class NoAudioStreamError(ValueError):
    pass


@dataclass(frozen=True)
class AudioTensorInfo:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    original_device: torch.device
    is_disk_backed = True
    requires_grad = False
    layout = torch.strided

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self):
        return self.shape.numel()

    def element_size(self):
        return self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def clone(self):
        return replace(self)

    detach = clone

    def to(self, device=None, dtype=None):
        if isinstance(device, torch.dtype):
            device, dtype = None, device
        return replace(self, device=torch.device(device) if device is not None else self.device,
                       dtype=dtype or self.dtype)

    def cpu(self):
        return self.to('cpu')

    def __getattr__(self, name):
        raise AttributeError(f'{name} requires waveform samples; materialize DiskAudio first')

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise TypeError('Materialize DiskAudio before tensor operations')


def _ffmpeg():
    path = shutil.which('ffmpeg')
    if not path:
        raise RuntimeError('DiskAudio needs FFmpeg on PATH for audio codecs')
    return path


def _run(args, **kwargs):
    result = subprocess.run([_ffmpeg(), '-v', 'error', '-nostdin', *args],
                            stderr=subprocess.PIPE, **kwargs)
    if result.returncode:
        raise RuntimeError('DiskAudio FFmpeg: ' + result.stderr.decode(errors='replace')[-4000:])
    return result


def _input_args(path, start=0, duration=0):
    args = ['-i', str(path)]
    if start:
        args += ['-ss', str(start)]
    if duration:
        args += ['-t', str(duration)]
    return args


def _identity(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _probe(path):
    # Count decoded frames in blocks: container duration estimates include codec padding.
    probe = shutil.which('ffprobe')
    if not probe:
        raise RuntimeError('DiskAudio needs ffprobe on PATH')
    result = subprocess.run([probe, '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                             'stream=sample_rate,channels', '-of', 'json', str(path)],
                            capture_output=True, check=True)
    streams = json.loads(result.stdout)['streams']
    if not streams:
        raise NoAudioStreamError('File has no audio stream')
    return int(streams[0]['sample_rate']), int(streams[0]['channels'])


def _decoded_frames(path, rate, channels, start=0, duration=0):
    args = [_ffmpeg(), '-v', 'error', '-nostdin', *_input_args(path, start, duration),
            '-map', '0:a:0', '-ar', str(rate), '-f', 'f32le', '-']
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors) as proc:
            size = 0
            while block := proc.stdout.read(1024 * 1024):
                size += len(block)
            if proc.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
    if size % (channels * 4):
        raise ValueError('Incomplete decoded audio frame')
    return size // (channels * 4)


class DiskAudio(Mapping):
    is_disk_backed = True

    def __init__(self, path, *, device=None, dtype=None):
        self.path = str(Path(path).expanduser().resolve())
        with open(self.path, encoding='utf8') as source:
            self.manifest = json.load(source)
        if self.manifest.get('version') != 1 or self.manifest.get('format') not in FORMATS:
            raise ValueError('Unsupported DiskAudio manifest')
        root = Path(self.path).parent
        self.files = []
        for filename in self.manifest['files']:
            path = (root / filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError('DiskAudio file escapes its manifest directory')
            self.files.append(str(path))
        self._device, self._dtype = device, dtype
        self._identities = {p: _identity(p) for p in [self.path, *self.files]}
        self.start = self.duration_limit = 0

    @classmethod
    def from_file(cls, path, start=0, duration=0):
        path = str(Path(path).expanduser().resolve())
        if path.endswith('.diskaudio'):
            if start or duration:
                raise ValueError('Trim a DiskAudio manifest with an audio processing node')
            return cls(path)
        if start < 0 or duration < 0:
            raise ValueError('Audio start and duration must be nonnegative')
        rate, channels = _probe(path)
        frames = _decoded_frames(path, rate, channels, start, duration)
        result = object.__new__(cls)
        result.path, result.files = path, [path]
        result.manifest = {'version': 1, 'format': 'Source', 'sample_rate': rate,
                           'shape': [1, channels, frames], 'dtype': 'float32', 'device': 'cpu',
                           'metadata': {}}
        result._device = result._dtype = None
        result.start, result.duration_limit = start, duration
        result._identities = {path: _identity(path)}
        return result

    def _check(self):
        for path, identity in self._identities.items():
            if _identity(path) != identity:
                raise ValueError('DiskAudio file changed since this reference was created: ' + path)

    def __getitem__(self, key):
        if key == 'waveform':
            return AudioTensorInfo(torch.Size(self.manifest['shape']),
                                   self._dtype or getattr(torch, self.manifest['dtype']),
                                   torch.device(self._device or self.manifest['device']),
                                   torch.device(self.manifest['device']))
        if key == 'sample_rate':
            return self.manifest['sample_rate']
        if self.manifest['format'] == 'Exact' and self.manifest.get('exact_encoding') != 'pcm_f32le_zstd':
            return DiskLatent(self.files[0])['audio'][key]
        return copy.deepcopy(self.manifest['metadata'][key])

    def __iter__(self):
        if self.manifest['format'] == 'Exact' and self.manifest.get('exact_encoding') != 'pcm_f32le_zstd':
            return iter(DiskLatent(self.files[0])['audio'])
        return iter(['waveform', 'sample_rate', *self.manifest['metadata']])

    def __len__(self):
        return sum(1 for _ in self)

    def to(self, device=None, dtype=None):
        if isinstance(device, torch.dtype):
            device, dtype = None, device
        result = copy.copy(self)
        result._device = str(torch.device(device)) if device is not None else self._device
        result._dtype = dtype or self._dtype
        return result

    def clone(self):
        return copy.copy(self)

    detach = clone

    def cpu(self):
        return self.to('cpu')

    def ffmpeg_input_args(self):
        self._check()
        if self.manifest['format'] == 'Exact' or len(self.files) != 1:
            return None
        # Input-level seek/duration also works when another video input follows.
        return (['-ss', str(self.start)] if self.start else []) + (
            ['-t', str(self.duration_limit)] if self.duration_limit else []) + ['-i', self.files[0]]

    def materialize(self, device=None, dtype=None):
        self._check()
        if self.manifest.get('exact_encoding') == 'pcm_f32le_zstd':
            waveform = self._materialize_pcm()
            return {**copy.deepcopy(self.manifest['metadata']), 'sample_rate': self['sample_rate'],
                    'waveform': waveform.to(device=device or self._device or self.manifest['device'],
                                            dtype=dtype or self._dtype or torch.float32)}
        if self.manifest['format'] == 'Exact':
            return DiskLatent(self.files[0]).materialize(device=device or self._device,
                                                       dtype=dtype or self._dtype)['audio']
        target = device or self._device or self.manifest['device']
        dtype = dtype or self._dtype or getattr(torch, self.manifest['dtype'])
        waves = []
        channels = self.manifest['shape'][1]
        for path in self.files:
            raw = _run([*_input_args(path, self.start, self.duration_limit), '-map', '0:a:0',
                        '-ar', str(self['sample_rate']), '-f', 'f32le', '-'], stdout=subprocess.PIPE).stdout
            data = np.frombuffer(raw, dtype='<f4').copy().reshape(-1, channels).T
            waves.append(torch.from_numpy(data))
        waveform = torch.stack(waves)
        if list(waveform.shape) != self.manifest['shape']:
            raise ValueError('Decoded DiskAudio shape differs from its metadata')
        return {**copy.deepcopy(self.manifest['metadata']), 'sample_rate': self['sample_rate'],
                'waveform': waveform.to(device=target, dtype=dtype)}

    def _materialize_pcm(self):
        batch, channels, frames = self.manifest['shape']
        # Fill the destination incrementally, with no second full decoded byte buffer.
        interleaved = torch.empty(batch, frames, channels, dtype=torch.float32)
        for index, path in enumerate(self.files):
            target = memoryview(interleaved[index].numpy()).cast('B')
            digest, offset = hashlib.sha256(), 0
            with open(path, 'rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
                while block := reader.read(1024 * 1024):
                    if offset + len(block) > len(target):
                        raise ValueError('Exact audio exceeds its declared size')
                    target[offset:offset + len(block)] = block
                    digest.update(block)
                    offset += len(block)
            if offset != len(target) or digest.hexdigest() != self.manifest['sha256'][index]:
                raise ValueError('Exact audio integrity check failed')
        return interleaved.transpose(1, 2)


def _codec_args(format, bitrate, channels):
    if format == 'FLAC':
        return ['-c:a', 'flac', '-sample_fmt', 's32', '-bits_per_raw_sample', '24', '-compression_level', '5']
    if format == 'MP3' and channels > 2:
        raise ValueError('MP3 supports mono/stereo; choose FLAC or Exact for more channels')
    bitrate = (128 if format == 'Opus' else 192) if str(bitrate) == 'Auto' else int(bitrate)
    if bitrate <= 0:
        raise ValueError('Audio bitrate must be positive')
    return ['-c:a', 'libopus' if format == 'Opus' else 'libmp3lame', '-b:a', f'{bitrate}k']


def save_audio(audio, prefix='audio', output_dir=DEFAULT_AUDIO_DIR, format='FLAC', bitrate_kbps='Auto'):
    if format not in FORMATS:
        raise ValueError('Unknown DiskAudio format')
    if not prefix or prefix in ('.', '..') or any(c in prefix for c in '/\\\0'):
        raise ValueError('audio_prefix must be a filename prefix, not a path')
    if isinstance(audio, DiskAudio) and audio.manifest['format'] == format:
        if format not in ('Opus', 'MP3') or str(audio.manifest.get('bitrate_kbps')) == str(bitrate_kbps):
            audio._check()
            return audio
    if hasattr(audio, 'to_disk_audio'):
        audio = audio.to_disk_audio()
        if audio is None:
            return None
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix, sequence = resolve_list_mapped_output_identity(prefix, 0)
    name = f'{prefix}_{sequence:06d}_{uuid.uuid4().hex}'
    scratch = Path(tempfile.mkdtemp(prefix='.diskaudio-', dir=directory))
    destination = directory / name
    try:
        if isinstance(audio, DiskAudio):
            audio._check()
        if (format == 'Exact' and isinstance(audio, DiskAudio) and audio.manifest['format'] != 'Exact'
                and audio._dtype in (None, torch.float32)):
            # File/chunk sources already decode as float32. Compress that stream directly.
            manifest = _save_exact_source(audio, scratch, prefix)
        elif format == 'Exact':
            native = audio.materialize() if isinstance(audio, DiskAudio) else audio
            wave = native['waveform']
            _validate_audio(native)
            exact = save_latent({'samples': wave, 'audio': dict(native)}, prefix, str(scratch))
            manifest = {'shape': list(wave.shape), 'dtype': str(wave.dtype)[6:],
                        'sample_rate': int(native['sample_rate']), 'device': str(wave.device),
                        'files': [Path(exact.path).name], 'metadata': {}}
        else:
            disk = isinstance(audio, DiskAudio)
            # Exact requires materialization. Codec files can transcode in bounded memory.
            if disk and audio.manifest['format'] == 'Exact':
                audio, disk = audio.materialize(device='cpu'), False
            if not disk:
                _validate_audio(audio)
            wave = audio['waveform']
            rate, channels = int(audio['sample_rate']), wave.shape[1]
            metadata = {key: audio[key] for key in audio if key not in ('waveform', 'sample_rate')}
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError('Non-JSON audio metadata requires Exact storage') from error
            extension = {'FLAC': 'flac', 'Opus': 'opus', 'MP3': 'mp3'}[format]
            files, shapes, rates = [], [], []
            for index in range(wave.shape[0]):
                output = scratch / f'{prefix}_{index:06d}.{extension}'
                if disk:
                    source = audio.files[index]
                    if format == 'FLAC':
                        # Inspect float blocks before integer conversion; never silently clip.
                        _check_file_range(source, audio.start, audio.duration_limit)
                    args = [*_input_args(source, audio.start, audio.duration_limit), '-map', '0:a:0']
                    if format == 'Opus':
                        args += ['-ar', '48000']
                    _run([*args, *_codec_args(format, bitrate_kbps, channels), str(output)])
                else:
                    samples = wave[index].detach().to(device='cpu', dtype=torch.float32).T.contiguous().numpy()
                    if format == 'FLAC':
                        _check_range(samples)
                        sf.write(output, samples, rate, subtype='PCM_24', format='FLAC')
                    else:
                        args = ['-f', 'f32le', '-ar', str(rate), '-ac', str(channels), '-i', '-']
                        if format == 'Opus':
                            args += ['-ar', '48000']
                        _run([*args, *_codec_args(format, bitrate_kbps, channels), str(output)], input=samples.tobytes())
                decoded_rate, decoded_channels = _probe(output)
                frames = _decoded_frames(output, decoded_rate, decoded_channels)
                files.append(output.name)
                shapes.append([decoded_channels, frames])
                rates.append(decoded_rate)
            if len(set(rates)) != 1 or any(shape != shapes[0] for shape in shapes):
                raise ValueError('Decoded audio batch items must have matching shapes and sample rates')
            manifest = {'shape': [len(files), *shapes[0]], 'dtype': 'float32', 'sample_rate': rates[0],
                        'device': str(wave.device), 'files': files, 'metadata': metadata}
        manifest.update(version=1, format=format, bitrate_kbps=str(bitrate_kbps))
        (scratch / 'audio.diskaudio').write_text(json.dumps(manifest, allow_nan=False), encoding='utf8')
        os.replace(scratch, destination)
        return DiskAudio(destination / 'audio.diskaudio')
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _validate_audio(audio):
    wave = audio['waveform']
    if not isinstance(wave, torch.Tensor) or wave.ndim != 3 or not wave.is_floating_point():
        raise ValueError('AUDIO waveform must be a floating tensor shaped [batch, channels, samples]')
    if any(size <= 0 for size in wave.shape) or int(audio['sample_rate']) <= 0:
        raise ValueError('DiskAudio requires nonempty audio and a positive sample rate')


def _save_exact_source(audio, directory, prefix):
    files, hashes = [], []
    for index, path in enumerate(audio.files):
        destination = directory / f'{prefix}_{index:06d}.pcm.zst'
        command = [_ffmpeg(), '-v', 'error', '-nostdin', *_input_args(path, audio.start, audio.duration_limit),
                   '-map', '0:a:0', '-ar', str(audio['sample_rate']), '-f', 'f32le', '-']
        digest, size = hashlib.sha256(), 0
        with tempfile.TemporaryFile() as errors, open(destination, 'wb') as output:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as proc:
                try:
                    with zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(output) as writer:
                        while block := proc.stdout.read(1024 * 1024):
                            digest.update(block)
                            size += len(block)
                            writer.write(block)
                except BaseException:
                    proc.kill()
                    proc.wait()
                    raise
                if proc.wait():
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors='replace'))
        if size != audio['waveform'].shape[1] * audio['waveform'].shape[2] * 4:
            raise ValueError('Decoded audio changed while saving Exact storage')
        files.append(destination.name)
        hashes.append(digest.hexdigest())
    return {'shape': list(audio['waveform'].shape), 'dtype': 'float32', 'device': str(audio['waveform'].device),
            'sample_rate': audio['sample_rate'], 'metadata': copy.deepcopy(audio.manifest['metadata']),
            'files': files, 'sha256': hashes, 'exact_encoding': 'pcm_f32le_zstd'}


def _check_range(samples):
    if not np.isfinite(samples).all() or samples.min() < -1 or samples.max() > 1:
        raise ValueError('FLAC requires finite samples in [-1, 1]; choose Exact or adjust the waveform explicitly')


def _check_file_range(path, start, duration):
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen([_ffmpeg(), '-v', 'error', '-nostdin', *_input_args(path, start, duration),
                               '-map', '0:a:0', '-f', 'f32le', '-'], stdout=subprocess.PIPE, stderr=errors) as proc:
            try:
                while block := proc.stdout.read(1024 * 1024):
                    _check_range(np.frombuffer(block, dtype='<f4'))
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            if proc.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))


def materialize_audio(value, device_policy='Original', device='cpu', memo=None):
    if device_policy not in ('Original', 'CPU', 'Specified'):
        raise ValueError('Unknown audio device policy')
    memo = {} if memo is None else memo
    if isinstance(value, DiskAudio):
        if id(value) not in memo:
            memo[id(value)] = value.materialize(device=None if device_policy == 'Original' else
                                               ('cpu' if device_policy == 'CPU' else device))
        return memo[id(value)]
    if isinstance(value, dict):
        if isinstance(value.get('waveform'), torch.Tensor):
            return value
        converted = {k: materialize_audio(v, device_policy, device, memo) for k, v in value.items()}
        return value if all(converted[k] is v for k, v in value.items()) else converted
    if isinstance(value, (list, tuple)):
        converted = type(value)(materialize_audio(v, device_policy, device, memo) for v in value)
        return value if all(a is b for a, b in zip(value, converted)) else converted
    return value


def audio_controls(prefix, has_input=True, has_output=True, control_prefix='audio_', match_input=True):
    controls = {}
    if has_output:
        modes = ['Tensor', 'DiskAudio', 'Input'] if has_input and match_input else ['Tensor', 'DiskAudio']
        controls.update({
            'return_type': (modes, {'default': 'Tensor', 'tooltip': 'Tensor keeps audio in memory (existing behavior). DiskAudio writes compressed files. Input matches the single audio input storage type.'}),
            'format': (list(FORMATS), {'default': 'FLAC', 'tooltip': FORMAT_HELP}),
            'bitrate_kbps': (['Auto', '32', '48', '64', '96', '128', '160', '192', '256', '320'], {'default': 'Auto', 'tooltip': BITRATE_HELP}),
            'output_dir': ('STRING', {'default': DEFAULT_AUDIO_DIR, 'tooltip': 'DiskAudio folder, relative to ComfyUI working directory unless absolute. Default ./tmp/diskaudio. Independent of image and latent folders.'}),
            'prefix': ('STRING', {'default': '_'.join(prefix.split()), 'tooltip': 'Audio filename prefix, derived from the node name. List position and a unique suffix prevent collisions.'}),
        })
    if has_input:
        controls.update({
            'device_policy': (['Original', 'CPU', 'Specified'], {'default': 'Original', 'advanced': True, 'tooltip': 'DiskAudio reload placement: Original restores the saved device or deferred override; CPU uses RAM; Specified uses audio_device. Native tensors are unchanged.'}),
            'device': ('STRING', {'default': 'cpu', 'advanced': True, 'tooltip': 'Used only with Specified: cpu = system RAM; cuda:0 = first visible GPU; cuda:1 requires a second GPU. Controls waveform loading, not model placement.'}),
        })
    return {control_prefix + key: spec for key, spec in controls.items()}
