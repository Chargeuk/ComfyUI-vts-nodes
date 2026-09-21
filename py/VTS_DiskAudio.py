import json
import os
import sys

_utils = os.path.join(os.path.dirname(__file__), 'vtsUtils')
if _utils not in sys.path:
    sys.path.append(_utils)

from vts_disk_audio import DiskAudio, INPUT_HELP, OUTPUT_HELP, audio_controls, materialize_audio, save_audio


class VTSAudioToDisk:
    VTS_DISK_AUDIO_SUPPORT = True

    @classmethod
    def INPUT_TYPES(cls):
        controls = audio_controls('VTS_Audio_To_Disk')
        controls.pop('audio_return_type')
        return {'required': {'audio': ('AUDIO', {'tooltip': INPUT_HELP})}, 'optional': controls}

    RETURN_TYPES = ('AUDIO',)
    OUTPUT_TOOLTIPS = (OUTPUT_HELP,)
    FUNCTION = 'save'
    CATEGORY = 'VTS/audio/disk'

    def save(self, audio, audio_format='FLAC', audio_bitrate_kbps='Auto',
             audio_output_dir='./tmp/diskaudio', audio_prefix='VTS_Audio_To_Disk',
             audio_device_policy='Original', audio_device='cpu'):
        if audio_device_policy != 'Original':
            audio = materialize_audio(audio, audio_device_policy, audio_device)
        return (save_audio(audio, audio_prefix, audio_output_dir, audio_format, audio_bitrate_kbps),)


class VTSDiskAudioFromFile:
    VTS_DISK_AUDIO_SUPPORT = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio_path': ('STRING', {'default': '', 'tooltip': 'Local audio/video file or .diskaudio manifest. References the source without copying. Media files are scanned in blocks to determine exact decoded duration.'})},
                'optional': {'start_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'tooltip': 'Start offset in seconds for a media file. Use 0 for DiskAudio manifests.'}),
                             'duration_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'tooltip': 'Clip length in seconds; 0 means the remaining file. Use an audio trim node for DiskAudio manifests.'})}}

    RETURN_TYPES = ('AUDIO',)
    OUTPUT_TOOLTIPS = ('DiskAudio reference; waveform data remains on disk until required.',)
    FUNCTION = 'load'
    CATEGORY = 'VTS/audio/disk'

    @classmethod
    def IS_CHANGED(cls, audio_path, **kwargs):
        # Rebuild source metadata on execution, including changed codec payload files.
        return float('nan')

    def load(self, audio_path, start_seconds=0, duration_seconds=0):
        return (DiskAudio.from_file(audio_path, start_seconds, duration_seconds),)


class VTSMaterializeAudio:
    VTS_DISK_AUDIO_SUPPORT = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO', {'tooltip': INPUT_HELP})},
                'optional': audio_controls('', has_output=False)}

    RETURN_TYPES = ('AUDIO',)
    OUTPUT_TOOLTIPS = ('Native AUDIO with waveform tensors in memory. Already-native input is unchanged.',)
    FUNCTION = 'load'
    CATEGORY = 'VTS/audio/disk'

    def load(self, audio, audio_device_policy='Original', audio_device='cpu'):
        return (materialize_audio(audio, audio_device_policy, audio_device),)


class VTSInspectAudio:
    VTS_DISK_AUDIO_SUPPORT = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO', {'tooltip': 'Native audio or DiskAudio. Disk inspection reads metadata only.'})}}

    RETURN_TYPES = ('STRING',)
    OUTPUT_TOOLTIPS = ('JSON description of shape, duration, sample rate, dtype, device and storage. Does not decode DiskAudio.',)
    FUNCTION = 'inspect'
    CATEGORY = 'VTS/audio/disk'
    OUTPUT_NODE = True

    def inspect(self, audio):
        waveform = audio['waveform']
        report = {'shape': list(waveform.shape), 'sample_rate': audio['sample_rate'],
                  'duration_seconds': waveform.shape[-1] / audio['sample_rate'],
                  'dtype': str(waveform.dtype), 'device': str(waveform.device),
                  'is_disk_backed': isinstance(audio, DiskAudio)}
        if isinstance(audio, DiskAudio):
            report.update(path=audio.path, format=audio.manifest['format'],
                          stored_bytes=sum(os.path.getsize(path) for path in set([audio.path, *audio.files])))
        text = json.dumps(report, indent=2)
        return {'ui': {'text': [text]}, 'result': (text,)}


NODE_CLASS_MAPPINGS = {'VTS Audio To Disk': VTSAudioToDisk, 'VTS DiskAudio From File': VTSDiskAudioFromFile,
                       'VTS Materialize Audio': VTSMaterializeAudio, 'VTS Inspect Audio': VTSInspectAudio}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
