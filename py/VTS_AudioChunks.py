import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
from contextlib import contextmanager

import soundfile as sf
import torch

import folder_paths
import comfy.model_management as model_management


BLOCK_FRAMES = 65536
MANIFEST_TYPE = "VTS_AUDIO_CHUNKS"


def _output_root():
    return Path(folder_paths.get_output_directory()).resolve()


def _relative_path(root, value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Audio paths must be nonempty relative paths under the output directory")
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(value).drive or any(part in ("", ".", "..") for part in value.replace("\\", "/").split("/")):
        raise ValueError("Audio paths must stay under the output directory; absolute paths and traversal are not allowed")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Audio path resolves outside the output directory")
    return resolved


@contextmanager
def _operation_directory(root, filename_prefix):
    prefix = _relative_path(root, filename_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    parent = prefix.parent.resolve()
    if not parent.is_relative_to(root):
        raise ValueError("Audio output directory resolves outside the output directory")
    directory = Path(tempfile.mkdtemp(prefix=prefix.name + "_", dir=parent))
    completed = False
    try:
        yield directory
        completed = True
    finally:
        if not completed:
            # Only this operation's freshly-created private directory is owned here.
            shutil.rmtree(directory)


def _publish(temporary, final):
    if final.exists():
        raise FileExistsError(f"Audio output already exists: {final}")
    os.rename(temporary, final)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_record(root, relative):
    path = _relative_path(root, relative)
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("Unsupported audio chunk manifest version")
    for name in ("sample_rate", "channels", "frames", "total_frames", "chunk_count"):
        _positive_int(record.get(name), name)
    if record.get("chunk") != "audio.wav":
        raise ValueError("Invalid audio chunk filename")
    return path, record


def _head(root, chunks):
    if not isinstance(chunks, dict) or chunks.get("version") != 1:
        raise ValueError("Expected a VTS audio chunk manifest")
    path, record = _read_record(root, chunks.get("manifest"))
    for key, field in (("sample_rate", "sample_rate"), ("channels", "channels"), ("frames", "total_frames"), ("chunks", "chunk_count")):
        if type(chunks.get(key)) is not int or chunks[key] != record[field]:
            raise ValueError(f"Audio chunk descriptor disagrees with its manifest: {key}")
    return path, record


def save_audio_chunk(audio, filename_prefix="audio/chunk", chunks=None):
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Save Audio Chunk requires waveform shape [1, channels, samples]")
    if waveform.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Save Audio Chunk supports float32, float16, and bfloat16 audio")
    sample_rate = _positive_int(audio["sample_rate"], "sample_rate")
    channels = _positive_int(waveform.shape[1], "channels")
    frames = _positive_int(waveform.shape[2], "samples")
    root = _output_root()
    previous, count, total = None, 0, 0
    if chunks is not None:
        previous_path, record = _head(root, chunks)
        if (record["sample_rate"], record["channels"]) != (sample_rate, channels):
            raise ValueError("Audio chunks must have the same sample rate and channel count")
        previous = previous_path.relative_to(root).as_posix()
        count, total = record["chunk_count"], record["total_frames"]
    with _operation_directory(root, filename_prefix) as directory:
        temporary = directory / "audio.partial.wav"
        with sf.SoundFile(temporary, "w", samplerate=sample_rate, channels=channels, format="RF64", subtype="FLOAT") as sink:
            for start in range(0, frames, BLOCK_FRAMES):
                model_management.throw_exception_if_processing_interrupted()
                block = waveform[0, :, start:start + BLOCK_FRAMES].detach().to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous().numpy()
                sink.write(block)
                del block
        model_management.throw_exception_if_processing_interrupted()
        _publish(temporary, directory / "audio.wav")
        record = {"version": 1, "chunk": "audio.wav", "previous": previous,
                  "sample_rate": sample_rate, "channels": channels, "frames": frames,
                  "total_frames": total + frames, "chunk_count": count + 1}
        temporary = directory / "manifest.partial.json"
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(record, handle)
        _publish(temporary, directory / "manifest.json")
        descriptor = {"version": 1, "manifest": (directory / "manifest.json").relative_to(root).as_posix(),
                      "sample_rate": sample_rate, "channels": channels, "frames": total + frames, "chunks": count + 1}
    return descriptor, str(directory / "audio.wav")


def _ordered_chunks(root, chunks):
    path, record = _head(root, chunks)
    sample_rate, channels = record["sample_rate"], record["channels"]
    expected_count, expected_frames = record["chunk_count"], record["total_frames"]
    ordered = []
    seen = set()
    while True:
        model_management.throw_exception_if_processing_interrupted()
        if path in seen:
            raise ValueError("Audio chunk manifest contains a cycle")
        seen.add(path)
        if (record["sample_rate"], record["channels"]) != (sample_rate, channels):
            raise ValueError("Audio chunks must have the same sample rate and channel count")
        if record["chunk_count"] != expected_count or record["total_frames"] != expected_frames:
            raise ValueError("Audio chunk manifest order or sample counts are inconsistent")
        chunk_path = _relative_path(root, (path.parent / record["chunk"]).relative_to(root).as_posix())
        ordered.append((chunk_path, record["frames"]))
        expected_count -= 1
        expected_frames -= record["frames"]
        previous = record.get("previous")
        if previous is None:
            if expected_count != 0 or expected_frames != 0:
                raise ValueError("Audio chunk manifest history is incomplete")
            break
        if expected_count <= 0 or expected_frames <= 0:
            raise ValueError("Audio chunk manifest history is inconsistent")
        path, record = _read_record(root, previous)
    ordered.reverse()
    return ordered


def assemble_audio_chunks(chunks, filename_prefix="audio/assembled", return_audio=False):
    if type(return_audio) is not bool:
        raise ValueError("return_audio must be a boolean")
    root = _output_root()
    ordered = _ordered_chunks(root, chunks)
    sample_rate, channels = chunks["sample_rate"], chunks["channels"]
    audio = None
    with _operation_directory(root, filename_prefix) as directory:
        temporary = directory / "audio.partial.wav"
        with sf.SoundFile(temporary, "w", samplerate=sample_rate, channels=channels, format="RF64", subtype="FLOAT") as sink:
            for path, expected_frames in ordered:
                with sf.SoundFile(path, "r") as source:
                    if (source.samplerate, source.channels, len(source)) != (sample_rate, channels, expected_frames):
                        raise ValueError("Audio chunk file disagrees with its manifest")
                    if source.format != "RF64" or source.subtype != "FLOAT":
                        raise ValueError("Audio chunks must use RF64 float32 WAV")
                    for block in source.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
                        model_management.throw_exception_if_processing_interrupted()
                        sink.write(block)
                        del block
        model_management.throw_exception_if_processing_interrupted()
        if return_audio:
            waveform, rate = sf.read(temporary, dtype="float32", always_2d=True)
            audio = {"waveform": torch.from_numpy(waveform.T).unsqueeze(0), "sample_rate": rate}
        _publish(temporary, directory / "audio.wav")
    return str(directory / "audio.wav"), audio


class VTSSaveAudioChunk:
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "filename_prefix": ("STRING", {"default": "audio/chunk"})},
                "optional": {"chunks": (MANIFEST_TYPE,)}}

    RETURN_TYPES = (MANIFEST_TYPE, "STRING")
    RETURN_NAMES = ("chunks", "chunk_path")
    FUNCTION = "save"
    CATEGORY = "VTS/audio"
    OUTPUT_NODE = True
    DESCRIPTION = "Save already overlap-trimmed audio losslessly as float32 WAV and append a small disk manifest. Prefix is relative to the ComfyUI output directory."

    def save(self, audio, filename_prefix="audio/chunk", chunks=None):
        return save_audio_chunk(audio, filename_prefix, chunks)


class VTSAssembleAudioChunks:
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"chunks": (MANIFEST_TYPE,), "filename_prefix": ("STRING", {"default": "audio/assembled"}),
                             "return_audio": ("BOOLEAN", {"default": False, "tooltip": "Loads the entire final waveform into CPU RAM when enabled. The tensor shares this buffer; file-only mode returns no AUDIO tensor."})}}

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("audio_path", "audio")
    FUNCTION = "assemble"
    CATEGORY = "VTS/audio"
    OUTPUT_NODE = True
    DESCRIPTION = "Stream saved audio chunks into one lossless RF64 WAV in manifest order. File-only mode keeps audio RAM bounded by a block; full AUDIO output is opt-in."

    def assemble(self, chunks, filename_prefix="audio/assembled", return_audio=False):
        return assemble_audio_chunks(chunks, filename_prefix, return_audio)


NODE_CLASS_MAPPINGS = {"VTS Save Audio Chunk": VTSSaveAudioChunk, "VTS Assemble Audio Chunks": VTSAssembleAudioChunks}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
