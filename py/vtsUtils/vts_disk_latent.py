"""Lossless, disk-backed LATENT values. Metadata access never loads tensor data."""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import safetensors.torch
import torch
import zstandard
from comfy.nested_tensor import NestedTensor

from vtsUtils import resolve_list_mapped_output_identity


DEFAULT_LATENT_DIR = "./tmp/disklatents"
MAGIC = b"VTSDLAT1"
MAX_MANIFEST_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


def _conversion(device, dtype):
    if isinstance(device, torch.dtype):
        if dtype is not None:
            raise TypeError("Specify dtype only once")
        device, dtype = None, device
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    return str(torch.device(device)) if device is not None else None, dtype


def _needs_materialization(*args, **kwargs):
    raise TypeError("DiskLatent tensor data is on disk; call DiskLatent.materialize() before tensor operations")


@dataclass(frozen=True)
class DiskTensorInfo:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    original_device: torch.device
    is_disk_backed = True
    is_nested = False
    layout = torch.strided
    requires_grad = False

    def __getattr__(self, name):
        raise AttributeError(f"DiskTensorInfo.{name} needs a native tensor; call DiskLatent.materialize() first")

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def clone(self):
        return replace(self)

    detach = clone

    def to(self, device=None, dtype=None):
        device, dtype = _conversion(device, dtype)
        return replace(self, device=torch.device(device) if device else self.device,
                       dtype=dtype or self.dtype)

    def cpu(self):
        return self.to("cpu")

    def float(self):
        return self.to(dtype=torch.float32)

    __getitem__ = __add__ = __sub__ = __mul__ = __truediv__ = _needs_materialization
    __radd__ = __rsub__ = __rmul__ = __rtruediv__ = _needs_materialization

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return _needs_materialization()


@dataclass(frozen=True)
class DiskNestedTensorInfo:
    tensors: tuple
    is_disk_backed = True
    is_nested = True

    def __getattr__(self, name):
        raise AttributeError(f"DiskNestedTensorInfo.{name} needs native tensors; call DiskLatent.materialize() first")

    def unbind(self):
        return list(self.tensors)

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def ndim(self):
        return max(t.ndim for t in self.tensors)

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def device(self):
        return self.tensors[0].device

    @property
    def layout(self):
        return torch.strided

    def size(self):
        return self.shape

    def to(self, device=None, dtype=None):
        return DiskNestedTensorInfo(tuple(t.to(device, dtype) for t in self.tensors))

    def clone(self):
        return replace(self)

    detach = clone

    def cpu(self):
        return self.to("cpu")

    def float(self):
        return self.to(dtype=torch.float32)

    __getitem__ = __add__ = __sub__ = __mul__ = __truediv__ = _needs_materialization


def _encode(value, tensors, descriptions, seen):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_nested or value.is_quantized:
            raise TypeError("DiskLatent supports dense strided tensors and ComfyUI NestedTensor")
        key = seen.get(id(value))
        if key is None:
            key = f"tensor_{len(tensors)}"
            seen[id(value)] = key
            # Own each allocation: safetensors cannot serialize overlapping storage.
            tensors[key] = value.detach().to(device="cpu", copy=True).contiguous()
            descriptions[key] = {"shape": list(value.shape), "dtype": str(value.dtype)[6:],
                                 "device": str(value.device), "bytes": value.numel() * value.element_size()}
        return {"type": "tensor", "key": key}
    if isinstance(value, NestedTensor):
        return {"type": "nested", "items": [_encode(t, tensors, descriptions, seen) for t in value.tensors]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("DiskLatent dictionary keys must be strings")
        return {"type": "dict", "items": {k: _encode(v, tensors, descriptions, seen) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(v, tensors, descriptions, seen) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return {"type": "value", "value": value}
    raise TypeError(f"Unsupported DiskLatent value: {type(value).__name__}")


def _decode(tree, tensor, metadata=False):
    kind = tree["type"]
    if kind == "tensor":
        return tensor(tree["key"])
    if kind == "value":
        return tree["value"]
    if kind == "dict":
        return {k: _decode(v, tensor, metadata) for k, v in tree["items"].items()}
    items = [_decode(v, tensor, metadata) for v in tree["items"]]
    if kind == "list":
        return items
    if kind == "tuple":
        return tuple(items)
    if kind == "nested":
        return DiskNestedTensorInfo(tuple(items)) if metadata else NestedTensor(items)
    raise ValueError(f"Unknown DiskLatent structure type: {kind}")


def _read_manifest(stream):
    if stream.read(8) != MAGIC:
        raise ValueError("Not a VTS DiskLatent file")
    length = stream.read(8)
    if len(length) != 8:
        raise ValueError("Truncated DiskLatent header")
    size = struct.unpack("<Q", length)[0]
    if not 0 < size <= MAX_MANIFEST_SIZE:
        raise ValueError("Invalid DiskLatent manifest size")
    encoded = stream.read(size)
    if len(encoded) != size:
        raise ValueError("Truncated DiskLatent manifest")
    manifest = json.loads(encoded)
    if manifest["version"] != 1 or manifest["compression"] != "zstd":
        raise ValueError("Unsupported DiskLatent format version or compression")
    if manifest["structure"]["type"] != "dict" or "samples" not in manifest["structure"]["items"]:
        raise ValueError("DiskLatent must contain a LATENT dictionary with samples")
    if not isinstance(manifest["payload_size"], int) or manifest["payload_size"] < 0:
        raise ValueError("Invalid DiskLatent payload size")
    return manifest, hashlib.sha256(encoded).hexdigest()


class DiskLatent(Mapping):
    is_disk_backed = True

    def __init__(self, path, device=None, dtype=None):
        self.path = str(Path(path).expanduser().resolve())
        self._device, self._dtype = _conversion(device, dtype)
        with open(self.path, "rb") as stream:
            self._manifest, self._identity = _read_manifest(stream)
        # Validate metadata without touching the compressed payload.
        self._metadata()

    def _metadata(self):
        def describe(key):
            info = self._manifest["tensors"][key]
            dtype = getattr(torch, info["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Unsupported tensor dtype: {info['dtype']}")
            if any(type(n) is not int or n < 0 for n in info["shape"]):
                raise ValueError("Invalid tensor shape")
            if math.prod(info["shape"]) * dtype.itemsize != info["bytes"]:
                raise ValueError("Tensor size does not match DiskLatent metadata")
            return DiskTensorInfo(torch.Size(info["shape"]), self._dtype or dtype,
                                  torch.device(self._device or info["device"]), torch.device(info["device"]))
        return _decode(self._manifest["structure"], describe, metadata=True)

    def __getitem__(self, key):
        return self._metadata()[key]

    def __iter__(self):
        return iter(self._manifest["structure"]["items"])

    def __len__(self):
        return len(self._manifest["structure"]["items"])

    def copy(self):
        return dict(self)

    def clone(self):
        result = copy.copy(self)
        result._manifest = copy.deepcopy(self._manifest)
        return result

    detach = clone

    def to(self, device=None, dtype=None):
        device, dtype = _conversion(device, dtype)
        result = self.clone()
        result._device = device or self._device
        result._dtype = dtype or self._dtype
        return result

    def cpu(self):
        return self.to("cpu")

    @property
    def stored_size_bytes(self):
        return os.path.getsize(self.path)

    @property
    def tensor_size_bytes(self):
        return sum(t["bytes"] for t in self._manifest["tensors"].values())

    def materialize(self, device=None, dtype=None):
        device, dtype = _conversion(device, dtype)
        device, dtype = device or self._device, dtype or self._dtype
        with open(self.path, "rb") as source, tempfile.NamedTemporaryFile(suffix=".safetensors") as raw:
            manifest, identity = _read_manifest(source)
            if identity != self._identity:
                raise ValueError("DiskLatent file changed since this reference was created")
            if os.fstat(source.fileno()).st_size - source.tell() != manifest["compressed_size"]:
                raise ValueError("DiskLatent compressed payload is truncated or has trailing data")
            digest, size = hashlib.sha256(), 0
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                while block := reader.read(CHUNK_SIZE):
                    size += len(block)
                    if size > manifest["payload_size"]:
                        raise ValueError("DiskLatent payload exceeds its declared size")
                    digest.update(block)
                    raw.write(block)
            if size != manifest["payload_size"] or digest.hexdigest() != manifest["sha256"]:
                raise ValueError("DiskLatent payload integrity check failed")
            raw.flush()
            stored = safetensors.torch.load_file(raw.name, device="cpu")
            if set(stored) != set(manifest["tensors"]):
                raise ValueError("DiskLatent tensor keys do not match its manifest")
            loaded = {}
            for key, value in stored.items():
                info = manifest["tensors"][key]
                if list(value.shape) != info["shape"] or str(value.dtype)[6:] != info["dtype"]:
                    raise ValueError("DiskLatent tensor does not match its manifest")
                # copy=True releases the decompressed mmap even for CPU destinations.
                loaded[key] = value.to(device=device or info["device"], dtype=dtype or value.dtype, copy=True)
            del stored
        return _decode(manifest["structure"], loaded.__getitem__)

    def __repr__(self):
        return f"DiskLatent(path={self.path!r}, metadata={self._metadata()!r})"


def save_latent(latent, prefix="latent", output_dir=DEFAULT_LATENT_DIR,
                start_sequence=0, compression_level=3):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("Expected a native LATENT dictionary containing samples")
    if not isinstance(prefix, str) or not prefix.strip() or any(c in prefix for c in '/\\\0') or prefix in (".", ".."):
        raise ValueError("latent_prefix must be a filename prefix, not a path")
    if type(compression_level) is not int or not 1 <= compression_level <= 19:
        raise ValueError("latent_compression_level must be between 1 and 19")
    if type(start_sequence) is not int or start_sequence < 0:
        raise ValueError("latent_start_sequence must be a nonnegative integer")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix, start_sequence = resolve_list_mapped_output_identity(prefix, start_sequence)
    destination = directory / f"{prefix}_{start_sequence:06d}_{uuid.uuid4().hex}.disklatent"
    tensors, descriptions = {}, {}
    structure = _encode(latent, tensors, descriptions, {})
    temporary = None
    try:
        with tempfile.TemporaryDirectory(dir=directory, prefix=".disklatent-") as scratch:
            raw_path = os.path.join(scratch, "tensors.safetensors")
            safetensors.torch.save_file(tensors, raw_path)
            del tensors
            digest = hashlib.sha256()
            with open(raw_path, "rb") as raw:
                while block := raw.read(CHUNK_SIZE):
                    digest.update(block)
                size = raw.tell()
            compressed_path = os.path.join(scratch, "tensors.zst")
            with open(raw_path, "rb") as raw, open(compressed_path, "wb") as compressed:
                zstandard.ZstdCompressor(level=compression_level, write_checksum=True).copy_stream(raw, compressed)
            manifest = {"version": 1, "compression": "zstd", "compression_level": compression_level,
                        "structure": structure, "tensors": descriptions,
                        "payload_size": size, "sha256": digest.hexdigest(),
                        "compressed_size": os.path.getsize(compressed_path)}
            encoded = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            if len(encoded) > MAX_MANIFEST_SIZE:
                raise ValueError("DiskLatent metadata is too large")
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".disklatent.tmp", delete=False) as out:
                temporary = out.name
                out.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
                with open(compressed_path, "rb") as compressed:
                    shutil.copyfileobj(compressed, out, CHUNK_SIZE)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, destination)
            temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return DiskLatent(destination)


def materialize_latents(value, device_policy="Original", device="cpu", memo=None):
    if device_policy not in ("Original", "CPU", "Specified"):
        raise ValueError(f"Unknown latent device policy: {device_policy}")
    if memo is None:
        memo = {}
    if isinstance(value, DiskLatent):
        key = id(value)
        if key not in memo:
            target = None if device_policy == "Original" else "cpu" if device_policy == "CPU" else device
            memo[key] = value.materialize(device=target)
        return memo[key]
    if isinstance(value, (dict, list, tuple)):
        items = value.values() if isinstance(value, dict) else value
        converted = [materialize_latents(v, device_policy, device, memo) for v in items]
        items = list(value.values()) if isinstance(value, dict) else value
        if all(a is b for a, b in zip(items, converted)):
            return value
        if isinstance(value, dict):
            return dict(zip(value, converted))
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


from vts_tooltips import DEVICE_HELP, POLICY_HELP


def latent_controls(prefix, has_output=True, has_input=True, control_prefix="latent_", match_input=True):
    controls = {}
    if has_output:
        modes = ["Tensor", "DiskLatent", "Input", "Input or DiskLatent"] if has_input and match_input else ["Tensor", "DiskLatent"]
        controls.update({
            "return_type": (modes, {"default": "Tensor", "tooltip": "Tensor returns native latents. DiskLatent writes a lossless compressed file. Input matches the latent input; Input or DiskLatent keeps disk outputs or saves native outputs."}),
            "output_dir": ("STRING", {"default": DEFAULT_LATENT_DIR, "tooltip": "Folder for losslessly compressed DiskLatent files. Default: ./tmp/disklatents, relative to ComfyUI working directory. Independent of the DiskImage output folder."}),
            "prefix": ("STRING", {"default": re.sub(r"\s+", "_", prefix.strip()), "tooltip": "DiskLatent filename prefix, using the same node-name logic as DiskImage. Sequence, list position and a unique suffix distinguish saved files."}),
            "start_sequence": ("INT", {"default": 0, "min": 0, "tooltip": "Starting sequence number for DiskLatent filenames. Independent of the image sequence counter."}),
            "compression_level": ("INT", {"default": 3, "min": 1, "max": 19, "tooltip": "Lossless Zstandard compression. Higher levels take longer and may not save much more space."}),
        })
    if has_input:
        controls.update({
            "device_policy": (["Original", "CPU", "Specified"], {"default": "Original", "tooltip": POLICY_HELP}),
            "device": ("STRING", {"default": "cpu", "tooltip": DEVICE_HELP}),
        })
    return {control_prefix + key: spec for key, spec in controls.items()}


def take_latent_controls(kwargs, specs, control_prefix="latent_", input_is_list=False):
    controls = {}
    for key, spec in specs.items():
        value = kwargs.pop(key, spec[1]["default"])
        if input_is_list and isinstance(value, list):
            value = value[0] if value else spec[1]["default"]
        controls[key[len(control_prefix):]] = value
    return controls


def _disk_inputs(value):
    if isinstance(value, DiskLatent):
        return [True]
    if isinstance(value, dict) and "samples" in value:
        return [False]
    if isinstance(value, dict):
        return [item for v in value.values() for item in _disk_inputs(v)]
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in _disk_inputs(v)]
    return []


def latent_output_mode(controls, inputs):
    mode = controls.get("return_type", "Tensor")
    if mode == "Input":
        representations = [flag for value in inputs for flag in _disk_inputs(value)]
        if not representations or len(set(representations)) != 1:
            raise ValueError("latent_return_type=Input needs latent inputs of one representation; choose Tensor or DiskLatent")
        return "DiskLatent" if representations[0] else "Tensor"
    if mode == "Input or DiskLatent":
        return "DiskLatent"
    if mode not in ("Tensor", "DiskLatent"):
        raise ValueError(f"Unknown latent return type: {mode}")
    return mode


def process_latent_outputs(outputs, indexes, names, controls, mode):
    outputs = list(outputs)

    def convert(value, prefix):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            values = [convert(v, f"{prefix}_item_{i:06d}") for i, v in enumerate(value)]
            return tuple(values) if isinstance(value, tuple) else values
        if mode == "Tensor":
            return materialize_latents(value, controls.get("device_policy", "Original"), controls.get("device", "cpu"))
        if isinstance(value, DiskLatent):
            return value
        return save_latent(value, prefix=prefix, output_dir=controls["output_dir"],
                           start_sequence=controls["start_sequence"], compression_level=controls["compression_level"])

    for index in indexes:
        prefix = controls.get("prefix", "latent")
        if len(indexes) > 1:
            prefix += "_" + (re.sub(r"[^A-Za-z0-9_-]+", "_", names[index]).strip("_") or "latent")
        outputs[index] = convert(outputs[index], prefix)
    return tuple(outputs)
