import json
import os
import sys
from collections.abc import Mapping

import torch
from comfy.nested_tensor import NestedTensor

_utils = os.path.join(os.path.dirname(__file__), "vtsUtils")
if _utils not in sys.path:
    sys.path.append(_utils)

from vts_disk_latent import DiskLatent, DiskNestedTensorInfo, DiskTensorInfo, latent_controls, materialize_latents, save_latent


def _describe(value):
    if isinstance(value, (torch.Tensor, DiskTensorInfo)):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (NestedTensor, DiskNestedTensorInfo)):
        return {"nested_tensors": [_describe(item) for item in value.tensors]}
    if isinstance(value, Mapping):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value]
    return value


class VTSLatentToDisk:
    OUTPUT_TOOLTIPS = ('DiskLatent reference to a losslessly compressed file, including tensor shape, dtype and saved device metadata.',)
    @classmethod
    def INPUT_TYPES(cls):
        controls = latent_controls("VTS_Latent_To_Disk")
        controls.pop("latent_return_type")
        return {"required": {"latent": ("LATENT", {"tooltip": "Native latent dictionary or DiskLatent file reference."})}, "optional": controls}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    CATEGORY = "VTS/latent/disk"

    def save(self, latent, latent_output_dir="./tmp/disklatents", latent_prefix="VTS_Latent_To_Disk",
             latent_start_sequence=0, latent_compression_level=3,
             latent_device_policy="Original", latent_device="cpu"):
        native = materialize_latents(latent, latent_device_policy, latent_device)
        return (save_latent(native, latent_prefix, latent_output_dir,
                            latent_start_sequence, latent_compression_level),)


class VTSDiskLatentFromFile:
    OUTPUT_TOOLTIPS = ('DiskLatent reference with metadata loaded; tensor data remains on disk.',)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_path": ("STRING", {"default": "", "tooltip": "Path to a VTS DiskLatent file. Reads only metadata here; tensors load when a processing node needs them. This is not a standard ComfyUI .latent file loader."})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "VTS/latent/disk"

    @classmethod
    def IS_CHANGED(cls, latent_path):
        # Re-read the manifest when a file changes; no tensor decompression here.
        stat = os.stat(os.path.expanduser(latent_path))
        return f"{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_size}"

    def load(self, latent_path):
        return (DiskLatent(latent_path),)


class VTSMaterializeLatent:
    OUTPUT_TOOLTIPS = ('Native latent dictionary with tensors loaded into memory using the selected device policy. Already-native inputs pass through unchanged.',)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT", {"tooltip": "Native latent dictionary or DiskLatent file reference."})},
                "optional": latent_controls("", has_output=False)}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "VTS/latent/disk"

    def load(self, latent, latent_device_policy="Original", latent_device="cpu"):
        return (materialize_latents(latent, latent_device_policy, latent_device),)


class VTSInspectLatent:
    OUTPUT_TOOLTIPS = ('JSON summary of shape, dtype and device, plus file size and compression ratio for DiskLatent. Inspection does not load disk-backed tensor data.',)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT", {"tooltip": "Native latent dictionary or DiskLatent file reference."})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = "VTS/latent/disk"
    OUTPUT_NODE = True

    def inspect(self, latent):
        if isinstance(latent, DiskLatent):
            report = {"path": latent.path, "stored_size_bytes": latent.stored_size_bytes,
                      "tensor_size_bytes": latent.tensor_size_bytes,
                      "compression_ratio": latent.tensor_size_bytes / latent.stored_size_bytes,
                      "is_disk_backed": True, "metadata": _describe(latent)}
        else:
            report = {"is_disk_backed": False, "metadata": _describe(latent)}
        text = json.dumps(report, indent=2)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "VTS Latent To Disk": VTSLatentToDisk,
    "VTS DiskLatent From File": VTSDiskLatentFromFile,
    "VTS Materialize Latent": VTSMaterializeLatent,
    "VTS Inspect Latent": VTSInspectLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
