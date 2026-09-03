"""Disk-backed Qwen reference latent caching for ComfyUI.

The save/load nodes intentionally use files rather than a process-global cache so
the values survive prompt boundaries and their use is observable.  The Qwen
conditioning node keeps normal Qwen-VL image conditioning, but injects the
already encoded fixed/previous reference latents without asking the VAE to
encode those reference images again.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

import comfy.model_management
import comfy.utils
import node_helpers


_FORMAT = "vts_reference_latent_v1"


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _normalise_path(path: str) -> Path:
    value = os.path.expandvars(os.path.expanduser(path.strip()))
    if not value:
        raise ValueError("Latent cache path cannot be empty")
    result = Path(value)
    if result.suffix == "":
        result = result.with_suffix(".pt")
    return result.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, dict):
        return {str(key): _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return copy.deepcopy(value)
    raise TypeError(f"Unsupported LATENT cache value type: {type(value).__name__}")


def _move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def _samples(latent: dict, label: str) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{label} must be a LATENT dictionary containing 'samples'")
    tensor = latent["samples"]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{label}['samples'] must be a torch.Tensor")
    if tensor.ndim not in (4, 5):
        raise ValueError(f"{label} samples must be 4D or 5D, got {tuple(tensor.shape)}")
    if tensor.shape[1] != 16:
        raise ValueError(
            f"{label} is not a Qwen image latent: expected 16 channels, got {tensor.shape[1]}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains NaN or infinity")
    return tensor


def _append_audit(path: str, event: dict) -> None:
    if not path.strip():
        return
    audit_path = _normalise_path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class VTSSaveReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "path": (
                    "STRING",
                    {
                        "default": "/opt/ComfyUI/output/vts_latent_cache/first_latent_fisheye180.pt",
                        "multiline": False,
                    },
                ),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "label": ("STRING", {"default": "qwen_reference"}),
                "model_id": ("STRING", {"default": ""}),
                "vae_id": ("STRING", {"default": ""}),
                "audit_log_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "cache_path", "save_report")
    FUNCTION = "save"
    CATEGORY = "VTS/latent/cache"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def save(
        self,
        latent,
        path,
        overwrite=True,
        label="qwen_reference",
        model_id="",
        vae_id="",
        audit_log_path="",
    ):
        tensor = _samples(latent, "latent")
        destination = _normalise_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Latent cache already exists: {destination}")

        metadata = {
            "format": _FORMAT,
            "created_utc": _utc_now(),
            "label": label,
            "model_id": model_id,
            "vae_id": vae_id,
            "samples_shape": list(tensor.shape),
            "samples_dtype": str(tensor.dtype),
            "source_device": str(tensor.device),
        }
        payload = {
            "format": _FORMAT,
            "metadata": metadata,
            "latent": _cpu_copy(latent),
        }

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
            try:
                directory_fd = os.open(destination.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except (AttributeError, OSError):
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        digest = _sha256(destination)
        report = {
            "event": "save",
            "saved_utc": _utc_now(),
            "path": str(destination),
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
            "atomic_replace": True,
            "overwrite": bool(overwrite),
            **metadata,
        }
        _append_audit(audit_log_path, report)
        print("[VTS latent cache] " + json.dumps(report, sort_keys=True))
        return (latent, str(destination), json.dumps(report, indent=2, sort_keys=True))


class VTSLoadReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "default": "/opt/ComfyUI/output/vts_latent_cache/first_latent_fisheye180.pt",
                        "multiline": False,
                    },
                ),
                "device": (["intermediate", "cpu"], {"default": "intermediate"}),
            },
            "optional": {
                "expected_sha256": ("STRING", {"default": ""}),
                "expected_model_id": ("STRING", {"default": ""}),
                "expected_vae_id": ("STRING", {"default": ""}),
                "audit_log_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "metadata", "load_report")
    FUNCTION = "load"
    CATEGORY = "VTS/latent/cache"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # A cache path such as previous_latent_fisheye180.pt may be overwritten
        # between prompt submissions.  Always execute and really read the file.
        return float("nan")

    def load(
        self,
        path,
        device="intermediate",
        expected_sha256="",
        expected_model_id="",
        expected_vae_id="",
        audit_log_path="",
    ):
        source = _normalise_path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Latent cache not found: {source}")
        digest = _sha256(source)
        if expected_sha256.strip() and digest.lower() != expected_sha256.strip().lower():
            raise ValueError(
                f"Latent cache SHA-256 mismatch for {source}: expected "
                f"{expected_sha256.strip()}, got {digest}"
            )

        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise ValueError(f"Unsupported or corrupt VTS latent cache: {source}")
        metadata = payload.get("metadata")
        latent = payload.get("latent")
        if not isinstance(metadata, dict) or not isinstance(latent, dict):
            raise ValueError(f"Incomplete VTS latent cache: {source}")
        tensor = _samples(latent, "cached latent")
        if list(tensor.shape) != metadata.get("samples_shape"):
            raise ValueError(f"Cached latent metadata shape mismatch: {source}")
        if expected_model_id and metadata.get("model_id") != expected_model_id:
            raise ValueError(
                f"Cached latent model mismatch: expected {expected_model_id!r}, "
                f"got {metadata.get('model_id')!r}"
            )
        if expected_vae_id and metadata.get("vae_id") != expected_vae_id:
            raise ValueError(
                f"Cached latent VAE mismatch: expected {expected_vae_id!r}, "
                f"got {metadata.get('vae_id')!r}"
            )

        target_device = (
            comfy.model_management.intermediate_device()
            if device == "intermediate"
            else torch.device("cpu")
        )
        latent = _move_tensors(latent, target_device)
        loaded_tensor = _samples(latent, "loaded latent")
        report = {
            "event": "load",
            "loaded_utc": _utc_now(),
            "path": str(source),
            "sha256": digest,
            "size_bytes": source.stat().st_size,
            "file_mtime_ns": source.stat().st_mtime_ns,
            "loaded_from_disk": True,
            "target_device": str(loaded_tensor.device),
            "samples_shape": list(loaded_tensor.shape),
            "samples_dtype": str(loaded_tensor.dtype),
        }
        _append_audit(audit_log_path, report)
        print("[VTS latent cache] " + json.dumps(report, sort_keys=True))
        return (
            latent,
            json.dumps(metadata, indent=2, sort_keys=True),
            json.dumps(report, indent=2, sort_keys=True),
        )


def _qwen_vl_image(image: torch.Tensor) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    total = int(384 * 384)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    return comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)


def _qwen_vae_image(image: torch.Tensor) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    total = int(1024 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by / 8.0) * 8
    height = round(samples.shape[2] * scale_by / 8.0) * 8
    return comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)[:, :, :, :3]


class VTSQwenImageEditCachedReferences:
    """Build positive/negative Qwen conditioning with disk-loaded ref latents.

    Reference images still enter Qwen-VL so visual understanding is unchanged.
    Their matching latent inputs are injected directly and are never VAE encoded.
    The current image is VAE encoded once and the resulting latent is returned for
    both conditioning and sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamic_prompts": True, "default": " "}),
                "current_image": ("IMAGE",),
            },
            "optional": {
                "fixed_reference_image": ("IMAGE",),
                "fixed_reference_latent": ("LATENT",),
                "previous_reference_image": ("IMAGE",),
                "previous_reference_latent": ("LATENT",),
                "audit_log_path": ("STRING", {"default": ""}),
                "audit_label": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "current_latent", "encode_report")
    FUNCTION = "encode"
    CATEGORY = "VTS/conditioning/qwen image"

    def encode(
        self,
        clip,
        vae,
        positive_prompt,
        negative_prompt,
        current_image,
        fixed_reference_image=None,
        fixed_reference_latent=None,
        previous_reference_image=None,
        previous_reference_latent=None,
        audit_log_path="",
        audit_label="",
    ):
        pairs = [
            ("fixed", fixed_reference_image, fixed_reference_latent),
            ("previous", previous_reference_image, previous_reference_latent),
        ]
        for label, image, latent in pairs:
            if (image is None) != (latent is None):
                raise ValueError(
                    f"{label}_reference_image and {label}_reference_latent must be supplied together"
                )

        ordered_images = [current_image]
        ordered_latents = []
        current_tensor = vae.encode(_qwen_vae_image(current_image))
        _samples({"samples": current_tensor}, "current latent")
        ordered_latents.append(current_tensor)
        cached_labels = []
        for label, image, latent in pairs:
            if image is not None:
                ordered_images.append(image)
                ordered_latents.append(_samples(latent, f"{label} reference latent"))
                cached_labels.append(label)

        images_vl = [_qwen_vl_image(image) for image in ordered_images]
        image_prompt = "".join(
            f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(1, len(images_vl) + 1)
        )
        llama_template = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain how "
            "the user's text instruction should alter or modify the image. Generate "
            "a new image that meets the user's requirements while maintaining "
            "consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

        def make_conditioning(prompt):
            tokens = clip.tokenize(
                image_prompt + prompt,
                images=images_vl,
                llama_template=llama_template,
            )
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            return node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ordered_latents},
                append=True,
            )

        positive = make_conditioning(positive_prompt)
        negative = make_conditioning(negative_prompt)
        report = {
            "event": "qwen_cached_reference_encode",
            "encoded_utc": _utc_now(),
            "audit_label": audit_label,
            "current_vae_encode_count": 1,
            "cached_reference_vae_encode_count": 0,
            "cached_reference_labels": cached_labels,
            "qwen_vl_image_count": len(images_vl),
            "positive_and_negative_prompt_encodes": 2,
            "prompt_conditioning_cached": False,
            "current_latent_shape": list(current_tensor.shape),
        }
        _append_audit(audit_log_path, report)
        print("[VTS Qwen cached references] " + json.dumps(report, sort_keys=True))
        return (positive, negative, {"samples": current_tensor}, json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {
    "VTS_Save_Reference_Latent": VTSSaveReferenceLatent,
    "VTS_Load_Reference_Latent": VTSLoadReferenceLatent,
    "VTS_Qwen_Image_Edit_Cached_References": VTSQwenImageEditCachedReferences,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_Save_Reference_Latent": "VTS Save Reference Latent (Disk)",
    "VTS_Load_Reference_Latent": "VTS Load Reference Latent (Disk)",
    "VTS_Qwen_Image_Edit_Cached_References": "VTS Qwen Image Edit (Cached Reference Latents)",
}
