"""Reference-derived colour transforms, fitted once or over time.

All stages are baked into an RGB lookup table so histogram and statistical
methods share the same temporal behavior. No state survives an execution.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import kornia.color

from comfy import model_management
from vtsUtils import DiskImage


METHODS = ["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm",
           "hm-mkl-hm", "reinhard_lab_gpu"]
MODES = ["fixed_per_clip", "per_frame", "smoothed_over_time"]


def _thumbnail(frame, size):
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError("Colour reference must contain RGB images.")
    if not torch.isfinite(frame).all():
        raise ValueError("Colour correction received non-finite image values.")
    rgb = frame[..., :3].float().clamp(0, 1)
    return F.interpolate(rgb.permute(2, 0, 1)[None], size=size,
                         mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def _reference_count(reference):
    if isinstance(reference, DiskImage):
        count = reference.number_of_images
    elif isinstance(reference, torch.Tensor) and reference.ndim == 4:
        count = reference.shape[0]
    else:
        raise ValueError("color_ref must be an IMAGE tensor or VTS DiskImage.")
    if not count:
        raise ValueError("color_ref contains no images.")
    return int(count)


def _reference_frame(reference, index):
    if isinstance(reference, DiskImage):
        return reference.materialize(start=index, count=1)[0]
    return reference[index]


def _indices(count, limit=16):
    if count <= limit:
        return list(range(count))
    return [round(i * (count - 1) / (limit - 1)) for i in range(limit)]


def _luminance(rgb):
    return rgb @ rgb.new_tensor([0.299, 0.587, 0.114])


def _lab_transform(src, ref):
    def to_lab(rgb):
        return kornia.color.rgb_to_lab(rgb.permute(2, 0, 1)[None])[0].permute(1, 2, 0)

    src_lab, ref_lab = to_lab(src), to_lab(ref)
    src_std, src_mean = torch.std_mean(src_lab.reshape(-1, 3), dim=0, unbiased=False)
    ref_std, ref_mean = torch.std_mean(ref_lab.reshape(-1, 3), dim=0, unbiased=False)
    # A flat source contains no contrast to expand. Match its mean safely.
    gain = torch.where(src_std > 1e-6, ref_std / src_std.clamp_min(1e-6), 1.0)

    def apply(rgb):
        lab = (to_lab(rgb) - src_mean) * gain + ref_mean
        return kornia.color.lab_to_rgb(lab.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
    return apply


def _cpu_color_transform(src, ref, method):
    # Optional backend, used only by the six color-matcher methods.
    from color_matcher import ColorMatcher
    from color_matcher.reinhard_matcher import LMS_MAT, LMS_MAT_INV

    source = src.cpu().numpy().astype(np.float64, copy=True)
    reference = ref.cpu().numpy().astype(np.float64, copy=True)
    stages = []
    methods = ["hm", "mvgd" if "mvgd" in method else "mkl", "hm"] if method.startswith("hm-") else [method]
    for stage in methods:
        if stage == "hm":
            curves = []
            for channel in range(3):
                x, counts = np.unique(source[..., channel], return_counts=True)
                y, ref_counts = np.unique(reference[..., channel], return_counts=True)
                mapped = np.interp(np.cumsum(counts) / counts.sum(),
                                   np.cumsum(ref_counts) / ref_counts.sum(), y)
                curves.append((x, mapped))

            def apply(rgb, curves=curves):
                return np.stack([np.interp(rgb[..., c], *curves[c]) for c in range(3)], axis=-1)
        elif stage == "reinhard":
            basis = np.diag([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]) @ np.array(
                [[1, 1, 1], [1, 1, -2], [1, -1, 0]])

            def to_log_lab(rgb):
                return np.log10(np.maximum(np.maximum(rgb, 1e-8) @ LMS_MAT.T, 1e-8)) @ basis.T

            a, b = to_log_lab(source).reshape(-1, 3), to_log_lab(reference).reshape(-1, 3)
            mean_a, mean_b = a.mean(0), b.mean(0)
            std_a, std_b = a.std(0), b.std(0)
            gain = np.divide(std_b, std_a, out=np.ones(3), where=std_a > 1e-6)

            def apply(rgb, mean_a=mean_a, mean_b=mean_b, gain=gain):
                lab = (to_log_lab(rgb) - mean_a) * gain + mean_b
                return np.power(10.0, np.clip(lab @ basis, -12, 6)) @ LMS_MAT_INV.T
        else:
            matcher = ColorMatcher(method=stage)
            # MVGD needs equal sample counts; reference thumbnails are resized
            # to the source analysis dimensions, without resizing output frames.
            if reference.shape != source.shape:
                ref_tensor = torch.from_numpy(reference).permute(2, 0, 1)[None]
                reference = F.interpolate(ref_tensor, size=source.shape[:2], mode="bilinear",
                                          align_corners=False)[0].permute(1, 2, 0).numpy()
            if np.max(source.reshape(-1, 3).std(0)) < 1e-6:
                matrix = np.eye(3)
                mean_a, mean_b = source.mean((0, 1)), reference.mean((0, 1))
            else:
                matcher.transfer(src=source.copy(), ref=reference.copy(), method=stage)
                matrix = np.real_if_close(matcher.transfer_mat)
                mean_a, mean_b = matcher.mu_r.ravel(), matcher.mu_z.ravel()

            def apply(rgb, matrix=matrix, mean_a=mean_a, mean_b=mean_b):
                return (rgb - mean_a) @ matrix.T + mean_b
        source = apply(source)
        stages.append(apply)

    def transform(rgb):
        values = rgb.cpu().numpy().astype(np.float64, copy=True)
        for apply in stages:
            values = apply(values)
        if np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("Colour matching produced an invalid transform; try reinhard_lab_gpu.")
        return torch.as_tensor(values, dtype=torch.float32, device=rgb.device)
    return transform


def _fit_lut(src, ref, method, color_weight, white_weight, brightness_weight,
             contrast_weight, brightness_method, resolution):
    axis = torch.linspace(0, 1, resolution, device=src.device)
    lattice = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    mapped = lattice.reshape(resolution * resolution, resolution, 3)
    current = src

    if color_weight:
        transform = _lab_transform(src, ref) if method == "reinhard_lab_gpu" else _cpu_color_transform(src, ref, method)
        mapped = torch.lerp(mapped, transform(mapped), color_weight).clamp(0, 1)
        current = torch.lerp(current, transform(current), color_weight).clamp(0, 1)

    if white_weight:
        src_mean, ref_mean = current.mean((0, 1)), ref.mean((0, 1))
        gain = ref_mean / src_mean.clamp_min(1e-6)
        gain = gain / (_luminance(src_mean * gain) / _luminance(src_mean).clamp_min(1e-6)).clamp_min(1e-6)
        gain = gain.clamp(0.25, 4.0)
        mapped = torch.lerp(mapped, mapped * gain, white_weight).clamp(0, 1)
        current = torch.lerp(current, current * gain, white_weight).clamp(0, 1)

    if brightness_weight:
        target = _luminance(ref).median().clamp(0.001, 0.999)
        if brightness_method == "gamma":
            # Fit actual output luminance; pow(mean, gamma) is only approximate.
            low, high = 0.25, 4.0
            for _ in range(20):
                gamma = (low + high) / 2
                if _luminance(current.clamp_min(1e-8).pow(gamma)).median() > target:
                    low = gamma
                else:
                    high = gamma
            corrected = current.clamp_min(1e-8).pow(gamma)
            mapped_new = mapped.clamp_min(1e-8).pow(gamma)
        else:
            gain = (target / _luminance(current).median().clamp_min(1e-6)).clamp(0.25, 4.0)
            corrected, mapped_new = current * gain, mapped * gain
        mapped = torch.lerp(mapped, mapped_new, brightness_weight).clamp(0, 1)
        current = torch.lerp(current, corrected, brightness_weight).clamp(0, 1)

    if contrast_weight:
        quantiles = current.new_tensor([0.01, 0.99])
        low, high = torch.quantile(_luminance(current).flatten(), quantiles)
        ref_low, ref_high = torch.quantile(_luminance(ref).flatten(), quantiles)
        gain = torch.where(high - low > 1e-6, (ref_high - ref_low) / (high - low).clamp_min(1e-6), 1.0).clamp(0.25, 4.0)
        mapped = torch.lerp(mapped, (mapped - low) * gain + ref_low, contrast_weight).clamp(0, 1)

    if not torch.isfinite(mapped).all():
        raise ValueError("Colour correction produced a non-finite lookup table.")
    return mapped.reshape(resolution, resolution, resolution, 3).permute(3, 0, 1, 2)[None].contiguous()


def _apply_lut(rgb, lut):
    # grid_sample coordinates are W,H,D, whereas the lattice axes are R,G,B.
    grid = rgb[..., [2, 1, 0]].clamp(0, 1).mul(2).sub(1)[None, None]
    return F.grid_sample(lut, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0, :, 0].permute(1, 2, 0)


def correct_images(images, reference, method="reinhard_lab_gpu", color_weight=0.5,
                   white_weight=0.0, brightness_weight=0.0, contrast_weight=0.0,
                   brightness_method="gamma", mode="fixed_per_clip", smoothing=0.9,
                   overall_weight=1.0, analysis_size=128, lut_resolution=33):
    """Yield corrected frames; references are loaded at most one frame at a time."""
    weights = (color_weight, white_weight, brightness_weight, contrast_weight, overall_weight)
    if any(not math.isfinite(w) or not 0 <= w <= 1 for w in weights):
        raise ValueError("Correction weights must be between 0 and 1.")
    if reference is None or overall_weight == 0 or not any(weights[:4]):
        yield from images
        return
    if method not in METHODS or mode not in MODES or brightness_method not in ("gamma", "exposure"):
        raise ValueError("Unknown colour correction method or calculation mode.")
    if not 0 <= smoothing <= 1 or not 16 <= analysis_size <= 512 or lut_resolution not in (17, 33, 65):
        raise ValueError("Invalid smoothing, analysis size, or lookup-table resolution.")
    if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 3:
        raise ValueError("Colour correction requires a non-empty RGB frame batch.")
    ref_count = _reference_count(reference)
    device = model_management.get_torch_device() if method == "reinhard_lab_gpu" else torch.device("cpu")
    h, w = images.shape[1:3]
    scale = min(1.0, analysis_size / max(h, w))
    size = (max(1, round(h * scale)), max(1, round(w * scale)))

    def sample(frame):
        return _thumbnail(frame.to(device), size)

    def fit(src, ref):
        return _fit_lut(src, ref, method, color_weight, white_weight,
                        brightness_weight, contrast_weight, brightness_method, lut_resolution)

    lut = None
    single_ref = sample(_reference_frame(reference, 0)) if ref_count == 1 else None
    if mode == "fixed_per_clip":
        source = torch.cat([sample(images[i]) for i in _indices(len(images))], dim=0)
        ref = single_ref if single_ref is not None else torch.cat(
            [sample(_reference_frame(reference, i)) for i in _indices(ref_count)], dim=0)
        lut = fit(source, ref)
        del source, ref

    for i, frame in enumerate(images):
        if mode != "fixed_per_clip":
            ref = single_ref if single_ref is not None else sample(_reference_frame(reference, min(i, ref_count - 1)))
            fresh = fit(sample(frame), ref)
            lut = torch.lerp(fresh, lut, smoothing) if mode == "smoothed_over_time" and lut is not None else fresh
        result = frame.clone()
        # Bound temporary GPU storage for high-resolution/VR180 frames.
        for row in range(0, h, 128):
            original = frame[row:row + 128, :, :3].to(device=device, dtype=torch.float32)
            corrected = _apply_lut(original, lut)
            result[row:row + 128, :, :3] = torch.lerp(original, corrected, overall_weight).clamp(0, 1).to(frame)
        yield result
