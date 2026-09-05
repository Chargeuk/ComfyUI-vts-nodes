# VAE Decode VTS (Tiled + Colour Match)

A separate decoder with the same tiled VAE inputs, Tensor/DiskImage outputs,
output paths, numbering and list suffixes as **VAE Decode VTS (Tiled)**.
Connect an optional `color_ref` to correct decoded frames before saving.
No reference, zero overall weight, or all component weights at zero bypasses
correction. Audio and input latents are not modified.

## Controls

- `color_match_method`: MKL, histogram matching, CPU Reinhard, MVGD,
  HM-MVGD-HM, HM-MKL-HM, or `reinhard_lab_gpu` (Kornia Lab statistics).
- `color_match_weight`: selected colour-transfer strength (default 0.5).
- `white_balance_weight`: reference-derived RGB balance, normalized to retain
  mean luminance. This preserves the reference's intended warm/cool cast.
- `brightness_method`: gamma or exposure, fitted to reference median luminance.
- `brightness_weight`: reference brightness strength.
- `contrast_weight`: match the reference's 1st/99th luminance percentiles.
- `overall_weight`: blend the combined result with the uncorrected frame.

Weights range from 0 (disabled) to 1 (full). The order is colour match, white
balance, brightness, contrast, then overall blend. Each additional stage measures
the result of the previous stage. Tone controls default to zero. These are
reference-derived adaptations of Donut-style operations, not calls to Donut's
automatic normalization node. Gain/gamma are limited to 0.25–4 to bound extreme
adjustments. No correction can recover detail already clipped during generation.

## Calculation modes

- `fixed_per_clip` (default): sample up to 16 evenly spaced frames across the
  decoded clip and reference sequence; fit one correction for the entire clip.
  This includes generated frames, not just the copied motion-context head.
- `per_frame`: calculate a correction for each frame. A single reference is reused.
  Multiple references are paired by relative index; a shorter sequence repeats
  its final frame and unused reference frames are ignored.
- `smoothed_over_time`: calculate per frame, then blend correction lookup tables.
  `smoothing=0` equals per-frame; `0.9` retains 90% of the previous transform;
  `1` holds the first transform. Video pixels from adjacent frames are never mixed.
  Temporal state resets for each node execution/list item.

Tensor and DiskImage references both work. Disk references load one selected
frame at a time. References are resized to the target analysis dimensions using
bilinear interpolation, without changing output resolution. Resizing fixes size
differences but cannot align different scenes. Fixed mode pools each sequence's
sampled thumbnails; it does not require matching sequence lengths.

## Precision and memory

Statistics use thumbnails (`analysis_size`, default maximum dimension 128).
Combined transforms are approximated by a trilinearly interpolated RGB lookup
table (`lut_resolution`: 17, 33 default, or 65). Results are therefore not
pixel-identical to running full-resolution KJ/VTS matchers. CPU Reinhard uses
log-LMS statistics; GPU Lab uses CIE Lab, so these are different algorithms.
The GPU option uses ComfyUI's selected compute device (CPU in CPU-only mode).
Transforms are applied in row strips, without uploading the full video to GPU.
The underlying tiled decoder still assembles its decoded frame tensor in memory,
just like the original node; DiskImage output does not make VAE decoding a stream.

Try the defaults for a clip-wide colour offset. For gradually changing drift,
try smoothed mode. To correct brightness alone, set colour weight to zero and
raise brightness weight. A fixed reference helps avoid cumulative grading drift,
but a different composition or intentional lighting change can affect matching.

The CPU matching methods require `color-matcher`; GPU Lab requires Kornia
(also a ComfyUI dependency). No KJNodes or Donut runtime dependency is needed.
