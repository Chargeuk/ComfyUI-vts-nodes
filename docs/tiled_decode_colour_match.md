# VAE Decode VTS (Tiled + Colour Match)

A separate decoder with the same tiled VAE inputs, Tensor/DiskImage outputs,
output paths, numbering and list suffixes as **VAE Decode VTS (Tiled)**.
Connect an optional `color_ref` to correct decoded frames before saving.
No reference, zero overall weight, or all component weights at zero bypasses
correction. Optional Merserk processing runs after colour correction. Audio and
input latents are not modified.

## Corrected H3 continuation context

`encode_corrected_context` defaults to **false**, preserving the existing image
output and doing no extra encoding. The second output,
`corrected_video_context`, is `None` when disabled. Existing image links stay on
output 0.

Enable it to encode only the final `context_length` selected frames directly
from the in-memory image tensor, before disk compression. Lengths are 5, 22
(default), 39 or 56 frames; shorter clips use the largest valid length available
(including one frame). It requires a single H3 video and its matching video VAE.
It uses the VAE's normal encode method and memory management, independently of
the decoder's tile settings. Encoding adds work and temporary memory use; this
is an optional colour-continuity experiment, not a speed optimisation.

Connect this output to `corrected_video_context` on **VTS H3 Prepare Loop Context**.
Keep the original AV sampler latent connected to Prepare's `context_latent` so
audio and its timing remain unchanged. Set the same `context_length` on both
nodes, and use the same source clip. Prepare rejects mismatched lengths and
resolutions; it cannot identify unrelated clips with identical shapes.
Apply Loop Context uses the corrected video for both its guide and masked prefix.
Repeat this wiring for the initial clip and for the loop body as applicable.

Context selection does not change the main Tensor/DiskImage output. No saved images are read back,
and the new output owns only the small encoded tail, not the full images or
original AV latent. If colour correction is bypassed while encoding is enabled,
the tail is still re-encoded. With Merserk excluded from context it uses the
unchanged decoded images; with Merserk included it uses the processed result. Disconnecting
the new output alone does not disable encoding: turn its flag off as well.
Use a stable colour reference to avoid repeatedly compounding a grade. Compare
several continuations for colour, detail and motion before relying on this path.

## Optional Merserk processing

`enable_merserk` defaults to **false**, preserving existing workflows. When on,
the order is:

1. Decode the entire sequence using the existing tiled VAE settings.
2. Apply the enabled colour correction.
3. Run the existing **VTS Merserk Temporal Enhance** processing: resize, neural
   enhancement, then optional interpolation.
4. Encode the selected continuation tail if requested, then save/return the
   final image sequence.

The `merserk_` controls reuse the standalone temporal node's settings, defaults
and tooltips, including server URL/timeout, independent scaling/neural/
interpolation switches, Scale to Min or Multiplier sizing, native neural passes,
style, tone/skin/detail controls and shimmer suppression. There are no outer
iterations. Scale to Min accepts reversed side limits, enlargement uses RTX VSR,
and reduction uses local Lanczos. The master switch overrides every Merserk
setting. Local reduction and a complete bypass do not require a server.

The decoder's existing `return_type`, path, prefix, numbering, image format,
compression and quality controls apply to the final output only. Merserk receives
and returns lossless 8-bit PNG images; floating-point decoded values are
clamped/quantized to 8 bits for this processing, so this is not HDR or lossless
floating-point transport. JPEG is available for final disk output independently
of transport. No intermediate images are saved by this integration.

Interpolation supports 2x, 3x, 4x and 8x, producing `(N - 1) * multiplier + 1`
frames. For example, 5 input frames at 2x produce 9 output frames. 3x uses the
existing approximate positions at 37.5% and 62.5%, not exact thirds. A single
frame remains one frame. Set the downstream video output FPS separately to
source FPS times the multiplier; this node returns images, not a video file.

### Which frames enter the continuation context?

`encode_corrected_context` still controls whether encoding happens at all.
`use_merserk_for_context` defaults to **false**:

- **Off:** encode the colour-corrected tail before Merserk. Main image output
  still includes all enabled Merserk operations.
- **On:** encode the Merserk result when `enable_merserk` is on. If the master
  switch is off, use the usual colour-corrected frames.

With Merserk included and interpolation enabled, `context_frame_selection`
chooses the tail's cadence:

- **Original** (default): select enhanced source frames, skipping inserted
  frames. With 2x interpolation these are output indices 0, 2, 4, and so on.
- **Interpolated:** use consecutive frames from the end of the expanded output,
  including inserted frames. A five-frame context covers a shorter period of
  source motion than five original-cadence frames.

Both choices keep the existing `context_length` rules, capped by the original
clip's available H3 length, and preserve compatibility with Prepare Loop Context.
Selected frames automatically resize back to the source latent's width and
height using Lanczos before VAE encoding. There is no resize toggle: the
replacement latent must match the source geometry. This does not undo any crop
already applied by Merserk; it restores dimensions, not removed image content.
The main output retains its processed resolution and expanded frame count.

**H3 context still uses its fixed 24 FPS timebase.** Interpolated selection does
not enable true 48/72/96/192 FPS conditioning. A denser tail represents less
motion per conditioned frame and can slow apparent continuation motion or
misalign that motion with the unchanged original audio. Original selection is
the default for preserving the source cadence. Neither option changes source
audio, H3 timing metadata, or the downstream video writer's FPS.

The context is encoded directly from selected in-memory frames before any final
JPEG/WebP/PNG save. Saved files are never read back for encoding. Network errors
or cancellation propagate through the existing Merserk client instead of silently
returning an unenhanced result. Its queue, worker-recovery retry and cancellation
behavior are shared with the standalone node.

This first integration is sequential: Merserk starts after decoding and colour
correction finish. It reuses the client's bounded network buffers, but retains
full input and processed image batches during processing, even for DiskImage
output. Interpolation and enlargement can substantially increase RAM use. Only
the selected context tail is retained separately; returned context latents own
their storage independently of the full clip.

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
