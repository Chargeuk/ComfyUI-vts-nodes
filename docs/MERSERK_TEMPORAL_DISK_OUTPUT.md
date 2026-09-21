# Merserk Temporal Enhance: disk output

Select `return_type = DiskImage` to write final frames on the ComfyUI machine.
Disk encoding is independent of the lossless PNG transport to/from Merserk.

| Setting | Behaviour |
| --- | --- |
| `format` | `jpg`, `webp`, or `png`. PNG remains the default for compatibility. |
| `quality` | JPG or lossy WebP quality 1-100; 101 forces lossless WebP and uses quality 100 for JPG. Ignored for PNG. |
| `webp_lossless` | On by default, preserving WebP pixels and alpha exactly. Turn off for lossy WebP at quality 1-100; quality 101 stays lossless. |
| `compression_level` | PNG compression 0-9; WebP method clamped to 0-6; ignored for JPG. Default 1 preserves the old PNG setting. |
| `prefix` | Frame filename prefix; default `frame`. Paths are not accepted here. |
| `start_sequence` | First frame number; default 0. Does not select/skip input frames. |
| `output_dir` | Parent directory. Blank uses ComfyUI output/merserk_temporal. Each execution creates a unique sequence subfolder to avoid overwriting earlier outputs. |
| `num_workers` | Parallel disk writers, 0-16; default 1. Zero writes synchronously. Pending saves are bounded by this setting, not by video length. |

For example, JPG with prefix `enhanced` and start sequence 42 writes
`enhanced_000042.jpg`, `enhanced_000043.jpg`, etc. JPEG composites alpha onto white; its
DiskImage metadata correctly reports RGB. PNG and WebP retain alpha.

Tensor output ignores disk settings. `Input` follows input storage type. A full
processing bypass with `Input` preserves the original DiskImage unchanged;
select explicit `DiskImage` to re-encode it even when enhancement/scaling are off.
Only final files use the chosen codec: transport and in-session temporal
processing are unchanged. Existing workflows retain lossless PNG disk defaults.

Tests in `tests/test_merserk_temporal.py` exercise actual JPEG/PNG/WebP files,
lossless pixel round trips, alpha/metadata, ordered lossless transport using a
mock server, numbering, offline conversion and failure cleanup. They do not
require neural inference. Load the updated node on the next ComfyUI restart;
refresh the browser afterwards to obtain the new controls.
