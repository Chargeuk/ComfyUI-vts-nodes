# VTS Merserk Frame Interpolate

Find **VTS Merserk Frame Interpolate** under **VTS/video**. Connect an ordered
IMAGE batch or VTS DiskImage sequence, enter the Windows Merserk URL, and choose
the multiplier and output storage type. The server must have the VTS WebSocket
interpolation endpoint installed; upstream Merserk alone does not provide it.

## Frame count and timing

The output contains `(input_count - 1) * multiplier + 1` frames, including every
original. For example, two inputs A and B at 3x produce A, new frame, new frame, B.
Set the downstream video's frame rate to the source rate times the multiplier.
Single-frame input passes through or converts storage without contacting Merserk.

| Multiplier | New frames between each pair | Positions between originals |
| --- | --- | --- |
| 2x | 1 | 50% |
| 3x | 2 | 37.5%, 62.5% (approximate thirds) |
| 4x | 3 | 25%, 50%, 75% |
| 8x | 7 | Every 12.5% |

The server reuses native DLSS Frame Generation sessions in a 2x cascade.
3x selects two results from an 8x cascade: it costs approximately as much GPU
work as 8x and does not provide exact, evenly spaced thirds. Higher cascades can
accumulate interpolation artifacts. At a detected scene cut, the client repeats
the nearest original frames locally, preserving the expected frame count.

## Transport and memory

- Each source frame is uploaded once, as losslessly compressed PNG. Only new
  intermediate frames are downloaded. Original frames are inserted locally.
- PNG messages are split into chunks of at most 256 KiB. One source can be sent
  ahead while the current interval is processed; buffers stay bounded as the
  clip gets longer. No raw-pixel or lossy network option is used.
- `Input` keeps the input storage type. `Tensor` allocates one complete output
  batch in CPU memory. `DiskImage` writes final frames incrementally on the
  ComfyUI client and is the preferred option for long clips.
- DiskImage output supports lossless PNG and lossless WebP, including transparent
  RGB pixels. Each run creates a separate directory under `output_dir`, or
  `output/merserk_interpolate` when blank. `compression_level` controls local
  PNG compression only; network PNG uses fast compression level 1.
- Merserk does not save the streamed images, videos or Gradio image-cache files.
  Its native runtime may still write diagnostic logs. Cancellation or errors
  close the request's GPU sessions and remove incomplete client output folders.

PNG transport preserves the **8-bit image pixels** exactly. ComfyUI float input
is clamped to 0–1 and quantized to 8-bit for this native runtime; this is not a
float/HDR preservation path. Original frames remain unchanged in Tensor output.
Generated alpha is blended between the two source alpha channels.

## Requirements and protocol

Install VTS requirements (including `websockets>=15.0.1`) into the ComfyUI Python
environment and restart ComfyUI. Windows Merserk also needs WebSocket support and
the matching `/vts/interpolate` route. NVIDIA must report DLSS Frame Generation
as available on that Windows host. If initialization reports disabled hardware-
accelerated GPU scheduling, check that Windows setting and reboot after changing it.

The connection uses ws/wss derived from the server's http/https URL. Protocol v1
starts with JSON containing `version`, `width`, `height`, `frame_count`,
`multiplier` and `timeout_seconds`. The server replies `ready`, including output
count and fractional interpolation positions. Each input has a numbered `frame`
header with PNG byte length, followed by binary chunks; the client finishes with
`end`. Replies are `generated` headers plus PNG chunks, local `repeat` references,
`frame_done`, then `done` after GPU cleanup. Failures return `error` and close the
connection. One GPU job runs at a time; a busy server rejects another request
without cancelling its current job.
