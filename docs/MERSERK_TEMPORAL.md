# VTS Merserk Temporal Enhance

Find **VTS Merserk Temporal Enhance** under **VTS/video**. Supply an ordered IMAGE
batch or VTS DiskImage sequence. The node sends lossless PNGs to Windows Merserk
and returns enhanced images in sequence, optionally inserting interpolated frames.
It does not open or encode video files, handle audio/HDR, or expose outer iterations.

Unlike the still-image Merserk node, this node keeps one neural rendering session
alive through the sequence. History continues between adjacent frames and resets
at detected scene cuts. Each new node execution starts fresh history.

## Controls

- **NR Passes**: native neural evaluations per source frame, before interpolation, 1-4. More passes can change
  the result more strongly and increase processing time; they are not guaranteed
  to improve quality.
- **Shimmer Suppression**: upstream motion-based stabilization of enhanced detail,
  from 0 (extra filter off) to 1 (strongest). Higher values can reduce flicker but
  may smear detail when motion is estimated poorly. Scene-cut resets still apply.
- Neural style, intensity, tone/structure, automatic skin mask, colour strength,
  tone preservation, face/skin protection and grain preservation work as in the
  still-image node. Skin Structure needs Automatic Mask enabled.
- Scaling and neural enhancement are independent toggles. RTX VSR enlarges before
  enhancement; Lanczos reduction happens locally before transfer. No resizing is
  repeated for additional NR Passes.
- Scale to Min uses the existing shared dimension helper, including reversed
  min/max values, small/large/max modes, divisibility and centre cropping. Mixed
  resizing reduces the necessary axis locally, then enlarges the other with VSR.
- Multiplier scales both dimensions proportionally and rounds to even pixels.
- Every input has a tooltip explaining its effect and when it applies.

Defaults: Scale to Min, side limits 512/512, NR Passes 1, Shimmer Suppression 0.7,
scaling and enhancement enabled; frame interpolation disabled. Review the side limits before processing large
source frames if you want to keep or increase their resolution.

## Storage and network

`return_type=Input` follows the input's storage type. Tensor allocates the entire
output IMAGE batch in RAM. DiskImage writes only final PNG, WebP or JPEG frames
in a new subfolder of `output_dir` on the ComfyUI client. Blank uses
`output/merserk_temporal`. An incomplete folder is removed on failure.
Disk writers are synchronous with `num_workers=0` or bounded parallel writers
with 1-16 workers (default 1). Frame processing and output order stay fixed.

Use the Merserk LAN URL, currently **http://192.168.1.1:7865**. The server needs the
new `/vts/enhance_sequence` WebSocket route. Existing image and interpolation APIs
are unaffected. No additional dependency beyond the existing VTS requirements is
needed.

Each frame is uploaded once and its final result downloaded once as PNG chunks.
At most two uploads are outstanding. Server frame buffers do not grow with clip
length. The server writes no image/video outputs or transfer-cache images; native
diagnostic logs may still be written. Tensor output memory naturally grows with
the number of returned frames; use DiskImage for long sequences.

RGB/RGBA transport is lossless for 8-bit pixels. ComfyUI float input is clamped and
quantized to 8-bit; HDR and floating-point preservation are out of scope. Alpha is
preserved and resized when necessary. A true bypass returning the same storage
type returns the original object without quantization.

The timeout applies to server readiness or one output frame, not the whole
sequence. Connection setup is capped at 30 seconds. Interruption/disconnect cancels
this request; queued requests wait for their turn without interrupting another render.

Pure local reduction and bypass work offline when neural rendering and frame
interpolation are disabled.
A single frame can be enhanced. Frames must share dimensions and channel count.
Neural output must be at least 64 pixels per side and fit within 7680 by 4320 in
either orientation. VSR has a 16384-pixel side limit and the stream a 100-megapixel
frame limit; GPU memory may impose lower practical limits.

The implementation reuses Merserk's existing motion estimation and stabilization.
It does not promise perfect temporal consistency, and does not add a new learned
optical-flow model. The host frame path uses DIS flow for final enhancement
stabilization; it is not a GPU video decode/encode pipeline.

## Concurrent requests

Merserk queues GPU work from GUI and VTS clients in arrival order. Updated streaming
nodes accept periodic queue status messages, so waiting behind another render does
not exhaust the per-frame timeout. A disconnected queued client leaves the queue
without cancelling the active job. Rendering and silent-server timeouts still apply.
Older streaming clients can also wait, up to their existing readiness timeout.

## Native worker recovery

If Merserk reports a fatal neural worker failure, this node reconnects and replays
the complete original sequence once. This rebuilds VSR, scene detection and neural
history instead of continuing with missing history. Partial DiskImage output from
the failed attempt is removed; only a complete result is returned. A second worker
failure is reported without another retry. Cancellation and unrelated errors are
not retried. Merserk keeps its web server and queue alive during worker recovery.

## Optional frame interpolation

Enable **enable_frame_interpolation** on this same node to insert frames after
resizing and temporal enhancement. **interpolation_multiplier** offers 2x, 3x, 4x
and 8x. The toggle defaults to off, so existing workflows retain their frame count.

- Output count is `(input_count - 1) * multiplier + 1`. Three inputs at 2x produce
  five outputs: enhanced A, A/B intermediate, enhanced B, B/C intermediate, enhanced C.
- 3x inserts frames at 37.5% and 62.5%, not exact thirds, and costs roughly as much
  as 8x. Higher cascades can accumulate artifacts.
- Scene cuts repeat the nearest enhanced originals. Generated alpha is blended
  between endpoint alpha channels. One input cannot interpolate and returns one frame.
- Neural rendering and scaling remain independent switches; interpolation-only is
  supported. NR Passes apply to source frames, not to generated intermediates.
- Choose source FPS times the multiplier when encoding downstream for the intended
  playback rate. This image node does not manage FPS metadata or audio.

There is one network stream and one queue entry. Each source is uploaded once,
and the server keeps enhanced frames for interpolation instead of downloading and
re-uploading them. Only final enhanced originals and generated frames are returned.
The server needs the updated version-2 `/vts/enhance_sequence` protocol. With
interpolation off, the existing version-1 protocol is used.

DiskImage **format** selects PNG, WebP or JPEG. **compression_level** controls
PNG compression from 0 to 9 and WebP encoder method from 0 to 6 (higher values
use 6). **prefix** and **start_sequence** control filenames and numbering.
**quality** sets JPEG or lossy WebP quality from 1 to 100 (default 95); 101
selects lossless WebP and JPEG uses 100. **webp_lossless** defaults to on,
so WebP remains lossless unless switched off; quality 101 always stays lossless.
JPEG composites transparency onto white and returns three-channel RGB DiskImages.
These controls apply whether interpolation is enabled or disabled. Tensor output
ignores them. Network transport always remains PNG.

The combined pipeline retains both enhancement and interpolation GPU sessions,
so it can use more peak VRAM than separate stages. It keeps the existing motion
estimators and does not inject external vectors into the neural DLL. All sessions
close at job completion; Merserk's idle worker policy then applies.
