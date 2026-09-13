# VTS Merserk Temporal Enhance

Find **VTS Merserk Temporal Enhance** under **VTS/video**. Supply an ordered IMAGE
batch or VTS DiskImage sequence. The node sends lossless PNGs to Windows Merserk
and returns one enhanced image per input, in the same order. It does not open or
encode video files, change frame count/FPS, handle audio/HDR, or expose outer
iterations.

Unlike the still-image Merserk node, this node keeps one neural rendering session
alive through the sequence. History continues between adjacent frames and resets
at detected scene cuts. Each new node execution starts fresh history.

## Controls

- **NR Passes**: native neural evaluations per frame, 1-4. More passes can change
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
scaling and enhancement enabled. Review the side limits before processing large
source frames if you want to keep or increase their resolution.

## Storage and network

`return_type=Input` follows the input's storage type. Tensor allocates the entire
output IMAGE batch in RAM. DiskImage writes only final PNG frames, one at a time,
in a new subfolder of `output_dir` on the ComfyUI client. Blank uses
`output/merserk_temporal`. An incomplete folder is removed on failure.

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

Pure local reduction and bypass work offline when neural rendering is disabled.
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
