# DiskAudio

DiskAudio keeps AUDIO waveforms on disk and passes metadata between nodes. It uses the standard AUDIO socket. VTS audio nodes, eligible generated wrappers, and the Chargeuk VideoHelperSuite fork load samples before computation.

Existing workflows still return native tensors by default. Select **DiskAudio** in `audio_return_type` (or `vts_audio_return_type` on generated wrappers) to enable disk output. Storage defaults to **FLAC**.

## Storage choices

| Format | Behavior |
| --- | --- |
| FLAC | Compressed 24-bit integer PCM. The codec is lossless, but float-to-PCM conversion rounds samples; +1 maps to the maximum positive PCM value. Non-finite values or values outside [-1, 1] raise an error rather than being silently clipped. Decodes to float32. |
| Exact | Zstandard compression, shared with DiskLatent. Native audio uses the DiskLatent serializer to preserve tensor bits, dtype and supported extra metadata. File-backed sources stream their decoded float32 PCM into checksummed Zstd files without loading a full waveform. |
| Opus | Lossy audio. Output is encoded at 48 kHz; metadata describes the decoded rate and sample count. |
| MP3 | Lossy mono/stereo audio. The encoder may choose a supported sample rate; metadata reports the actual decoded rate. No automatic multichannel downmix is performed. |

Lossy storage changes samples. Repeated re-encoding can accumulate degradation. FLAC and Exact have no bitrate setting; compression effort is automatic. Exact preserves the samples it receives, not information already lost in an earlier FLAC conversion or lossy encode.

## Controls

Native VTS and VHS controls use `audio_`; generated wrappers use `vts_audio_`. Audio controls are independent of image/latent controls.

| Control | Default | Meaning |
| --- | --- | --- |
| `audio_return_type` | Tensor | Tensor: native AUDIO. DiskAudio: compressed disk output. Input: match the single AUDIO input, where available. |
| `audio_format` | FLAC | FLAC, Exact, Opus or MP3. |
| `audio_bitrate_kbps` | Auto | Total clip bitrate, not per channel. Auto selects Opus 128 kbps or MP3 192 kbps. Ignored for FLAC/Exact. |
| `audio_output_dir` | ./tmp/diskaudio | Relative to ComfyUI's working directory; absolute paths are supported. |
| `audio_prefix` | Node name | Filename prefix. List position and unique suffixes prevent collisions. |
| `audio_device_policy` | Original | Advanced input option: restore the saved device/deferred override, load to CPU, or use Specified. Native tensors are unchanged. |
| `audio_device` | cpu | Advanced, used only by Specified: cpu, cuda:0, cuda:1 if a second GPU exists. Affects waveforms, not model placement. |

Bitrate starting points (also included in the widget tooltip):

- **Opus:** 32–48 kbps mono speech; 64 kbps higher-quality speech; 96–128 kbps stereo music; 160–192 kbps more headroom for demanding stereo.
- **MP3:** 64–96 kbps mono speech; 128 kbps compact stereo; 192 kbps general music; 256–320 kbps higher-quality stereo.

These are guidelines, not listening-quality guarantees. Larger bitrates generally mean larger files.

## Files and inspection

Each saved value has a uniquely named directory containing `audio.diskaudio` and its payload. Codec formats use one ordinary playable file per batch item. Move or retain the whole directory to preserve the manifest and batch metadata. Failed saves remove their partial directory. Existing files are never overwritten.

`audio["sample_rate"]`, `audio["waveform"].shape`, `.size()`, `.ndim`, `.dtype`, `.device`, and `.numel()` inspect metadata without decoding. `.to()` records a deferred device/dtype override. Arithmetic requires materialization. Nodes never retain a decoded waveform cache on DiskAudio objects.

- **VTS Audio To Disk:** save native audio or transcode DiskAudio.
- **VTS DiskAudio From File:** reference a media file or a saved `.diskaudio` manifest without copying. Media files are scanned in blocks to obtain exact decoded sample counts; optional start/duration selects a source clip.
- **VTS Materialize Audio:** explicitly return native waveform tensors.
- **VTS Inspect Audio:** display shape, rate, duration, dtype, device, format and file size without decoding DiskAudio.

Unchanged DiskAudio values reuse their files when the requested format/bitrate matches; changing the output directory or prefix alone does not force a copy. Codec-backed metadata must be JSON-compatible; choose Exact for additional tensor-valued metadata.

## VideoHelperSuite and long audio

Install the matching Chargeuk VideoHelperSuite update for native loader and Video Combine support. VHS remains usable with ordinary audio without VTS installed.

VHS audio/video loaders offer native DiskAudio output. Video Combine reads single-clip codec files directly through FFmpeg; the final video's audio encoding is still controlled by the video format preset. Exact input is materialized for muxing. Device options only matter when a waveform actually needs loading.

VTS chunk assembly streams to disk and returns DiskAudio when selected, overriding `return_audio` without loading the assembled recording. FLAC, Opus, MP3 and file-to-Exact saving are block-based. Normal processing nodes and VHS Unbatch still materialize the samples they need; Unbatch returns native batched audio.

FFmpeg and ffprobe must be on PATH, with FLAC, libopus and libmp3lame encoding available. No new Python dependencies are required beyond the existing VTS audio and DiskLatent dependencies.

## Validation

Run `tests/run_disk_latent_tests.py` using the ComfyUI Python environment for audio/latent/wrapper regressions. Run the VHS fork's `tests/run_disk_audio_tests.py` for native VHS integration and existing loader regressions. Tests run on CPU and do not stop a running ComfyUI instance.
