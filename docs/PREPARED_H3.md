# Prepared H3 models (experimental)

Three nodes save an already configured H3 model and load it without merging its
saved weight adapters again:

- **VTS Save Prepared H3**: one-time export.
- **VTS Load Prepared H3**: normal generation loader.
- **VTS Prepared H3 Runtime Options**: optional overrides connected to the loader.

## First export

Keep the existing model loader, LoRAs, VDN/SLA/hybrid and sigma-shift nodes.
Connect the final MODEL to **Save Prepared H3**, choose a filename prefix, enable
`save_enabled`, and queue once. Export itself can be slow: it performs the weight
patching that subsequent loads will avoid. Do not put this saver in a loop.

Set `output_directory` to the absolute WSL folder you want, for example
`/mnt/external-lan2/comfyui/models/prepared_h3`. Spaces are supported. Leave it
blank to use the active ComfyUI **output/prepared_h3** folder, following
`--output-directory`. Each execution creates a new uniquely named `.safetensors`
there; it never overwrites the original model. It is remote only if the selected
directory is on a remote mount. Allow space for a complete diffusion model plus VDN branches.
An interrupted export removes its own incomplete `.partial` file when Python can
handle the error; an abrupt process termination may leave that partial file.

For SLA, `sla_dense_backend = from workflow` reads the explicit backend choice
from the incoming MODEL chain. If that choice is dynamically connected or the
chain cannot be traced, enter its actual value (for example `pytorch`). The saver
does not guess which kernel an opaque attention callable represents.

## Normal generation

Remove/disable the export branch. In **Load Prepared H3**, enter the full file
path in `input_path`, including the `.safetensors` filename. This overrides the
dropdown and works outside the default discovery folders. Alternatively, leave
it blank, refresh the model list, and select a discovered file from the dropdown.
Replace the original loader + baked LoRA + attention chain
with this loader. **Do not reapply the baked LoRAs or Apply VDN-H3.** The optional
`prepared_file` socket can instead accept the saver's returned filename for an
initial round-trip test, but leave the saver out of the ongoing workflow.
Use either the `input_path` field or the connected `prepared_file`, not both.
Paths are WSL/Linux paths (or `~/...`), not Windows drive-letter paths. The loader
never downloads or copies the file to a local cache just because it is remote.

The loader also discovers files in `prepared_h3` beneath configured diffusion
model folders. Its `settings` STRING output describes the effective runtime
configuration. CLIP, VAEs, conditioning, sampler, seed, step count and scheduler
remain in the workflow; this file contains the diffusion model, not those models
or the entire workflow.

Works with `--cache-none`: no saved adapter merge is needed when the loader runs
again. It still reads the file, creates the model, transfers weights and installs
small attention wrappers. It does not save CUDA graphs, warmed kernels or live
GPU state, and does not guarantee that the whole first denoising step becomes
fast. Network I/O and normal model offloading can still dominate.

## Runtime overrides

Without an options connection, the saved values are used. On the options node,
`-1` and `saved` mean unchanged. **Zero is a real value**, including zero dense
first/last steps. These settings do not change the baked weights.

For example, to test all-sparse hybrid steps, set both dense-step widgets to `0`.
To change branch memory policy on another GPU, choose `auto` for branch weights
and retained buffers. `local_sparsity = 0` disables the hybrid overlay, leaving
VDN's original attention. A different runtime setting can change output quality;
runtime-adjustable does not mean mathematically identical.

Common widgets cover sparsity, first/last dense steps, minimum sparse window size,
branch residency, retained buffers, prefetch and grouped/flex VDN attention.
All remaining supported controls are available through `advanced_json`:

```json
{
  "hybrid": {"dense_blocks": "0-2", "fallback_to_exact": true},
  "vdn": {"compile_scan": false, "fuse_statistics": false},
  "sampling": {"shift": 12.0, "audio_shift": 3.0}
}
```

Only include groups present in the saved variant. Setting the same control in a
widget and JSON is an error, to avoid unclear precedence. Supported JSON keys:

| Group | Runtime keys |
| --- | --- |
| `hybrid` | `enabled`, `local_sparsity`, `min_window_tokens`, `dense_first_steps`, `dense_last_steps`, `dense_blocks`, `fallback_to_exact`, `verbose` |
| `vdn` | `radius`, `chunk`, `anchor_frames`, `delta_rule`, `bridge`, `a_fp32`, `short_conv`, `enable_text_state`, `enable_softmax_gate`, `linear_enabled`, `branch_weights`, `retain_buffers`, `prefetch`, `attention_backend`, `fast_kernels`, `compile_scan`, `fuse_statistics`, `verbose` |
| `sla` | `sparsity_ratio`, `block_size`, `min_seq_len`, `dense_first_steps`, `dense_last_steps`, `dense_steps`, `protect_audio`, `dense_backend`, `disable_fp16_accum`, `stabilize_motion`, `reference_protection`, `tail_correction`, `use_int8_qk`, `use_int8_pv`, `engine` |
| `sampling` | `shift`, `audio_shift`, `timesteps`, `multiplier`, `noise_scale` |

SLA's `dense_steps` is an independent, explicit zero-based list. Setting first
and last to zero does not erase that list; also use `"sla":{"dense_steps":""}`
to remove explicit dense steps. `short_conv` is a JSON list such as `["k","v"]`.
Use the same value names/types as the original nodes. `sampling.timesteps` is the
model's internal schedule resolution, **not the number of denoising steps**.

For a directly connected VDN node, the saver preserves the requested
`retain_buffers` policy, including `auto`. If graph tracing is impossible, it
records the effective on/off value; choose `auto` explicitly on load if desired.
Branch *placement* is adjustable; changing the bundled branch's quantized versus
BF16 representation requires exporting the desired representation again.

## Format and limits

One safetensors file contains native ComfyUI diffusion weights with merged
patches, original VDN branch data/scales where applicable, and a JSON manifest.
Native ComfyUI save/load routines handle quantization and Comfy Kitchen ops;
there is no alternative quantizer, pickle or serialized Python code here.
VDN restoration installs its trained branch without loading/applying adapters.
SLA/hybrid restoration reinstalls runtime attention functions from installed
code. The original branch/adapter files are not required to load the bundle.

VDN, SLA and hybrid configurations may be saved as separate files. Switching a
hybrid file back to VDN attention is a runtime change; switching to an SLA-trained
weight configuration is not. Baked LoRA identity/strength and weight dimensions
cannot be overridden. A new LoRA applied after loading is an additional patch,
not an edit to the baked one, and brings its own patching cost.

Only native H3 with supported VDN/SLA/hybrid wrappers and H3 sampling settings is
exported. Unknown runtime wrappers, object patches, callbacks, hooks and model
injections are rejected rather than silently lost. In particular, VDN
`lora_mode=bypass` injections are not a merged-weight export. Any adapter deltas
already skipped by the original Apply VDN node remain skipped; export cannot
recover them.

Attention-code fingerprints are checked by default. After updating dependencies,
re-export or explicitly select `allow changed attention code` and re-test.
Required nodes/kernels must still be installed on the target machine. No GPU
architecture-specific compiled artifacts are bundled.

`weight_dtype = saved` uses native automatic loading of the stored weights.
The other choices pass a dtype preference to ComfyUI; they are not offline
requantization controls. Ordinary model-load launch flags still apply.

Validation uses small real H3 CPU save/load fixtures, including quantized weights,
adapter-once checks and runtime restoration. This is not a full-size video-quality
or cold-start benchmark; measure the actual workflow before assuming a speedup.
