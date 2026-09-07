# VTS H3 Hybrid Attention (Experimental)

An opt-in SLA-style sparse local-attention path for the Chargeuk VDN-H3 fork.
This is an experiment, not a validated replacement for VDN's trained attention.
It may change faces, motion, audio and loop continuity, or run slower/use more memory.

## Connect

`H3 model / existing LoRAs → Apply VDN-H3 → VTS H3 Hybrid Attention → sampler`

Keep VDN's checkpoint, turbo adapter, weights/residency settings and sampler
settings unchanged. For the existing DMD checkpoint, keep the turbo adapter on
and use its usual eight-step setup. Do **not** add an SLA turbo LoRA or a separate
H3 SLA Attention / Block Sparse Attention patch for this comparison.

The node requires the VDN local-attention API hook supplied with this change and
the installed PlagueKind H3 SLA Triton kernel. Restart ComfyUI after installing.
No model downloads or new Python dependencies are added.

## First comparison

- Use VDN's **grouped** attention backend for both runs.
- Keep the seed, input image, prompt, clip length, resolution and sampler identical.
- Baseline: `enabled=false` (or bypass this node).
- Hybrid: `enabled=true`, `local_sparsity=0.70`, `min_window_tokens=4096`,
  `dense_first_steps=1`, `dense_last_steps=1`, `dense_blocks` blank,
  `fallback_to_exact=true`, `verbose=true`.
- Compare another run at `local_sparsity=0.50` if quality changes too much. This
  keeps more attention and may be slower than exact VDN. These are test settings,
  not a measured optimal configuration.
- Warm up each mode before comparing timing. Measure sampling time and peak GPU
  **allocated** memory as well as total GPU usage. Do not infer savings from speed alone.

`local_sparsity` is the fraction of local key blocks omitted **before** protection
is added. Text, reference and audio key tokens, and VDN's anchor-key frames, are
always included. Their query rows remain exact too. Protected key blocks share
the same softmax normalization with selected video blocks; this does not preserve
the original numerical output, since other video keys may be absent.

The first/last-step controls count logical sampler steps, not model calls. If
step metadata is missing and either guard is nonzero, the node stays exact.
`dense_blocks` excludes zero-based transformer layers, e.g. `0-3,46-49`.

The console reports actual sparse window calls, exact window fallbacks and the
mean fraction of selected key blocks. The counters do not include global/anchor
queries or whole exact steps. **Zero sparse calls means no hybrid acceleration
ran**: the VDN window may cover the entire clip, the windows may be too small,
the protected budget may include every block, or the step/layer guards may exclude
all work. Full-coverage VDN fallback is deliberately unchanged.

The implementation uses native BF16/FP16 SLA kernel math, with INT8 and pooled-tail
correction disabled. It leaves VDN's trained recurrent branch untouched. Sparse
steps use grouped windows even if VDN was set to Flex; exact steps retain the
original backend. The node itself stores no activation/routing tensors between
calls or loop iterations. Existing VDN memory policies are not altered.

Recoverable sparse-kernel failures use exact VDN for the rest of that sampling
run and emit a warning. OOM and fatal CUDA errors propagate. Disable
`fallback_to_exact` to investigate a kernel failure without fallback.

## Scope and dependencies

VTS uses ComfyUI's model-patcher wrappers; it does not monkey-patch global attention
or change core ComfyUI. Two small opt-in hooks in VDN expose the already-gathered
local windows and their protected key ranges. Existing workflows never select
the hook and retain their exact attention path.

ComfyKitchen's current `sol_attn` API requires equal Q/K lengths; VDN's windows
have unequal lengths. This node therefore imports the installed PlagueKind
`get_block_map` and `block_sparse_attention` helpers directly, without copying
their third-party kernel source. Upstream changes to those helpers may require
an integration update.

Sources: [VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3),
[Chargeuk ComfyUI VDN fork](https://github.com/Chargeuk/ComfyUI-VDN-H3),
[PlagueKind SLA / LightX2V-derived kernels](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/tree/main/ComfyUI-H3-SLA-Attention).

Run CPU tests from the ComfyUI directory:

```sh
python -m unittest discover -s custom_nodes/ComfyUI-vts-nodes/tests -p test_h3_hybrid_attention.py
```

Set `VTS_HYBRID_CUDA_TEST=1` to include the small rectangular-kernel checks on CUDA.
These are implementation checks, not an end-to-end generation quality benchmark.
