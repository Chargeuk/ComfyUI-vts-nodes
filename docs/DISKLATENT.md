# DiskLatent

DiskLatent stores a complete ComfyUI latent dictionary in a losslessly compressed
`.disklatent` file. The graph carries metadata instead of retaining the output's
tensors. Existing VTS latent nodes materialize inputs before computation; generated
VTS wrappers provide the same boundary for supported third-party and built-in nodes.
Restart ComfyUI to load these changes. Existing node IDs, sockets and output order
are unchanged, and the new optional controls default to native tensor outputs.

## Controls

Native VTS nodes use `latent_` names. Generated wrappers use `vts_latent_` names.
Image controls are independent and retain their existing names.

| Native setting | Default | Behaviour |
| --- | --- | --- |
| `latent_return_type` | `Tensor` | `Tensor` returns native values; `DiskLatent` saves the result. |
| `latent_output_dir` | `./tmp/disklatents` | Relative to the ComfyUI process working directory. |
| `latent_prefix` | Node display name, spaces replaced by underscores | Same display-name convention as generated DiskImage wrappers. |
| `latent_start_sequence` | `0` | Six-digit sequence component in the filename. |
| `latent_compression_level` | `3` | Zstandard, levels 1–19; all levels preserve exact tensor bytes. |
| `latent_device_policy` | `Original` | Restore disk inputs to saved/deferred devices, CPU, or a specified device. |
| `latent_device` | `cpu` | Used when the policy is `Specified`. |

Nodes with a single latent input also offer `Input` (match its representation) and
`Input or DiskLatent` (retain disk outputs or save native outputs). A list input
must have one consistent representation for `Input`. Nodes with multiple latent
input sockets offer an explicit Tensor/DiskLatent choice. Input-only nodes expose
device controls; output-only nodes expose storage controls.

Names include the output name when there are multiple latent outputs, the existing
ComfyUI list-mapping suffix when applicable, and a unique identity to prevent later
runs from overwriting files still referenced by cached results. Each output is a
complete latent file, not one file per image frame. Prefixes must be filenames,
not paths; select the directory using `latent_output_dir`.

## Utility nodes

- **VTS Latent To Disk** writes a native latent or a materialized disk input.
- **VTS DiskLatent From File** reads only a `.disklatent` manifest.
- **VTS Materialize Latent** returns native tensors for an unwrapped consumer.
- **VTS Inspect Latent** reports metadata and storage sizes without decompressing a disk latent.

Example: VTS VAE Encode with `latent_return_type=DiskLatent` → VTS KSampler with
`latent_return_type=Input` → VTS VAE Decode Tiled. An unwrapped upstream node needs
VTS Materialize Latent immediately before it. The `LATENT` socket label alone does
not make an arbitrary upstream node understand disk references.

See `examples/disklatent-api.json` for a small model-free API workflow that creates
a disk latent, splits/recombines its batch, processes it through a generated wrapper,
and inspects disk and materialized outputs.

## Metadata interface

```python
latent["samples"].shape
latent["samples"].dtype
latent["samples"].device            # intended materialization device
latent["samples"].original_device
latent["samples"].ndim
latent["samples"].size()
latent["samples"].numel()
latent["samples"].element_size()
latent.keys()
latent.tensor_size_bytes
latent.stored_size_bytes

cpu_reference = latent.to("cpu")   # metadata-only, does not rewrite the file
native = cpu_reference.materialize()
```

`DiskLatent` implements a read-only mapping, not `dict` or `torch.Tensor`.
`clone()`/`detach()` return references; `copy()` returns an ordinary dictionary of
metadata descriptors. Metadata containers returned by lookup can be inspected or
modified without changing the stored manifest. Descriptors expose shape and device
information, not tensor values: arithmetic and indexing require materialization.
Nested descriptors expose `tensors`/`unbind()` and reconstruct ComfyUI NestedTensor
on load. Device and dtype overrides are deferred; an explicit dtype conversion may
change the loaded values but never the original file. Unavailable devices and OOM
raise normally rather than silently changing placement.

## Storage and lifetime

Version 1 contains a JSON manifest and a Zstandard-compressed Safetensors payload.
It preserves supported tensor dtypes and values, string-keyed dictionaries, lists,
tuples, primitive metadata, masks and ComfyUI NestedTensor audio/video structures.
Repeated references to the same tensor object remain shared after loading; arbitrary
view strides/storage aliasing are not preserved. Sparse, quantized, PyTorch-native
nested tensors and unsupported Python objects are rejected rather than discarded.
The format is not interchangeable with stock `.latent`, Qwen `.pt`, or other
packs' `.safetensors` files. Load those using their existing nodes before converting.

Writes use temporary files and atomic rename. Payload length, SHA-256 and Zstandard
checksums detect incomplete or corrupt data. An existing reference rejects a changed
manifest. No pickle or executable object reconstruction is used.

Tensor metadata is readable without decompression. Materialization decompresses the
whole payload to a temporary file, validates it and copies tensors to their intended
devices. Saving also needs temporary uncompressed storage in the selected directory.
There is no persistent in-memory tensor cache.

Files persist until explicitly deleted. They are not automatically removed by
DiskImage cleanup. Delete them only when no cached workflow or saved reference needs
them; restarting a workflow that holds a deleted reference will fail on loading.

This reduces memory retained between nodes. It does not change a sampler/decoder's
working-memory requirements, chunk temporal computation, or offload tensor references
retained inside conditioning or other non-LATENT outputs. Noisy latents may compress
only modestly. Level 3 is the default speed/size compromise, not a size guarantee.

## Native node coverage

Explicit integration covers all 21 existing VTS latent classes: standard/advanced
KSampler, looping sampler, VAE encode/tiled encode, tiled decode/colour-match decode,
TAE video encode/decode, latent compare/convert, list-to-batch/batch-to-list, H3 motion
context and loop prepare/apply, masked H3 video conditioning, LTX IC-LoRA guide, and
Qwen save/load/cached-reference conditioning. Wildcard routing nodes preserve the
reference as an opaque value. Generated wrappers additionally handle supported legacy
and V3 LATENT schemas, including Autogrow inputs and mixed IMAGE/LATENT outputs.

Run `python tests/run_disk_latent_tests.py` using the ComfyUI Python environment
from this node pack. The runner hides GPUs from optional dependency probes and
checks storage, wrappers and affected native-node regressions without a server.
