# Memory-aware loops (experimental)

These opt-in nodes leave existing VTS and EasyUse nodes unchanged. They are intended to reduce historical loop-data retention, not to unload models or guarantee that process RAM drops after each iteration.

## Nodes

| Node | Purpose |
|---|---|
| VTS Memory-Aware Loop Start | Begin a counted loop with a variable set of carried values. |
| VTS Memory-Aware Loop End | Supply the next values, complete iteration dependencies, and optionally perform allocator cleanup. |
| VTS Loop Value | Select an individual carried value by key, such as `item0`. |
| VTS H3 Prepare Loop Context | Copy only the required video/audio latent tails, preserving H3 temporal alignment. |
| VTS H3 Apply Loop Context | Apply the compact context to the next clip's conditioning and masked latent. |
| VTS Save Audio Chunk | Save an already overlap-trimmed audio segment and extend a disk-backed manifest. |
| VTS Assemble Audio Chunks | Stream the ordered chunks to a final WAV, optionally returning a full ComfyUI AUDIO tensor. |

Start's `initial.item0`, `initial.item1`, etc. correspond to End's `values.item0`, `values.item1`, etc. Connect Start's `flow` output to End's `flow` input. Use Loop Value on Start's values to read the current iteration, and on End's values to read the final result. The index is zero-based; `total=3` executes the body three times.

## Memory ownership

The loop keeps carry payloads in a per-execution store and places only identifiers in expanded graph inputs. It releases its own previous carry references when safe; it does not delete other nodes' outputs or rewrite ComfyUI's graphs. Prompt lifecycle callbacks clean up execution-owned storage at the end. No ComfyUI source modification or runtime monkey patch is installed.

Ordinary ComfyUI output caching can retain node results independently. `--cache-none` complements the loop by disabling that cache. Allocators can also keep freed memory for reuse, so live tensor bytes, CUDA allocated/reserved bytes and process RSS are different measurements.

The loop also snapshots loop-invariant inputs for reuse during its execution. This intentionally keeps inputs needed by future iterations available instead of making no-cache recreate their upstream work. Connect any value that should change per iteration to the loop index or carried values so it belongs to the loop body.

Connected lazy/conditional body inputs are not supported in this first release. The loop rejects them before capturing external inputs, rather than accidentally executing an unselected branch. Move that choice upstream and carry the selected result. The H3 example uses supported eager body inputs. Each Start must connect directly to exactly one End through `flow`; unused Start/End nodes should be removed or disabled.

Connect side-effect work that must finish before advancing to End's `after` inputs using an output from that work. Do not rely only on a node's position on the canvas. Node-specific dependencies must remain connected.

## Optional cleanup controls

- `release_previous`: release VTS-owned historical carries; enabled by default.
- `collect_garbage`: run Python garbage collection at a cleanup boundary; off by default.
- `empty_cuda_cache`: release unused CUDA allocator blocks without initializing CUDA or unloading models; off by default.
- `trim_cpu`: ask supported Linux/WSL allocators to return unused CPU pages to the OS; off by default. This cannot release live tensors.
- `cleanup_every`: run optional cleanup every N iterations.
- `log_memory`: log memory snapshots and cleanup duration; off by default.

Start with only `release_previous` enabled. Add optional cleanup flags separately and compare elapsed time as well as memory. Aggressive cleanup may trade speed for lower idle allocator residency. No RTX 5090 or DGX Spark performance claim follows from tests on an RTX 4080 Super.

The experimental workflow preserves its existing VTS Clear Ram nodes. Their `gc_collect` option already performs CPU allocator trimming as well as garbage collection, so the new Loop End allocator-cleanup options start disabled to avoid adding redundant cleanup passes.

## H3 context

Prepare belongs after the current sampler and before the loop hand-off. The current decoder still consumes the full sampler latent. Carry Prepare's `VTS_H3_CONTEXT` output to Apply in the next iteration; Apply replaces the existing motion-context node in that experimental path.

Keep `context_length` and `audio_context_length` equal to the settings used by the original workflow when checking equivalence. The compact package owns independent copies of the required native latent tails, not views keeping the entire original tensor alive. It omits source noise masks; Apply reconstructs the next target's mask using the established VTS helpers.

## Disk audio and paths

Save Audio Chunk expects overlap trimming to be done upstream. It stores lossless float32 RF64 WAV chunks and linked JSON manifests in unique directories under ComfyUI's configured output directory. Relative prefixes such as `audio/chunk` are supported; absolute paths, traversal and resolved paths outside the output directory are rejected. When output is a network mount, these files are stored on that mount. Completed files are persistent outputs, not automatically deleted temporary data.

The carried descriptor contains paths and counts, not audio tensors or the entire manifest history. Assembly validates sample rates, channel counts and frame counts and streams audio in blocks. File-only assembly (`return_audio=false`) returns `None` on its AUDIO socket; do not connect that socket to a node requiring AUDIO. Enabling it loads the final waveform once, which is necessary for consumers such as VHS Video Combine.

`soundfile>=0.13.1` is declared in requirements. It is already present on the development machine.

## Experimental workflow

The example is a reduced, neutral-prompt copy of the user's H3 workflow: 3-second clips, 0.3 megapixels, 22-frame context, eight sampling steps. Subgraphs are flattened for explicit wiring. The original workflow is not overwritten.

Load `examples/vts_memory_loops/vts_memory_loop_counter.json` for a model-free example (the preview should show 3), or `minimax_generate_fisheye180_video_013_vts_memory_experimental_3s.json` in that same folder for the H3 example. Files ending `.api.json` are API prompts, not canvas workflows.

The packaged H3 example uses `/mnt/external-lan2/comfyui/output/vts_memory_loop_example_20260906/images` for frames. This is the development machine's network mount; change all four directory settings together if your mount differs, and use a fresh directory for a separate run. Select an available reference image in Load Image. The GPU-tested diagnostic used a local `/tmp` frame directory; the packaged network-path variant was schema-validated, not GPU-repeated. Audio and video outputs follow the launcher's configured output directory.

It saves frames and audio chunks each iteration and assembles the growing final video once after the loop. Repeated intermediate growing-video encodes are omitted; consequently, whole-workflow timing is not an identical-work comparison with the original. Sampler/VDN timing must be compared separately. Final assembly enables AUDIO output for VHS compatibility.

## Validation

See the accompanying implementation report for the tests actually run, their results and remaining limitations. Do not infer full model-output equivalence or large-memory savings solely from CPU ownership tests.

The final validation passed 60 tests (including 11 workflow-builder checks held with the investigation scripts). Installed node regressions can be rerun from ComfyUI with `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. COMFYUI_ROOT="$PWD" python -m unittest discover -s custom_nodes/ComfyUI-vts-nodes/tests -p 'test_memory_aware_loop*.py' -v`, and the corresponding `test_h3*.py` / `test_audio_chunks.py` patterns. A real 4080 Super no-cache run completed the initial clip and three continuations, with zero tensor bytes left in the VTS loop store after completion; this is not a promise that normal ComfyUI output caches or unrelated nodes release their own references.

Development ComfyUI revision: `b78cec879b9460d5cb25228a83a942fb78d2cd24`. The loop depends on the installed V3 Autogrow/raw-link schema and cache-provider lifecycle interfaces. Its own input-schema adapter handles lazy Autogrow lookup in this scheduler; this is version-sensitive integration that must be re-tested after ComfyUI upgrades, not a promise of compatibility with arbitrary older versions.
