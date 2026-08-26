# VTS Ref2VA helper-node audit

## Scope and isolation

| Check | Result |
|---|---|
| Host | `DanDesktop` |
| Account | `d_a_s` |
| Authoritative VTS repository | `/home/d_a_s/code/comfyui/custom_nodes/ComfyUI-vts-nodes` |
| Authoritative remote | `https://github.com/Chargeuk/ComfyUI-vts-nodes.git` |
| Baseline | `main` at `64f4f3d2dbf86a3adc92bdd813da3b5efa0f57e6` |
| Isolated worktree | `/home/d_a_s/code/minimax-h3-vr180-lora-vts-nodes` |
| Isolated branch | `codex/vts-ref2va-dandesktop-v1` |
| Installed VTS state | Clean before and after the investigation; not modified or restarted |
| Large source collection | `vrvids` was excluded from searches and was not read or copied |

The enclosing ComfyUI checkout already contained unrelated dirty files. They
were recorded and left untouched. The installed AI-Toolkit MiniMax H3 checkout
was clean and was inspected read-only.

## Existing VTS inventory

The baseline loader discovers every `py/*.py` module and merges each module's
`NODE_CLASS_MAPPINGS`; no root-loader edit is needed for a new node file. Static
inventory found 62 registered baseline IDs, grouped as follows.

| Area | Existing capability examples | Ref2VA relevance |
|---|---|---|
| Image and mask | Color Mask To Mask, Image Composite Masked, Images Scale, Images Crop From Masks, Image From Batch, Sharpen, Frame Interpolate | General operations only. None performs exact marker injection plus known-pixel collision escape. Existing composite nodes may resize/interpolate masks and are not an exact-marker boundary. |
| Batch, list, and text | Image Batch Extend With Overlap, Reduce Batch Size, To/List/Text nodes, prompt builders, text merge/clean nodes | No Ref1/Ref2 semantics. |
| Latent, VAE, and sampling | KSampler, Looping KSampler, VAE encode/decode/tiled nodes, latent conversion/comparison, VAE loader | No reference-image preparation. |
| Model wrappers | Generic Image Wrapper, LTX guide, MiniMax H3 Motion Context | Motion Context adds continuation guides; it does not construct Ref2VA image references or marker masks. |
| Disk/output | Image To Disk, Delete Saved Images | Not used by the new pure helpers, so invalid or interrupted node calls cannot publish files. |

Relevant existing nodes were rejected as substitutes:

- `VTS Color Mask To Mask` extracts a mask from an image; it does not write or
  validate `#FF00FF`, escape collisions, preserve fixed reference order, or
  report counts.
- `VTS Image Composite Masked` is a general compositor. Its interpolation and
  in-place destination update are unsuitable for the byte-exact marker
  invariant.
- `VTS Images Scale` and `Scale To Min` resize but do not perform spherical
  projection or true-aspect letterboxing tied to a matching geometry mask.

## Reviewed shared projection core

The reviewed staging source was
`control/vts_patch/py/vtsUtils/vr180_projection.py`, text SHA-256
`59ce9126d48f4e8e5d4f104784b7fe8c2cf35928151e35c5ab8dbe3e6b42573c`
after removing 416 trailing NUL bytes from the staged file. Its pixel-centred
half-equirectangular geometry, camera basis, rotations, sampling mode, masks,
and chunking were reused.

The only core change is retention policy: a grid/mask item larger than 64 MiB
uses the same uncached builder. The four-entry CPU LRU remains for smaller
items. A direct diff against the cleaned reviewed source confirms that no
geometry calculations changed.

## Installed MiniMax H3 and AI-Toolkit contracts

| Component | Revision and observed contract |
|---|---|
| Native ComfyUI H3 | ComfyUI `b78cec879b9460d5cb25228a83a942fb78d2cd24`, `comfy_extras/nodes_minimax_h3.py`. Reference images are ComfyUI `IMAGE` tensors `[B,H,W,C]`; RGB is taken from the last dimension. `ref_image_1`, then `ref_image_2`, are consumed through ordered `ref_images.values()`. Each still-image reference uses `img[:1]`, is aspect-preserving and downscaled only, then VAE encoded. There is no reference `MASK` input. |
| AI-Toolkit H3 helper | `ComfyUI-AIToolkit-MiniMaxH3` `4bf42cd782b6625d48420537259f7a93dba14bff`. Its output frames are `[T,H,W,C]` float images, resampled to 24 fps and resized to the reference aspect on a `/32` grid. It prepares reference video timing/size only and exposes no Ref2VA marker mask convention. |
| H3 model mask logic | `denoise_mask` is the target sampling mask, not a reference mask. It max-reduces 2×2 latent patches, corresponding to 32×32 target pixels with the H3 VAE. Reusing it as a Ref2 marker mask would invent a second convention. |

For LoRA inference, wire helper output `ref1` to native H3 `ref_image_1` and
helper output `ref2` to `ref_image_2`. The fixed order is therefore explicit at
both output names and destination sockets.

## Required-function decision table

| Required function | Decision | Evidence and implementation |
|---|---|---|
| Stereo repair: left Ref1 + incomplete right + binary effective mask → exact-magenta Ref2 | **new node** | No existing VTS node combines these semantics. `VTSRef2VAStereoReferences` returns Ref1 then Ref2, validates exact binary size, preserves Ref1 object identity, and never mutates the right input. |
| Projection: left VR180 → true-aspect letterboxed rectilinear Ref1 + reprojected partial VR180 Ref2 + exact mask | **extend** | The reviewed pure projection core already supplies both geometric warps and masks but no combined workflow. `VTSRef2VAProjectionReferences` composes the core, adds 16:9/4:3 letterboxing, and keeps the VR180 input unchanged. |
| Exact marker safety and collision counts | **new node/shared helper** | `mark_ref2` detects pixels that encode as `#FF00FF` in known Ref2 space, changes only their blue byte to `254`, writes exact magenta only under the effective mask, verifies the iff invariant, and reports per-batch counts. Ref1 and the projection target are never passed to this escape operation. |
| Optional MiniMax effective-mask/block diagnostic | **already exists / no new node** | Native H3 already handles its target denoise mask. References have no mask socket, so a new block diagnostic would be misleading and was not added. |

## Conclusion

Two thin nodes are justified. A third MiniMax mask diagnostic is not. No
production installation, package installation, ComfyUI restart, model load, or
source-media access was required.
