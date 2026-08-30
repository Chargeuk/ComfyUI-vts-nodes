# Staged VTS rectilinear / VR180 projection patch

This folder is intentionally **not installed**. It contains a reviewable patch
for ComfyUI-vts-nodes.

## Files to install after review

- Copy `py/vtsUtils/vr180_projection.py` to the matching `py/vtsUtils/` folder.
- Copy `py/VTS_Rectilinear_To_VR180.py` to the plugin's `py/` folder.
- Restart ComfyUI. The existing VTS loader discovers the node file
  automatically; its root `__init__.py` does not need editing.

The two registered nodes are:

- **VTS Rectilinear Video To VR180**: creates a partial half-equirectangular
  VR180 frame plus exact known/unknown masks. Unknown RGB pixels are exactly
  `#FF00FF` (`1, 0, 1` in ComfyUI floats).
- **VTS VR180 To Rectilinear Video**: reverse-projects a VR180 frame for
  diagnostics, dataset creation, and round-trip tests. An optional known mask
  prevents unknown source pixels from silently becoming valid.

Both nodes accept ComfyUI image batches. `chunk_rows` limits transient GPU
memory without changing results. A bounded four-entry LRU stores grids in CPU
memory only when an individual grid is no larger than 64 MiB. Larger grids,
including 8192-square grids, are computed but deliberately not retained; the
cache can therefore retain at most 256 MiB per projection direction.

Sampling grids stay float32 for FP16/BF16 images. On a backend that cannot mix
a low-precision image with a float32 grid, the image is safely sampled in
float32 and the result is converted back to the original dtype. This costs
temporary memory but avoids reducing projection-coordinate precision.

## Geometry contract

- Projection: `half_equirectangular_180`.
- Longitude: -90 degrees at the left edge to +90 at the right edge.
- Latitude: +90 degrees at the top edge to -90 at the bottom edge.
- Camera coordinates: +X right, +Y up, +Z forward.
- Pixel-centred rays and PyTorch `align_corners=False` throughout.
- Positive yaw turns right, positive pitch turns up, and roll is around the
  camera's local optical axis.
- A coordinate is known only when it lies within the source's outermost pixel
  centres: normalized `±(1-1/width)` and `±(1-1/height)`. Sampling uses zero
  padding, while exact magenta is applied explicitly wherever the mask is
  unknown.
- Masks are raw geometry. H3's 32x32 block expansion belongs downstream in the
  workflow and is deliberately not hidden inside this node.

## Tests

From this staged folder, with PyTorch installed:

```bash
python tests/test_vr180_projection.py
```

The suite checks centre alignment, exact magenta, masks, batching, row chunks,
pose handling, reverse projection, bounded caching, 1024/2048 scale
consistency, validation errors, and CPU/CUDA agreement when CUDA is available.

## Installation checks still required

Before merging into either installed plugin:

1. Run the tests in that ComfyUI Python environment.
2. Start ComfyUI and confirm both nodes import without warnings.
3. Use a calibration grid to confirm orientation and FOV against the existing
   VR180 production convention.
4. Save one 1024-square and one 2048-square workflow result, verify unknown
   pixels remain exactly `#FF00FF`, and record peak VRAM.
5. Commit the VTS repository change before copying it to the other machine.
