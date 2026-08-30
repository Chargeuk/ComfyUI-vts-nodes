# MiniMax H3 VR180 workflow integration

No additional VTS node is required. The existing installed nodes already cover
the two missing workflow operations:

- `VTSRectilinearToVR180` produces a partial half-equirectangular VR180 image,
  exact known/unknown masks, and geometry metadata from a rectilinear image or
  decoded video batch.
- `VTSVR180ToRectilinear` performs the reverse diagnostic and accepts the
  forward known mask, preventing magenta/unknown source pixels from being
  treated as valid.
- `VTSRef2VAProjectionReferences` produces fixed-order MiniMax H3 Ref1 and Ref2
  plus the exact effective generation mask from genuine left-eye VR180 data.

## Portable examples

`examples/VTS_VR180_Roundtrip_Diagnostic.json` is a model-free, CPU-safe smoke
workflow. Replace `EmptyImage` with `LoadImage` or decoded video frames for real
evaluation. Keep all FOV and pose widgets identical on the forward and reverse
nodes. The forward `known_mask` must remain connected to the reverse
`source_known_mask`.

`examples/VTS_Ref2VA_Projection_Dataset_Eval.json` demonstrates the exact
training/evaluation reference contract:

1. Genuine square left-eye VR180 enters `left_vr180`.
2. `ref1` is the letterboxed 16:9 rectilinear reference.
3. `ref2` is the partial VR180 reference with exact `#FF00FF` markers.
4. `effective_mask` is the area MiniMax must generate.
5. The diagnostic branch inverts that mask before reverse projection, so only
   genuine known Ref2 pixels can contribute.

Both examples use a synthetic `EmptyImage`, so they load without media paths,
models, LoRAs, or checkpoints. They exercise geometry only.

## Insert into the existing MiniMax H3 workflow

The inspected base workflow is:

`/mnt/external-lan/comfyui/workflows/minimax-h3/droz_MiniMaxH3_PerRowMasking_Example_v2_0005.json`

For projection-LoRA evaluation, insert the example's outputs immediately before
the existing reference-image preparation route:

- VTS `ref1` -> MiniMax first reference image (Ref1).
- VTS `ref2` -> MiniMax second reference image (Ref2).
- VTS `effective_mask` -> the existing generation/inpaint mask route.

Reference order is not interchangeable. Ref1 is the rectilinear semantic
reference; Ref2 is the partial VR180 spatial reference.

For a live rectilinear source rather than a genuine VR180 training target,
letterbox the rectilinear input to the square MiniMax reference canvas, and use
`VTSRectilinearToVR180` to create Ref2. Use its `unknown_mask` as the generation
mask. The same FOV/yaw/pitch/roll values must be recorded and reused for reverse
diagnostics.

Do not splice these nodes into the 119 KB model workflow until a trained LoRA is
available. Keeping the geometry example separate prevents accidental model
execution and makes failures attributable to projection versus inference.

## Validation

Run from the VTS repository with the ComfyUI Python environment:

```bash
PYTHONPATH=/home/d_a_s/code/comfyui \
  /home/d_a_s/comfyui-env/bin/python -m unittest discover -s tests -v
```

`tests/test_vr180_workflows.py` verifies unique graph IDs, every link endpoint
and type, the forward-known-to-reverse-mask connection, fixed Ref1/Ref2 order,
16:9-to-1024x576 diagnostic geometry, and required node presence.
