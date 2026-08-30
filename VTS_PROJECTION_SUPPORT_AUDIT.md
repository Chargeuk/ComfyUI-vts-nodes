# VTS rectilinear-to-VR180 support audit

## Result

The installed VTS nodes already provide the complete model-free geometry layer
needed to prepare and evaluate the separate projection LoRA. No third geometry
node is needed.

- `VTSRectilinearToVR180` maps an ordinary rectilinear image or decoded video
  batch into a square half-equirectangular left-eye VR180 canvas.
- It returns exact complementary `known_mask` and `unknown_mask` outputs. The
  unknown RGB pixels are exact `#FF00FF`, suitable for the established Ref2VA
  marker convention.
- `VTSVR180ToRectilinear` performs the analytical inverse view. Feeding the
  forward `known_mask` into its optional `source_known_mask` input prevents an
  unknown source pixel from being presented as a valid reconstruction.
- Both directions expose the same explicit FOV, yaw, pitch, roll, sampling,
  row-chunk, and frame-batch controls. Sorted JSON metadata records these
  values and the per-frame validity counts.
- `VTSRef2VAProjectionReferences` supplies the fixed Ref1/Ref2/effective-mask
  ordering expected by the projection-LoRA workflow.

The projection is reversible geometry, not learned content generation. For
example, a 16:9 camera view only covers one region of the 180-degree canvas;
the mask marks the rest as unknown for the LoRA to generate.

## Production checks

`tests/test_vr180_roundtrip_contract.py` adds public-wrapper contract coverage
for both 16:9 and 4:3 inputs, non-zero yaw/pitch/roll, bounded frame batching,
forward-to-reverse mask propagation, metadata agreement, reconstruction error,
and exact confinement of marker magenta to unknown geometry.

The portable examples remain:

- `examples/VTS_VR180_Roundtrip_Diagnostic.json`
- `examples/VTS_Ref2VA_Projection_Dataset_Eval.json`

They are model-free and contain no machine-specific media or checkpoint paths.
Use the round-trip workflow to separate a projection/geometry problem from a
LoRA-generation problem before comparing model outputs.

## Known limitations

- The nodes model one pinhole rectilinear camera inside a 180-by-180-degree
  half-equirectangular view. They do not infer camera FOV or pose.
- A single rectilinear source cannot contain pixels outside its camera view;
  those pixels are intentionally left magenta/unknown.
- Bilinear or bicubic round trips resample twice, so they are close rather than
  byte-identical. Evaluate only where the returned reverse mask is known.
- FOV and pose must match between forward and reverse nodes. The JSON metadata
  is the authoritative record for this check.

## Validation command

Run in DanDesktop's actual ComfyUI environment:

```bash
PYTHONPATH=/home/d_a_s/code/comfyui PYTHONDONTWRITEBYTECODE=1 \
  /home/d_a_s/comfyui-env/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

Because the node files were added to the installed custom-node repository,
ComfyUI must be restarted once before these node classes appear in an already
running UI. No dependency installation or model download is required.
