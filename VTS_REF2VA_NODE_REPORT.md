# VTS Ref2VA node report

## Outcome

Implementation commit `c3795aef0e3fdbfcc9f7c679d4c60626bf18582f`
adds the two helpers justified by the audit:

- `VTS Ref2VA Stereo References`
- `VTS Ref2VA Projection References`

Both return `(ref1, ref2, effective_mask, diagnostics)`. This is the required
LoRA reference order. Diagnostics are sorted JSON containing dimensions,
per-batch effective/marker counts, escaped known-Ref2 collision counts, and
projection geometry where applicable.

The implementation is pure PyTorch. It accepts batched ComfyUI `IMAGE`
`[B,H,W,C]` and `MASK` `[B,H,W]` tensors, keeps tensor devices, does not use
Pillow, does not write files, and stores no image tensors in persistent caches.

## Exact behavior

- Masks must exactly match Ref2 `[B,H,W]` and contain only exact zero/one
  values. Invalid sizes and non-binary values fail before output is returned.
- Known Ref2 pixels that would encode to byte RGB `[255,0,255]` are changed to
  `[255,0,254]`. No other known pixel changes.
- Effective-mask pixels are set to float RGB `[1,0,1]`, then independently
  re-counted to verify `#FF00FF iff effective_mask`.
- Stereo Ref1 is returned as the identical tensor object; the incomplete-right
  input is cloned before changes.
- Projection uses the input VR180 only as a source. Ref1 is a centred black
  letterbox around the requested 16:9 or 4:3 rectilinear view. Ref2 is the
  geometric reprojection of the unletterboxed view. The effective mask is the
  reviewed core's exact unknown-geometry mask.
- A pose/FOV whose rectilinear camera extends outside the source 180-degree
  hemisphere is rejected rather than silently treating unavailable pixels as
  known.
- Geometry grids are built in row chunks. CPU LRU items are capped at 64 MiB;
  larger grids are not retained. Source/output images are never cached.

## Test command and output summary

Environment: Python 3.12.12, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 4080
SUPER (16,376 MiB), driver 591.86. CUDA was available and tested.

Focused Ref2VA command:

```bash
PYTHONPATH=/home/d_a_s/code/comfyui PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/time -v /home/d_a_s/comfyui-env/bin/python \
  -m unittest discover -v -s tests -p 'test_ref2va.py'
```

Output summary: `Ran 8 tests in 1.555s` / `OK`; elapsed wall time 3.31s;
maximum resident set 1,612,312 KiB. This includes batch-two synthetic 1024²
and 2048² CPU frames and the CUDA agreement/device test.

Shared projection command:

```bash
PYTHONPATH=/home/d_a_s/code/comfyui PYTHONDWRITEBYTECODE=1 \
  /home/d_a_s/comfyui-env/bin/python \
  -m unittest discover -v -s tests -p 'test_vr180_projection.py'
```

Output summary: `Ran 9 tests in 0.732s` / `OK`, including 1024/2048 grid
consistency and CUDA/CPU agreement.

Repository-wide command:

```bash
PYTHONPATH=/home/d_a_s/code/comfyui PYTHONDONTWRITEBYTECODE=1 \
  /home/d_a_s/comfyui-env/bin/python \
  -m unittest discover -v -s tests -p 'test_*.py'
```

Output summary: `Ran 26 tests in 1.667s` / `OK`. Two pre-existing motion
context warnings about intentionally dropped overlapping guides were emitted;
there were no failures.

Coverage includes node mapping import, fixed Ref1/Ref2 output semantics, exact
marker iff mask, collision escape, input identity/non-mutation, empty/full/
one-pixel masks, invalid sizes and non-binary masks, batch size greater than
one, 1024/2048 frames, CPU/CUDA, direct shared-core equality, bounded cache,
and failure paths that leave an empty publication directory.

## Production state and gated installation

The installed VTS checkout remains clean on `main`; no production ComfyUI was
modified or restarted. Do not run the following until coordinator review:

```bash
git -C /home/d_a_s/code/comfyui/custom_nodes/ComfyUI-vts-nodes \
  merge --ff-only codex/vts-ref2va-dandesktop-v1
```

After that merge, use the existing operator-controlled ComfyUI restart
procedure. The loader discovers `py/VTS_Ref2VA.py` automatically; no loader
file or dependency installation is needed.

For a different clone, fetch the delivered bundle first, review the branch,
then fast-forward or cherry-pick the reviewed commits. Installation remains
gated; the delivery bundle is not itself an installation.
