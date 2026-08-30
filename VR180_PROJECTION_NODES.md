# VTS rectilinear / VR180 projection nodes

These nodes expose the existing reviewed half-equirectangular projection core
to ComfyUI:

- **VTS Rectilinear Video To VR180** converts an ordinary rectilinear image or
  decoded video batch into a partial square VR180 conditioning/reference view.
- **VTS VR180 To Rectilinear Video** performs the explicit reverse view for
  diagnostics and round-trip checks. Its optional `source_known_mask` prevents
  unknown source pixels from becoming apparently valid.

Both nodes return the image, exact known and unknown masks, and sorted JSON
metadata recording every geometry setting. Use identical output size, FOV,
yaw, pitch, roll, and sampling settings when results must be reproduced.

The geometry is square half-equirectangular 180 by 180 degrees, with pixel
centres and `align_corners=False`. Camera axes are +X right, +Y up, +Z forward.
Unknown pixels are exact `#FF00FF`.

The angle signs follow the implementation exactly. Positive pitch moves the
projected rectilinear view downward on the VR180 canvas. Camera rays are
rotated by `Rz(roll) @ Rx(pitch) @ Ry(yaw)` (the rightmost rotation is applied
first). Positive roll is therefore a right-handed rotation about fixed +Z
after yaw and pitch; at zero yaw and pitch it moves the view's right side
toward +Y/up and its left side toward -Y/down. In screen terms, that tilts the
view counter-clockwise. With non-zero yaw or pitch, do not reinterpret this as
a separate post-pose roll about the camera's new local optical axis.

`chunk_rows` bounds the temporary sampling-grid transfer. `frame_batch_size`
bounds the number of video frames sent through `grid_sample` at once. The
required output batch is allocated once; the node does not retain source or
output images after the call. Only the shared core's bounded CPU geometry-grid
LRU persists.

This is a geometric projection, not a learned outpainting operation. Areas
outside the rectilinear camera view remain unknown and must be handled by the
projection LoRA or another downstream generator.
