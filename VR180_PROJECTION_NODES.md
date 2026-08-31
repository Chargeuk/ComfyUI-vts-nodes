# VTS rectilinear / VR180 projection nodes

These nodes expose the existing reviewed half-equirectangular projection core
to ComfyUI:

- **VTS Rectilinear Video To VR180** converts an ordinary rectilinear image or
  decoded video batch into a partial square VR180 conditioning/reference view.
- **VTS VR180 To Rectilinear Video** performs the explicit reverse view for
  diagnostics and round-trip checks. Its optional `source_known_mask` prevents
  unknown source pixels from becoming apparently valid.
- **VTS VR180 Square To ERP Outpaint Canvas** converts one square VR180 eye
  view into a full 360 by 180 degree, 2:1 equirectangular canvas for learned
  outpainting. It accepts a normal `IMAGE` tensor or a VTS `DiskImage`.

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

## Square VR180 to full-ERP outpaint canvas

The outpaint node does not invent pixels. It copies the angular region supplied
by the source into a full equirectangular canvas, fills everything else with
`fill_color`, and returns three tensors:

1. `erp_canvas` (`IMAGE`) - the projected full-ERP image or video batch.
2. `known_mask` (`MASK`) - pixels genuinely supplied by the source.
3. `outpaint_mask` (`MASK`) - the exact inverse region that a model must create.

The source may be either:

- a ComfyUI `IMAGE` tensor with shape `[frames, height, width, channels]`; or
- a VTS `DiskImage`. Disk-backed sequences are loaded in bounded groups set by
  `frame_batch_size`.

Outputs are always ordinary in-memory tensors. The source must be square and
the requested output must be exactly 2:1, for example 2048 by 1024.

### Projection modes

- `half_equirectangular_ideal_180` - a rectangular 180 by 180 degree source;
  it occupies the centre half of the full ERP canvas.
- `half_equirectangular_production_calibrated` - the current production-video
  fit, with effective spans of 202.611324 by 160.978888 degrees.
- `half_equirectangular_custom` - user-supplied rectangular angular spans.
- `equidistant_fisheye_180` - a circular 180-degree equidistant fisheye.
- `equidistant_fisheye_custom` - a circular equidistant fisheye with a custom
  angular diameter; horizontal and vertical values must match.

The production preset reflects the measured source encoding used by the VTS
pipeline. The ideal half-ERP mode is appropriate for mathematically standard
square VR180 frames. The fisheye modes are for genuinely circular lens images;
they must not be used merely because a rectangular VR180 frame looks curved.

`yaw_degrees`, `pitch_degrees`, and `roll_degrees` rotate the supplied view on
the sphere. `chunk_rows` bounds temporary projection work. `frame_batch_size`
bounds disk loading and per-call frame processing. `sampling` controls image
resampling; masks remain geometric and exact.

`trim_left`, `trim_right`, `trim_top`, and `trim_bottom` optionally crop that
many pixels from the corresponding edges of the square source image. Their
defaults are zero. The crop is evaluated in source coordinates and then
projected, so its new edges correctly follow calibrated, rotated, and fisheye
views instead of trimming the outer full-ERP canvas. Projected crop bands are
exact black in `erp_canvas`, zero in `known_mask`, and one in `outpaint_mask`.
The outpaint mask therefore follows the pasted image's new cropped edges.
Opposing trims must leave at least one source row or column. Output tensor
dimensions remain unchanged.

For an outpaint workflow, pass `erp_canvas` as the starting image and
`outpaint_mask` as the region to generate. The known mask is useful when the
generated result must later be composited without changing the source view.
The two masks remain exact inverses when trims are non-zero.
