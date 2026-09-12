# ComfyUI-vts-nodes

## Prepared H3 models

Experimental save/load nodes bundle merged MiniMax-H3 weights with VDN, SLA or
hybrid attention settings. A separate runtime-options node overrides sparsity,
dense steps, memory policy and sampling settings without rebaking the weights.
See [Prepared H3 models](docs/PREPARED_H3.md) for the one-time export workflow,
supported overrides and limitations.

## VR180 projection and outpainting

The package includes geometric nodes for converting between rectilinear and
square VR180 views. It also includes **VTS VR180 Square To ERP Outpaint
Canvas**, which places a square VR180 eye view on a full 2:1 equirectangular
canvas and produces exact known/outpaint masks.

The outpaint node accepts either a normal ComfyUI `IMAGE` tensor or a VTS
`DiskImage` sequence. Its outputs are ordinary in-memory tensors, ready for an
outpainting or harmonisation workflow. It supports both rectangular
half-equirectangular VR180 sources and circular equidistant-fisheye sources.
Optional per-edge source-pixel trims crop the image being projected. The
projected crop bands become black, leave the known mask, and enter the
outpaint mask without changing the full-ERP tensor dimensions.
The node also reports the final post-trim left and right projected x positions
as zero-based output-canvas pixel coordinates.

See [VR180_PROJECTION_NODES.md](VR180_PROJECTION_NODES.md) for projection
conventions, presets, inputs, outputs, and memory behaviour.

## Positioned seam strip mask

**VTS Positioned Seam Strip Mask** creates a vertical repaint mask at an
explicit pixel column. It has the same `width`, `height`, `seam_width`, and
in-strip `feather` controls as a centred panorama seam mask, plus
`strip_center_x`, measured from the left edge. A strip that extends beyond the
canvas is clipped while retaining its original feather profile.

## Merserk frame interpolation

**VTS Merserk Frame Interpolate** sends IMAGE or DiskImage sequences to a Windows
Merserk server using lossless PNG streaming. It uploads each original once and
downloads only intermediate frames, with 2x, approximate 3x, 4x and 8x options.
See [Merserk frame interpolation](docs/MERSERK_INTERPOLATION.md) for setup,
timing, storage choices and the required server endpoint.
