# ComfyUI-vts-nodes

## VR180 projection and outpainting

The package includes geometric nodes for converting between rectilinear and
square VR180 views. It also includes **VTS VR180 Square To ERP Outpaint
Canvas**, which places a square VR180 eye view on a full 2:1 equirectangular
canvas and produces exact known/outpaint masks.

The outpaint node accepts either a normal ComfyUI `IMAGE` tensor or a VTS
`DiskImage` sequence. Its outputs are ordinary in-memory tensors, ready for an
outpainting or harmonisation workflow. It supports both rectangular
half-equirectangular VR180 sources and circular equidistant-fisheye sources.
Optional per-edge pixel trims create black matte borders that are excluded
from both output masks without changing the tensor dimensions.

See [VR180_PROJECTION_NODES.md](VR180_PROJECTION_NODES.md) for projection
conventions, presets, inputs, outputs, and memory behaviour.
