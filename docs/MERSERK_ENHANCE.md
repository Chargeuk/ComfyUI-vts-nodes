# Merserk Neural Enhance VTS

Search for **Merserk Neural Enhance VTS** (node ID `VTS Merserk Enhance`). Connect an IMAGE batch or VTS DiskImage. The output is an IMAGE socket carrying either a normal **Tensor** or **DiskImage**, as selected by `return_type`.

## Scaling and enhancement

| Enable scaling | Enable neural rendering | Result |
|---|---|---|
| Off | Off | Unchanged image; no server needed. Converts storage type only if requested. |
| On | Off | RTX VSR for enlargement; local Lanczos for reduction. |
| Off | On | Neural enhancement at the original dimensions. |
| On | On | DLSS enlargement and neural enhancement, or local Lanczos reduction followed by neural enhancement at 1×. |

The direction is evaluated **after cropping**. If either dimension needs to shrink, including a resize that enlarges the other dimension, the resize uses local Lanczos. A crop with no remaining resize needs no VSR call. Local-only paths work without a running server or valid server URL.

`sizing_mode = Scale to Min` shares the exact dimension helper with **VTS Images Scale To Min**:

- `smallMaxSize` and `largeMaxSize` are sorted, so entering them backwards gives the same result.
- `scale_type = small` fits the short-side target within the long-side limit, preserving aspect ratio apart from the existing near-aspect snapping and divisibility rules.
- `scale_type = large` uses both side sizes, oriented to the source image. This can change aspect ratio.
- `scale_type = max` sets the longest side to the larger size and calculates the other side from the source aspect ratio.
- `divisible_by` rounds each calculated side down to a multiple; 0 or 1 disables this rounding. A result of zero pixels raises an actionable error.
- `crop = center` crops centrally to the target aspect ratio before resizing. `disabled` stretches when the aspect ratio changes. Neither adds borders.

For example, a 1920×1080 image with side sizes **720 and 1280** becomes **1280×720** using local Lanczos. Turn neural rendering on to enhance that smaller result. A 640×360 image with the same settings enlarges to **1280×720**, using RTX VSR with neural rendering off or DLSS with neural rendering on.

`dlss_quality` selects the DLSS working resolution for enlargement in Scale to Min mode; it does not alter the requested output dimensions. `vsr_quality` selects RTX VSR quality when neural rendering is off. All existing neural preset, style, strength, mask, and model-preset controls remain available.

`iterations` counts neural passes. Scaling happens once; later passes enhance the preceding output at 1×. Only the final image per input is saved. Iterations and neural settings are ignored when neural rendering is off.

## Existing workflows

`sizing_mode = Multiplier` retains `upscaling_factor` and its original even-pixel rounding. Old API prompts that omit the new inputs keep this mode. The included browser extension selects Multiplier when loading workflows saved with the original fourteen widgets. New nodes default to Scale to Min. Refresh ComfyUI's browser after updating the package.

## Windows server and limits

Use **http://192.168.1.1:7865** on this LAN. This also opens Merserk's GUI. The Windows server listens on the Ethernet LAN address, with its firewall rule allowing the local `192.168.1.0/24` subnet on the Private network profile.

The server requires the custom `/vts_enhance_memory` and `/vts_cancel` API, including the exact-size/VSR extension. An unmodified upstream Merserk installation does not expose these endpoints. Update the Windows server before using this node version. The memory endpoint accepts and returns base64 PNG strings; it takes the same JSON parameters and request ID as the older `/vts_enhance` file endpoint. Parameters include `operation` (`neural` or `vsr`), `target_width`, `target_height`, and `vsr_quality` (1–4). Both target dimensions must be supplied together. Neural settings and `iterations` have their existing names. Exact-size requests apply the supplied image to the full target without adding letterboxing; the node performs any requested crop before transfer.

DLSS requires source and negotiated working dimensions of at least 64×64, with target longest side at most 7680 and shortest side at most 4320 (portrait also works). Higher DLSS performance modes need larger targets to keep their working dimensions above 64. RTX VSR accepts only non-shrinking dimensions and has a 16384-pixel texture limit per side; GPU memory can impose a lower practical limit. These GPU limits do not restrict local downscaling or bypass. GPU errors are reported, not silently replaced with another method.

Transfers encode PNG in memory, with no uploaded/downloaded image files, shared drive, or Windows/WSL path translation. RGB/RGBA channels are preserved. VTS calls do not save server output images, success reports, or ZIPs. Tensor output stays in memory; selecting DiskImage deliberately saves the final frames on the ComfyUI client. The API uses the node's settings without changing saved GUI settings. A conflicting GUI GPU render reports busy. Interrupts and timeouts cancel only this node's current request.

In Merserk's Neural Rendering Image and RTX Upscale Image tabs, **Output format → Do not save** returns previews without saving render files or ZIPs. **PNG remains the GUI default**, and choosing PNG or another file format keeps the existing save behavior. The selected setting can be saved in settings/presets. GUI uploads/previews still use Gradio's temporary cache; the VTS memory endpoint avoids that cache. The older file endpoint remains compatible but still uses Gradio's image-transfer cache. Native runtime logs and failure diagnostics may still be written.

DiskImage inputs and outputs are processed one frame at a time; Tensor outputs hold the complete result batch in RAM. `output_dir` belongs to the **ComfyUI client**. Blank uses `output/merserk/render-...`. An incomplete output directory is removed on failure. Bypass with the same storage type returns the original object without rewriting files or quantizing tensors.

To use another ComfyUI installation, update the **whole VTS package**, including `py/vtsUtils/vts_image_sizing.py` and `web/vts_merserk.js`, install its requirements, and restart that ComfyUI installation. Point the node at the LAN server above.
