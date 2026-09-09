# Merserk Neural Enhance VTS

Installed in this machine's WSL ComfyUI VTS package. Search for **Merserk Neural Enhance VTS** (node ID: `VTS Merserk Enhance`).

Server URL: **http://192.168.1.1:7865**. The Windows Merserk server must be running. This address also opens its GUI. The server listens on this machine's Ethernet LAN address; its firewall rule permits the local `192.168.1.0/24` subnet on the Private network profile.

## Use

Connect a normal IMAGE batch or VTS DiskImage to `image`. Choose `return_type`:

- **Tensor:** a normal ComfyUI IMAGE batch, ready for Preview Image, Save Image, or other image nodes.
- **DiskImage:** final lossless PNGs in a separate folder for each execution. The returned object connects to VTS nodes that accept DiskImage.

The output socket is IMAGE in both cases, following VTS's convention.

Set the upscale factor, neural rendering settings, and iterations. For example, factor **2** and **3 iterations** means upscale/enhance once, then enhance the previous result twice at the same resolution. Only the final result of each input frame is returned.

DiskImage inputs are read one frame at a time. DiskImage outputs are saved one frame at a time; Tensor outputs necessarily hold the complete returned batch in RAM. `output_dir` is a path on the **ComfyUI client**, not on the Windows server. Blank uses ComfyUI's configured output directory under `merserk/render-...`. Files use `image_000000.png`, `image_000001.png`, etc. Each run gets its own directory.

Transfers use PNG image data, so no shared drive or Windows/WSL path translation is required. The API uses the node's settings without changing the GUI's saved settings. Merserk can render one job at a time; a conflicting GUI render is reported as busy. Interrupting the node requests cancellation of that node's current render. An incomplete DiskImage batch is removed on failure.

## Add this node to another ComfyUI installation

1. Install the VTS node package there if it is not already present.
2. Copy `py/VTS_MerserkEnhance.py` from this folder into `ComfyUI/custom_nodes/ComfyUI-vts-nodes/py/`.
3. In that ComfyUI installation's Python environment, run `python -m pip install -r requirements.txt` using this folder's requirements file.
4. Restart ComfyUI and refresh the browser. Use **http://192.168.1.1:7865** in the node.

The server already has the required `/vts_enhance` and `/vts_cancel` Gradio API endpoints installed. An unmodified upstream Merserk installation does not expose these endpoints.

## Verification

Live tests from WSL through the Windows server passed for all four input/output combinations, each with a two-image batch, 1.5× initial upscale, and two iterations. Inputs of 256×256 returned 384×384 images. DiskImage tests included a source sequence beginning at index 7. Unit checks covered partial-output cleanup, cancellation, timeout, and URL validation.
