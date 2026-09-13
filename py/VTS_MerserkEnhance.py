import base64
import io
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import TimeoutError
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image
from gradio_client import Client

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)
from vtsUtils import DiskImage
from vts_image_sizing import ScaleToMinDimensions


class VTSMerserkEnhance(ScaleToMinDimensions):
    @classmethod
    def INPUT_TYPES(cls):
        strength = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image or batch to process, supplied as a normal ComfyUI IMAGE or VTS DiskImage. Each image is enhanced independently; this node does not share temporal history between video frames."}),
            "server_url": ("STRING", {"default": "http://192.168.1.1:7865", "tooltip": "Address of your Windows Merserk server, for example http://192.168.1.1:7865. Required for neural enhancement or enlargement; local downscaling and bypass work without it."}),
            "return_type": (["Tensor", "DiskImage"], {"default": "Tensor", "tooltip": "Tensor returns a normal IMAGE batch held in RAM. DiskImage saves only the final images on this ComfyUI machine and returns their file references, which is more memory-efficient for large batches."}),
            "upscaling_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16384.0, "step": 0.05, "tooltip": "Used only when scaling is enabled and sizing_mode is Multiplier. Multiplies width and height: 2 doubles each side, 1 keeps the size, and 0.5 halves each side. Dimensions are rounded to even pixels. Enlargement uses RTX VSR; reduction uses Lanczos."}),
            "iterations": ("INT", {"default": 1, "min": 1, "tooltip": "Our outer enhancement loop: feed each result back through neural rendering this many times. Resizing happens once before the loop, and only the final result is returned. 3 iterations with 2 nr_passes means 6 neural evaluations. More iterations take longer and can overprocess detail. Ignored when neural rendering is disabled."}),
            "nr_passes": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Merserk native neural passes within EACH iteration, from 1 to 4. Total neural evaluations = iterations x nr_passes: 2 iterations with 3 passes means 6. Increasing this adds processing and may strengthen changes; it does not resize again. Ignored when neural rendering is disabled."}),
            "nr_style": (["Default", "Natural", "Cinematic"], {"tooltip": "Overall look requested from the neural model. Default uses its standard style; Natural aims for a more restrained look; Cinematic aims for a more stylized look. The visible difference depends on the image and model. Ignored when neural rendering is disabled."}),
            "nr_intensity": ("FLOAT", dict(strength, tooltip="Overall neural enhancement strength. Lower values reduce the requested effect; 1 is the standard setting. Higher values request a stronger effect, but the model may limit the response, so 2 is not guaranteed to look twice as strong. Use enable_neural_rendering to bypass enhancement completely.")),
            "local_tone_strength": ("FLOAT", dict(strength, tooltip="Strength of local tone changes: how the model adjusts brightness and contrast within parts of the image. Lower values request less tonal change; higher values can reshape lighting and contrast more strongly. 1 is the standard setting. Tone Preservation can bring the final tone back toward the source.")),
            "local_structure_strength": ("FLOAT", dict(strength, tooltip="Strength of local detail and texture reconstruction, such as edges, hair and surface detail. Lower values request less reconstruction; higher values request stronger detail changes and can change the original texture. 1 is the standard setting. This is neural reconstruction, not a conventional sharpening filter.")),
            "skin_structure_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": "Skin and pore reconstruction inside the automatic skin mask. -1 uses the native/model default, not negative sharpening. 0 is the lowest manual strength; positive values up to 2 request stronger skin reconstruction. Intermediate negative values are not documented as a smoothing scale: use -1 for the default or 0-2 for manual control. Requires automatic_mask=True; this node does not switch it on automatically."}),
            "automatic_mask": ("BOOLEAN", {"default": False, "tooltip": "Let the neural model identify skin regions so Skin Structure Strength can act on them. Enable this when adjusting skin structure. With it off, changing skin structure has no useful targeted effect. This is model-generated skin masking, not a custom mask input."}),
            "nr_color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How much of the neural result's colour is kept in the final blend. 0 keeps the source colour while allowing detail changes; 1 keeps the full neural colour contribution. Values between them blend the two. Pair 0 with Tone Preservation 1 for a detail-focused result."}),
            "tone_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How strongly the final blend keeps the original brightness and contrast. 0 adds no tone preservation; 1 gives maximum source-tone preservation while allowing neural detail changes. This preserves source tone, whereas Local Tone Strength controls the tone changes requested from the model."}),
            "face_skin_protection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Reduce neural changes in detected face/skin regions to keep them closer to the original. 0 adds no extra protection; 1 gives the strongest protection. This limits the final effect on faces, whereas Skin Structure Strength controls the requested skin reconstruction."}),
            "grain_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How strongly to preserve fine source grain/noise through the final blend. 0 adds no extra grain preservation; 1 gives maximum preservation. Higher values can retain a film-like texture, but can also retain unwanted source noise."}),
            "output_dir": ("STRING", {"default": "", "tooltip": "Used only for DiskImage output. Folder on this ComfyUI machine, not the Merserk server. Blank uses ComfyUI output/merserk. Each run creates its own subfolder containing only the final lossless PNG images."}),
            "timeout_seconds": ("INT", {"default": 600, "min": 1, "tooltip": "Maximum wait for each image request, including queueing, upscaling and all neural iterations/passes. Increase this for large images or many passes. On timeout, the node requests cancellation of its own server job."}),
        }, "optional": {
            "enable_scaling": ("BOOLEAN", {"default": True, "tooltip": "Enable resizing using the sizing controls below. Enlargement uses RTX VSR once; reduction uses local Lanczos before enhancement. Off preserves the input dimensions and ignores sizing/crop controls. Neural enhancement can still run."}),
            "enable_neural_rendering": ("BOOLEAN", {"default": True, "tooltip": "Apply Neuroframe enhancement after any resizing. Off skips all neural controls, iterations and NR Passes, but scaling can still run. Turn both this and Enable Scaling off to pass images through, converting storage type only if requested."}),
            "sizing_mode": (["Scale to Min", "Multiplier"], {"default": "Scale to Min", "tooltip": "Scale to Min uses the two side sizes, scale_type and divisible_by rules from VTS Scale To Min. Multiplier uses upscaling_factor to resize both sides proportionally. Used only when Enable Scaling is on."}),
            "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1, "tooltip": "Smaller side target/limit in pixels for Scale to Min mode. Together with largeMaxSize and scale_type it determines output dimensions. The two values are sorted automatically, so entering them backwards is fine. For a 16:9 landscape image, 720 and 1280 typically give 1280 x 720."}),
            "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1, "tooltip": "Larger side target/limit in pixels for Scale to Min mode. The two side values are sorted automatically, so this may be entered smaller than smallMaxSize. Its exact role depends on scale_type; max uses this larger value as the longest output side."}),
            "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1, "tooltip": "Scale to Min only: round each calculated dimension down to a multiple of this number. For example, 1001 becomes 1000 with 8. Use 0 or 1 to disable rounding. Rounding can slightly change aspect ratio; choose sizes large enough that neither side becomes zero."}),
            "crop": (["disabled", "center"], {"default": "disabled", "tooltip": "When scaling, center removes image edges to match the target aspect ratio before resizing. Disabled keeps all content but stretches it if the target aspect ratio differs. For a wide image resized to a square, center cuts off the left/right edges; disabled squeezes the width. No borders are added."}),
            "scale_type": (["small", "large", "max"], {"default": "small", "tooltip": "Scale to Min only. small fits the image within the short/long side limits, largely preserving aspect ratio. large uses both side sizes as the output dimensions, which may change aspect ratio. max sets the longest side to the larger limit and calculates the other proportionally. Divisibility rounding applies afterwards."}),
            "vsr_quality": (["Low", "Medium", "High", "Ultra"], {"default": "Ultra", "tooltip": "RTX VSR quality for enlargement, whether or not neural enhancement follows. Low is lighter; Ultra requests the highest quality and may take longer. This changes upscaling quality, not output dimensions or neural pass counts. Ignored when no enlargement is needed."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "VTS/image"
    DESCRIPTION = "VTS Scale To Min sizing with RTX VSR upscaling followed by optional Neuroframe enhancement. Downscales locally with Lanczos; repeated enhancement keeps the final dimensions."

    @staticmethod
    def _request(client, source, parameters, timeout_seconds, index):
        request_id = uuid.uuid4().hex
        job = client.submit(source, json.dumps(parameters), request_id, api_name="/vts_enhance_memory")
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                model_management.throw_exception_if_processing_interrupted()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Merserk exceeded {timeout_seconds} seconds for image {index + 1}.")
                try:
                    return job.result(timeout=.25)
                except TimeoutError:
                    continue
        except BaseException:
            job.cancel()
            with suppress(Exception):
                client.submit(request_id, api_name="/vts_cancel").result(timeout=10)
            raise

    def enhance(self, image, server_url="http://192.168.1.1:7865", return_type="Tensor",
                upscaling_factor=1.0, iterations=1, nr_passes=1, nr_style="Default",
                nr_intensity=1.0, local_tone_strength=1.0, local_structure_strength=1.0,
                skin_structure_strength=-1.0, automatic_mask=False,
                nr_color_strength=1.0, tone_preservation=0.0,
                face_skin_protection=0.0, grain_preservation=0.0,
                output_dir="", timeout_seconds=600, enable_scaling=True,
                enable_neural_rendering=True, sizing_mode="Multiplier", smallMaxSize=512,
                largeMaxSize=512, divisible_by=2, crop="disabled", scale_type="small",
                vsr_quality="Ultra"):
        if return_type not in {"Tensor", "DiskImage"}:
            raise ValueError("Return type must be Tensor or DiskImage.")
        count = len(image)
        if count == 0:
            raise ValueError("Merserk needs at least one input image.")
        if not enable_scaling and not enable_neural_rendering:
            if (return_type == "Tensor" and isinstance(image, torch.Tensor)) or (return_type == "DiskImage" and isinstance(image, DiskImage)):
                return (image,)
        neural_parameters = dict(
            iterations=iterations, nr_passes=nr_passes,
            nr_style=nr_style, nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength, local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength, automatic_mask=automatic_mask,
            nr_color_strength=nr_color_strength, tone_preservation=tone_preservation,
            face_skin_protection=face_skin_protection, grain_preservation=grain_preservation,
        )
        saved_directory = None
        complete = False
        try:
            if return_type == "DiskImage":
                destination = Path(os.path.expandvars(output_dir)).expanduser() if output_dir.strip() else Path(folder_paths.get_output_directory()) / "merserk"
                destination.mkdir(parents=True, exist_ok=True)
                saved_directory = Path(tempfile.mkdtemp(prefix="render-", dir=destination.resolve()))
            frames = []
            progress = ProgressBar(count)
            client = None
            try:
                for index in range(count):
                    model_management.throw_exception_if_processing_interrupted()
                    frame = image[index]
                    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                        raise ValueError("Merserk expects RGB or RGBA images in BHWC layout.")
                    mode = "RGBA" if frame.shape[-1] == 4 else "RGB"
                    height, width = frame.shape[:2]
                    target_width, target_height = width, height
                    factor = 1.0
                    if enable_scaling:
                        if sizing_mode == "Scale to Min":
                            target_width, target_height = self._calculate_target_dimensions(
                                width, height, smallMaxSize, largeMaxSize, divisible_by, scale_type)
                        elif sizing_mode == "Multiplier":
                            factor = float(upscaling_factor)
                            # Match Merserk's legacy rounding to the nearest even pixel.
                            target_width = max(2, int(width * factor / 2 + .5) * 2)
                            target_height = max(2, int(height * factor / 2 + .5) * 2)
                        else:
                            raise ValueError("Unknown sizing mode.")
                        if min(target_width, target_height) < 1:
                            raise ValueError("Sizing produced a zero dimension. Increase the side sizes or reduce divisible_by.")
                        if crop == "center" and (width, height) != (target_width, target_height):
                            frame = self._center_crop_for_aspect(frame.unsqueeze(0), target_width, target_height)[0]
                    height, width = frame.shape[:2]
                    shrinking = target_width < width or target_height < height
                    resizing = (target_width, target_height) != (width, height)
                    enlarging = target_width > width or target_height > height
                    remote = enable_neural_rendering or enlarging
                    parameters = None
                    if remote:
                        if enable_neural_rendering:
                            parameters = dict(neural_parameters, operation="neural")
                        else:
                            parameters = dict(operation="vsr")
                        parameters.update(target_width=target_width, target_height=target_height,
                                          vsr_quality={"Low": 1, "Medium": 2, "High": 3, "Ultra": 4}[vsr_quality])
                    if not remote and not resizing and saved_directory is None:
                        processed_frame = frame.detach().cpu()
                        enhanced = None
                    else:
                        pixels = frame.detach().cpu().float().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                        enhanced = Image.fromarray(pixels)
                        del pixels
                        if shrinking:
                            enhanced = enhanced.resize((min(width, target_width), min(height, target_height)), Image.Resampling.LANCZOS)
                        if remote:
                            if client is None:
                                server_url = server_url.strip().rstrip("/")
                                address = urlsplit(server_url)
                                if address.scheme not in {"http", "https"} or not address.netloc or address.query or address.fragment:
                                    raise ValueError("Enter the Merserk server URL, for example http://192.168.1.1:7865.")
                                client = Client(server_url, verbose=False, download_files=False,
                                                analytics_enabled=False, httpx_kwargs={"timeout": timeout_seconds})
                            with io.BytesIO() as buffer:
                                enhanced.save(buffer, format="PNG")
                                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                            returned_png = self._request(client, encoded, parameters, timeout_seconds, index)
                            del encoded
                            model_management.throw_exception_if_processing_interrupted()
                            with Image.open(io.BytesIO(base64.b64decode(returned_png, validate=True))) as returned:
                                enhanced = returned.convert(mode)
                            del returned_png
                            if enhanced.size != (target_width, target_height):
                                raise ValueError("Merserk returned the wrong dimensions. Install the updated VTS API on the server.")
                        if saved_directory is None:
                            processed_frame = torch.from_numpy(np.array(enhanced, dtype=np.float32) / 255.0)
                    shape = (count, target_height, target_width, len(mode))
                    if index and shape != output_shape:
                        raise ValueError("All returned images must have the same dimensions for an IMAGE batch.")
                    output_shape = shape
                    if saved_directory is not None:
                        enhanced.save(saved_directory / f"image_{index:06d}.png")
                    else:
                        frames.append(processed_frame)
                    del frame, enhanced
                    progress.update(1)
            finally:
                if client is not None:
                    client.close()
            if saved_directory is not None:
                output = DiskImage(prefix="image", start_sequence=0, number_of_images=count,
                                   output_dir=str(saved_directory), format="png", image=None)
                output.shape, output.dtype, output.ndim = output_shape, torch.float32, 4
            else:
                output = torch.stack(frames)
            complete = True
            return (output,)
        finally:
            if saved_directory is not None and not complete:
                # This directory was created exclusively for this execution.
                shutil.rmtree(saved_directory)


NODE_CLASS_MAPPINGS = {"VTS Merserk Enhance": VTSMerserkEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS Merserk Enhance": "Merserk Neural Enhance VTS"}
