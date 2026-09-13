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
            "image": ("IMAGE", {"tooltip": "Accepts a regular IMAGE batch or a VTS DiskImage."}),
            "server_url": ("STRING", {"default": "http://192.168.1.1:7865", "tooltip": "Windows Merserk server with the VTS image API installed."}),
            "return_type": (["Tensor", "DiskImage"], {"default": "Tensor", "tooltip": "Tensor is a normal ComfyUI IMAGE. DiskImage keeps the returned frames on disk."}),
            "upscaling_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16384.0, "step": 0.05, "tooltip": "Size multiplier, used only in Multiplier sizing mode. Values below 1 downscale locally."}),
            "iterations": ("INT", {"default": 1, "min": 1, "tooltip": "Enhance the previous result this many times on the server. Resize happens once before the loop; only the final image is returned."}),
            "nr_passes": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Native Merserk neural passes per iteration. Total neural evaluations = iterations x nr_passes."}),
            "nr_style": (["Default", "Natural", "Cinematic"],),
            "nr_intensity": ("FLOAT", dict(strength)),
            "local_tone_strength": ("FLOAT", dict(strength)),
            "local_structure_strength": ("FLOAT", dict(strength)),
            "skin_structure_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "automatic_mask": ("BOOLEAN", {"default": False}),
            "nr_color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "tone_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "face_skin_protection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "grain_preservation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "output_dir": ("STRING", {"default": "", "tooltip": "DiskImage only. Blank uses ComfyUI's output/merserk folder. Each run creates its own subfolder of lossless PNGs."}),
            "timeout_seconds": ("INT", {"default": 600, "min": 1, "tooltip": "Maximum time to wait for each image."}),
        }, "optional": {
            "enable_scaling": ("BOOLEAN", {"default": True}),
            "enable_neural_rendering": ("BOOLEAN", {"default": True}),
            "sizing_mode": (["Scale to Min", "Multiplier"], {"default": "Scale to Min", "tooltip": "Scale to Min uses the same dimension rules as VTS Images Scale To Min. Multiplier preserves older workflows."}),
            "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
            "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
            "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            "crop": (["disabled", "center"], {"default": "disabled", "tooltip": "Center crops to the target aspect ratio. Disabled stretches when the aspect ratio changes."}),
            "scale_type": (["small", "large", "max"], {"default": "small", "tooltip": "Same as VTS Scale To Min. Large uses both side sizes; small fits within them; max sets the longest side."}),
            "vsr_quality": (["Low", "Medium", "High", "Ultra"], {"default": "Ultra", "tooltip": "RTX VSR quality for all upscaling, before optional neural enhancement."}),
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
