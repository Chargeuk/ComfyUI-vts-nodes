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
from gradio_client import Client, handle_file

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)
from vtsUtils import DiskImage


class VTSMerserkEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        strength = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Accepts a regular IMAGE batch or a VTS DiskImage."}),
            "server_url": ("STRING", {"default": "http://192.168.1.1:7865", "tooltip": "Windows Merserk server with the VTS image API installed."}),
            "return_type": (["Tensor", "DiskImage"], {"default": "Tensor", "tooltip": "Tensor is a normal ComfyUI IMAGE. DiskImage keeps the returned frames on disk."}),
            "upscaling_factor": (["1", "1.5", "1.724", "2", "3"], {"default": "1"}),
            "iterations": ("INT", {"default": 1, "min": 1, "tooltip": "Total passes. Only the first upscales; subsequent passes enhance the previous output at the same size."}),
            "nr_preset": (["Default", "Preset #1", "Preset #2", "Preset #3"],),
            "nr_style": (["Default", "Natural", "Cinematic"],),
            "nr_intensity": ("FLOAT", dict(strength)),
            "local_tone_strength": ("FLOAT", dict(strength)),
            "local_structure_strength": ("FLOAT", dict(strength)),
            "skin_structure_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "automatic_mask": ("BOOLEAN", {"default": False}),
            "dlss_model_preset": (["Default", "J", "K", "L", "M"],),
            "output_dir": ("STRING", {"default": "", "tooltip": "DiskImage only. Blank uses ComfyUI's output/merserk folder. Each run creates its own subfolder of lossless PNGs."}),
            "timeout_seconds": ("INT", {"default": 600, "min": 1, "tooltip": "Maximum time to wait for each image."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "VTS/image"
    DESCRIPTION = "Render images on a Windows Merserk server. Supports VTS DiskImage input/output and repeated neural enhancement."

    def enhance(self, image, server_url="http://192.168.1.1:7865", return_type="Tensor",
                upscaling_factor="1", iterations=1, nr_preset="Default", nr_style="Default",
                nr_intensity=1.0, local_tone_strength=1.0, local_structure_strength=1.0,
                skin_structure_strength=-1.0, automatic_mask=False, dlss_model_preset="Default",
                output_dir="", timeout_seconds=600):
        server_url = server_url.strip().rstrip("/")
        address = urlsplit(server_url)
        if address.scheme not in {"http", "https"} or not address.netloc or address.query or address.fragment:
            raise ValueError("Enter the Merserk server URL, for example http://127.0.0.1:7865.")
        if return_type not in {"Tensor", "DiskImage"}:
            raise ValueError("Return type must be Tensor or DiskImage.")
        count = len(image)
        if count == 0:
            raise ValueError("Merserk needs at least one input image.")
        parameters = json.dumps(dict(
            upscaling_factor=float(upscaling_factor), iterations=iterations,
            nr_preset=nr_preset, nr_style=nr_style, nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength, local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength, automatic_mask=automatic_mask,
            dlss_model_preset=dlss_model_preset,
        ))
        saved_directory = None
        complete = False
        try:
            if return_type == "DiskImage":
                destination = Path(os.path.expandvars(output_dir)).expanduser() if output_dir.strip() else Path(folder_paths.get_output_directory()) / "merserk"
                destination.mkdir(parents=True, exist_ok=True)
                saved_directory = Path(tempfile.mkdtemp(prefix="render-", dir=destination.resolve()))
            frames = []
            progress = ProgressBar(count)
            with tempfile.TemporaryDirectory(prefix="vts-merserk-") as temporary:
                client = Client(server_url, verbose=False, download_files=temporary,
                                analytics_enabled=False, httpx_kwargs={"timeout": timeout_seconds})
                try:
                    for index in range(count):
                        model_management.throw_exception_if_processing_interrupted()
                        frame = image[index]
                        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                            raise ValueError("Merserk expects RGB or RGBA images in BHWC layout.")
                        mode = "RGBA" if frame.shape[-1] == 4 else "RGB"
                        pixels = frame.detach().cpu().float().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                        source = Path(temporary) / "input.png"
                        Image.fromarray(pixels).save(source)
                        del pixels, frame
                        request_id = uuid.uuid4().hex
                        job = client.submit(handle_file(str(source)), parameters, request_id, api_name="/vts_enhance")
                        deadline = time.monotonic() + timeout_seconds
                        try:
                            while True:
                                model_management.throw_exception_if_processing_interrupted()
                                if time.monotonic() >= deadline:
                                    raise TimeoutError(f"Merserk exceeded {timeout_seconds} seconds for image {index + 1}.")
                                try:
                                    returned_path = job.result(timeout=.25)
                                    break
                                except TimeoutError:
                                    continue
                        except BaseException:
                            # Cancel only this node's request, never a render started in the GUI.
                            job.cancel()
                            with suppress(Exception):
                                client.submit(request_id, api_name="/vts_cancel").result(timeout=10)
                            raise
                        model_management.throw_exception_if_processing_interrupted()
                        with Image.open(returned_path) as returned:
                            enhanced = returned.convert(mode)
                            shape = (count, enhanced.height, enhanced.width, len(mode))
                            if index and shape != output_shape:
                                raise ValueError("All returned images must have the same dimensions for an IMAGE batch.")
                            output_shape = shape
                            if saved_directory is not None:
                                enhanced.save(saved_directory / f"image_{index:06d}.png")
                            else:
                                frames.append(torch.from_numpy(np.array(enhanced, dtype=np.float32) / 255.0))
                        Path(returned_path).unlink()
                        progress.update(1)
                finally:
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
