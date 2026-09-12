import io
import json
import os
import shutil
import sys
import tempfile
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
from PIL import Image
from websockets.sync.client import connect

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)
from vtsUtils import DiskImage

CHUNK_BYTES = 256 * 1024


class VTSMerserkFrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Ordered IMAGE batch or VTS DiskImage sequence. At least two frames for interpolation."}),
            "server_url": ("STRING", {"default": "http://192.168.1.1:7865", "tooltip": "Merserk server with the VTS interpolation streaming API installed."}),
            "multiplier": ([2, 3, 4, 8], {"default": 2, "tooltip": "2/4/8 use evenly spaced interpolation. 3x selects 37.5% and 62.5% positions, approximating one-third and two-thirds."}),
            "return_type": (["Input", "Tensor", "DiskImage"], {"default": "Input", "tooltip": "Input preserves the storage type. Tensor holds the final batch in memory. DiskImage writes final frames on this ComfyUI client."}),
        }, "optional": {
            "output_dir": ("STRING", {"default": "", "tooltip": "DiskImage only. Blank uses output/merserk_interpolate. Each run gets a separate folder."}),
            "prefix": ("STRING", {"default": "interpolated"}),
            "start_sequence": ("INT", {"default": 0, "min": 0}),
            "format": (["png", "webp"], {"default": "png", "tooltip": "DiskImage format; both choices are lossless. Network transport is always PNG."}),
            "compression_level": ("INT", {"default": 1, "min": 0, "max": 9, "tooltip": "DiskImage PNG compression; every level is lossless. Network PNGs use fast level 1."}),
            "timeout_seconds": ("INT", {"default": 600, "min": 1, "tooltip": "Maximum wait for server startup or one input frame's results."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "VTS/video"
    DESCRIPTION = "Lossless PNG streaming to Windows Merserk. Sends each source once, receives only intermediate frames, and retains original frames locally. Scene cuts repeat source frames to preserve timing."

    @staticmethod
    def _receive(socket, deadline):
        while True:
            model_management.throw_exception_if_processing_interrupted()
            if time.monotonic() >= deadline:
                raise TimeoutError("Merserk interpolation timed out.")
            try:
                return socket.recv(timeout=min(.25, max(.001, deadline - time.monotonic())))
            except TimeoutError:
                continue

    @classmethod
    def _message(cls, socket, deadline):
        raw = cls._receive(socket, deadline)
        if not isinstance(raw, str):
            raise ValueError("Unexpected binary data in the Merserk interpolation protocol.")
        value = json.loads(raw)
        if value.get("type") == "error":
            raise RuntimeError("Merserk interpolation: " + value.get("message", "unknown error"))
        return value

    @staticmethod
    def _load_frame(images, index, shape):
        frame = images[index].detach().cpu()
        if tuple(frame.shape) != shape:
            raise ValueError("All interpolation frames must have identical dimensions and channels.")
        pixels = frame.float().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        with Image.fromarray(pixels) as image, io.BytesIO() as buffer:
            image.save(buffer, format="PNG", compress_level=1)
            return frame, buffer.getvalue()

    def interpolate(self, images, server_url="http://192.168.1.1:7865", multiplier=2,
                    return_type="Input", output_dir="", prefix="interpolated", start_sequence=0,
                    format="png", compression_level=1, timeout_seconds=600):
        if multiplier not in (2, 3, 4, 8):
            raise ValueError("Multiplier must be 2, 3, 4 or 8.")
        if return_type == "Input":
            return_type = "DiskImage" if isinstance(images, DiskImage) else "Tensor"
        if return_type not in ("Tensor", "DiskImage"):
            raise ValueError("Return type must be Input, Tensor or DiskImage.")
        if len(images.shape) != 4 or images.shape[-1] not in (3, 4) or len(images) < 1:
            raise ValueError("Expected a non-empty RGB/RGBA image sequence in BHWC layout.")
        count, height, width, channels = images.shape
        if count == 1 and ((return_type == "Tensor" and isinstance(images, torch.Tensor)) or
                           (return_type == "DiskImage" and isinstance(images, DiskImage))):
            return (images,)
        shape = (height, width, channels)
        output_count = (count - 1) * multiplier + 1
        saved_directory = None
        success = False
        try:
            if return_type == "DiskImage":
                if format not in ("png", "webp"):
                    raise ValueError("DiskImage format must be png or webp.")
                if not prefix or prefix in (".", "..") or any(c in prefix for c in '/\\:\x00'):
                    raise ValueError("Prefix must be a file name without a directory or drive.")
                if start_sequence < 0:
                    raise ValueError("Start sequence must not be negative.")
                destination = Path(os.path.expandvars(output_dir)).expanduser() if output_dir.strip() else Path(folder_paths.get_output_directory()) / "merserk_interpolate"
                destination.mkdir(parents=True, exist_ok=True)
                saved_directory = Path(tempfile.mkdtemp(prefix="interpolate-", dir=destination.resolve()))
                output = None
            else:
                # Allocate once instead of collecting chunks and doubling peak memory with torch.cat.
                output = torch.empty((output_count, height, width, channels), dtype=torch.float32, device="cpu")
            output_index = 0
            progress = ProgressBar(output_count)

            def emit(frame):
                nonlocal output_index
                model_management.throw_exception_if_processing_interrupted()
                if output_index >= output_count:
                    raise ValueError("Merserk returned too many intermediate frames.")
                if saved_directory is None:
                    output[output_index].copy_(frame)
                else:
                    pixels = frame.float().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                    with Image.fromarray(pixels) as image:
                        options = {"compress_level": compression_level} if format == "png" else {"lossless": True, "exact": True, "method": 1}
                        image.save(saved_directory / f"{prefix}_{start_sequence + output_index:06d}.{format}", **options)
                output_index += 1
                progress.update(1)

            if count == 1:
                emit(images[0].detach().cpu())
            else:
                address = urlsplit(server_url.strip().rstrip('/'))
                if address.scheme not in ("http", "https", "ws", "wss") or not address.netloc or address.query or address.fragment:
                    raise ValueError("Enter the Merserk server URL, for example http://192.168.1.1:7865.")
                url = urlunsplit(("wss" if address.scheme in ("https", "wss") else "ws", address.netloc,
                                  address.path.rstrip('/') + '/vts/interpolate', '', ''))
                with connect(url, open_timeout=min(30, timeout_seconds), close_timeout=2,
                             max_size=CHUNK_BYTES + 1024, max_queue=2, compression=None, proxy=None) as socket, \
                     ThreadPoolExecutor(max_workers=1, thread_name_prefix="vts-interpolation-input") as loader:
                    socket.send(json.dumps(dict(version=1, width=width, height=height, frame_count=count, multiplier=multiplier, timeout_seconds=timeout_seconds)))
                    ready = self._message(socket, time.monotonic() + timeout_seconds)
                    if ready.get("type") != "ready" or ready.get("output_count") != output_count:
                        raise ValueError("Merserk does not support the expected interpolation protocol.")
                    frames = Queue(maxsize=2)
                    credits = threading.Semaphore(2)
                    stopped = threading.Event()

                    def upload():
                        for index in range(count):
                            while not credits.acquire(timeout=.25):
                                if stopped.is_set():
                                    return
                            if stopped.is_set():
                                return
                            frame, png = self._load_frame(images, index, shape)
                            frames.put((index, frame))
                            socket.send(json.dumps(dict(type="frame", index=index, bytes=len(png))))
                            for offset in range(0, len(png), CHUNK_BYTES):
                                if stopped.is_set():
                                    return
                                socket.send(png[offset:offset + CHUNK_BYTES])
                            del png, frame
                        socket.send(json.dumps(dict(type="end")))

                    def guarded_upload():
                        try:
                            upload()
                        except BaseException:
                            socket.close()
                            raise

                    sending = loader.submit(guarded_upload)
                    previous = None
                    try:
                        for index in range(count):
                            deadline = time.monotonic() + timeout_seconds
                            while True:
                                model_management.throw_exception_if_processing_interrupted()
                                if sending.done() and sending.exception() is not None:
                                    raise sending.exception()
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Loading interpolation input timed out.")
                                try:
                                    actual_index, current = frames.get(timeout=.25)
                                    break
                                except Empty:
                                    continue
                            if actual_index != index:
                                raise ValueError("Input frame order changed during streaming.")
                            expected_slot = 1
                            while True:
                                message = self._message(socket, deadline)
                                if message.get("type") == "frame_done":
                                    if message.get("index") != index or expected_slot != (multiplier if index else 1):
                                        raise ValueError("Merserk returned an incomplete or out-of-order frame interval.")
                                    break
                                if not index or message.get("slot") != expected_slot:
                                    raise ValueError("Merserk returned out-of-order intermediate frames.")
                                if message.get("type") == "repeat":
                                    if message.get("source_index") not in (index - 1, index):
                                        raise ValueError("Invalid source-frame reference from Merserk.")
                                    emit(previous if message["source_index"] == index - 1 else current)
                                elif message.get("type") == "generated":
                                    size = message.get("bytes")
                                    if not isinstance(size, int) or not 0 < size <= width * height * 8 + 1024 * 1024:
                                        raise ValueError("Invalid returned PNG size.")
                                    data = bytearray()
                                    while len(data) < size:
                                        chunk = self._receive(socket, deadline)
                                        if not isinstance(chunk, bytes) or not chunk or len(data) + len(chunk) > size:
                                            raise ValueError("Invalid returned PNG chunks.")
                                        data.extend(chunk)
                                    with Image.open(io.BytesIO(data)) as image:
                                        if image.format != "PNG" or image.size != (width, height):
                                            raise ValueError("Merserk returned an invalid interpolated image.")
                                        frame = torch.from_numpy(np.array(image.convert("RGBA" if channels == 4 else "RGB"), dtype=np.float32) / 255)
                                    emit(frame)
                                    del data, frame
                                else:
                                    raise ValueError("Unexpected Merserk interpolation response.")
                                expected_slot += 1
                            emit(current)
                            previous = current
                            credits.release()
                        sending.result()
                        if self._message(socket, time.monotonic() + timeout_seconds).get("type") != "done":
                            raise ValueError("Merserk did not finish the interpolation sequence.")
                    finally:
                        stopped.set()
                        socket.close()
                        sending.cancel()
            if output_index != output_count:
                raise ValueError("Interpolation output frame count does not match the requested multiplier.")
            if saved_directory is not None:
                output = DiskImage(prefix=prefix, start_sequence=start_sequence, number_of_images=output_count,
                                   output_dir=str(saved_directory), format=format, image=None)
                output.shape, output.dtype, output.ndim = (output_count, height, width, channels), torch.float32, 4
            success = True
            return (output,)
        finally:
            if saved_directory is not None and not success:
                shutil.rmtree(saved_directory)


NODE_CLASS_MAPPINGS = {"VTS Merserk Frame Interpolate": VTSMerserkFrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS Merserk Frame Interpolate": "VTS Merserk Frame Interpolate"}
