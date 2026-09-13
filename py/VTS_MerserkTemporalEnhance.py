"""Ordered lossless image streaming through Merserk temporal enhancement."""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
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
from vts_image_sizing import ScaleToMinDimensions

CHUNK_BYTES = 256 * 1024


class MerserkWorkerRestarted(RuntimeError):
    pass


class VTSMerserkTemporalEnhance(ScaleToMinDimensions):
    @classmethod
    def INPUT_TYPES(cls):
        strength = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Ordered RGB/RGBA IMAGE batch or VTS DiskImage sequence with matching dimensions. Merserk preserves neural history between frames and resets it at scene cuts. Frame count/order are unchanged. Transport preserves 8-bit pixels, not HDR or arbitrary floating-point values."}),
            "server_url": ("STRING", {"default": "http://192.168.1.1:7865", "tooltip": "Address of your Windows Merserk server, for example http://192.168.1.1:7865. Required for neural enhancement or enlargement; local downscaling and bypass work without it."}),
            "return_type": (["Input", "Tensor", "DiskImage"], {"default": "Input", "tooltip": "Input keeps the input storage type. Tensor returns a normal IMAGE batch held in RAM. DiskImage saves only the final images on this ComfyUI machine and returns their file references, which is more memory-efficient for large batches."}),
            "upscaling_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16384.0, "step": 0.05, "tooltip": "Used only when scaling is enabled and sizing_mode is Multiplier. Multiplies width and height: 2 doubles each side, 1 keeps the size, and 0.5 halves each side. Dimensions are rounded to even pixels. Enlargement uses RTX VSR; reduction uses Lanczos."}),
            "nr_passes": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Native Merserk neural passes per frame. More passes can strengthen enhancement and cost more processing time; more is not always better. Temporal history is retained across successive frames. There is no outer iteration loop."}),
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
            "shimmer_suppression": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Stabilize model-created detail using motion between successive frames. 0 disables the extra shimmer filter; 1 requests strongest stabilization. Higher values can reduce flicker but may smear detail if motion is estimated poorly. Neural history still uses scene-cut resets. Ignored when neural rendering is disabled."}),
            "output_dir": ("STRING", {"default": "", "tooltip": "Used only for DiskImage output. Folder on this ComfyUI machine, not the Merserk server. Blank uses ComfyUI output/merserk_temporal. Each run creates its own subfolder containing only the final lossless PNG images."}),
            "timeout_seconds": ("INT", {"default": 600, "min": 1, "tooltip": "Maximum wait for a server response or one enhanced frame. Queue status messages keep the readiness wait alive while other jobs finish; the limit still applies during rendering. Increase for large frames or high NR Passes. A timeout closes this stream and cancels its GPU job. Connection setup is capped at 30 seconds."}),
        }, "optional": {
            "enable_scaling": ("BOOLEAN", {"default": True, "tooltip": "Enable resizing using the sizing controls below. Enlargement uses RTX VSR once; reduction uses local Lanczos before enhancement. Off preserves the input dimensions and ignores sizing/crop controls. Neural enhancement can still run."}),
            "enable_neural_rendering": ("BOOLEAN", {"default": True, "tooltip": "Apply temporally consistent Neuroframe enhancement after any resizing. Off skips all neural controls and NR Passes, but scaling can still run. Turn both this and Enable Scaling off to pass images through, converting storage type only if requested."}),
            "sizing_mode": (["Scale to Min", "Multiplier"], {"default": "Scale to Min", "tooltip": "Scale to Min uses the two side sizes, scale_type and divisible_by rules from VTS Scale To Min. Multiplier uses upscaling_factor to resize both sides proportionally. Used only when Enable Scaling is on."}),
            "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1, "tooltip": "Smaller side target/limit in pixels for Scale to Min mode. Together with largeMaxSize and scale_type it determines output dimensions. The two values are sorted automatically, so entering them backwards is fine. For a 16:9 landscape image, 720 and 1280 typically give 1280 x 720."}),
            "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1, "tooltip": "Larger side target/limit in pixels for Scale to Min mode. The two side values are sorted automatically, so this may be entered smaller than smallMaxSize. Its exact role depends on scale_type; max uses this larger value as the longest output side."}),
            "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1, "tooltip": "Scale to Min only: round each calculated dimension down to a multiple of this number. For example, 1001 becomes 1000 with 8. Use 0 or 1 to disable rounding. Rounding can slightly change aspect ratio; choose sizes large enough that neither side becomes zero."}),
            "crop": (["disabled", "center"], {"default": "disabled", "tooltip": "When scaling, center removes image edges to match the target aspect ratio before resizing. Disabled keeps all content but stretches it if the target aspect ratio differs. For a wide image resized to a square, center cuts off the left/right edges; disabled squeezes the width. No borders are added."}),
            "scale_type": (["small", "large", "max"], {"default": "small", "tooltip": "Scale to Min only. small fits the image within the short/long side limits, largely preserving aspect ratio. large uses both side sizes as the output dimensions, which may change aspect ratio. max sets the longest side to the larger limit and calculates the other proportionally. Divisibility rounding applies afterwards."}),
            "vsr_quality": (["Low", "Medium", "High", "Ultra"], {"default": "Ultra", "tooltip": "RTX VSR quality for enlargement, whether or not neural enhancement follows. Low is lighter; Ultra requests the highest quality and may take longer. This changes upscaling quality, not output dimensions or neural pass counts. Ignored when no enlargement is needed."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "enhance"
    CATEGORY = "VTS/video"
    DESCRIPTION = "Temporal neural enhancement of ordered images with optional RTX VSR scaling. Persistent server sessions, lossless PNG transport, unchanged frame count, and bounded streaming buffers. No outer iterations, video files or HDR."

    @staticmethod
    def _receive(socket, deadline):
        while True:
            model_management.throw_exception_if_processing_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Merserk temporal enhancement timed out.")
            try:
                return socket.recv(timeout=min(.25, remaining))
            except TimeoutError:
                continue

    @classmethod
    def _message(cls, socket, deadline):
        raw = cls._receive(socket, deadline)
        if not isinstance(raw, str):
            raise ValueError("Expected a Merserk sequence message.")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Invalid Merserk sequence message.")
        if value.get("type") == "error":
            if value.get("code") == "NEURAL_WORKER_RESTARTED":
                raise MerserkWorkerRestarted(value.get("message", "Neural worker restarted"))
            raise RuntimeError("Merserk: " + value.get("message", "sequence failed"))
        return value

    @classmethod
    def _ready(cls, socket, timeout_seconds):
        # Queue heartbeats extend the readiness wait, never a rendering timeout.
        while True:
            message = cls._message(socket, time.monotonic() + timeout_seconds)
            if message.get("type") != "queued":
                return message
            position = message.get("position")
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                raise ValueError("Invalid Merserk queue status.")

    def enhance(self, *args, **kwargs):
        for attempt in range(2):
            try:
                return self._enhance_once(*args, **kwargs)
            except MerserkWorkerRestarted:
                if attempt:
                    raise
                model_management.throw_exception_if_processing_interrupted()
                # The failed attempt cleans its own partial output. Replay every
                # original frame so VSR, scene detection and NR history agree.

    def _enhance_once(self, images, server_url="http://192.168.1.1:7865", return_type="Input",
                upscaling_factor=1.0, nr_passes=1, nr_style="Default", nr_intensity=1.0,
                local_tone_strength=1.0, local_structure_strength=1.0, skin_structure_strength=-1.0,
                automatic_mask=False, nr_color_strength=1.0, tone_preservation=0.0,
                face_skin_protection=0.0, grain_preservation=0.0, shimmer_suppression=.7,
                output_dir="", timeout_seconds=600, enable_scaling=True, enable_neural_rendering=True,
                sizing_mode="Scale to Min", smallMaxSize=512, largeMaxSize=512, divisible_by=2,
                crop="disabled", scale_type="small", vsr_quality="Ultra"):
        if return_type == "Input":
            return_type = "DiskImage" if isinstance(images, DiskImage) else "Tensor"
        if return_type not in {"Tensor", "DiskImage"}:
            raise ValueError("Return type must be Input, Tensor or DiskImage.")
        if len(images.shape) != 4 or images.shape[-1] not in (3, 4) or len(images) < 1:
            raise ValueError("Expected a non-empty, equally sized RGB/RGBA image sequence.")
        count, height, width, channels = images.shape
        original_shape = (height, width, channels)
        tw, th = width, height
        if enable_scaling:
            if sizing_mode == "Scale to Min":
                tw, th = self._calculate_target_dimensions(width, height, smallMaxSize, largeMaxSize, divisible_by, scale_type)
            elif sizing_mode == "Multiplier":
                factor = float(upscaling_factor)
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Size multiplier must be positive and finite.")
                tw, th = max(2, int(width * factor / 2 + .5) * 2), max(2, int(height * factor / 2 + .5) * 2)
            else:
                raise ValueError("Unknown sizing mode.")
            if min(tw, th) < 1:
                raise ValueError("Sizing produced a zero dimension. Increase the side sizes or reduce divisible_by.")
            if crop not in {"disabled", "center"}:
                raise ValueError("Crop must be disabled or center.")
        if not enable_neural_rendering and (tw, th) == (width, height):
            if (return_type == "Tensor" and isinstance(images, torch.Tensor)) or (return_type == "DiskImage" and isinstance(images, DiskImage)):
                return (images,)
        first = images[0]
        sample = self._center_crop_for_aspect(first.unsqueeze(0), tw, th)[0] if enable_scaling and crop == "center" else first
        ch, cw = sample.shape[:2]
        upload_size = (min(cw, tw), min(ch, th))
        del sample
        remote = enable_neural_rendering or upload_size != (tw, th)
        parameters = dict(nr_passes=nr_passes, nr_style=nr_style, nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength, local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength, automatic_mask=automatic_mask,
            nr_color_strength=nr_color_strength, tone_preservation=tone_preservation,
            face_skin_protection=face_skin_protection, grain_preservation=grain_preservation,
            shimmer_suppression=shimmer_suppression) if enable_neural_rendering else {}
        if remote:
            if max(*upload_size, tw, th) > 16384 or max(upload_size[0] * upload_size[1], tw * th) > 100_000_000:
                raise ValueError("Requested frames exceed the server's size limit.")
            if enable_neural_rendering and (min(tw, th) < 64 or max(tw, th) > 7680 or min(tw, th) > 4320):
                raise ValueError("Neural output must be at least 64 per side and fit within 7680 by 4320 (either orientation).")
        saved_directory = None
        success = False
        try:
            if return_type == "DiskImage":
                destination = Path(os.path.expandvars(output_dir)).expanduser() if output_dir.strip() else Path(folder_paths.get_output_directory()) / "merserk_temporal"
                destination.mkdir(parents=True, exist_ok=True)
                saved_directory = Path(tempfile.mkdtemp(prefix="sequence-", dir=destination.resolve()))
                output = None
            else:
                output = torch.empty((count, th, tw, channels), dtype=torch.float32, device="cpu")
            progress = ProgressBar(count)

            def prepare(index):
                model_management.throw_exception_if_processing_interrupted()
                frame = (first if index == 0 else images[index]).detach().cpu()
                if tuple(frame.shape) != original_shape:
                    raise ValueError("All frames must have identical dimensions and channels.")
                if enable_scaling and crop == "center":
                    frame = self._center_crop_for_aspect(frame.unsqueeze(0), tw, th)[0]
                pixels = frame.float().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                image = Image.fromarray(pixels)
                if image.size != upload_size:
                    reduced = image.resize(upload_size, Image.Resampling.LANCZOS)
                    image.close()
                    image = reduced
                return image

            def emit(index, image):
                model_management.throw_exception_if_processing_interrupted()
                if image.size != (tw, th):
                    raise ValueError("Merserk returned incorrect frame dimensions.")
                if saved_directory is not None:
                    image.save(saved_directory / f"frame_{index:06d}.png", compress_level=1)
                else:
                    output[index].copy_(torch.from_numpy(np.array(image, dtype=np.float32) / 255))
                progress.update(1)

            if not remote:
                for index in range(count):
                    with prepare(index) as image:
                        emit(index, image)
            else:
                address = urlsplit(server_url.strip().rstrip('/'))
                if address.scheme not in ('http', 'https', 'ws', 'wss') or not address.netloc or address.query or address.fragment:
                    raise ValueError("Enter the Merserk server URL, for example http://192.168.1.1:7865.")
                url = urlunsplit(('wss' if address.scheme in ('https','wss') else 'ws', address.netloc,
                    address.path.rstrip('/') + '/vts/enhance_sequence', '', ''))
                with connect(url, open_timeout=min(30, timeout_seconds), close_timeout=2,
                             max_size=CHUNK_BYTES + 1024, max_queue=2, compression=None, proxy=None) as socket, \
                     ThreadPoolExecutor(max_workers=1, thread_name_prefix='vts-temporal-upload') as loader:
                    setup = dict(version=1, queue_status=True, width=upload_size[0], height=upload_size[1], target_width=tw,
                        target_height=th, channels=channels, frame_count=count, enable_neural_rendering=enable_neural_rendering,
                        vsr_quality={'Low':1,'Medium':2,'High':3,'Ultra':4}[vsr_quality],
                        parameters=parameters, timeout_seconds=timeout_seconds)
                    socket.send(json.dumps(setup))
                    ready = self._ready(socket, timeout_seconds)
                    if any(ready.get(k) != v for k,v in dict(type='ready',version=1,output_count=count,
                            width=tw,height=th,channels=channels,chunk_bytes=CHUNK_BYTES).items()):
                        raise ValueError("Merserk does not support the expected temporal enhancement protocol.")
                    credits = threading.Semaphore(2)
                    stopped = threading.Event()

                    def upload():
                        try:
                            for index in range(count):
                                while not credits.acquire(timeout=.25):
                                    if stopped.is_set(): return
                                if stopped.is_set(): return
                                with prepare(index) as image, io.BytesIO() as buffer:
                                    image.save(buffer, format='PNG', compress_level=1)
                                    png = buffer.getvalue()
                                socket.send(json.dumps(dict(type='frame',index=index,bytes=len(png))))
                                for offset in range(0,len(png),CHUNK_BYTES):
                                    if stopped.is_set(): return
                                    socket.send(png[offset:offset+CHUNK_BYTES])
                                del png
                            socket.send(json.dumps(dict(type='end')))
                        except BaseException:
                            socket.close()
                            raise

                    sending = loader.submit(upload)
                    try:
                        for index in range(count):
                            deadline = time.monotonic() + timeout_seconds
                            if sending.done() and sending.exception() is not None:
                                raise sending.exception()
                            header = self._message(socket, deadline)
                            if header.get('type') != 'enhanced' or header.get('index') != index:
                                raise ValueError("Merserk returned frames out of order.")
                            size = header.get('bytes')
                            if isinstance(size,bool) or not isinstance(size,int) or not 0 < size <= tw * th * 8 + 1024 * 1024:
                                raise ValueError("Invalid returned PNG length.")
                            data = bytearray()
                            while len(data) < size:
                                chunk = self._receive(socket,deadline)
                                if not isinstance(chunk,bytes) or not chunk or len(chunk)>CHUNK_BYTES or len(data)+len(chunk)>size:
                                    raise ValueError("Invalid returned PNG chunk.")
                                data.extend(chunk)
                            with Image.open(io.BytesIO(data)) as image:
                                if image.format != 'PNG' or image.mode != ('RGBA' if channels==4 else 'RGB'):
                                    raise ValueError("Expected an 8-bit PNG with matching channels.")
                                emit(index,image)
                            del data
                            credits.release()
                        sending.result()
                        done = self._message(socket,time.monotonic()+timeout_seconds)
                        if done.get('type') != 'done' or done.get('stats',{}).get('frames') != count:
                            raise ValueError("Merserk did not complete the full sequence.")
                    finally:
                        stopped.set()
                        socket.close()
                        sending.cancel()
            if saved_directory is not None:
                output = DiskImage(prefix='frame',start_sequence=0,number_of_images=count,
                    output_dir=str(saved_directory),format='png',image=None)
                output.shape, output.dtype, output.ndim = (count,th,tw,channels),torch.float32,4
            success = True
            return (output,)
        finally:
            if saved_directory is not None and not success:
                shutil.rmtree(saved_directory)


NODE_CLASS_MAPPINGS = {"VTS Merserk Temporal Enhance": VTSMerserkTemporalEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS Merserk Temporal Enhance": "VTS Merserk Temporal Enhance"}
