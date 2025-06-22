

import logging
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor
from comfy import model_management
import torch
import comfy.utils
import folder_paths 

class VTSImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "device_preference": (["auto", "cuda", "cpu"],),  # New input type
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(
        self,
        upscale_model: ImageModelDescriptor,
        image: torch.Tensor,
        device_preference: str = "auto",
        # override_upscale_amount: float = None,  # Optional parameter
    ) -> tuple:
        # Use override_upscale_amount if provided, otherwise use the model's scale
        # upscale_amount = override_upscale_amount if override_upscale_amount else upscale_model.scale
        upscale_amount = upscale_model.scale
        # # Wrap the upscale_model in CustomImageModelDescriptor
        # upscale_model = CustomImageModelDescriptor(upscale_model, upscale_amount)

        # Determine the device based on device_preference
        if device_preference == "auto":
            device = model_management.get_torch_device()
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = model_management.get_torch_device()  # Fallback to auto
        elif device_preference == "cpu":
            device = "cpu"
        else:
            raise ValueError(f"Invalid device preference: {device_preference}")

        logging.info(f"VTSImageUpscaleWithModel - upscale_model.scale = {upscale_model.scale}, size_requirements= {upscale_model.size_requirements}, device = {device}")

        # Memory management
        if device != "cpu":
            memory_required = model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_amount, 1.0) * 384.0  # Estimate
            memory_required += image.nelement() * image.element_size()
            model_management.free_memory(memory_required, device)

        # Move model to the selected device
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                # Calculate the number of steps for tiling
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)

                # Perform tiled scaling
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_amount,  # Pass the correct upscale amount
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # Move model back to CPU and clamp output
        if upscale_model.device != "cpu":
            upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Image Upscale With Model": VTSImageUpscaleWithModel
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image Upscale With Model": "Image Upscale With Model"
}
