

import logging
from spandrel import ModelLoader, ImageModelDescriptor
from spandrel.__helpers.size_req import pad_tensor
from comfy import model_management
import torch
import comfy.utils
import folder_paths


# class UpscaleModelLoader:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"), ),
#                              }}
#     RETURN_TYPES = ("UPSCALE_MODEL",)
#     FUNCTION = "load_model"

#     CATEGORY = "loaders"

#     def load_model(self, model_name):
#         model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
#         sd = comfy.utils.load_torch_file(model_path, safe_load=True)
#         if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
#             sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
#         out = ModelLoader().load_from_state_dict(sd).eval()

#         if not isinstance(out, ImageModelDescriptor):
#             raise Exception("Upscale model must be a single-image model.")

#         return (out, )
    

# class CustomImageModelDescriptor:
#     def __init__(self, original_descriptor: ImageModelDescriptor, scaleOverride: float = None):
#         self._original_descriptor = original_descriptor
#         if scaleOverride is not None and scaleOverride > 0:
#             self.scale = scaleOverride

#     def __getattr__(self, name):
#         """
#         Delegate attribute access to the original descriptor.
#         """
#         return getattr(self._original_descriptor, name)
    


#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Custom implementation of the __call__ method.
#         """
#         if len(image.shape) != 4:
#             raise ValueError(
#                 f"Expected image tensor to have 4 dimensions, but got {image.shape}"
#             )

#         _, _, h, w = image.shape

#         logging.info(f"CustomImageModelDescriptor - Processing image of size {w}x{h} with scale {self.scale}")


#         # Example: Custom logic for padding
#         did_pad, image = pad_tensor(image, self.size_requirements)

#         # Example: Custom inference logic
#         if self.model.training:
#             self.model.eval()

#         output = self.model(image)  # Replace with your custom logic
#         assert isinstance(
#             output, torch.Tensor
#         ), f"Expected {type(self.model).__name__} model to return a tensor, but got {type(output)}"

#         # Guarantee range
#         output = output.clamp_(0, 1)

#         # Remove padding
#         if did_pad:
#             newWidth = int(w * self.scale)
#             newHeight = int(h * self.scale)
#             output = output[..., : newHeight, : newWidth]

#         logging.info(f"CustomImageModelDescriptor - did_pad= {did_pad}, output shape = {output.shape}")

#         return output
    

class VTSImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "device_preference": (["auto", "cuda", "cpu"],),  # New input type
            },
            # "optional": {
            #     "override_upscale_amount": ("FLOAT",),  # Optional parameter to override upscale amount
            # },
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
