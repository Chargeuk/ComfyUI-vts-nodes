

import logging
from comfy import model_management
import torch
import torch.nn.functional as F

def gaussian_kernel(kernel_size: int, sigma: float, device=None):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device), torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()

# class VTSImageUpscaleWithModel:
class VTSSharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "batch_size": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "image/postprocessing"

    def sharpen(self, image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float, batch_size: int, passthrough: bool = False) -> tuple:
        if passthrough:
            logging.info("VTSSharpen - passthrough is True, returning original image without processing")
            return (image,)
        
        if sharpen_radius == 0:
            return (image,)

        total_batch_size, height, width, channels = image.shape
        logging.info(f"VTSSharpen - total_batch_size: {total_batch_size}, height: {height}, width: {width}, channels: {channels}, sharpen_radius: {sharpen_radius}, sigma: {sigma}, alpha: {alpha}, batch_size: {batch_size}")
        
        # Process images in batches to avoid memory issues
        # Use in-place processing for maximum memory efficiency
        if image.device != model_management.intermediate_device():
            final_result = image.clone()
            logging.info("VTSSharpen - Input images cloned to intermediate device for processing")
        else:
            final_result = image
            logging.info("VTSSharpen - Processing input images in-place (no clone needed)")
        
        for i in range(0, total_batch_size, batch_size):
            logging.info(f"Processing batch {i // batch_size + 1} of {total_batch_size // batch_size + 1}")
            # Get current batch
            end_idx = min(i + batch_size, total_batch_size)
            batch_images = final_result[i:end_idx]
            
            # Move batch to GPU
            logging.info(f"Moving batch images to device: {model_management.get_torch_device()}")
            batch_images = batch_images.to(model_management.get_torch_device())

            kernel_size = sharpen_radius * 2 + 1
            kernel = gaussian_kernel(kernel_size, sigma, device=batch_images.device) * -(alpha*10)
            center = kernel_size // 2
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
            kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

            logging.info(f"sharpening batch with kernel size {kernel_size}, sigma {sigma}, alpha {alpha}")
            tensor_image = batch_images.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
            tensor_image = F.pad(tensor_image, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
            sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
            sharpened = sharpened.permute(0, 2, 3, 1)

            batch_result = torch.clamp(sharpened, 0, 1)
            
            # Write directly back to the result tensor
            logging.info(f"Writing batch result directly to final tensor")
            final_result[i:end_idx] = batch_result.to(final_result.device)
            
            # Clear GPU memory for this batch
            logging.info("Clearing GPU memory for this batch")
            del batch_images, tensor_image, sharpened, batch_result
            if hasattr(torch.cuda, 'empty_cache'):
                logging.info("Emptying CUDA cache")
                torch.cuda.empty_cache()
        
        return (final_result,)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Sharpen": VTSSharpen
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Sharpen": "Image Sharpen VTS"
}
