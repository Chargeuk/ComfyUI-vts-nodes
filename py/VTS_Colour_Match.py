import comfy
import comfy.utils
import node_helpers
import torch
from nodes import MAX_RESOLUTION
from comfy import model_management
import os
import sys

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, save_images, vtsImageTypes, vtsReturnTypes, default_output_dir

def colormatch(image_ref, image_target, method, strength=1.0, editInPlace=False, gc_interval=50):
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    
    # Early validation
    if image_ref.dim() != 4 or image_target.dim() != 4:
        raise ValueError("ColorMatch: Expected 4D tensors (batch, height, width, channels)")
    
    batch_size = image_target.size(0)
    ref_batch_size = image_ref.size(0)
    
    # Validate batch sizes early
    if ref_batch_size > 1 and ref_batch_size != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")
    
    # Move to CPU efficiently (avoid redundant moves)
    if image_ref.device != torch.device('cpu'):
        image_ref = image_ref.cpu()
    if image_target.device != torch.device('cpu'):
        image_target = image_target.cpu()
    
    # Handle output tensor allocation
    if editInPlace:
        out = image_target
    else:
        out = torch.empty_like(image_target, dtype=torch.float32, device='cpu')
    
    # Initialize ColorMatcher once
    cm = ColorMatcher()
    
    # Process each image in the batch
    for i in range(batch_size):
        # Get individual images (avoid squeeze - use direct indexing)
        target_img = image_target[i]  # Shape: [H, W, C]
        ref_img = image_ref[0] if ref_batch_size == 1 else image_ref[i]  # Shape: [H, W, C]
        
        # Convert to numpy only when needed
        target_np = target_img.numpy()
        ref_np = ref_img.numpy()
        
        try:
            # Perform color matching
            result_np = cm.transfer(src=target_np, ref=ref_np, method=method)
            
            # Apply strength multiplier efficiently
            if strength != 1.0:
                result_np = target_np + strength * (result_np - target_np)
            
            # Convert back to tensor and update output
            result_tensor = torch.from_numpy(result_np)
            
            if editInPlace:
                image_target[i].copy_(result_tensor)
            else:
                out[i].copy_(result_tensor)
            
            # Clean up intermediate variables
            del target_np, ref_np, result_np, result_tensor
            
            # Garbage collection at intervals
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error occurred during transfer for image {i}: {e}")
            # Continue processing other images rather than breaking
            continue
    
    # Ensure output is float32 and properly clamped
    if not editInPlace and out.dtype != torch.float32:
        out = out.to(torch.float32)
    out.clamp_(0, 1)
    
    return (out,)


class VTS_ColourMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "When true, bypass processing and return images unchanged"}),
                "return_type": (vtsReturnTypes, {"default": "Input", "tooltip": "Output as the same type as the target input, force DiskImage output, or force Tensor output."}),
                "batch_size": ("INT", {"default": 20, "min": 1}),
                "prefix": ("STRING", {"default": "color_match", "multiline": False}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "editInPlace": ("BOOLEAN", {"default": False, "tooltip": "When true, modify the input image_target tensor directly instead of creating a new tensor"}),
                "gc_interval": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1, "tooltip": "Garbage collection interval. Set to 0 to disable automatic garbage collection. For large batches, lower values can help manage memory"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""

    CATEGORY = "VTS"

    def _is_tensor(self, image):
        return isinstance(image, torch.Tensor)

    def _image_count(self, image):
        return image.shape[0] if self._is_tensor(image) else image.number_of_images

    def _image_shape(self, image):
        if self._is_tensor(image):
            return tuple(image.shape)
        return image.shape

    def _load_batch(self, image, batch_start, batch_count):
        if self._is_tensor(image):
            batch_end = min(batch_start + batch_count, image.shape[0])
            return image[batch_start:batch_end]
        return image.load_images(start_sequence=image.start_sequence + batch_start, count=batch_count)

    def _resolve_return_type(self, return_type, image_target):
        if return_type == "Input":
            return "Tensor" if self._is_tensor(image_target) else "DiskImage"
        if return_type == "Input or DiskImage":
            return "DiskImage"
        return return_type

    def _resolve_output_config(self, image_target, return_type, editInPlace, prefix, start_sequence, output_dir, format, compression_level, quality):
        if quality > 100:
            quality = None

        if not self._is_tensor(image_target) and (return_type == "Input" or return_type == "Input or DiskImage" or editInPlace):
            prefix = image_target.prefix
            start_sequence = image_target.start_sequence
            output_dir = image_target.output_dir
            format = image_target.format
            compression_level = image_target.compression_level
            quality = image_target.quality

        return {
            "prefix": prefix,
            "start_sequence": start_sequence,
            "output_dir": output_dir,
            "format": format,
            "compression_level": compression_level,
            "quality": quality,
        }

    def colormatch(self, image_ref, image_target, method, passthrough, return_type="Input", batch_size=20, prefix="color_match", start_sequence=0, output_dir=default_output_dir, format="png", num_workers=16, compression_level=9, quality=95, strength=1.0, editInPlace=False, gc_interval=50):
        target_count = self._image_count(image_target)
        ref_count = self._image_count(image_ref)

        if ref_count > 1 and ref_count != target_count:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        resolved_return_type = self._resolve_return_type(return_type, image_target)
        output_config = self._resolve_output_config(
            image_target,
            return_type,
            editInPlace,
            prefix,
            start_sequence,
            output_dir,
            format,
            compression_level,
            quality,
        )

        print(
            f"VTS_ColourMatch - processing images. method: {method}, strength: {strength}, "
            f"editInPlace: {editInPlace}, gc_interval: {gc_interval}, batch_size: {batch_size}, "
            f"return_type: {return_type} -> {resolved_return_type}, passthrough: {passthrough}"
        )

        ref_single = None
        if ref_count == 1:
            ref_single = self._load_batch(image_ref, 0, 1)

        tensor_batches = []

        for batch_start in range(0, target_count, batch_size):
            batch_count = min(batch_size, target_count - batch_start)
            target_batch = self._load_batch(image_target, batch_start, batch_count)

            if ref_single is not None:
                ref_batch = ref_single
            else:
                ref_batch = self._load_batch(image_ref, batch_start, batch_count)

            if passthrough:
                processed_batch = target_batch
            else:
                processed_batch = colormatch(ref_batch, target_batch, method, strength, False, gc_interval)[0]

            if resolved_return_type == "Tensor":
                if self._is_tensor(image_target) and editInPlace:
                    batch_end = batch_start + batch_count
                    image_target[batch_start:batch_end] = processed_batch.to(
                        device=image_target.device,
                        dtype=image_target.dtype,
                    )
                else:
                    tensor_batches.append(processed_batch.cpu())
            else:
                save_images(
                    image=processed_batch,
                    prefix=output_config["prefix"],
                    start_sequence=output_config["start_sequence"] + batch_start,
                    output_dir=output_config["output_dir"],
                    format=output_config["format"],
                    num_workers=num_workers,
                    compression_level=output_config["compression_level"],
                    quality=output_config["quality"],
                )
                del processed_batch
                model_management.soft_empty_cache()

            del target_batch
            if ref_single is None:
                del ref_batch

        if ref_single is not None:
            del ref_single

        if resolved_return_type == "Tensor":
            if self._is_tensor(image_target) and editInPlace:
                print("VTS_ColourMatch - finished processing images in-place")
                return (image_target,)

            if len(tensor_batches) == 0:
                raise RuntimeError("VTS_ColourMatch - no output batches were produced")

            result = torch.cat(tensor_batches, dim=0)
            print("VTS_ColourMatch - finished processing images and returning tensor output")
            return (result,)

        result = DiskImage(
            prefix=output_config["prefix"],
            start_sequence=output_config["start_sequence"],
            number_of_images=target_count,
            output_dir=output_config["output_dir"],
            format=output_config["format"],
            image=None,
            compression_level=output_config["compression_level"],
            quality=output_config["quality"],
        )
        result.shape = self._image_shape(image_target)
        result.dtype = torch.float32
        result.ndim = len(result.shape) if result.shape is not None else 4

        print("VTS_ColourMatch - finished processing images and returning DiskImage output")
        return (result,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Colour Match": VTS_ColourMatch
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Colour Match": "VTS Colour Match"
}
