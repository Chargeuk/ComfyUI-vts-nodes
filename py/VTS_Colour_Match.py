import comfy
import comfy.utils
import node_helpers
import torch
from nodes import MAX_RESOLUTION

def colormatch(image_ref, image_target, method, strength=1.0, editInPlace=False, gc_interval=50):
    try:
        from color_matcher import ColorMatcher
    except:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    
    cm = ColorMatcher()
    
    # Move to CPU only once and reuse
    if image_ref.device != torch.device('cpu'):
        image_ref = image_ref.cpu()
    if image_target.device != torch.device('cpu'):
        image_target = image_target.cpu()
    
    batch_size = image_target.size(0)
    
    # Handle output tensor allocation
    if editInPlace:
        out = image_target
    else:
        # Use the same dtype as input to avoid unnecessary conversions
        out = torch.empty_like(image_target, dtype=image_target.dtype, device='cpu')
    
    # Re-introduce squeeze logic - this is essential for proper tensor handling
    images_target = image_target.squeeze()
    images_ref = image_ref.squeeze()

    image_ref_np = images_ref.numpy()
    images_target_np = images_target.numpy()

    # Validate batch sizes early
    if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

    for i in range(batch_size):
        image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
        image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
        
        try:
            # Perform color matching
            image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            
            # Apply strength multiplier - optimized version
            if strength != 1.0:
                image_result = image_target_np + strength * (image_result - image_target_np)
            
            # Update the pre-allocated tensor directly
            if editInPlace:
                if batch_size == 1:
                    image_target.copy_(torch.from_numpy(image_result))
                else:
                    image_target[i].copy_(torch.from_numpy(image_result))
            else:
                if batch_size == 1:
                    out.copy_(torch.from_numpy(image_result))
                else:
                    out[i].copy_(torch.from_numpy(image_result))
            
            # Explicitly clear intermediate variables for memory management
            del image_target_np, image_ref_np_i, image_result
            
            # Force garbage collection at specified intervals for large batches
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                import gc
                gc.collect()
                
        except BaseException as e:
            print(f"Error occurred during transfer: {e}")
            break
    
    # Convert to float32 and clamp in-place
    if out.dtype != torch.float32:
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

    def colormatch(self, image_ref, image_target, method, passthrough, strength=1.0, editInPlace=False, gc_interval=50):
        if passthrough:
            print("VTS_ColourMatch - passthrough is True, returning original image_target without processing")
            return (image_target,)
        print(f"VTS_ColourMatch - passthrough is False, processing images. method: {method}, strength: {strength}, editInPlace: {editInPlace}, gc_interval: {gc_interval}")
        output = colormatch(image_ref, image_target, method, strength, editInPlace, gc_interval)
        print(f"VTS_ColourMatch - finished processing images. method: {method}, strength: {strength}, editInPlace: {editInPlace}, gc_interval: {gc_interval}")
        return (output[0],)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Colour Match": VTS_ColourMatch
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Colour Match": "VTS Colour Match"
}