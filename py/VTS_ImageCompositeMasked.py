import comfy
import comfy.utils
import node_helpers
import torch
from nodes import MAX_RESOLUTION

def colormatch(image_ref, image_target, method, strength=1.0):
    try:
        from color_matcher import ColorMatcher
    except:
        raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
    cm = ColorMatcher()
    image_ref = image_ref.cpu()
    image_target = image_target.cpu()
    batch_size = image_target.size(0)
    out = []
    images_target = image_target.squeeze()
    images_ref = image_ref.squeeze()

    image_ref_np = images_ref.numpy()
    images_target_np = images_target.numpy()

    if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
        raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

    for i in range(batch_size):
        image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
        image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
        try:
            image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
        except BaseException as e:
            print(f"Error occurred during transfer: {e}")
            break
        # Apply the strength multiplier
        image_result = image_target_np + strength * (image_result - image_target_np)
        out.append(torch.from_numpy(image_result))
        
    out = torch.stack(out, dim=0).to(torch.float32)
    out.clamp_(0, 1)
    return (out,)

def composite(destination, source, x, y, color_match_method, color_match_strength = 0.0, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    # Crop source and mask to visible area
    source_cropped = source[:, :, :visible_height, :visible_width]
    mask_cropped = mask[:, :, :visible_height, :visible_width]
    
    # Apply color matching if strength > 0
    if color_match_strength > 0.0:
        print(f"VTS_ImageCompositeMasked - applying color match with method {color_match_method} and strength {color_match_strength}")
        # Extract the destination pixels where source will be pasted
        image_dest = destination[:, :, top:bottom, left:right]
        # Convert from BCHW to BHWC format for colormatch function
        source_for_colormatch = source_cropped.movedim(1, -1)
        dest_for_colormatch = image_dest[:, :, :visible_height, :visible_width].movedim(1, -1)
        
        # Apply color matching
        color_matched_source = colormatch(dest_for_colormatch, source_for_colormatch, color_match_method, color_match_strength)[0]
        
        # Convert back to BCHW format
        source_cropped = color_matched_source.movedim(-1, 1)
    else:
        print("VTS_ImageCompositeMasked - no color match applied")

    inverse_mask = torch.ones_like(mask_cropped) - mask_cropped

    source_portion = mask_cropped * source_cropped
    destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination


class VTS_ImageCompositeMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "color_match_method": (
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
            },
            "optional": {
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "VTS"

    def composite(self, destination, source, x, y, resize_source, color_match_method, color_match_strength = 0.0, mask = None):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, color_match_method, color_match_strength, mask, 1, resize_source).movedim(1, -1)
        return (output,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Image Composite Masked": VTS_ImageCompositeMasked
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Image Composite Masked": "VTS Image Composite Masked"
}