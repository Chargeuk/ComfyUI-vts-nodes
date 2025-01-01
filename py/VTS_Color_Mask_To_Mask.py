import torch
import webcolors
import numpy as np
import cv2

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask

def dilate_mask(mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
    """Dilate a mask using a square kernel with a given dilation factor."""
    kernel_size = int(dilation_factor * 2) + 1
    kernel = np.ones((abs(kernel_size), abs(kernel_size)), np.uint8)

    masks = make_3d_mask(mask).numpy()
    dilated_masks = []
    for m in masks:
        if dilation_factor > 0:
            m2 = cv2.dilate(m, kernel, iterations=1)
        else:
            m2 = cv2.erode(m, kernel, iterations=1)

        dilated_masks.append(torch.from_numpy(m2))

    return torch.stack(dilated_masks)

def color_to_mask(color_mask, mask_colors):
    selected_colors = []
    for mask_color in mask_colors.split(','):
        mask_color = mask_color.strip()
        try:
            if mask_color.startswith("#") or mask_color.isalpha():
                hex = mask_color[1:] if mask_color.startswith("#") else webcolors.name_to_hex(mask_color)[1:]
                selected = int(hex, 16)
            else:
                selected = int(mask_color, 10)
            selected_colors.append(selected)
        except Exception:
            raise Exception(f"[ERROR] Invalid mask_color value. mask_color should be a color value for RGB")

    temp = (torch.clamp(color_mask, 0, 1.0) * 255.0).round().to(torch.int)
    temp = torch.bitwise_left_shift(temp[:, :, :, 0], 16) + torch.bitwise_left_shift(temp[:, :, :, 1], 8) + temp[:, :, :, 2]
    
    mask = torch.zeros_like(temp, dtype=torch.float)
    for selected in selected_colors:
        mask = torch.where(temp == selected, 1.0, mask)
    
    return mask

class VTS_Color_Mask_To_Mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_colors": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
            },
            "optional": {
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "VTS"

    @staticmethod
    def doit(color_mask, mask_colors, dilation=0):
        mask = color_to_mask(color_mask, mask_colors)

        if dilation != 0:
            mask = dilate_mask(mask, dilation)

        return (mask, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Color Mask To Mask": VTS_Color_Mask_To_Mask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Color Mask To Mask": "Color Mask To Mask"
}