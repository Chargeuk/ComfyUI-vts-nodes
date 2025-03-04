import comfy
import comfy.utils

MAX_RESOLUTION = 16384

class VTS_Images_ScaleToMin:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, smallMaxSize, largeMaxSize, divisible_by, crop):
        # Get the dimensions of the image
        height, width = image.shape[2], image.shape[3]
        
        # Identify the largest and smallest sides
        largest_side = max(height, width)
        smallest_side = min(height, width)

        # Calculate the aspect ratio
        aspect_ratio = largest_side / smallest_side

        # Calculate new dimensions based on aspect ratio and max sizes
        new_largest_side = int(smallMaxSize * aspect_ratio)
        new_smallest_side = int(largeMaxSize / aspect_ratio)

        # Determine final dimensions
        if new_largest_side <= largeMaxSize:
            width = smallMaxSize if width < height else new_largest_side
            height = smallMaxSize if height < width else new_largest_side
        else:
            width = new_smallest_side if width < height else largeMaxSize
            height = new_smallest_side if height < width else largeMaxSize

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # Move dimensions for processing
        samples = image.movedim(-1, 1)
        
        # Perform the upscale
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1, -1)
        
        return (s,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Images Scale To Min": VTS_Images_ScaleToMin
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Images Scale To Min": "VTS Images Scale To Min"
}