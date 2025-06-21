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

    CATEGORY = "VTS"

    def upscale(self, image, upscale_method, smallMaxSize, largeMaxSize, divisible_by, crop):
        # Get the dimensions of the image
        height, width = image.shape[1], image.shape[2]
        original_height, original_width = height, width
        
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
            width = smallMaxSize if original_width < original_height else new_largest_side
            height = smallMaxSize if original_height < original_width else new_largest_side
        else:
            width = new_smallest_side if original_width < original_height else largeMaxSize
            height = new_smallest_side if original_height < original_width else largeMaxSize

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # if we are not actually scaling, just return the original image
        if width == original_width and height == original_height:
            print(f"VTS_Images_ScaleToMin - no scaling needed from {original_width}x{original_height} to {width}x{height}")
            return (image,)

        print(f"VTS_Images_ScaleToMin - scaling from {original_width}x{original_height} to {width}x{height}")

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