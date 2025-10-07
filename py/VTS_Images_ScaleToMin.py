import comfy
import comfy.utils

MAX_RESOLUTION = 16384

class VTS_Images_ScaleToMin:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]
    scale_types = ["small", "large", "max"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "upscale_method": (s.upscale_methods,),
                              "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                              "crop": (s.crop_methods,),
                              "scale_type": (s.scale_types, {"default": "small"}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = "VTS"

    def scale(self, image, upscale_method, smallMaxSize, largeMaxSize, divisible_by, crop, scale_type):
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
        if scale_type == "small":
            width, height = self.getSmallDimensions(original_width, original_height, smallMaxSize, largeMaxSize, new_largest_side, new_smallest_side)
        elif scale_type == "large":
            width, height = self.getLargeDimensions(original_width, original_height, smallMaxSize, largeMaxSize)
        else:
            width, height = self.getMaxDimensions(original_width, original_height, smallMaxSize, largeMaxSize)

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # if we are not actually scaling, just return the original image
        if width == original_width and height == original_height:
            print(f"VTS_Images_ScaleToMin - no scaling needed from {original_width}x{original_height} to {width}x{height}")
            return (image,)

        print(f"VTS_Images_ScaleToMin - scaling from {original_width}x{original_height} to {width}x{height}")
        old_aspect = original_width / original_height
        new_aspect = width / height
        print(f"Aspect ratios - old: {old_aspect:.10f}, new: {new_aspect:.10f}, diff: {abs(old_aspect - new_aspect):.2e}")

        if abs(old_aspect - new_aspect) < 1e-6:
            print("Aspect ratios are essentially equal - forcing crop='disabled'")
            crop = "disabled"
        try:
            # Move dimensions for processing
            samples = image.movedim(-1, 1)
            print(f"VTS_Images_ScaleToMin - moved dimensions for processing")
            # Perform the upscale
            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            print(f"VTS_Images_ScaleToMin - completed upscale")
            s = s.movedim(1, -1)
            print(f"VTS_Images_ScaleToMin - moved dimensions back to original")
            
            return (s,)
        except Exception as e:
            print(f"VTS_Images_ScaleToMin - error during scaling: {e}. returning original image")
            return (image,)

    def getSmallDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize, new_largest_side, new_smallest_side):
        if new_largest_side <= largeMaxSize:
            width = smallMaxSize if original_width < original_height else new_largest_side
            height = smallMaxSize if original_height < original_width else new_largest_side
        else:
            width = new_smallest_side if original_width < original_height else largeMaxSize
            height = new_smallest_side if original_height < original_width else largeMaxSize
        return width, height

    def getLargeDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize):
        if original_width < original_height:
            width = smallMaxSize
            height = largeMaxSize
        else:
            width = largeMaxSize
            height = smallMaxSize
        return width, height
    
    def getMaxDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize):
        if original_width < original_height:
            height = largeMaxSize
            heightRatio = height / original_height
            width = int(original_width * heightRatio)
        else:
            width = largeMaxSize
            widthRatio = width / original_width
            height = int(original_height * widthRatio)
        return width, height


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Images Scale To Min": VTS_Images_ScaleToMin
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Images Scale To Min": "VTS Images Scale To Min"
}