import comfy
import comfy.utils

MAX_RESOLUTION=16384


class VTS_Images_Scale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image_list": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image_list",)
    FUNCTION = "upscale"

    INPUT_IS_LIST = True

    OUTPUT_IS_LIST = (
        True,
    )

    CATEGORY = "VTS"

    def upscale(self, image_list: list, upscale_method, width, height, crop):
        # as INPUT_IS_LIST is true, all input parameters are lists
        # the only one we want to use as a list is image_list
        # the rest of the parameters are single values

        upscale_method = upscale_method[0]
        width = width[0]
        height = height[0]
        crop = crop[0]

        results = []

        for image in image_list:
            if width == 0 and height == 0:
                s = image
            else:
                samples = image.movedim(-1,1)

                if width == 0:
                    width = max(1, round(samples.shape[3] * height / samples.shape[2]))
                elif height == 0:
                    height = max(1, round(samples.shape[2] * width / samples.shape[3]))

                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                s = s.movedim(1,-1)
            results.append(s)

        return (results,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Images Scale": VTS_Images_Scale
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Images Scale": "VTS Images Scale"
}