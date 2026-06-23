import comfy
import comfy.utils

# Add the py directory to sys.path to allow imports
import sys
import os
import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, transform_and_save_images, get_default_image_input_types, deep_merge, ensure_image_defaults

MAX_RESOLUTION = 16384

class VTS_Images_ScaleToMin:
    upscale_methods = [
        "nearest-exact", "bilinear", "area", "bicubic", "lanczos",
        "rtx-vsr-low", "rtx-vsr-medium", "rtx-vsr-high", "rtx-vsr-ultra",
    ]
    crop_methods = ["disabled", "center"]
    scale_types = ["small", "large", "max"]

    @classmethod
    def INPUT_TYPES(s):
        defaults = get_default_image_input_types()
        input_types = {"required": { "image": ("IMAGE",),
                              "upscale_method": (s.upscale_methods,),
                              "smallMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "largeMaxSize": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                              "crop": (s.crop_methods,),
                              "scale_type": (s.scale_types, {"default": "small", "tooltip": "The type of scaling to apply. Large does not maintain aspect ratio."}),
                              "edit_in_place": ("BOOLEAN", {"default": False}),
                              "batch_size": ("INT", {"default": 80, "min": 1, "max": 1000, "step": 1}),
                            }
                }
        result = deep_merge(defaults, input_types)
        return result
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = "VTS"


    def scale(self, upscale_method, smallMaxSize, largeMaxSize, divisible_by, crop, scale_type, **kwargs):
        kwargs = ensure_image_defaults(kwargs)
        image = kwargs.get("image", None)
        # Check if input is a DiskImage
        # if isinstance(image, DiskImage):
        return self._scale_disk_image(image, upscale_method, smallMaxSize, largeMaxSize, 
                                        divisible_by, crop, scale_type, kwargs)
        

    def _scale_disk_image(self, image, upscale_method, smallMaxSize, largeMaxSize, 
                          divisible_by, crop, scale_type, kwargs):
        """Handle scaling of DiskImage input"""

        total_batch_size, height, width, channels = image.shape
        print(f"VTS_Images_ScaleToMin - processing DiskImage with {total_batch_size} images")
        
        
        # Create transform function that captures all the scaling parameters
        def transform_fn(batch_tensor):
            return self._scale_tensor(batch_tensor, upscale_method, smallMaxSize, 
                                     largeMaxSize, divisible_by, crop, scale_type)[0]
        
        result = transform_and_save_images(
            transform_fn=transform_fn,
            **kwargs
        )
        print(f"VTS_Images_ScaleToMin - DiskImage scaling complete")
        return (result,)
    

    def _scale_tensor(self, image, upscale_method, smallMaxSize, largeMaxSize, 
                      divisible_by, crop, scale_type):
        """Original tensor-based scaling logic"""
        # Get the dimensions of the image
        height, width = image.shape[1], image.shape[2]
        original_height, original_width = height, width
        
        # Identify the largest and smallest sides
        largest_side = max(height, width)
        smallest_side = min(height, width)

        # Calculate the aspect ratio
        aspect_ratio = largest_side / smallest_side

        # Calculate new dimensions based on aspect ratio and max sizes
        new_largest_side = round(smallMaxSize * aspect_ratio)
        new_smallest_side = round(largeMaxSize / aspect_ratio)

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

        # Force safer upscale methods for large downscaling operations
        if (original_width * original_height > 1000000 and  # > 1MP
            width * height < original_width * original_height and  # downscaling
            upscale_method == "bilinear"):
            print(f"VTS_Images_ScaleToMin - large downscale detected, using area instead of bilinear")
            upscale_method = "area"  # Much safer for downscaling

        original_upscale_method = upscale_method
        if self._is_rtx_vsr_method(upscale_method):
            rtx_input = image
            if crop == "center":
                rtx_input = self._center_crop_for_aspect(image, width, height)

            cropped_height, cropped_width = rtx_input.shape[1], rtx_input.shape[2]
            if width < cropped_width and height < cropped_height:
                print(
                    "VTS_Images_ScaleToMin - RTX VSR target would shrink both dimensions "
                    f"after crop ({cropped_width}x{cropped_height} -> {width}x{height}); "
                    "falling back to bicubic with original image/settings"
                )
                upscale_method = "bicubic"
            else:
                try:
                    quality = self._rtx_vsr_quality(original_upscale_method)
                    print(
                        f"VTS_Images_ScaleToMin - using RTX VSR {quality} "
                        f"from {cropped_width}x{cropped_height} to {width}x{height}"
                    )
                    return (self._scale_tensor_rtx_vsr(rtx_input, width, height, quality),)
                except Exception as e:
                    print(
                        "VTS_Images_ScaleToMin - RTX VSR failed; "
                        f"falling back to bicubic with original image/settings: {e}"
                    )
                    upscale_method = "bicubic"

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

    def _is_rtx_vsr_method(self, upscale_method):
        return isinstance(upscale_method, str) and upscale_method.startswith("rtx-vsr-")

    def _rtx_vsr_quality(self, upscale_method):
        return upscale_method.removeprefix("rtx-vsr-").upper()

    def _center_crop_for_aspect(self, image, width, height):
        old_height, old_width = image.shape[1], image.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)

        if x == 0 and y == 0:
            return image

        cropped = image[:, y:old_height - y, x:old_width - x, :]
        print(f"VTS_Images_ScaleToMin - RTX VSR center-cropped from {old_width}x{old_height} to {cropped.shape[2]}x{cropped.shape[1]}")
        return cropped

    def _scale_tensor_rtx_vsr(self, image, width, height, quality):
        import torch
        import nvvfx

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)
        output_width = max(8, round(width / 8) * 8)
        output_height = max(8, round(height / 8) * 8)

        if output_width != width or output_height != height:
            print(f"VTS_Images_ScaleToMin - RTX VSR rounded target from {width}x{height} to {output_width}x{output_height}")

        out_tensor = torch.empty(
            (image.shape[0], output_height, output_width, image.shape[-1]),
            device=image.device,
            dtype=image.dtype,
        )

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            for i in range(image.shape[0]):
                input_frame = image[i].cuda().permute(2, 0, 1).float().contiguous()
                dlpack_out = sr.run(input_frame).image
                output_frame = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)
                out_tensor[i:i + 1] = output_frame.to(device=image.device, dtype=image.dtype)

        if output_width != width or output_height != height:
            samples = out_tensor.movedim(-1, 1)
            out_tensor = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled").movedim(1, -1)

        return out_tensor.clamp(0, 1)

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
            width = round(original_width * heightRatio)
        else:
            width = largeMaxSize
            widthRatio = width / original_width
            height = round(original_height * widthRatio)
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
