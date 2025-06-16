import torch
import logging

class VTSCalculateUpscaleAmount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Batch of images
                "target_small_resolution": ("INT",),  # Target resolution for the smallest side
                "target_long_resolution": ("INT",),  # Target resolution for the longest side
            },
        }
    RETURN_TYPES = ("FLOAT","FLOAT",)
    FUNCTION = "calculate_upscale_amount"

    CATEGORY = "image/upscaling"
    RETURN_NAMES = (
        "max_multiplier",
        "min_multiplier",
    )

    def calculate_upscale_amount(
        self,
        image: torch.Tensor,
        target_small_resolution: int,
        target_long_resolution: int,
    ) -> tuple:
        """
        Calculate the upscale multiplier for a batch of images based on target resolutions.

        Args:
            images (torch.Tensor): Batch of images with shape (batch_size, channels, height, width).
            target_small_resolution (int): Target resolution for the smallest side.
            target_long_resolution (int): Target resolution for the longest side.

        Returns:
            tuple: A tuple containing the highest multiplier for each image in the batch.
        """
        # Ensure images are in the correct shape (batch_size, channels, height, width)
        if len(image.shape) != 4:
            raise ValueError("Input images must be a 4D tensor (batch_size, channels, height, width).")

        # Extract height and width for each image in the batch
        height, width = image.shape[1], image.shape[2]

        # Determine the smallest and longest sides
        smallest_side = min(height, width)
        longest_side = max(height, width)

        # Calculate multipliers
        small_multiplier = target_small_resolution / smallest_side
        long_multiplier = target_long_resolution / longest_side

        # Return the 2 multipliers
        max_multiplier = max(small_multiplier, long_multiplier)
        min_multiplier = min(small_multiplier, long_multiplier)
        logging.info(f"VTSCalculateUpscaleAmount - Calculated multipliers: max={max_multiplier}, min={min_multiplier} from image size {height}x{width} to target sizes {target_small_resolution}x{target_long_resolution}")
        return (max_multiplier, min_multiplier,)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Calculate Upscale Amount": VTSCalculateUpscaleAmount
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Calculate Upscale Amount": "Calculate Upscale Amount"
}
