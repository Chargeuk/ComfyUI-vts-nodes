import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import crop

# taken from comfyUi samplers.py to match the behavior of the sampler function
def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

class VTS_Images_Crop_From_Masks:
    @classmethod
    def INPUT_TYPES(s):
        return {
          "required": {
              "image": ("IMAGE",),
              "mask": ("MASK",),
          },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX",)
    RETURN_NAMES = ("crop_image", "crop_mask", "bboxes",)
    OUTPUT_IS_LIST = (
        False,
        False,
        False
    )
    FUNCTION = "crop"
    CATEGORY = "VTS"

    def cropimage(self, original_images, masks):
        if len(masks.shape) == 2:
                print(f"VTS_Images_Crop_From_Masks - len(mask.shape) == 2. mask.shape={masks.shape}. Unsqueezing the mask.")
                masks = masks.unsqueeze(0)

        bounds_mask = torch.max(torch.abs(masks),dim=0).values.unsqueeze(0)
        print(f"VTS_Images_Crop_From_Masks bounds.shape={bounds_mask.shape}")

        # Get the size of the first image in the original_images batch
        first_image_size = original_images[0].shape[:2]  # (height, width)
        print(f"VTS_Images_Crop_From_Masks first_image_size={first_image_size}")
        bounds_mask_size = bounds_mask.shape[1:]  # (height, width)
        print(f"VTS_Images_Crop_From_Masks bounds_mask_size={bounds_mask_size}")

        if not all(x == y for x, y in zip(first_image_size, bounds_mask_size)):
            # Rescale bounds_mask to match the size of the first image
            bounds_mask = Resize(first_image_size)(bounds_mask)
            print(f"VTS_Images_Crop_From_Masks rescaled bounds.shape={bounds_mask.shape}")

        boxes, is_empty = get_mask_aabb(bounds_mask)
        if is_empty[0]:
            # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
            area = (8, 8, 0, 0)
        else:
            box = boxes[0]
            H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
            H = max(8, H)
            W = max(8, W)
            area = (int(H), int(W), int(Y), int(X))
            print(f"VTS_Images_Crop_From_Masks - calculated ares. area={area}. H, W, Y, X = {H, W, Y, X}")

        # now crop the provided original_images using the calculated area
        cropped_images = []
        for img_count, img in enumerate(original_images):
            print(f"\nVTS_Images_Crop_From_Masks - image[{img_count}] to be cropped shape={img.shape}.")
            # Permute the image to (colors, height, width)
            img_permuted = img.permute(2, 0, 1)
            print(f"VTS_Images_Crop_From_Masks - permuted image[{img_count}] to be cropped shape={img_permuted.shape}.")
            # Crop the image
            cropped_img = crop(img_permuted, area[2], area[3], area[0], area[1])
            print(f"VTS_Images_Crop_From_Masks - cropped image[{img_count}] shape={cropped_img.shape}.")
            # Permute the cropped image back to (height, width, colors)
            cropped_img = cropped_img.permute(1, 2, 0)
            print(f"VTS_Images_Crop_From_Masks - permuted cropped image[{img_count}] shape={cropped_img.shape}.")
            cropped_images.append(cropped_img)

        # Calculate the bounding box in a format that can be visualised by other nodes
        min_x = area[3]
        min_y = area[2]
        max_x = min_x + area[1]
        max_y = min_y + area[0]
        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        return cropped_images, area, bounds_mask, bounding_box


    def crop(self, image, mask):
      print(f"VTS_Images_Crop_From_Masks image.shape={image.shape}, mask.shape={mask.shape}")
      cropped_images, area, bounds_mask, bounding_box = self.cropimage(image, mask)
      cropped_image_out = torch.stack(cropped_images, dim=0)
      print(f"VTS_Images_Crop_From_Masks cropped_image_out.shape={cropped_image_out.shape}, bounds_mask.shape={bounds_mask.shape}, bounding_box={bounding_box}")
      return (cropped_image_out, bounds_mask, [bounding_box])


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Images Crop From Masks": VTS_Images_Crop_From_Masks
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Images Crop From Masks": "Images Crop From Masks"
}