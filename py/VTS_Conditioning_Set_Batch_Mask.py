def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c

class VTS_Conditioning_Set_Batch_Mask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "masks": ("MASK", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "set_cond_area": (["default", "mask bounds"],),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_batch_masks"

    CATEGORY = "VTS"

    def apply_batch_masks(self, conditioning, masks, set_cond_area, strength):
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        conditionings = []
        
        # Check if masks is a batch (4D tensor)
        if len(masks.shape) == 4:
            for mask in masks:
                if len(mask.shape) < 3:
                    mask = mask.unsqueeze(0)
                c = conditioning_set_values(conditioning, {"mask": mask,
                                                            "set_area_to_bounds": set_area_to_bounds,
                                                            "mask_strength": strength})
                conditionings.append(c)
        else:
            # Handle the case where masks is not a batch
            if len(masks.shape) < 3:
                masks = masks.unsqueeze(0)
            c = conditioning_set_values(conditioning, {"mask": masks,
                                                        "set_area_to_bounds": set_area_to_bounds,
                                                        "mask_strength": strength})
            conditionings.append(c)
        
        print(f"VTS conditionings: {conditionings}")
        return conditionings

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Conditioning Set Batch Mask": VTS_Conditioning_Set_Batch_Mask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Conditioning Set Batch Mask": "Conditioning Set Batch Mask"
}