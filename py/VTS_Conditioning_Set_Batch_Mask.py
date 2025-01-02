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
    RETURN_TYPES = (
        "CONDITIONING",
    )

    # OUTPUT_IS_LIST = (
    #     True
    # )

    FUNCTION = "apply_batch_masks"

    CATEGORY = "VTS"

    @staticmethod
    def printConditioningInfo(data, name: str):
        if isinstance(data, list):
            print(f"!!!VTS_Conditioning_Set_Batch_Mask item '{name}' contains a list of length [{len(data)}].")
            # iterate through each item and call this method
            for i, d in enumerate(data):
                VTS_Conditioning_Set_Batch_Mask.printConditioningInfo(d, f"{name}[{i}]")
        if isinstance(data, dict):
            # print out the keys
            print(f"!!!VTS_Conditioning_Set_Batch_Mask item '{name}' contains a dict with keys {data.keys()}.")
            # iterate through each item and call this method
            for k, v in data.items():
                VTS_Conditioning_Set_Batch_Mask.printConditioningInfo(v, f"{name}.{k}")
        elif hasattr(data, 'shape'):
            print(f"!!!VTS_Conditioning_Set_Batch_Mask item '{name}' contains a conditioning shape[{len(data.shape)}]={data.shape}.")
        else:
            print(f"!!!VTS_Conditioning_Set_Batch_Mask item '{name}' contains an unexpected type: {type(data)}.")

    @staticmethod
    def get_conditionings_from_shape_masks(masks, conditioning, set_area_to_bounds, strength):
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
        return conditionings

    @staticmethod
    def is_list_of_multiple_conditionings(conditioning):
        return isinstance(conditioning, list) and len(conditioning) > 0 and isinstance(conditioning[0], list) and len(conditioning[0]) > 0 and isinstance(conditioning[0][0], list) and len(conditioning[0][0]) > 0

    @staticmethod
    def is_list_of_masks(masks):
        return isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], torch.Tensor)

    def apply_batch_masks(self, conditioning, masks, set_cond_area, strength):
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        

        VTS_Conditioning_Set_Batch_Mask.printConditioningInfo(conditioning, "conditioning")

        is_list_of_multiple_conditionings = VTS_Conditioning_Set_Batch_Mask.is_list_of_multiple_conditionings(conditioning)
        is_list_of_masks = VTS_Conditioning_Set_Batch_Mask.is_list_of_masks(masks)

        if is_list_of_multiple_conditionings != is_list_of_masks:
            raise Exception(f"Conditioning and masks must both be lists of multiple items or not, but got conditioning: {is_list_of_multiple_conditionings} and masks: {is_list_of_masks}.")

        if is_list_of_multiple_conditionings and is_list_of_masks:
            all_conditionings = []
            conditioning_length = len(conditioning)
            masks_length = len(masks)
            if conditioning_length != masks_length:
                raise Exception(f"Conditioning length {conditioning_length} does not match masks length {masks_length}.")
            for conditioning_item, masks_item in zip(conditioning, masks):
                all_conditionings.append(VTS_Conditioning_Set_Batch_Mask.get_conditionings_from_shape_masks(masks_item, conditioning_item, set_area_to_bounds, strength))
            return all_conditionings
        
        conditionings = VTS_Conditioning_Set_Batch_Mask.get_conditionings_from_shape_masks(masks, conditioning, set_area_to_bounds, strength)
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