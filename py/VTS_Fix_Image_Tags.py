import re

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class VTS_Fix_Image_Tags:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    DEPRECATED (`bool`):
        Indicates whether the node is deprecated. Deprecated nodes are hidden by default in the UI, but remain
        functional in existing workflows that use them.
    EXPERIMENTAL (`bool`):
        Indicates whether the node is experimental. Experimental nodes are marked as such in the UI and may be subject to
        significant changes or removal in future versions. Use with caution in production workflows.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "tags_list": ("STRING", ),
                "cutoff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum cutoff value for positive tags."}),
                "cutoff_applied_at": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "Minimum number of positive tags before we start to apply the cutoff."}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("positive_tags","negative_tags",)
    OUTPUT_IS_LIST = (
        True,
        True,
    )

    FUNCTION = "notify"
    INPUT_IS_LIST = True

    CATEGORY = "VTS"


    def notify(self, tags_list, cutoff, cutoff_applied_at):
        cutoff = cutoff[0]
        cutoff_applied_at = cutoff_applied_at[0]
        positive_tags_list = []
        negative_tags_list = []
        for text in tags_list:
            positive_tags = ""
            negative_tags = ""

            # Regular expression to match tags in the format (tag:float)
            tag_pattern = r"\(([^:]+):([-+]?\d*\.\d+|\d+)\)"
            matches = re.findall(tag_pattern, text)

            positive_dict = {}
            negative_list = []

            for tag, value in matches:
                try:
                    value = float(value)
                    if value > 0:
                        # Keep only the highest value for each tag
                        if tag not in positive_dict or value > positive_dict[tag]:
                            positive_dict[tag] = value
                    else:
                        negative_list.append(f"({tag}:1.0)")
                except ValueError:
                    # Ignore invalid values
                    continue

            # Convert positive_dict to a list of tuples and sort by value in descending order
            positive_list = sorted(positive_dict.items(), key=lambda x: x[1], reverse=True)

            # Filter positive tags based on cutoff and cutoff_applied_at
            filtered_positive_list = []
            for tag, value in positive_list:
                if value > cutoff or len(filtered_positive_list) < cutoff_applied_at:
                    filtered_positive_list.append(f"({tag}:{value})")

            positive_tags = ",".join(filtered_positive_list)
            negative_tags = ",".join(negative_list)
            if positive_tags:
                positive_tags_list.append(positive_tags)
            if negative_tags:
                negative_tags_list.append(negative_tags)

        return (positive_tags_list, negative_tags_list)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Fix Image Tags": VTS_Fix_Image_Tags
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Fix Image Tags": "Fix Image Tags"
}
