import json

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class VTS_Merge_Text_Lists:
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
                "string_list_1": ("STRING",),
                "string_list_2": ("STRING",),
                "seperator": ("STRING", {"default": "-----"})
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    OUTPUT_IS_LIST = (
        True, # question answers output is a list of string, 1 item per image
    )

    RETURN_NAMES = (
        "merged_string_lists", # question answers output
    )
    #RETURN_NAMES = ("image_output_name",)
    INPUT_IS_LIST = True

    FUNCTION = "merge_string_arrays"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"


    def merge_string_arrays(self, string_list_1, string_list_2, seperator):
        seperator = seperator[0]  # Get the first item from the list, as INPUT_IS_LIST is True
        # Determine the length of the longer array
        max_length = max(len(string_list_1), len(string_list_2))
        
        # Merge strings at each index or append remaining elements
        merged_array = []
        for i in range(max_length):
            if i < len(string_list_1) and i < len(string_list_2):
                merged_array.append(f"{string_list_1[i]}{seperator}{string_list_2[i]}")
            elif i < len(string_list_1):
                merged_array.append(string_list_1[i])
            elif i < len(string_list_2):
                merged_array.append(string_list_2[i])
        
        return (merged_array,)


# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Merge Text Lists": VTS_Merge_Text_Lists
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Merge Text Lists": "Merge Text Lists"
}
