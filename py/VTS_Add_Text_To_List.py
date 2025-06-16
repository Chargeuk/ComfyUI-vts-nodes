import json

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class VTS_Add_Text_To_list:

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
                "position": ("STRING", {"default": "start", "values": ["start", "end"]}),
                "string_to_add": ("STRING", {"default": ""}),
                "string_list": ("STRING", {"default": ""}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "end_index": ("INT", {"default": 9999999, "min": 0, "step": 1})
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    OUTPUT_IS_LIST = (
        True, 
    )

    RETURN_NAMES = (
        "string_list", # question answers output
    )

    INPUT_IS_LIST = True

    FUNCTION = "add_text_to_list"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"


    def add_text_to_list(
            self,
            position,
            string_to_add,
            string_list,
            start_index,
            end_index):
        # Extract values from input lists
        position = position[0]
        string_to_add = string_to_add[0]
        start_index = start_index[0]
        end_index = end_index[0]

        # Initialize the result list
        result = []

        # Iterate through the string_list
        for i, string in enumerate(string_list):
            if start_index <= i <= end_index:  # Check if index is within range
                if position == "start":
                    string = string_to_add + string  # Prepend string_to_add
                else:
                    string = string + string_to_add  # Append string_to_add
            result.append(string)  # Add the modified or original string to result

        return (result,)


# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Add Text To list": VTS_Add_Text_To_list
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Add Text To list": "Add Text To list"
}
