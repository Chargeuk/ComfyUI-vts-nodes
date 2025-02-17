from itertools import zip_longest

class VTS_Merge_Delimited_Text:
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
                "delimiter": ("STRING", {"default": ", "}),
                "input_string1": ("STRING", ),
                "input_string2": ("STRING", ),
                "merge_delimiter": ("STRING", {"default": ", "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "notify"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"


    def notify(self, delimiter, input_string1, input_string2, merge_delimiter):
        # Replace all occurrences of "\n" (literal newline) in the delimiter with an actual newline character.
        delimiter = delimiter.replace("\\n", "\n")
        
        # Split the input strings by the delimiter
        list1 = input_string1.split(delimiter)
        #print(f"\nlist1: ", list1)

        list2 = input_string2.split(delimiter)
        #print(f"\nlist2: ", list2)
        
        # Merge the lists at corresponding indexes, placing merge_delimiter between the merged strings if both are non-empty
        merged_list = [a + (merge_delimiter if a and b else '') + b for a, b in zip_longest(list1, list2, fillvalue='')]
        #print(f"\nmerged_list: {merged_list}")
        
        # Join the merged list back into a single string with the delimiter
        result_string = delimiter.join(merged_list)

        #print(f"\nresult_string: {result_string}")
        
        return (result_string,)



# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Merge Delimited Text": VTS_Merge_Delimited_Text
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Merge Delimited Text": "Merge Delimited Text"
}
