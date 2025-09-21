import json

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class VTS_To_Text_List:
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
                "input_data": (any_typ, )
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_text_list", "list_length")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (
        True,
        False,
    )

    FUNCTION = "notify"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"


    def notify(self, input_data):
        print("*********** VTS_To_Text_List CALLED!! *************")
        output_text_list = []

        # if either delimiter or input_data is None, return empty text
        if input_data is None:
            print("VTS_To_Text_List input_data is None, returning empty output")
            return (output_text_list, len(output_text_list),)

        for index, input_data_item in enumerate(input_data):
            # Try to convert input_data_item to a list or dict if it's a string
            if isinstance(input_data_item, str):
                try:
                    input_data_item = json.loads(input_data_item)
                    print("VTS_To_Text_List input_data successfully converted from string to list or dict")
                except json.JSONDecodeError:
                    pass
                    print("VTS_To_Text_List input_data_item is a string but not a valid JSON, proceeding as a single string value")

            if isinstance(input_data_item, list):
                for item in input_data_item:
                    output_text_list.append(str(item))
            elif isinstance(input_data_item, dict):
                for value in input_data_item.values():
                    output_text_list.append(str(value))
            else:
                output_text_list.append(str(input_data_item))

        return (output_text_list, len(output_text_list),)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS To Text List": VTS_To_Text_List
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS To Text List": "To Text List"
}
