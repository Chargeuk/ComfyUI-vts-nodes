# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class VTS_Clean_Text:
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
                "text": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "notify"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"


    def notify(self, text):
        cleaned_text = text.replace(",", ", ").replace(".", ". ").replace("\n", ", ").replace("\r", "").replace("\t", ", ")

        needs_cleaning = True
        while needs_cleaning:
            needs_cleaning = False
            # Remove double spaces
            while "  " in cleaned_text:
                cleaned_text = cleaned_text.replace("  ", " ")
                needs_cleaning = True

            # Remove double commas
            while ",," in cleaned_text:
                cleaned_text = cleaned_text.replace(",,", ",")
                needs_cleaning = True

            # Remove commas with only spaces separating them
            while ", ," in cleaned_text:
                cleaned_text = cleaned_text.replace(", ,", ",")
                needs_cleaning = True

            # remove double full stops
            while ".." in cleaned_text:
                cleaned_text = cleaned_text.replace("..", ".")
                needs_cleaning = True

            # Remove full stops with only spaces separating them
            while ". ." in cleaned_text:
                cleaned_text = cleaned_text.replace(". .", ".")
                needs_cleaning = True

            while ",." in cleaned_text:
                cleaned_text = cleaned_text.replace(",.", ".")
                needs_cleaning = True

            # Remove full stops with only spaces separating them
            while ", ." in cleaned_text:
                cleaned_text = cleaned_text.replace(", .", ".")
                needs_cleaning = True
            
            while ".," in cleaned_text:
                cleaned_text = cleaned_text.replace(".,", ".")
                needs_cleaning = True

            # Remove full stops with only spaces separating them
            while ". ," in cleaned_text:
                cleaned_text = cleaned_text.replace(". ,", ".")
                needs_cleaning = True

            # Remove duplicated words
            words = cleaned_text.split()
            removed_words_cleaned_text = ' '.join([word for i, word in enumerate(words) if i == 0 or word != words[i - 1]])
            if removed_words_cleaned_text != cleaned_text:
                needs_cleaning = True
            cleaned_text = removed_words_cleaned_text

        return (cleaned_text,)



# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Clean Text": VTS_Clean_Text
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Clean Text": "Clean Text"
}
