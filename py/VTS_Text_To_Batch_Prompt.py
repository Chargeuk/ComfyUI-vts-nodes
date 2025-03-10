class VTS_Text_To_Batch_Prompt:
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
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!",
                    "lazy": True
                }),
                "string_seperator": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "-----",
                    "lazy": True
                }),
                "origin_fps": ("INT", {"default": 1, "min": 1}),
                "result_fps": ("INT", {"default": 1, "min": 1}),
                "max_frames": ("INT", {"default": 1000, "min": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "notify"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"

    def check_lazy_status(self, print_to_screen, string_field, string_seperator, origin_fps, result_fps, max_frames):
        """
            Return a list of input names that need to be evaluated.

            This function will be called if there are any lazy inputs which have not yet been
            evaluated. As long as you return at least one field which has not yet been evaluated
            (and more exist), this function will be called again once the value of the requested
            field is available.

            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
            inputs will have the value None.
        """
        if print_to_screen == "enable":
            return ["string_field", "string_field", "origin_fps", "result_fps", "max_frames"]
        else:
            return []

    def notify(self, print_to_screen, string_field, string_seperator, origin_fps, result_fps, max_frames):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                string_seperator: {string_seperator}
                print_to_screen: {print_to_screen}
                origin_fps: {origin_fps}
                result_fps: {result_fps}
            """)
        if isinstance(string_field, list):
            string_field = string_seperator.join(map(str, string_field))
        workedStrings = string_field.split(string_seperator)

        skip_size = origin_fps / result_fps  # Use float division
        max_frames = max(max_frames - 1, 0)  # Ensure max_frames is at least 0

        # we need to merge all of the strings within workedStrings so that each string is prefixed by an incrementing
        # number that starts at 0 and ends with a comma and then a newline character. Each item must be within quotes,
        # as must each incrementing number. eg:
        # "0" :"Hello",
        # "1" :"World",
        # "2" :"!"
        # This is the format that the batch prompt expects.
        # We can use a list comprehension to achieve this:
        workedStrings = [f'"{min(round(i * skip_size), max_frames)}" :"{string}",\n' for i, string in enumerate(workedStrings)]
        # Now we need to join all of the strings together and remove the last comma and new line:
        workedStrings = "".join(workedStrings)[:-2]
        return (workedStrings,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, print_to_screen, string_field, string_seperator):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hello")
async def get_hello(request):
    return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Vts Text To Batch Prompt": VTS_Text_To_Batch_Prompt
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Vts Text To Batch Prompt": "Text To Batch Prompt"
}
