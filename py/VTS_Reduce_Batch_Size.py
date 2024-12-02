import torch

class VTS_Reduce_Batch_Size:
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
        return { "required": {
                "images": ("IMAGE",),
                "origin_fps": ("INT", {"default": 1, "min": 1}),
                "result_fps": ("INT", {"default": 1, "min": 1}),
                "add_end_frame": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "notify"

    #OUTPUT_NODE = False

    CATEGORY = "VTS"

    @staticmethod
    def resize_tensor(input_tensor, skip_size, add_end_frame):
        if skip_size > len(input_tensor):
            return input_tensor  # Return the original tensor if skip_size is greater than its length
        
        skip_size = float(skip_size)

        rolling_total = 0.0
        indices = []
        added_frame_this_loop = False
        for i in range(len(input_tensor)):
            added_frame_this_loop = False
            if rolling_total <= i:
                indices.append(i)
                rolling_total += skip_size
                added_frame_this_loop = True

        if add_end_frame and not added_frame_this_loop:
            indices.append(len(input_tensor)-1)
        # output_tensor = torch.cat([input_tensor[i].unsqueeze(0) for i in indices[:int(skip_size)]], dim=0)
        output_tensor = torch.cat([input_tensor[i].unsqueeze(0) for i in indices], dim=0)

        return output_tensor

    # def resize_tensor(input_tensor, skip_size):
    #     if skip_size > len(input_tensor):
    #         return input_tensor  # Return the original tensor if skip_size is greater than its length

    #     if isinstance(skip_size, float):
    #         rolling_total = 0.0
    #         output_tensor = []
    #         for i in range(len(input_tensor)):
    #             rolling_total += skip_size
    #             if rolling_total >= 1.0:
    #                 output_tensor.append(input_tensor[i].unsqueeze(0))
    #                 rolling_total -= 1.0
    #         output_tensor = torch.cat(output_tensor[:int(skip_size)], dim=0)
    #     else:
    #         step = len(input_tensor) // skip_size
    #         output_tensor = input_tensor[::step][:skip_size]
    #         output_tensor = torch.cat([t.unsqueeze(0) for t in output_tensor], dim=0)
    #     return output_tensor
    

    # def resize_tensor(input_tensor, skip_size):
    #     if skip_size > len(input_tensor):
    #         return input_tensor  # Return the original tensor if skip_size is greater than its length

    #     if isinstance(skip_size, float):
    #         rolling_total = 0.0
    #         output_tensor = []
    #         for i in range(len(input_tensor)):
    #             rolling_total += skip_size
    #             if rolling_total >= 1.0:
    #                 output_tensor.append(input_tensor[i])
    #                 rolling_total -= 1.0
    #         output_tensor = output_tensor[:int(skip_size)]
    #     else:
    #         step = len(input_tensor) // skip_size
    #         output_tensor = input_tensor[::step][:skip_size]
    #     return output_tensor


    def notify(self, images, origin_fps, result_fps, add_end_frame=True):
        skip_size = origin_fps / result_fps  # Use float division
        res = VTS_Reduce_Batch_Size.resize_tensor(images, skip_size, add_end_frame)
        return (res,)



# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Reduce Batch Size": VTS_Reduce_Batch_Size
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Reduce Batch Size": "Reduce Batch Size"
}
