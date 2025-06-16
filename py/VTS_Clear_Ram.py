import json
import model_management
import torch
import gc


# wildcard trick is taken from pythongossss's
class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")

class VTS_Clear_Ram:
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
      return {
        "required": {
            
            "empty_cache": ("BOOLEAN", {"default": True}),
            "gc_collect": ("BOOLEAN", {"default": True}),
            "unload_all_models": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "any_input": (any, {}),
            "any_input2": (any, {}),
            "any_input3": (any, {}),
            "any_input4": (any, {}),
            "any_input5": (any, {}),
            "any_input6": (any, {}),
            "any_input7": (any, {}),
            "any_input8": (any, {}),
            "any_input9": (any, {}),
            "image_pass": ("IMAGE",),
            "model_pass": ("MODEL",),
        }
	}
        
    RETURN_TYPES = (
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        "IMAGE",
        "MODEL",
        "INT",
        "INT",)
    RETURN_NAMES = (
        "any_output",
        "any_output2",
        "any_output3",
        "any_output4",
        "any_output5",
        "any_output6",
        "any_output7",
        "any_output8",
        "any_output9",
        "image_pass",
        "model_pass",
        "freemem_before",
        "freemem_after")
    FUNCTION = "VRAMdebug"
    CATEGORY = "VTS"
    DESCRIPTION = """
    Returns the inputs unchanged, they are only used as triggers,  
    and performs comfy model management functions and garbage collection,  
    reports free VRAM before and after the operations.
    """

    def trim_memory(self):
        try:
            import ctypes
            import os
            if os.name == 'posix':  # For Linux/Unix systems
                libc = ctypes.CDLL("libc.so.6")
                return libc.malloc_trim(0)
            elif os.name == 'nt':  # For Windows systems
                kernel32 = ctypes.WinDLL("kernel32.dll")
                process = kernel32.GetCurrentProcess()
                result = kernel32.SetProcessWorkingSetSize(process, -1, -1)
                return result != 0  # Returns True if successful
        except Exception as e:
            # Fallback for systems that may not support these methods
            print(f"VTS_Clear_Ram: Trim memory failure: {e}")
            return False

    def VRAMdebug(self,
        gc_collect,
        empty_cache,
        unload_all_models,
        image_pass=None,
        model_pass=None,
        any_input=None,
        any_input2=None,
        any_input3=None,
        any_input4=None,
        any_input5=None,
        any_input6=None,
        any_input7=None,
        any_input8=None,
        any_input9=None
        ):
        freemem_before = model_management.get_free_memory()
        freeram_before = model_management.get_free_memory(torch.device('cpu'))
        
        if empty_cache:
            model_management.soft_empty_cache()
            # torch.device('cpu').empty_cache()
        if unload_all_models:
            model_management.unload_all_models()
            model_management.free_memory(1e30, model_management.unet_offload_device())
            model_management.free_memory(1e30, torch.device('cpu'))
        if gc_collect:
            for _ in range(3):  # Run garbage collection multiple times
                self.trim_memory()
                gc.collect()
            
            self.trim_memory()

            # Handle CUDA memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # More aggressive memory cleanup
                # Force a synchronization point
                torch.cuda.synchronize()
                
                # Try to defragment memory
                if hasattr(torch.cuda, 'caching_allocator_delete_caches'):
                    torch.cuda.caching_allocator_delete_caches()

        freemem_after = model_management.get_free_memory()
        freeram_after = model_management.get_free_memory(torch.device('cpu'))
        print("VTS_Clear_Ram: free vram before: ", f"{freemem_before:,.0f}")
        print("VTS_Clear_Ram: free vram after: ", f"{freemem_after:,.0f}")
        print("VTS_Clear_Ram: freed vram: ", f"{freemem_after - freemem_before:,.0f}")

        print("VTS_Clear_Ram: free RAM before: ", f"{freeram_before:,.0f}")
        print("VTS_Clear_Ram: free RAM after: ", f"{freeram_after:,.0f}")
        print("VTS_Clear_Ram: freed RAM: ", f"{freeram_after - freeram_before:,.0f}")
        return {"ui": {
            "text": [f"{freemem_before:,.0f}x{freemem_after:,.0f}"]}, 
            "result": (
                any_input,
                any_input2,
                any_input3,
                any_input4,
                any_input5,
                any_input6,
                any_input7,
                any_input8,
                any_input9,
                image_pass,
                model_pass,
                freemem_before,
                freemem_after) 
        }


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Clear Ram": VTS_Clear_Ram
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Clear Ram": "VTS Clear Ram"
}
