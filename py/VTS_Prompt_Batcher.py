class VTS_Prompt_Batcher:
    """
    Split a single text field into a batch of prompts using a configurable delimiter.
    Each prompt is trimmed and empty prompts are discarded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Enter prompts separated by the delimiter",
                        "tooltip": "A single text block containing multiple prompts separated by the chosen delimiter.",
                    },
                ),
                "delimiter": (
                    "STRING",
                    {
                        "default": "|",
                        "multiline": False,
                        "tooltip": "Delimiter used to split the prompt string. Use \\n for newline.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "batch_prompts"
    CATEGORY = "VTS/text"
    OUTPUT_IS_LIST = (True,)

    def batch_prompts(self, prompts="", delimiter="|"):
        delimiter = delimiter.replace("\\n", "\n")

        if delimiter == "":
            prompt_list = [prompts.strip()] if prompts.strip() else []
        else:
            prompt_list = [part.strip() for part in prompts.split(delimiter) if part.strip()]

        if not prompt_list:
            print("[VTS Prompt Batcher] No prompts provided, returning empty prompt")
            return ([""],)

        print(f"[VTS Prompt Batcher] Batching {len(prompt_list)} prompts:")
        for i, prompt in enumerate(prompt_list, 1):
            preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"[VTS Prompt Batcher]   {i}. {preview}")

        return (prompt_list,)


NODE_CLASS_MAPPINGS = {
    "VTS Prompt Batcher": VTS_Prompt_Batcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Prompt Batcher": "VTS Prompt Batcher",
}
