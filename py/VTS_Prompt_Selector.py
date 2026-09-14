class VTS_Prompt_Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Text containing items separated by the delimiter. Surrounding whitespace and empty items are removed.",
                    },
                ),
                "delimiter": (
                    "STRING",
                    {
                        "default": "|",
                        "multiline": False,
                        "tooltip": "Delimiter used to split the text. Use \\n for newline. An empty delimiter keeps the text as one item.",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "tooltip": "Zero-based item index: 0 selects the first item. Indices past the end select the last item.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_prompt"
    CATEGORY = "VTS/text"
    DESCRIPTION = "Splits text like VTS Prompt Batcher and returns one item by index, repeating the last item past the end."

    def select_prompt(self, text="", delimiter="|", index=0):
        delimiter = delimiter.replace("\\n", "\n")
        if delimiter == "":
            items = [text.strip()] if text.strip() else []
        else:
            items = [part.strip() for part in text.split(delimiter) if part.strip()]

        if not items:
            return ("",)
        return (items[min(max(index, 0), len(items) - 1)],)


NODE_CLASS_MAPPINGS = {
    "VTS Prompt Selector": VTS_Prompt_Selector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Prompt Selector": "VTS Prompt Selector",
}
