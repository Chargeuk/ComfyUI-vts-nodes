import random


class VTS_Random_Prompt_Builder:
    """
    Build multiple semi-random prompts from a template containing optional
    choice blocks in square brackets, then join them into a single
    pipe-delimited string for use with VTS Prompt Batcher.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Example: a [red|blue|green] car in a [forest|city|desert]",
                        "tooltip": "Text template used to build prompts. Text outside brackets is always included. Inside each [...] block, one pipe-separated option is chosen per prompt.",
                    },
                ),
                "number_of_prompts": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 10000,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "build_prompts"
    CATEGORY = "VTS/text"

    def _parse_template(self, template_text):
        parts = []
        cursor = 0

        while cursor < len(template_text):
            start = template_text.find("[", cursor)
            if start == -1:
                parts.append(("literal", template_text[cursor:]))
                break

            end = template_text.find("]", start + 1)
            if end == -1:
                parts.append(("literal", template_text[cursor:]))
                break

            if start > cursor:
                parts.append(("literal", template_text[cursor:start]))

            block = template_text[start + 1:end]
            options = [option.strip() for option in block.split("|") if option.strip()]

            if options:
                parts.append(("choices", options))
            else:
                parts.append(("literal", ""))

            cursor = end + 1

        if not parts:
            parts.append(("literal", ""))

        return parts

    def _build_single_prompt(self, parsed_parts):
        built = []

        for part_type, value in parsed_parts:
            if part_type == "literal":
                built.append(value)
            else:
                built.append(random.choice(value))

        return "".join(built).strip()

    def build_prompts(self, template_text="", number_of_prompts=1):
        if not template_text.strip():
            print("[VTS Random Prompt Builder] Empty template provided, returning empty output")
            return ("",)

        parsed_parts = self._parse_template(template_text)
        prompts = [
            self._build_single_prompt(parsed_parts)
            for _ in range(number_of_prompts)
        ]

        result = " | ".join(prompts)

        print(f"[VTS Random Prompt Builder] Built {len(prompts)} prompts")
        for i, prompt in enumerate(prompts, 1):
            preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"[VTS Random Prompt Builder]   {i}. {preview}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "VTS Random Prompt Builder": VTS_Random_Prompt_Builder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Random Prompt Builder": "VTS Random Prompt Builder",
}
