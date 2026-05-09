import random


class VTS_Random_Prompt_Builder:
    """
    Build multiple semi-random prompts from a template containing optional
    choice blocks in square brackets, then join them into a single
    pipe-delimited string for use with VTS Prompt Batcher.
    """

    _last_used_seed_by_node = {}
    _pending_seed_by_node = {}
    _random_source = random.SystemRandom()

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
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1125899906842624,
                        "max": 1125899906842624,
                    },
                ),
                "seed_mode": (
                    ["fixed", "increment", "decrement", "randomise"],
                    {
                        "default": "fixed",
                        "tooltip": "fixed reuses the same seed and stays cached when inputs do not change. increment/decrement advance the seed on each execution. randomise chooses a fresh seed on each execution.",
                    },
                ),
            }
            ,
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "build_prompts"
    CATEGORY = "VTS/text"

    @classmethod
    def _node_key(cls, unique_id):
        return str(unique_id) if unique_id is not None else "__default__"

    @classmethod
    def _new_random_seed(cls):
        return cls._random_source.randint(1, 1125899906842624)

    @classmethod
    def _resolve_seed_for_mode(cls, seed, seed_mode, unique_id):
        key = cls._node_key(unique_id)
        last_used_seed = cls._last_used_seed_by_node.get(key)

        if seed_mode == "fixed":
            cls._pending_seed_by_node.pop(key, None)
            return seed
        if seed_mode == "increment":
            return seed if last_used_seed is None else last_used_seed + 1
        if seed_mode == "decrement":
            return seed if last_used_seed is None else last_used_seed - 1
        return cls._new_random_seed()

    @classmethod
    def _consume_execution_seed(cls, seed, seed_mode, unique_id):
        key = cls._node_key(unique_id)
        if key in cls._pending_seed_by_node:
            return cls._pending_seed_by_node.pop(key)
        return cls._resolve_seed_for_mode(seed, seed_mode, unique_id)

    @classmethod
    def _update_seed_metadata(cls, resolved_seed, unique_id, prompt=None, extra_pnginfo=None):
        if unique_id is None:
            return

        unique_id = str(unique_id)

        if isinstance(prompt, dict):
            prompt_node = prompt.get(unique_id)
            if isinstance(prompt_node, dict):
                prompt_inputs = prompt_node.setdefault("inputs", {})
                prompt_inputs["seed"] = resolved_seed

        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow")
            if isinstance(workflow, dict):
                workflow_nodes = workflow.get("nodes", [])
                for node in workflow_nodes:
                    if str(node.get("id")) != unique_id:
                        continue
                    widget_values = node.get("widgets_values")
                    if isinstance(widget_values, list) and len(widget_values) >= 3:
                        widget_values[2] = resolved_seed
                    break

    @classmethod
    def IS_CHANGED(
        cls,
        template_text="",
        number_of_prompts=1,
        seed=0,
        seed_mode="fixed",
        unique_id=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if seed_mode == "fixed":
            return ("fixed", seed)

        resolved_seed = cls._resolve_seed_for_mode(seed, seed_mode, unique_id)
        cls._pending_seed_by_node[cls._node_key(unique_id)] = resolved_seed
        return (seed_mode, resolved_seed)

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

    def _build_single_prompt(self, parsed_parts, rng):
        built = []

        for part_type, value in parsed_parts:
            if part_type == "literal":
                built.append(value)
            else:
                built.append(rng.choice(value))

        return "".join(built).strip()

    def build_prompts(
        self,
        template_text="",
        number_of_prompts=1,
        seed=0,
        seed_mode="fixed",
        unique_id=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if not template_text.strip():
            print("[VTS Random Prompt Builder] Empty template provided, returning empty output")
            return ("",)

        resolved_seed = self._consume_execution_seed(seed, seed_mode, unique_id)
        rng = random.Random(resolved_seed)

        parsed_parts = self._parse_template(template_text)
        prompts = [
            self._build_single_prompt(parsed_parts, rng)
            for _ in range(number_of_prompts)
        ]

        self._last_used_seed_by_node[self._node_key(unique_id)] = resolved_seed
        self._update_seed_metadata(
            resolved_seed,
            unique_id,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )

        result = " | ".join(prompts)

        print(
            f"[VTS Random Prompt Builder] Built {len(prompts)} prompts "
            f"using seed {resolved_seed} ({seed_mode})"
        )
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
