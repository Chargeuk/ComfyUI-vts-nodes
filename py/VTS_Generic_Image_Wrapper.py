import os
import sys
from collections import Counter

import torch
from comfy import model_management
from comfy_api.latest import io
import nodes as core_nodes

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, save_images, vtsImageTypes


VTS_WRAPPER_RETURN_TYPES = ["Input", "Tensor", "DiskImage"]
_NO_SUPPORTED_KEY = "__no_supported_nodes__"
_DYNAMIC_ANCHOR_INPUT = "__vts_dynamic_anchor"

_OBJECT_IO_MAP = {
    "IMAGE": io.Image,
    "MASK": io.Mask,
    "LATENT": io.Latent,
    "CONDITIONING": io.Conditioning,
    "MODEL": io.Model,
    "VAE": io.Vae,
    "CLIP": io.Clip,
    "CONTROL_NET": io.ControlNet,
    "UPSCALE_MODEL": io.UpscaleModel,
    "SIGMAS": io.Sigmas,
    "NOISE": io.Noise,
    "SAMPLER": io.Sampler,
    "GUIDER": io.Guider,
    "AUDIO": io.Audio,
}

_SUPPORTED_WIDGET_TYPES = {"BOOLEAN", "INT", "FLOAT", "STRING"}


def _is_builtin_node(node_cls):
    module_name = getattr(node_cls, "__module__", "")
    if module_name == "nodes":
        return True
    if module_name.startswith("comfy_extras."):
        return True
    return False


def _gather_node_mappings():
    mappings = {}
    display_mappings = {}

    node_mappings = getattr(core_nodes, "NODE_CLASS_MAPPINGS", {})
    display_name_mappings = getattr(core_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for node_name, node_cls in node_mappings.items():
        if not _is_builtin_node(node_cls):
            continue
        mappings[node_name] = node_cls
        if node_name in display_name_mappings:
            display_mappings[node_name] = display_name_mappings[node_name]

    return mappings, display_mappings


def _get_legacy_input_config(node_cls):
    input_types = getattr(node_cls, "INPUT_TYPES", None)
    if input_types is None or not callable(input_types):
        return None
    try:
        return input_types()
    except TypeError:
        return None
    except Exception:
        return None


def _normalize_config_value(config, key, default=None):
    value = config.get(key, default)
    return value


def _convert_legacy_input(input_name, legacy_spec, optional=False):
    if not isinstance(legacy_spec, (tuple, list)) or len(legacy_spec) == 0:
        return None

    raw_type = legacy_spec[0]
    config = legacy_spec[1] if len(legacy_spec) > 1 and isinstance(legacy_spec[1], dict) else {}

    tooltip = _normalize_config_value(config, "tooltip")
    lazy = _normalize_config_value(config, "lazy")
    advanced = _normalize_config_value(config, "advanced")
    default = _normalize_config_value(config, "default")

    if isinstance(raw_type, (list, tuple)):
        options = list(raw_type)
        if not all(isinstance(option, (str, int)) for option in options):
            return None
        return io.Combo.Input(
            input_name,
            options=options,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            default=default,
            advanced=advanced,
        )

    if raw_type == "BOOLEAN":
        return io.Boolean.Input(
            input_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            default=default,
            advanced=advanced,
        )

    if raw_type == "INT":
        return io.Int.Input(
            input_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            default=default,
            min=_normalize_config_value(config, "min"),
            max=_normalize_config_value(config, "max"),
            step=_normalize_config_value(config, "step"),
            advanced=advanced,
        )

    if raw_type == "FLOAT":
        return io.Float.Input(
            input_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            default=default,
            min=_normalize_config_value(config, "min"),
            max=_normalize_config_value(config, "max"),
            step=_normalize_config_value(config, "step"),
            round=_normalize_config_value(config, "round"),
            advanced=advanced,
        )

    if raw_type == "STRING":
        return io.String.Input(
            input_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            default=default,
            multiline=bool(_normalize_config_value(config, "multiline", False)),
            placeholder=_normalize_config_value(config, "placeholder"),
            dynamic_prompts=_normalize_config_value(config, "dynamicPrompts"),
            advanced=advanced,
        )

    io_cls = _OBJECT_IO_MAP.get(raw_type)
    if io_cls is not None:
        return io_cls.Input(
            input_name,
            optional=optional,
            tooltip=tooltip,
            lazy=lazy,
            advanced=advanced,
        )

    return None


def _build_wrappable_specs():
    mappings, display_mappings = _gather_node_mappings()
    specs = {}

    for node_name, node_cls in mappings.items():
        if node_name.startswith("VTS "):
            continue
        if node_name in {"VTS Generic Image Wrapper", "VTSGenericImageWrapper", "VTS Prompt Batcher"}:
            continue

        input_config = _get_legacy_input_config(node_cls)
        if not isinstance(input_config, dict):
            continue

        if getattr(node_cls, "INPUT_IS_LIST", False):
            continue
        output_is_list = getattr(node_cls, "OUTPUT_IS_LIST", False)
        if output_is_list not in (False, None):
            continue
        if getattr(node_cls, "OUTPUT_NODE", False):
            continue

        return_types = getattr(node_cls, "RETURN_TYPES", None)
        if not isinstance(return_types, (tuple, list)) or tuple(return_types) != ("IMAGE",):
            continue

        function_name = getattr(node_cls, "FUNCTION", None)
        if not isinstance(function_name, str):
            continue

        required_inputs = input_config.get("required", {})
        optional_inputs = input_config.get("optional", {})
        hidden_inputs = input_config.get("hidden", {})
        if hidden_inputs:
            continue

        option_inputs = [
            io.String.Input(
                _DYNAMIC_ANCHOR_INPUT,
                default="",
                socketless=True,
                extra_dict={"hidden": True},
            )
        ]
        image_input_names = []
        supported = True

        for group_name, group_inputs in (("required", required_inputs), ("optional", optional_inputs)):
            for input_name, legacy_spec in group_inputs.items():
                v3_input = _convert_legacy_input(
                    input_name,
                    legacy_spec,
                    optional=(group_name == "optional"),
                )
                if v3_input is None:
                    supported = False
                    break
                option_inputs.append(v3_input)

                raw_type = legacy_spec[0] if isinstance(legacy_spec, (tuple, list)) and len(legacy_spec) > 0 else None
                if raw_type == "IMAGE":
                    image_input_names.append(input_name)
            if not supported:
                break

        if not supported:
            continue

        display_name = display_mappings.get(node_name, node_name)
        specs[node_name] = {
            "node_name": node_name,
            "display_name": display_name,
            "class": node_cls,
            "function_name": function_name,
            "option_inputs": option_inputs,
            "image_input_names": set(image_input_names),
            "first_image_input_name": image_input_names[0] if image_input_names else None,
            "all_input_names": [v3_input.id for v3_input in option_inputs],
        }

    return specs


def _build_option_keys(specs):
    display_counts = Counter(spec["display_name"] for spec in specs.values())
    option_keys = {}
    for node_name, spec in sorted(specs.items(), key=lambda item: item[1]["display_name"].lower()):
        display_name = spec["display_name"]
        if display_counts[display_name] > 1 or display_name != node_name:
            key = f"{display_name} [{node_name}]"
        else:
            key = display_name
        option_keys[key] = node_name
    return option_keys


class VTS_Generic_Image_Wrapper(io.ComfyNode):
    _cached_specs = None
    _cached_option_keys = None

    @classmethod
    def _get_specs(cls):
        if cls._cached_specs is None or cls._cached_option_keys is None:
            cls._cached_specs = _build_wrappable_specs()
            cls._cached_option_keys = _build_option_keys(cls._cached_specs)
        return cls._cached_specs, cls._cached_option_keys

    @classmethod
    def define_schema(cls):
        specs, option_keys = cls._get_specs()

        options = []
        for option_key, node_name in option_keys.items():
            spec = specs[node_name]
            options.append(io.DynamicCombo.Option(option_key, spec["option_inputs"]))

        if not options:
            options = [io.DynamicCombo.Option(_NO_SUPPORTED_KEY, [])]

        return io.Schema(
            node_id="VTSGenericImageWrapper",
            display_name="VTS Generic Image Wrapper",
            category="VTS/image",
            description=(
                "Wrap a safe subset of old-style image nodes so DiskImage inputs can be materialized automatically "
                "and the single image output can optionally be written back to disk as a DiskImage."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "wrapped_node",
                    options=options,
                    display_name="Wrapped Node",
                    tooltip="Select a supported image node to wrap.",
                ),
                io.Combo.Input(
                    "return_type",
                    options=VTS_WRAPPER_RETURN_TYPES,
                    default="Input",
                    tooltip="Choose Tensor or DiskImage explicitly, or use Input to match the first IMAGE input when the wrapped node has one. If there is no IMAGE input, Input falls back to Tensor.",
                ),
                io.String.Input("prefix", default="generic_wrapper", tooltip="Filename prefix to use when writing DiskImage output."),
                io.Int.Input("start_sequence", default=0, min=0),
                io.String.Input("output_dir", default=default_output_dir),
                io.Combo.Input("format", options=vtsImageTypes, default=vtsImageTypes[0]),
                io.Int.Input("num_workers", default=16, min=1),
                io.Int.Input("compression_level", default=9, min=0, max=9, tooltip="Image compression level (0-9 for png and 0-6 for WebP)."),
                io.Int.Input("quality", default=95, min=1, max=101, tooltip="Image quality (1-100), or 101 for lossless. Only affects WebP."),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @staticmethod
    def _resolve_return_type(spec, wrapped_node, requested_return_type):
        if requested_return_type != "Input":
            return requested_return_type

        first_image_input_name = spec.get("first_image_input_name")
        if not first_image_input_name:
            return "Tensor"

        input_value = wrapped_node.get(first_image_input_name)
        if isinstance(input_value, DiskImage):
            return "DiskImage"

        return "Tensor"

    @classmethod
    def execute(
        cls,
        wrapped_node,
        return_type,
        prefix,
        start_sequence,
        output_dir,
        format,
        num_workers,
        compression_level,
        quality,
    ) -> io.NodeOutput:
        specs, option_keys = cls._get_specs()
        selected_key = wrapped_node.get("wrapped_node")

        if selected_key == _NO_SUPPORTED_KEY or selected_key not in option_keys:
            raise ValueError("No supported wrapped node is currently available.")

        node_name = option_keys[selected_key]
        spec = specs[node_name]
        resolved_return_type = cls._resolve_return_type(spec, wrapped_node, return_type)
        node_kwargs = {}
        materialized_inputs = []

        for input_name in spec["all_input_names"]:
            if input_name == _DYNAMIC_ANCHOR_INPUT:
                continue
            if input_name not in wrapped_node:
                continue
            value = wrapped_node[input_name]
            if input_name in spec["image_input_names"] and isinstance(value, DiskImage):
                value = value.materialize()
                materialized_inputs.append(value)
            node_kwargs[input_name] = value

        node_instance = spec["class"]()
        node_function = getattr(node_instance, spec["function_name"])

        try:
            result = node_function(**node_kwargs)
            if not isinstance(result, tuple):
                result = (result,)

            if len(result) == 0:
                raise ValueError(f"Wrapped node '{node_name}' returned no outputs.")

            image = result[0]

            if resolved_return_type == "Tensor":
                return io.NodeOutput(image)

            saved_paths = save_images(
                image=image,
                prefix=prefix,
                start_sequence=start_sequence,
                output_dir=output_dir,
                format=format,
                num_workers=num_workers,
                compression_level=compression_level,
                quality=None if quality > 100 else quality,
            )

            disk_image = DiskImage(
                prefix=prefix,
                start_sequence=start_sequence,
                number_of_images=len(saved_paths),
                output_dir=output_dir,
                format=format,
                image=image,
                compression_level=compression_level,
                quality=None if quality > 100 else quality,
            )

            del image
            model_management.soft_empty_cache()
            return io.NodeOutput(disk_image)
        finally:
            for materialized in materialized_inputs:
                del materialized
            if materialized_inputs:
                model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "VTSGenericImageWrapper": VTS_Generic_Image_Wrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSGenericImageWrapper": "VTS Generic Image Wrapper",
}
