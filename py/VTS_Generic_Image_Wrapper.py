import os
import sys
import inspect
from collections import Counter
from pathlib import Path

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
_NO_FILTER_VALUE = "All"
_DYNAMIC_ANCHOR_INPUT = "__vts_dynamic_anchor"
_WRAPPED_INPUT_PREFIX = "wrapped__"
_MAX_INPUT_COUNT = 12
_MAX_COMBO_OPTIONS = 128
_SAFE_SHARED_CONFIG_KEYS = {"tooltip", "lazy", "advanced"}
_SAFE_COMBO_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default"}
_SAFE_INT_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "min", "max", "step"}
_SAFE_FLOAT_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "min", "max", "step", "round"}
_SAFE_STRING_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "multiline", "placeholder", "dynamicPrompts"}
_CURATED_CUSTOM_NODE_ALLOWLIST = {
    "donutnodes": {
        "DonutGammaCorrection",
        "DonutAutoWhiteBalance",
        "DonutHistogramStretch",
        "DonutHiRaLoAm",
        "DonutCAS",
    }
}

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


def _is_builtin_or_extra_node(node_cls):
    module_name = getattr(node_cls, "__module__", "") or ""
    return module_name == "nodes" or module_name.startswith("comfy_extras.")


def _get_custom_node_folder_name(node_cls):
    try:
        source_file = inspect.getfile(node_cls)
    except (TypeError, OSError):
        return None

    source_parts = Path(source_file).parts
    if "custom_nodes" in source_parts:
        custom_index = source_parts.index("custom_nodes")
        if custom_index + 1 < len(source_parts):
            return source_parts[custom_index + 1]
    return None


def _is_curated_custom_node(node_name, node_cls):
    custom_folder = _get_custom_node_folder_name(node_cls)
    if not custom_folder:
        return False
    allowed_nodes = _CURATED_CUSTOM_NODE_ALLOWLIST.get(custom_folder)
    return node_name in allowed_nodes if allowed_nodes else False


def _gather_node_mappings(include_custom=False):
    mappings = {}
    display_mappings = {}

    node_mappings = getattr(core_nodes, "NODE_CLASS_MAPPINGS", {})
    display_name_mappings = getattr(core_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    for node_name, node_cls in node_mappings.items():
        if not include_custom:
            if not _is_builtin_or_extra_node(node_cls) and not _is_curated_custom_node(node_name, node_cls):
                continue
        mappings[node_name] = node_cls
        if node_name in display_name_mappings:
            display_mappings[node_name] = display_name_mappings[node_name]

    return mappings, display_mappings


def _get_node_category(node_cls):
    category = getattr(node_cls, "CATEGORY", None)
    if isinstance(category, str) and category.strip():
        return category.strip()
    return "uncategorized"


def _get_node_package(node_cls):
    module_name = getattr(node_cls, "__module__", "") or ""
    if module_name == "nodes":
        return "Built-in"
    if module_name.startswith("comfy_extras."):
        return "Comfy Extras"

    custom_folder = _get_custom_node_folder_name(node_cls)
    if custom_folder:
        return custom_folder

    if module_name:
        return module_name.split(".")[0]
    return "Other"


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


def _is_bool_or_none(value):
    return value is None or isinstance(value, bool)


def _is_number_or_none(value):
    return value is None or isinstance(value, (int, float))


def _is_string_or_none(value):
    return value is None or isinstance(value, str)


def _has_only_known_keys(config, allowed_keys):
    return all(key in allowed_keys for key in config)


def _is_safe_shared_config(config):
    return (
        _is_string_or_none(config.get("tooltip"))
        and _is_bool_or_none(config.get("lazy"))
        and _is_bool_or_none(config.get("advanced"))
    )


def _is_safe_combo_spec(raw_type, config):
    return (
        len(raw_type) <= _MAX_COMBO_OPTIONS
        and all(isinstance(option, (str, int)) for option in raw_type)
        and _has_only_known_keys(config, _SAFE_COMBO_CONFIG_KEYS)
        and _is_safe_shared_config(config)
        and (config.get("default") is None or isinstance(config.get("default"), (str, int)))
    )


def _is_safe_int_spec(config):
    return (
        _has_only_known_keys(config, _SAFE_INT_CONFIG_KEYS)
        and _is_safe_shared_config(config)
        and _is_number_or_none(config.get("default"))
        and _is_number_or_none(config.get("min"))
        and _is_number_or_none(config.get("max"))
        and _is_number_or_none(config.get("step"))
    )


def _is_safe_float_spec(config):
    return (
        _has_only_known_keys(config, _SAFE_FLOAT_CONFIG_KEYS)
        and _is_safe_shared_config(config)
        and _is_number_or_none(config.get("default"))
        and _is_number_or_none(config.get("min"))
        and _is_number_or_none(config.get("max"))
        and _is_number_or_none(config.get("step"))
        and _is_number_or_none(config.get("round"))
    )


def _is_safe_string_spec(config):
    return (
        _has_only_known_keys(config, _SAFE_STRING_CONFIG_KEYS)
        and _is_safe_shared_config(config)
        and _is_string_or_none(config.get("default"))
        and _is_bool_or_none(config.get("multiline"))
        and _is_string_or_none(config.get("placeholder"))
        and _is_bool_or_none(config.get("dynamicPrompts"))
    )


def _is_safe_boolean_spec(config):
    return (
        _has_only_known_keys(config, _SAFE_SHARED_CONFIG_KEYS | {"default"})
        and _is_safe_shared_config(config)
        and _is_bool_or_none(config.get("default"))
    )


def _is_safe_object_spec(config):
    return _has_only_known_keys(config, _SAFE_SHARED_CONFIG_KEYS) and _is_safe_shared_config(config)


def _is_safe_legacy_spec(legacy_spec):
    if not isinstance(legacy_spec, (tuple, list)) or len(legacy_spec) == 0:
        return False

    raw_type = legacy_spec[0]
    config = legacy_spec[1] if len(legacy_spec) > 1 and isinstance(legacy_spec[1], dict) else {}

    if isinstance(raw_type, (list, tuple)):
        return _is_safe_combo_spec(raw_type, config)
    if raw_type == "BOOLEAN":
        return _is_safe_boolean_spec(config)
    if raw_type == "INT":
        return _is_safe_int_spec(config)
    if raw_type == "FLOAT":
        return _is_safe_float_spec(config)
    if raw_type == "STRING":
        return _is_safe_string_spec(config)
    if raw_type in _OBJECT_IO_MAP:
        return _is_safe_object_spec(config)
    return False


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


def _build_wrappable_specs(include_custom=False):
    mappings, display_mappings = _gather_node_mappings(include_custom=include_custom)
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
        if len(required_inputs) + len(optional_inputs) > _MAX_INPUT_COUNT:
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
        legacy_inputs = []
        supported = True

        for group_name, group_inputs in (("required", required_inputs), ("optional", optional_inputs)):
            for input_name, legacy_spec in group_inputs.items():
                if not _is_safe_legacy_spec(legacy_spec):
                    supported = False
                    break
                v3_input = _convert_legacy_input(
                    input_name,
                    legacy_spec,
                    optional=(group_name == "optional"),
                )
                if v3_input is None:
                    supported = False
                    break
                option_inputs.append(v3_input)
                legacy_inputs.append({
                    "name": input_name,
                    "optional": group_name == "optional",
                    "legacy_spec": legacy_spec,
                })

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
            "category": _get_node_category(node_cls),
            "package": _get_node_package(node_cls),
            "option_inputs": option_inputs,
            "legacy_inputs": legacy_inputs,
            "image_input_names": set(image_input_names),
            "image_input_count": len(image_input_names),
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


def _build_filter_options(specs, field_name):
    values = sorted(
        {
            spec[field_name]
            for spec in specs.values()
            if isinstance(spec.get(field_name), str) and spec[field_name].strip()
        },
        key=str.lower,
    )
    return [_NO_FILTER_VALUE] + values


def _serialize_catalog_for_frontend(specs, option_keys):
    catalog = {}
    for option_key, node_name in option_keys.items():
        spec = specs[node_name]
        catalog[option_key] = {
            "node_name": spec["node_name"],
            "display_name": spec["display_name"],
            "category": spec["category"],
            "package": spec["package"],
            "has_image_input": spec["first_image_input_name"] is not None,
        }
    return catalog


def _resolve_requested_return_type(spec, kwargs, requested_return_type):
    if requested_return_type != "Input":
        return requested_return_type

    if spec.get("image_input_count") != 1:
        return "Tensor"

    first_image_input_name = spec.get("first_image_input_name")
    if not first_image_input_name:
        return "Tensor"

    wrapped_key = f"{_WRAPPED_INPUT_PREFIX}{first_image_input_name}"
    input_value = kwargs.get(wrapped_key)
    if isinstance(input_value, DiskImage):
        return "DiskImage"

    return "Tensor"


def _execute_wrapped_node(spec, requested_return_type, prefix, start_sequence, output_dir, format, num_workers, compression_level, quality, kwargs):
    resolved_return_type = _resolve_requested_return_type(spec, kwargs, requested_return_type)
    node_kwargs = {}
    materialized_inputs = []

    for input_name in spec["all_input_names"]:
        if input_name == _DYNAMIC_ANCHOR_INPUT:
            continue

        wrapped_key = f"{_WRAPPED_INPUT_PREFIX}{input_name}"
        if wrapped_key not in kwargs:
            continue

        value = kwargs[wrapped_key]
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
            raise ValueError(f"Wrapped node '{spec['node_name']}' returned no outputs.")

        image = result[0]

        if resolved_return_type == "Tensor":
            return (image,)

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
        return (disk_image,)
    finally:
        for materialized in materialized_inputs:
            del materialized
        if materialized_inputs:
            model_management.soft_empty_cache()


class VTS_Generic_Image_Wrapper(io.ComfyNode):
    _cached_specs = None
    _cached_option_keys = None
    _cached_category_options = None
    _cached_package_options = None

    @classmethod
    def _get_specs(cls):
        if (
            cls._cached_specs is None
            or cls._cached_option_keys is None
            or cls._cached_category_options is None
            or cls._cached_package_options is None
        ):
            cls._cached_specs = _build_wrappable_specs(include_custom=False)
            cls._cached_option_keys = _build_option_keys(cls._cached_specs)
            cls._cached_category_options = _build_filter_options(cls._cached_specs, "category")
            cls._cached_package_options = _build_filter_options(cls._cached_specs, "package")
        return (
            cls._cached_specs,
            cls._cached_option_keys,
            cls._cached_category_options,
            cls._cached_package_options,
        )

    @classmethod
    def define_schema(cls):
        specs, option_keys, category_options, package_options = cls._get_specs()

        options = []
        option_meta = {}
        for option_key, node_name in option_keys.items():
            spec = specs[node_name]
            options.append(io.DynamicCombo.Option(option_key, spec["option_inputs"]))
            option_meta[option_key] = {
                "node_name": spec["node_name"],
                "display_name": spec["display_name"],
                "category": spec["category"],
                "package": spec["package"],
                "has_image_input": spec["first_image_input_name"] is not None,
                "image_input_count": spec["image_input_count"],
            }

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
                io.Combo.Input(
                    "category_filter",
                    options=category_options,
                    default=_NO_FILTER_VALUE,
                    tooltip="Filter wrapped nodes by their ComfyUI category.",
                ),
                io.Combo.Input(
                    "package_filter",
                    options=package_options,
                    default=_NO_FILTER_VALUE,
                    tooltip="Filter wrapped nodes by the package or source they come from.",
                ),
                io.DynamicCombo.Input(
                    "wrapped_node",
                    options=options,
                    display_name="Wrapped Node",
                    tooltip="Select a supported image node to wrap.",
                    extra_dict={"vts_node_meta": option_meta},
                ),
                io.Combo.Input(
                    "return_type",
                    options=VTS_WRAPPER_RETURN_TYPES,
                    default="Input",
                    tooltip="Choose Tensor or DiskImage explicitly. Input is only available for wrapped nodes with exactly one IMAGE input; otherwise it falls back to Tensor.",
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
        category_filter,
        package_filter,
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
        specs, option_keys, _, _ = cls._get_specs()
        selected_key = wrapped_node.get("wrapped_node")

        if selected_key == _NO_SUPPORTED_KEY or selected_key not in option_keys:
            raise ValueError("No supported wrapped node is currently available.")

        node_name = option_keys[selected_key]
        spec = specs[node_name]
        flat_kwargs = {
            f"{_WRAPPED_INPUT_PREFIX}{key}": value
            for key, value in wrapped_node.items()
            if key != "wrapped_node"
        }
        return io.NodeOutput(*_execute_wrapped_node(
            spec,
            return_type,
            prefix,
            start_sequence,
            output_dir,
            format,
            num_workers,
            compression_level,
            quality,
            flat_kwargs,
        ))


class VTS_Generic_Image_Wrapper_V2:
    _cached_specs = None
    _cached_option_keys = None
    _cached_category_options = None
    _cached_package_options = None
    _cached_catalog = None

    @classmethod
    def _get_specs(cls):
        if (
            cls._cached_specs is None
            or cls._cached_option_keys is None
            or cls._cached_category_options is None
            or cls._cached_package_options is None
            or cls._cached_catalog is None
        ):
            # Keep V2 startup-safe for now. In the full ComfyUI process,
            # core_nodes.NODE_CLASS_MAPPINGS already includes loaded custom nodes,
            # and scanning that full live set during INPUT_TYPES()/object_info can
            # stall frontend startup. A future version should fetch broader support
            # lazily through a dedicated backend route instead.
            cls._cached_specs = _build_wrappable_specs(include_custom=False)
            cls._cached_option_keys = _build_option_keys(cls._cached_specs)
            cls._cached_category_options = _build_filter_options(cls._cached_specs, "category")
            cls._cached_package_options = _build_filter_options(cls._cached_specs, "package")
            cls._cached_catalog = _serialize_catalog_for_frontend(cls._cached_specs, cls._cached_option_keys)
        return (
            cls._cached_specs,
            cls._cached_option_keys,
            cls._cached_category_options,
            cls._cached_package_options,
            cls._cached_catalog,
        )

    @classmethod
    def INPUT_TYPES(cls):
        _, option_keys, category_options, package_options, catalog = cls._get_specs()
        wrapped_node_options = list(option_keys.keys()) if option_keys else [_NO_SUPPORTED_KEY]
        return {
            "required": {
                "vts_category_filter": (
                    category_options,
                    {
                        "default": _NO_FILTER_VALUE,
                        "tooltip": "Filter wrapped nodes by their ComfyUI category.",
                    },
                ),
                "vts_package_filter": (
                    package_options,
                    {
                        "default": _NO_FILTER_VALUE,
                        "tooltip": "Filter wrapped nodes by the package or source they come from.",
                    },
                ),
                "vts_wrapped_node_name": (
                    wrapped_node_options,
                    {
                        "default": wrapped_node_options[0],
                        "tooltip": "Select a supported node to wrap.",
                        "vts_node_catalog": catalog,
                    },
                ),
                "vts_return_type": (
                    VTS_WRAPPER_RETURN_TYPES,
                    {
                        "default": "Input",
                        "tooltip": "Choose Tensor or DiskImage explicitly, or use Input to match the first IMAGE input when the wrapped node has one. If there is no IMAGE input, Input falls back to Tensor.",
                    },
                ),
                "prefix": ("STRING", {"default": "generic_wrapper", "tooltip": "Filename prefix to use when writing DiskImage output."}),
                "start_sequence": ("INT", {"default": 0, "min": 0}),
                "output_dir": ("STRING", {"default": default_output_dir}),
                "format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
                "num_workers": ("INT", {"default": 16, "min": 1}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)."}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "VTS/image"

    def execute(
        self,
        vts_category_filter,
        vts_package_filter,
        vts_wrapped_node_name,
        vts_return_type,
        prefix,
        start_sequence,
        output_dir,
        format,
        num_workers,
        compression_level,
        quality,
        **kwargs,
    ):
        specs, option_keys, _, _, _ = self._get_specs()

        if vts_wrapped_node_name == _NO_SUPPORTED_KEY or vts_wrapped_node_name not in option_keys:
            raise ValueError("No supported wrapped node is currently available.")

        node_name = option_keys[vts_wrapped_node_name]
        spec = specs[node_name]

        return _execute_wrapped_node(
            spec,
            vts_return_type,
            prefix,
            start_sequence,
            output_dir,
            format,
            num_workers,
            compression_level,
            quality,
            kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "VTSGenericImageWrapper": VTS_Generic_Image_Wrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTSGenericImageWrapper": "VTS Generic Image Wrapper",
}
