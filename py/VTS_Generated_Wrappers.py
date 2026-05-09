import copy
import inspect
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import torch
from comfy import model_management
import nodes as core_nodes

import_dir = os.path.join(os.path.dirname(__file__), "vtsUtils")
if import_dir not in sys.path:
    sys.path.append(import_dir)

from vtsUtils import DiskImage, default_output_dir, save_images, vtsImageTypes


_ALLOW_ALL_CUSTOM_NODES = "*"
_SUPPORTED_EXTERNAL_PACKAGES = {
    "donutnodes": {
        "DonutGammaCorrection",
        "DonutAutoWhiteBalance",
        "DonutHistogramStretch",
        "DonutHiRaLoAm",
        "DonutCAS",
    },
    "comfyui-kjnodes": _ALLOW_ALL_CUSTOM_NODES,
}

_MAX_INPUT_COUNT = 12
_MAX_COMBO_OPTIONS = 128
_MAX_OUTPUT_COUNT = 8
_REGISTRATION_ATTEMPTS = 120
_REGISTRATION_DELAY_SECONDS = 0.5
_SAFE_SHARED_CONFIG_KEYS = {"tooltip", "lazy", "advanced"}
_SAFE_COMBO_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default"}
_SAFE_INT_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "min", "max", "step"}
_SAFE_FLOAT_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "min", "max", "step", "round"}
_SAFE_STRING_CONFIG_KEYS = _SAFE_SHARED_CONFIG_KEYS | {"default", "multiline", "placeholder", "dynamicPrompts"}

_OUTPUT_CONTROL_PREFIX = "vts_"
_OUTPUT_CONTROL_SPECS_SINGLE_INPUT = {
    "vts_return_type": (["Input", "Tensor", "DiskImage"], {"default": "Input", "tooltip": "Return images as tensors, DiskImages, or match the single IMAGE input."}),
    "vts_prefix": ("STRING", {"default": "generated_wrapper", "multiline": False}),
    "vts_start_sequence": ("INT", {"default": 0, "min": 0}),
    "vts_output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
    "vts_format": (vtsImageTypes, {"default": vtsImageTypes[0]}),
    "vts_num_workers": ("INT", {"default": 16, "min": 1}),
    "vts_compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "tooltip": "Image compression level (0-9 for png and 0-6 for WebP)."}),
    "vts_quality": ("INT", {"default": 95, "min": 1, "max": 101, "tooltip": "Image quality (1-100), or 101 for lossless. Only affects WebP."}),
}
_OUTPUT_CONTROL_SPECS_MULTI_OR_NONE = {
    **_OUTPUT_CONTROL_SPECS_SINGLE_INPUT,
    "vts_return_type": (["Tensor", "DiskImage"], {"default": "Tensor", "tooltip": "Return image outputs as tensors or save them as DiskImages."}),
}


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


def _is_allowed_wrapper_source(node_name, node_cls):
    if _is_builtin_or_extra_node(node_cls):
        return True

    custom_folder = _get_custom_node_folder_name(node_cls)
    if not custom_folder:
        return False
    allowed_nodes = _SUPPORTED_EXTERNAL_PACKAGES.get(custom_folder)
    if allowed_nodes == _ALLOW_ALL_CUSTOM_NODES:
        return True
    return node_name in allowed_nodes if allowed_nodes else False


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
    return config.get(key, default)


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
    if isinstance(raw_type, str):
        return _is_safe_object_spec(config)
    return False


def _sanitize_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "Node"


def _sanitize_prefix_part(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "image"


def _resolve_return_names(node_cls, return_types):
    return_names = getattr(node_cls, "RETURN_NAMES", None)
    if isinstance(return_names, (tuple, list)) and len(return_names) == len(return_types):
        return tuple(str(name) for name in return_names)
    return tuple(f"output_{index + 1}" for index in range(len(return_types)))


def _get_node_category(node_cls):
    category = getattr(node_cls, "CATEGORY", None)
    if isinstance(category, str) and category.strip():
        return category.strip()
    return "uncategorized"


def _resolve_return_type(spec, requested_return_type, node_kwargs):
    if requested_return_type != "Input":
        return requested_return_type

    if spec["image_input_count"] != 1:
        return "Tensor"

    first_image_input_name = spec["first_image_input_name"]
    if not first_image_input_name:
        return "Tensor"

    input_value = node_kwargs.get(first_image_input_name)
    return "DiskImage" if isinstance(input_value, DiskImage) else "Tensor"


def _make_disk_image(image, prefix, start_sequence, output_dir, format_name, compression_level, quality):
    return DiskImage(
        prefix=prefix,
        start_sequence=start_sequence,
        number_of_images=image.shape[0],
        output_dir=output_dir,
        format=format_name,
        image=image,
        compression_level=compression_level,
        quality=quality,
    )


def _process_image_outputs(spec, outputs, resolved_return_type, prefix, start_sequence, output_dir, format_name, num_workers, compression_level, quality):
    if resolved_return_type == "Tensor":
        return tuple(outputs)

    processed_outputs = list(outputs)
    normalized_quality = None if quality is not None and quality > 100 else quality

    for output_index in spec["image_output_indexes"]:
        image = processed_outputs[output_index]
        if not isinstance(image, torch.Tensor):
            continue

        output_name = spec["return_names"][output_index]
        if len(spec["image_output_indexes"]) > 1:
            image_prefix = f"{prefix}_{_sanitize_prefix_part(output_name)}"
        else:
            image_prefix = prefix

        save_images(
            image=image,
            prefix=image_prefix,
            start_sequence=start_sequence,
            output_dir=output_dir,
            format=format_name,
            num_workers=num_workers,
            compression_level=compression_level,
            quality=normalized_quality,
        )

        processed_outputs[output_index] = _make_disk_image(
            image=image,
            prefix=image_prefix,
            start_sequence=start_sequence,
            output_dir=output_dir,
            format_name=format_name,
            compression_level=compression_level,
            quality=normalized_quality,
        )

    model_management.soft_empty_cache()
    return tuple(processed_outputs)


def _execute_wrapped_node(spec, kwargs):
    image_controls = {}
    if spec["has_image_output"]:
        for key in _OUTPUT_CONTROL_SPECS_SINGLE_INPUT.keys():
            image_controls[key] = kwargs.pop(key)

    node_kwargs = {}
    materialized_inputs = []

    for input_name in spec["all_input_names"]:
        if input_name not in kwargs:
            continue

        value = kwargs[input_name]
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

        if not spec["has_image_output"]:
            return result

        resolved_return_type = _resolve_return_type(spec, image_controls["vts_return_type"], kwargs)
        if resolved_return_type == "Tensor":
            return result

        return _process_image_outputs(
            spec,
            result,
            resolved_return_type,
            image_controls["vts_prefix"],
            image_controls["vts_start_sequence"],
            image_controls["vts_output_dir"],
            image_controls["vts_format"],
            image_controls["vts_num_workers"],
            image_controls["vts_compression_level"],
            image_controls["vts_quality"],
        )
    finally:
        for materialized in materialized_inputs:
            del materialized
        if materialized_inputs:
            model_management.soft_empty_cache()


def _build_wrapper_specs():
    node_mappings = dict(getattr(core_nodes, "NODE_CLASS_MAPPINGS", {}))
    display_name_mappings = dict(getattr(core_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    specs = []

    for node_name, node_cls in node_mappings.items():
        if not _is_allowed_wrapper_source(node_name, node_cls):
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

        function_name = getattr(node_cls, "FUNCTION", None)
        if not isinstance(function_name, str):
            continue

        return_types = getattr(node_cls, "RETURN_TYPES", None)
        if not isinstance(return_types, (tuple, list)) or len(return_types) == 0 or len(return_types) > _MAX_OUTPUT_COUNT:
            continue
        if not all(isinstance(output_type, str) and output_type.strip() for output_type in return_types):
            continue

        required_inputs = input_config.get("required", {})
        optional_inputs = input_config.get("optional", {})
        hidden_inputs = input_config.get("hidden", {})
        if hidden_inputs:
            continue
        if len(required_inputs) + len(optional_inputs) > _MAX_INPUT_COUNT:
            continue

        image_input_names = []
        all_input_names = []
        safe = True
        for group_inputs in (required_inputs, optional_inputs):
            for input_name, legacy_spec in group_inputs.items():
                if not _is_safe_legacy_spec(legacy_spec):
                    safe = False
                    break
                raw_type = legacy_spec[0] if isinstance(legacy_spec, (tuple, list)) and legacy_spec else None
                if raw_type == "IMAGE":
                    image_input_names.append(input_name)
                all_input_names.append(input_name)
            if not safe:
                break
        if not safe:
            continue

        image_output_indexes = [index for index, output_type in enumerate(return_types) if output_type == "IMAGE"]
        if not image_input_names and not image_output_indexes:
            continue

        display_name = display_name_mappings.get(node_name, node_name)
        package_name = _get_custom_node_folder_name(node_cls) or (
            "builtins" if getattr(node_cls, "__module__", "") == "nodes" else "comfy_extras"
        )
        return_names = _resolve_return_names(node_cls, tuple(return_types))

        specs.append({
            "node_name": node_name,
            "display_name": display_name,
            "package": package_name,
            "category": _get_node_category(node_cls),
            "class": node_cls,
            "function_name": function_name,
            "input_config": copy.deepcopy(input_config),
            "all_input_names": list(all_input_names),
            "image_input_names": set(image_input_names),
            "image_input_count": len(image_input_names),
            "first_image_input_name": image_input_names[0] if image_input_names else None,
            "return_types": tuple(return_types),
            "return_names": return_names,
            "image_output_indexes": image_output_indexes,
            "has_image_output": bool(image_output_indexes),
        })

    return specs


def _build_input_types(spec, wrapper_display_name):
    input_types = copy.deepcopy(spec["input_config"])
    input_types.setdefault("required", {})
    input_types.setdefault("optional", {})

    if spec["has_image_output"]:
        controls = (
            _OUTPUT_CONTROL_SPECS_SINGLE_INPUT
            if spec["image_input_count"] == 1
            else _OUTPUT_CONTROL_SPECS_MULTI_OR_NONE
        )
        for key, value in controls.items():
            input_types["required"][key] = copy.deepcopy(value)
        input_types["required"]["vts_prefix"] = (
            "STRING",
            {
                **input_types["required"]["vts_prefix"][1],
                "default": re.sub(r"\s+", "_", wrapper_display_name.strip()),
            },
        )

    return input_types


def _create_wrapper_class(spec):
    wrapper_display_name = f"VTS {spec['display_name']} Wrapper"
    input_types = _build_input_types(spec, wrapper_display_name)
    category = f"VTS/wrappers/{spec['category']}"
    description = (
        f"VTS-generated wrapper around {spec['display_name']} from {spec['package']}. "
        "IMAGE inputs accept tensors or DiskImages."
    )
    if spec["has_image_output"]:
        description += " IMAGE outputs can be returned as tensors or written to disk as DiskImages."

    @classmethod
    def INPUT_TYPES(cls):
        return copy.deepcopy(input_types)

    def execute(self, **kwargs):
        return _execute_wrapped_node(spec, kwargs)

    attrs = {
        "INPUT_TYPES": INPUT_TYPES,
        "RETURN_TYPES": spec["return_types"],
        "RETURN_NAMES": spec["return_names"],
        "FUNCTION": "execute",
        "CATEGORY": category,
        "DESCRIPTION": description,
        "execute": execute,
    }

    class_name = _sanitize_identifier(f"VTSWrapper_{spec['package']}_{spec['node_name']}")
    return type(class_name, (), attrs), wrapper_display_name


def _build_generated_mappings():
    node_class_mappings = {}
    display_name_mappings = {}

    for spec in _build_wrapper_specs():
        wrapper_cls, wrapper_display_name = _create_wrapper_class(spec)
        wrapper_node_id = _sanitize_identifier(f"VTSWrapper_{spec['package']}_{spec['node_name']}")
        node_class_mappings[wrapper_node_id] = wrapper_cls
        display_name_mappings[wrapper_node_id] = wrapper_display_name

    return node_class_mappings, display_name_mappings


def _register_generated_wrappers_late():
    registered_ids = set()

    for _ in range(_REGISTRATION_ATTEMPTS):
        try:
            node_class_mappings, display_name_mappings = _build_generated_mappings()
        except Exception:
            logging.exception("VTS generated wrapper registration pass failed; retrying.")
            time.sleep(_REGISTRATION_DELAY_SECONDS)
            continue
        pending_ids = [node_id for node_id in node_class_mappings if node_id not in registered_ids]

        if pending_ids:
            core_nodes.NODE_CLASS_MAPPINGS.update(node_class_mappings)
            core_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
            registered_ids.update(pending_ids)

            # If we have already found at least one wrapper, give the remaining
            # custom nodes a few more chances to appear and then stop.
            if registered_ids:
                for _ in range(6):
                    time.sleep(_REGISTRATION_DELAY_SECONDS)
                    try:
                        node_class_mappings, display_name_mappings = _build_generated_mappings()
                    except Exception:
                        logging.exception("VTS generated wrapper late-registration pass failed; retrying.")
                        continue
                    late_ids = [node_id for node_id in node_class_mappings if node_id not in registered_ids]
                    if not late_ids:
                        continue
                    core_nodes.NODE_CLASS_MAPPINGS.update(node_class_mappings)
                    core_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
                    registered_ids.update(late_ids)
                return

        time.sleep(_REGISTRATION_DELAY_SECONDS)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
threading.Thread(target=_register_generated_wrappers_late, daemon=True).start()
