"""Explicit DiskLatent integration shared by native VTS nodes and wrappers."""

import copy
import functools
import inspect

from comfy_api.latest import io

from vts_disk_latent import (
    latent_controls, latent_output_mode, materialize_latents,
    process_latent_outputs, take_latent_controls,
)


def v3_latent_controls(specs):
    result = []
    for name, (kind, options) in specs.items():
        options = {**options, "optional": True}
        if isinstance(kind, list):
            result.append(io.Combo.Input(name, options=kind, **options))
        elif kind == "STRING":
            result.append(io.String.Input(name, **options))
        elif kind == "INT":
            result.append(io.Int.Input(name, **options))
    return result


def transform_latent_result(result, indexes, names, controls, mode):
    if not indexes:
        return result
    if isinstance(result, dict):
        if "result" not in result:
            return result
        outputs = result["result"]
    elif isinstance(result, io.NodeOutput):
        outputs = result.args
    else:
        outputs = result if isinstance(result, tuple) else (result,)
    outputs = process_latent_outputs(outputs, indexes, names, controls, mode)
    if isinstance(result, dict):
        return {**result, "result": outputs}
    if isinstance(result, io.NodeOutput):
        return io.NodeOutput(*outputs, ui=result.ui, expand=result.expand, block_execution=result.block_execution)
    return outputs


def disk_latent_node(*, inputs=(), outputs=(), prefix, output_names=None):
    """Opt a VTS class into native latent I/O without adding another graph node.

    New controls are optional so old API prompts and positional calls still work.
    Only explicitly listed latent inputs are materialized; conditioning stays opaque.
    """
    def decorate(cls):
        specs = latent_controls(prefix, has_input=bool(inputs), has_output=bool(outputs), match_input=len(inputs) == 1)
        is_v3 = issubclass(cls, io.ComfyNode)
        if is_v3:
            original_schema = cls.define_schema.__func__

            @classmethod
            def define_schema(current_cls):
                schema = copy.deepcopy(original_schema(current_cls))
                schema.inputs = [item for item in schema.inputs if item.id not in specs]
                schema.inputs.extend(v3_latent_controls(specs))
                return schema

            cls.define_schema = define_schema
        else:
            original_inputs = cls.INPUT_TYPES.__func__

            @classmethod
            def input_types(current_cls):
                schema = copy.deepcopy(original_inputs(current_cls))
                schema.setdefault("optional", {}).update(copy.deepcopy(specs))
                return schema

            cls.INPUT_TYPES = input_types

        function_name = "execute" if is_v3 else cls.FUNCTION
        method = inspect.getattr_static(cls, function_name)
        is_classmethod = isinstance(method, classmethod)
        original = method.__func__ if is_classmethod else method
        original = getattr(original, "_vts_latent_original", original)
        signature = inspect.signature(original)
        extra_name = next((name for name, p in signature.parameters.items()
                           if p.kind == inspect.Parameter.VAR_KEYWORD), None)
        names = output_names or getattr(cls, "RETURN_NAMES", None) or tuple(
            f"latent_{i}" for i in range(max(outputs, default=-1) + 1))

        @functools.wraps(original)
        def execute(instance, *args, **kwargs):
            controls = take_latent_controls(kwargs, specs, input_is_list=getattr(cls, "INPUT_IS_LIST", False))
            bound = signature.bind(instance, *args, **kwargs)
            extra = bound.arguments.get(extra_name, {})
            values = [bound.arguments.get(name, extra.get(name)) for name in inputs]
            mode = latent_output_mode(controls, values) if outputs else "Tensor"
            memo = {}
            for name in inputs:
                target = bound.arguments if name in bound.arguments else extra
                if name in target:
                    target[name] = materialize_latents(target[name], controls.get("device_policy", "Original"),
                                                      controls.get("device", "cpu"), memo)
            result = original(*bound.args, **bound.kwargs)
            return transform_latent_result(result, outputs, names, controls, mode)

        execute._vts_latent_original = original
        setattr(cls, function_name, classmethod(execute) if is_classmethod else execute)
        cls.VTS_DISK_LATENT_SUPPORT = {"inputs": inputs, "outputs": outputs}
        return cls
    return decorate
