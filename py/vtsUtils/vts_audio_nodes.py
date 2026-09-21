"""Native and generated node integration for DiskAudio."""
import copy
import functools
import inspect

from comfy_api.latest import io

from vts_disk_audio import (DiskAudio, INPUT_HELP, OUTPUT_HELP, audio_controls,
                            materialize_audio, save_audio)
from vts_latent_nodes import v3_latent_controls


def _contains_audio(item):
    if item.get_io_type() == 'AUDIO':
        return True
    child = getattr(getattr(item, 'template', None), 'input', None)
    return child is not None and _contains_audio(child)


def audio_ports(cls):
    if issubclass(cls, io.ComfyNode):
        schema = cls.define_schema()
        return ([item.id for item in schema.inputs if _contains_audio(item)],
                [i for i, item in enumerate(schema.outputs) if item.get_io_type() == 'AUDIO'])
    schema = cls.INPUT_TYPES()
    return ([name for group in ('required', 'optional') for name, spec in schema.get(group, {}).items()
             if str(spec[0]) == 'AUDIO'],
            [i for i, kind in enumerate(getattr(cls, 'RETURN_TYPES', ())) if str(kind) == 'AUDIO'])


def _disk_values(value):
    if isinstance(value, DiskAudio):
        return [True]
    if isinstance(value, dict) and 'waveform' in value:
        return [False]
    if isinstance(value, dict):
        return [disk for item in value.values() for disk in _disk_values(item)]
    if isinstance(value, (tuple, list)):
        return [disk for item in value for disk in _disk_values(item)]
    return []


def _append(old, new):
    return old if new in (old or '') else ((old + ' ') if old else '') + new


def disk_audio_node(cls, *, prefix=None, control_prefix='audio_', keep_disk_inputs=()):
    if cls.__dict__.get('VTS_DISK_AUDIO_SUPPORT'):
        return cls
    inputs, outputs = audio_ports(cls)
    if not inputs and not outputs:
        return cls
    prefix = prefix or cls.__name__
    specs = audio_controls(prefix, bool(inputs), bool(outputs), control_prefix, len(inputs) == 1)
    is_v3 = issubclass(cls, io.ComfyNode)
    if is_v3:
        original_schema = cls.define_schema.__func__

        @classmethod
        def define_schema(current_cls):
            schema = copy.deepcopy(original_schema(current_cls))
            schema.inputs.extend(v3_latent_controls(specs))
            for item in schema.inputs:
                if item.id in inputs:
                    item.tooltip = _append(item.tooltip, INPUT_HELP)
            for index in outputs:
                schema.outputs[index].tooltip = _append(schema.outputs[index].tooltip, OUTPUT_HELP)
            return schema
        cls.define_schema = define_schema
    else:
        original_inputs = cls.INPUT_TYPES.__func__

        @classmethod
        def input_types(current_cls):
            schema = copy.deepcopy(original_inputs(current_cls))
            schema.setdefault('optional', {}).update(copy.deepcopy(specs))
            for group in ('required', 'optional'):
                for name in inputs:
                    if name in schema.get(group, {}):
                        spec = schema[group][name]
                        config = dict(spec[1]) if len(spec) > 1 else {}
                        config['tooltip'] = _append(config.get('tooltip'), INPUT_HELP)
                        schema[group][name] = (spec[0], config, *spec[2:])
            return schema
        cls.INPUT_TYPES = input_types
        tips = list(getattr(cls, 'OUTPUT_TOOLTIPS', ()) or ())
        tips += [''] * (len(cls.RETURN_TYPES) - len(tips))
        for index in outputs:
            tips[index] = _append(tips[index], OUTPUT_HELP)
        cls.OUTPUT_TOOLTIPS = tuple(tips)

    function = 'execute' if is_v3 else cls.FUNCTION
    method = inspect.getattr_static(cls, function)
    classmethod_call, staticmethod_call = isinstance(method, classmethod), isinstance(method, staticmethod)
    original = method.__func__ if classmethod_call or staticmethod_call else method
    signature = inspect.signature(original)
    extra_name = next((name for name, p in signature.parameters.items() if p.kind == p.VAR_KEYWORD), None)

    @functools.wraps(original)
    def execute(*args, **kwargs):
        controls = {}
        for name, spec in specs.items():
            value = kwargs.pop(name, spec[1]['default'])
            if getattr(cls, 'INPUT_IS_LIST', False) and isinstance(value, list):
                value = value[0] if value else spec[1]['default']
            controls[name[len(control_prefix):]] = value
        # DiskLatent accepts controls beyond the wrapped method signature.
        forwarded = {}
        if extra_name is None and hasattr(original, '_vts_latent_original'):
            forwarded = {key: kwargs.pop(key) for key in list(kwargs)
                         if key.startswith('latent_') and key not in signature.parameters}
        bound = signature.bind(*args, **kwargs)
        extra = bound.arguments.get(extra_name, {})
        values = [bound.arguments.get(name, extra.get(name)) for name in inputs]
        mode = controls.get('return_type', 'Tensor')
        if mode == 'Input':
            storage = _disk_values(values)
            mode = 'DiskAudio' if storage and all(storage) else 'Tensor'
        if mode not in ('Tensor', 'DiskAudio'):
            raise ValueError('Unknown audio return type')
        memo = {}
        for name in inputs:
            target = bound.arguments if name in bound.arguments else extra
            if name in target and name not in keep_disk_inputs:
                target[name] = materialize_audio(target[name], controls['device_policy'], controls['device'], memo)
        # File-producing nodes can preserve their bounded-memory path for disk output.
        if mode == 'DiskAudio' and hasattr(cls, 'DISK_AUDIO_EXECUTE'):
            result = cls.DISK_AUDIO_EXECUTE(bound, controls)
        else:
            result = original(*bound.args, **bound.kwargs, **forwarded)
        if not outputs or mode == 'Tensor':
            return result
        if isinstance(result, dict):
            if 'result' not in result:
                return result
            values = list(result['result'])
        elif isinstance(result, io.NodeOutput):
            values = list(result.args)
        else:
            values = list(result if isinstance(result, tuple) else (result,))

        def save(value, output_index):
            if value is None:
                return None
            if isinstance(value, list):
                return [save(item, output_index) for item in value]
            name = controls['prefix'] + (f'_output_{output_index}' if len(outputs) > 1 else '')
            return save_audio(value, name, controls['output_dir'], controls['format'], controls['bitrate_kbps'])
        for index in outputs:
            values[index] = save(values[index], index)
        if isinstance(result, dict):
            return {**result, 'result': tuple(values)}
        if isinstance(result, io.NodeOutput):
            return io.NodeOutput(*values, ui=result.ui, expand=result.expand, block_execution=result.block_execution)
        return tuple(values)

    setattr(cls, function, classmethod(execute) if classmethod_call else staticmethod(execute) if staticmethod_call else execute)
    cls.VTS_DISK_AUDIO_SUPPORT = {'inputs': inputs, 'outputs': outputs}
    return cls
