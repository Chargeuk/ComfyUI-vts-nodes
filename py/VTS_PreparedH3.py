"""Self-contained H3 weights with an explicit, editable attention recipe."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import re
import sys
import uuid

import torch
from safetensors import safe_open

import folder_paths
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.sd
import comfy.utils


log = logging.getLogger("comfy.vts.prepared_h3")
META = "vts_prepared_h3"
BRANCH = "vts_vdn_branch."
HYBRID_KEY = "vts_h3_hybrid_attention"

# These are runtime controls, never adapter strengths or weight dimensions.
SCHEMA = {
    "vdn": {
        "radius": 1, "chunk": 5, "anchor_frames": "both",
        "delta_rule": "vdn_solve", "bridge": "alpha", "a_fp32": True,
        "short_conv": ["k", "v"], "enable_text_state": True,
        "enable_softmax_gate": True, "linear_enabled": True,
        "branch_weights": "auto", "retain_buffers": "auto", "prefetch": "auto",
        "attention_backend": "grouped", "fast_kernels": False,
        "compile_scan": False, "fuse_statistics": False, "verbose": True,
    },
    "hybrid": {
        "enabled": True, "local_sparsity": 0.7, "min_window_tokens": 4096,
        "dense_first_steps": 1, "dense_last_steps": 1, "dense_blocks": "",
        "fallback_to_exact": True, "verbose": True,
    },
    "sla": {
        "sparsity_ratio": 0.8, "block_size": 64, "min_seq_len": 8192,
        "dense_first_steps": 0, "dense_last_steps": 0, "dense_steps": "",
        "protect_audio": True, "dense_backend": "comfy_kitchen",
        "disable_fp16_accum": True, "stabilize_motion": False,
        "reference_protection": "Off", "tail_correction": False,
        "use_int8_qk": False, "use_int8_pv": False, "engine": "triton",
    },
    "sampling": {
        "shift": 1.0, "audio_shift": None, "timesteps": 1000,
        "multiplier": 1000.0, "noise_scale": 1.0,
    },
}
VDN_CFG = set(SCHEMA["vdn"]) - {
    "branch_weights", "retain_buffers", "prefetch", "attention_backend",
    "fast_kernels", "compile_scan", "fuse_statistics", "verbose",
}
CHOICES = {
    ("vdn", "branch_weights"): ("auto", "stream", "cache_gpu"),
    ("vdn", "retain_buffers"): ("auto", "on", "off"),
    ("vdn", "prefetch"): ("auto", "on", "off"),
    ("vdn", "attention_backend"): ("grouped", "flex"),
    ("vdn", "anchor_frames"): ("both", "columns", "rows", "none"),
    ("vdn", "delta_rule"): ("vdn_solve", "sana_scaled", "vdn_scaled"),
    ("vdn", "bridge"): ("alpha", "none"),
    ("sla", "engine"): ("triton", "comfy_kitchen"),
    ("sla", "reference_protection"): ("Off", "Light", "Heavy Enforcement"),
    ("sla", "dense_backend"): ("auto", "pytorch", "comfy_kitchen", "sage:auto",
        "sage:qk_int8_pv_fp16_cuda", "sage:qk_int8_pv_fp16_triton",
        "sage:qk_int8_pv_fp8_cuda", "sage:qk_int8_pv_fp8_cuda++"),
}


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _validate_options(options):
    if not isinstance(options, dict):
        raise ValueError("Runtime overrides must be a JSON object.")
    for group, values in options.items():
        if group not in SCHEMA or not isinstance(values, dict):
            raise ValueError(f"Unknown runtime group: {group}. Use vdn, hybrid, sla or sampling.")
        for key, value in values.items():
            if key not in SCHEMA[group]:
                raise ValueError(f"{group}.{key} is not a supported runtime setting. "
                                 "Weight patches/strengths require a new prepared file.")
            example = SCHEMA[group][key]
            if key == "audio_shift":
                valid = value is None or type(value) in (int, float)
            elif type(example) is float:
                valid = type(value) in (int, float)
            else:
                valid = type(value) is type(example)
            if not valid:
                raise ValueError(f"Invalid type for {group}.{key}.")
            choices = CHOICES.get((group, key))
            if choices and value not in choices:
                raise ValueError(f"{group}.{key} must be one of {choices}.")
            if key == "short_conv" and any(v not in ("q", "k", "v") for v in value):
                raise ValueError("short_conv accepts q, k and v only.")
    return _json_copy(options)


def _merge_options(saved, overrides):
    result = deepcopy(saved)
    overrides = _validate_options(overrides)
    for group, values in overrides.items():
        if group not in result:
            raise ValueError(f"This prepared model has no {group} configuration to override.")
        result[group].update(values)
    return result


def _apply_common(runtime, options):
    options = deepcopy(options or {})
    common = options.pop("common", {})
    target = "hybrid" if "hybrid" in runtime else "sla" if "sla" in runtime else None
    mapping = {"local_sparsity": "sparsity_ratio", "min_window_tokens": "min_seq_len"}
    for key, value in common.items():
        if key in ("branch_weights", "retain_buffers", "prefetch", "attention_backend"):
            group, field = "vdn", key
        else:
            group, field = target, mapping.get(key, key) if target == "sla" else key
        if group is None or group not in runtime:
            raise ValueError(f"{key} does not apply to this prepared attention variant.")
        if field in options.get(group, {}):
            raise ValueError(f"Set {group}.{field} in either the widget or advanced JSON, not both.")
        options.setdefault(group, {})[field] = value
    return _merge_options(runtime, options)


def _load_module(name, path, package=False):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)] if package else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _sla():
    for root in folder_paths.get_folder_paths("custom_nodes"):
        path = Path(root) / "ComfyUI-PlagueKind-Nodes/ComfyUI-H3-SLA-Attention/sla/__init__.py"
        if path.is_file():
            _load_module("_vts_prepared_sla", path, package=True)
            return sys.modules["_vts_prepared_sla.patch"]
    raise RuntimeError("Install ComfyUI-PlagueKind-Nodes to restore SLA attention.")


def _hybrid():
    return _load_module("_vts_prepared_hybrid", Path(__file__).with_name("VTS_H3HybridAttention.py"))


def _fingerprints(variant):
    paths = {}
    if variant in ("vdn", "hybrid"):
        from vdn_h3 import branch, hybrid, spec, window
        for name, module in (("vdn_branch", branch), ("vdn_hybrid", hybrid),
                             ("vdn_spec", spec), ("vdn_window", window)):
            paths[name] = Path(module.__file__)
    if variant in ("sla", "hybrid"):
        directory = Path(_sla().__file__).parent
        for name in ("patch", "kernel", "block_map"):
            paths["sla_" + name] = directory / (name + ".py")
    if variant == "hybrid":
        paths["vts_hybrid"] = Path(__file__).with_name("VTS_H3HybridAttention.py")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _one_wrapper(model, kind, key):
    wrappers = model.get_wrappers(kind, key)
    if len(wrappers) > 1:
        raise ValueError(f"Duplicate {key} wrappers cannot be exported.")
    return wrappers[0] if wrappers else None


def _check_supported(model):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("Prepared H3 nodes require a native MiniMax-H3 MODEL.")
    for field in ("injections", "hook_patches", "weight_wrapper_patches", "additional_models"):
        if getattr(model, field):
            raise ValueError(f"Cannot bake {field}; remove that patch before saving. "
                             "VDN lora_mode=bypass is not a merged-weight export.")
    if any(funcs for group in model.callbacks.values() for funcs in group.values()):
        raise ValueError("This MODEL has unsupported callbacks; export before applying them.")
    allowed = {("diffusion_model", "vdn_h3"), ("diffusion_model", "h3_sla_state"),
               ("diffusion_model", HYBRID_KEY), ("outer_sample", HYBRID_KEY)}
    for kind, groups in model.wrappers.items():
        for key, funcs in groups.items():
            if funcs and (kind, key) not in allowed:
                raise ValueError(f"Unsupported runtime wrapper {kind}/{key}; export before applying it.")
    for key, patch in model.object_patches.items():
        if key == "model_sampling":
            continue
        if re.fullmatch(r"diffusion_model\.blocks\.\d+\.attn\.forward", key) and getattr(patch, "_vdn_forward", False):
            continue
        raise ValueError(f"Unsupported object patch {key}; it cannot be silently dropped.")
    for key, value in model.model_options.items():
        if key != "transformer_options" and value:
            raise ValueError(f"Unsupported model option {key}; apply it after loading instead.")
    allowed_options = {"optimized_attention_override", "minimax_h3_sigma_shift_video",
                       "minimax_h3_sigma_shift_audio"}
    for key, value in model.model_options.get("transformer_options", {}).items():
        if key not in allowed_options and value:
            raise ValueError(f"Unsupported transformer option {key}; apply it after loading instead.")


def _workflow_setting(prompt, unique_id, node_types, setting, default):
    # Read the explicit choice from the graph, never infer it from an opaque
    # optimized attention callable. Ambiguous/dynamic inputs require confirmation.
    pending, seen, choices = [str(unique_id)], set(), []
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = (prompt or {}).get(node_id, {})
        inputs = node.get("inputs", {})
        if node.get("class_type") in node_types:
            value = inputs.get(setting, default)
            if isinstance(value, str):
                choices.append(value)
        # MODEL chains use model; do not search unrelated conditioning branches.
        value = inputs.get("model")
        if isinstance(value, list) and len(value) == 2:
            pending.append(str(value[0]))
    return choices[0] if len(choices) == 1 else None


def _capture(model, prompt=None, unique_id=None, sla_dense_backend="from workflow"):
    _check_supported(model)
    runtime, vdn_state = {}, None
    vdn = _one_wrapper(model, "diffusion_model", "vdn_h3")
    sla = _one_wrapper(model, "diffusion_model", "h3_sla_state")
    hybrid = _one_wrapper(model, "diffusion_model", HYBRID_KEY)
    if sla and (vdn or hybrid):
        raise ValueError("Separate SLA and VDN patches cannot share an export; use the VTS hybrid node.")
    if not sla and model.model_options.get("transformer_options", {}).get("optimized_attention_override") is not None:
        raise ValueError("Unsupported attention override; export before applying it.")
    if not vdn and not sla:
        raise ValueError("Apply VDN-H3, H3 SLA, or VTS Hybrid before saving a prepared model.")
    if vdn:
        vdn_state = inspect.getclosurevars(vdn).nonlocals["state"]
        cfg = _json_copy(vdn_state.cfg)
        unknown = set(cfg) - VDN_CFG - {"linear_head_dim"}
        if unknown:
            raise ValueError(f"Unsupported VDN configuration fields: {sorted(unknown)}. Update VTS.")
        runtime["vdn"] = dict(SCHEMA["vdn"], **{k: v for k, v in cfg.items() if k in VDN_CFG})
        runtime["vdn"].update(
            branch_weights=vdn_state.cache_mode, prefetch=vdn_state.prefetch_mode,
            retain_buffers="on" if vdn_state.retain_buffers else "off",
            attention_backend=vdn_state.softmax_backend, verbose=vdn_state.verbose,
            fast_kernels=vdn_state.branches[0].fuse_epilogue,
            compile_scan=vdn_state.branches[0].compile_scan,
            fuse_statistics=vdn_state.branches[0].fuse_statistics)
        requested = _workflow_setting(prompt, unique_id, {"ApplyVDNH3", "ApplyVDNH3Advanced"},
                                      "retain_buffers", "auto")
        attached = model.get_attachment(META) or {}
        requested = requested or attached.get("runtime", {}).get("vdn", {}).get("retain_buffers")
        if requested is not None:
            runtime["vdn"]["retain_buffers"] = requested
    if hybrid:
        state = hybrid.__self__
        runtime["hybrid"] = dict(
            enabled=True, local_sparsity=state.sparsity, min_window_tokens=state.min_tokens,
            dense_first_steps=state.dense_first_steps, dense_last_steps=state.dense_last_steps,
            dense_blocks=",".join(map(str, sorted(state.dense_blocks))),
            fallback_to_exact=state.fallback_to_exact, verbose=state.verbose)
    if sla:
        wrapper = inspect.getclosurevars(sla).nonlocals
        override = model.model_options["transformer_options"]["optimized_attention_override"]
        attention = inspect.getclosurevars(override).nonlocals
        attached = model.get_attachment(META) or {}
        backend = sla_dense_backend
        if backend == "from workflow":
            backend = _workflow_setting(prompt, unique_id, {"H3SLAAttention"}, "dense_backend", "comfy_kitchen") or attached.get("runtime", {}).get("sla", {}).get("dense_backend")
        if not backend:
            raise ValueError("Cannot recover SLA dense_backend from this graph. Select its actual "
                             "setting on Save Prepared H3's sla_dense_backend input.")
        reference = attention["reference_sparsity"]
        runtime["sla"] = dict(SCHEMA["sla"])
        for key in ("min_seq_len", "protect_audio", "stabilize_motion",
                    "tail_correction", "use_int8_qk", "use_int8_pv", "engine"):
            runtime["sla"][key] = attention[key]
        runtime["sla"].update(
            sparsity_ratio=wrapper["sparsity_ratio"],
            block_size=attention["blkq"], dense_last_steps=wrapper["dense_last_steps"],
            dense_steps=",".join(map(str, sorted(wrapper["dense_steps"]))),
            disable_fp16_accum=wrapper["disable_fp16_accum"], dense_backend=backend,
            reference_protection="Off" if reference is None else "Heavy Enforcement" if reference == 0 else "Light")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, comfy.model_sampling.ModelSamplingAV):
        raise ValueError("Only H3's ModelSamplingAV sampling configuration can be restored.")
    runtime["sampling"] = dict(shift=float(sampling.shift), audio_shift=sampling.audio_shift,
        timesteps=len(sampling.sigmas), multiplier=float(sampling.multiplier),
        noise_scale=float(sampling.noise_scale))
    variant = "hybrid" if hybrid else "vdn" if vdn else "sla"
    manifest = dict(version=1, variant=variant, runtime=_validate_options(runtime),
                    code=_fingerprints(variant), baked_patch_count=len(model.patches))
    if vdn_state is not None:
        manifest["vdn"] = dict(name=vdn_state.name, linear_head_dim=vdn_state.cfg["linear_head_dim"],
                               heads=vdn_state.num_heads, head_dim=vdn_state.head_dim)
    return manifest, vdn_state


def _pack_branches(state, stack):
    from vdn_h3.spec import LazyBranchTensor
    tensors, descriptors, handles = {}, [], {}
    for block, branch in enumerate(state.branches):
        entries = {}
        for name, weight in branch.w.items():
            key = f"{BRANCH}{block}.{name}"
            if not isinstance(weight, LazyBranchTensor):
                raise ValueError("Expected disk-backed VDN branch weights. Update the Chargeuk VDN fork.")
            if weight._path not in handles:
                handles[weight._path] = stack.enter_context(safe_open(weight._path, framework="pt", device="cpu"))
            handle = handles[weight._path]
            tensors[key] = handle.get_tensor(weight._key)
            entry = dict(key=key, shape=list(weight.shape), dtype=str(weight.dtype).removeprefix("torch."))
            if weight._conf is not None:
                scale_key = key + ".vts_scale"
                tensors[scale_key] = handle.get_tensor(weight._scale_key)
                entry.update(scale_key=scale_key, quant=_json_copy(weight._conf))
            entries[name] = entry
        descriptors.append(entries)
    return tensors, descriptors


def _default_output_directory():
    return Path(folder_paths.models_dir) / "diffusion_models" / "prepared-h3"


def _resolve_file(name, explicit=False):
    roots = [Path(path).resolve() for path in folder_paths.get_folder_paths("diffusion_models")]
    path = Path(name).expanduser()
    if explicit and not path.is_absolute():
        raise ValueError("Enter an absolute WSL input file path, for example /mnt/share/prepared_h3.safetensors.")
    if not path.is_absolute():
        resolved = folder_paths.get_full_path("diffusion_models", name)
        if resolved is None:
            raise ValueError(f"Prepared model not found: {name}")
        path = Path(resolved)
    path = path.resolve()
    if path.suffix != ".safetensors":
        raise ValueError("Prepared input files must use the .safetensors extension.")
    if not explicit and not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Dropdown selections must be inside a registered diffusion-model folder; use input_path for other locations.")
    if not path.is_file():
        raise ValueError(f"Prepared model not found: {path.name}")
    return path


class _QuantizedExportPiece(comfy.model_patcher.LazyCastingParamPiece):
    def __new__(cls, caster, state_dict_key, tensor):
        # Core's LazyCastingParamPiece defaults to requires_grad=True, which
        # fails for integer storage under dynamic VRAM (ComfyUI ea33b154).
        return torch.nn.Parameter.__new__(cls, tensor, requires_grad=False)


def _export_state_dict(model):
    """Native lazy weight patching, with integer-safe serialization pieces.

    Kept inside this exporter: no global class replacement or model mutation.
    The native caster still applies/requantizes one weight at a time.
    """
    diffusion = model.model.diffusion_model
    state_dict = diffusion.state_dict()
    for name, op in diffusion.named_modules():
        if not hasattr(op, "comfy_cast_weights") or getattr(op, "comfy_patched_weights", False):
            continue
        for parameter in ("weight", "bias"):
            key = f"{name}.{parameter}" if name else parameter
            if key not in state_dict:
                continue
            weight = getattr(op, parameter)
            full_key = "diffusion_model." + key
            if isinstance(weight, comfy.model_patcher.QuantizedTensor):
                caster = comfy.model_patcher.LazyCastingQuantizedParam(model, full_key)
                for piece_key in weight.state_dict(key):
                    if piece_key in state_dict:
                        state_dict[piece_key] = _QuantizedExportPiece(
                            caster, "diffusion_model." + piece_key, state_dict[piece_key])
            else:
                state_dict[key] = comfy.model_patcher.LazyCastingParam(model, full_key, weight)
    return model.model.state_dict_for_saving(state_dict)


def _write_checkpoint(path, model, metadata, extra_keys):
    comfy.model_management.load_models_gpu([model])
    state_dict = _export_state_dict(model)
    state_dict.update(extra_keys)
    for key, tensor in state_dict.items():
        if not tensor.is_contiguous():
            state_dict[key] = tensor.contiguous()
    comfy.utils.save_torch_file(state_dict, path, metadata=metadata)


def _save(model, prefix, prompt=None, unique_id=None, sla_dense_backend="from workflow", output_directory=""):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", prefix):
        raise ValueError("Use a plain filename prefix (letters, numbers, dots, underscores or hyphens).")
    root = Path(output_directory).expanduser() if output_directory else _default_output_directory()
    if not root.is_absolute():
        raise ValueError("Enter an absolute WSL output directory, for example /mnt/share/prepared_h3.")
    root = root.resolve()
    manifest, state = _capture(model, prompt, unique_id, sla_dense_backend)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}_{uuid.uuid4().hex}.safetensors"
    temporary = path.with_suffix(".partial")
    try:
        with ExitStack() as stack:
            tensors = {}
            if state is not None:
                tensors, descriptors = _pack_branches(state, stack)
                manifest["vdn"]["branches"] = descriptors
            log.info("[VTS Prepared H3] Baking %d patched weights into %s. This is a one-time export.",
                     len(model.patches), path)
            _write_checkpoint(str(temporary), model,
                metadata={META: json.dumps(manifest, allow_nan=False)}, extra_keys=tensors)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    log.info("[VTS Prepared H3] Saved %s", path)
    return str(path)


def _read_manifest(handle):
    raw = (handle.metadata() or {}).get(META)
    if raw is None:
        raise ValueError("Not a VTS prepared H3 file. Use the ordinary model loader for raw checkpoints.")
    manifest = json.loads(raw)
    if manifest.get("version") != 1 or manifest.get("variant") not in ("vdn", "sla", "hybrid"):
        raise ValueError("Unsupported prepared H3 format/version.")
    expected = {"sampling"}
    expected.update({"sla"} if manifest["variant"] == "sla" else {"vdn"})
    if manifest["variant"] == "hybrid":
        expected.add("hybrid")
    if set(manifest["runtime"]) != expected:
        raise ValueError("Prepared H3 runtime does not match its attention variant.")
    _validate_options(manifest["runtime"])
    for group in expected:
        if set(manifest["runtime"][group]) != set(SCHEMA[group]):
            raise ValueError(f"Incomplete prepared {group} runtime configuration.")
    return manifest


def _unpack_branches(path, handle, descriptor):
    from vdn_h3.spec import LazyBranchTensor
    types = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    branches, used, stage_bytes = [], set(), 0
    for block, entries in enumerate(descriptor["branches"]):
        weights = {}
        for name, entry in entries.items():
            key = entry["key"]
            if key != f"{BRANCH}{block}.{name}" or entry["dtype"] not in types:
                raise ValueError("Invalid VDN branch tensor descriptor.")
            tensor = handle.get_tensor(key)
            if list(tensor.shape) != entry["shape"]:
                raise ValueError(f"Branch shape mismatch: {key}")
            used.add(key)
            stage_bytes += tensor.numel() * tensor.element_size()
            quant, scale_key = entry.get("quant"), entry.get("scale_key")
            if quant is not None:
                if (scale_key != key + ".vts_scale" or tensor.dtype != torch.int8
                        or not isinstance(quant, dict) or quant.get("format") != "int8_tensorwise"):
                    raise ValueError(f"Invalid INT8 branch data: {key}")
                scale = handle.get_tensor(scale_key)
                used.add(scale_key)
                stage_bytes += scale.numel() * scale.element_size()
            elif tensor.dtype != types[entry["dtype"]]:
                raise ValueError(f"Branch dtype mismatch: {key}")
            weights[name] = LazyBranchTensor(str(path), key, torch.Size(entry["shape"]),
                                             types[entry["dtype"]], scale_key, quant)
        branches.append(weights)
    if not branches or used != {key for key in handle.keys() if key.startswith(BRANCH)}:
        raise ValueError("Incomplete VDN branch payload.")
    return branches, stage_bytes


def _restore_vdn(model, descriptor, weights, stage_bytes, runtime):
    from vdn_h3 import hybrid
    from vdn_h3.branch import LinearBranch
    from vdn_h3.nodes import _disable_comfy_compiler_on_broken_builds
    dm = model.get_model_object("diffusion_model")
    if (len(dm.blocks) != len(weights) or dm.blocks[0].attn.heads != descriptor["heads"]
            or dm.blocks[0].attn.head_dim != descriptor["head_dim"]):
        raise ValueError("Prepared VDN branches do not match the base model.")
    heads, linear, hidden = descriptor["heads"], descriptor["linear_head_dim"], dm.hidden_size
    expected = {"to_out_linear.weight": (hidden, heads * linear), "beta_proj.weight": (heads, hidden),
        "alpha.A_log": (heads,), "alpha.dt_bias": (heads * linear,),
        "alpha.down.weight": (linear, hidden), "alpha.up.weight": (heads * linear, linear),
        "output_gate.down.weight": (linear, hidden), "output_gate.up.weight": (heads * linear, linear),
        "output_gate.up.bias": (heads * linear,), "norm.weight": (linear,)}
    if runtime["enable_softmax_gate"]:
        expected.update({"softmax_gate.up.weight": (heads, hidden), "softmax_gate.up.bias": (heads,)})
    for block, values in enumerate(weights):
        for key, shape in expected.items():
            if key not in values or tuple(values[key].shape) != shape:
                raise ValueError(f"Prepared VDN block {block}/{key} must have shape {shape}.")
    cfg = {key: runtime[key] for key in VDN_CFG}
    cfg["linear_head_dim"] = descriptor["linear_head_dim"]
    free = comfy.model_management.get_free_memory(comfy.model_management.get_torch_device())
    retain = free >= stage_bytes + 10 * (1 << 30) if runtime["retain_buffers"] == "auto" else runtime["retain_buffers"] == "on"
    branches = [LinearBranch(w, descriptor["heads"], descriptor["head_dim"],
        delta_rule=cfg["delta_rule"], bridge=cfg["bridge"], a_fp32=cfg["a_fp32"],
        short_conv=cfg["short_conv"], enable_text_state=cfg["enable_text_state"], retain_buffers=retain)
        for w in weights]
    for branch in branches:
        branch.fuse_epilogue = runtime["fast_kernels"]
        branch.compile_scan = runtime["compile_scan"]
        branch.fuse_statistics = runtime["fuse_statistics"]
    state = hybrid.VDNState(descriptor["name"], cfg, branches, descriptor["heads"], descriptor["head_dim"])
    state.owns_compiler_switch = _disable_comfy_compiler_on_broken_builds()
    state.retain_buffers = retain
    state.cache_mode = runtime["branch_weights"]
    state.cache_gpu = state.cache_mode == "cache_gpu"
    state.prefetch_mode = runtime["prefetch"]
    state.stage_bytes = stage_bytes
    state.block_bytes = max(sum(t.shape.numel() * t.dtype.itemsize for t in w.values()) for w in weights) * 1.25
    state.base_stream_bytes = 4 * model.model_size() / len(branches)
    state.verbose = runtime["verbose"]
    state.softmax_backend = runtime["attention_backend"]
    cloned = model.clone()
    state.unloaded_bytes = lambda: max(0, cloned.model_size() - cloned.loaded_size())
    hybrid.apply_vdn(cloned, state)
    return cloned


class _H3Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


def _restore_runtime(model, manifest, weights, stage_bytes, runtime):
    if "vdn" in runtime:
        model = _restore_vdn(model, manifest["vdn"], weights, stage_bytes, runtime["vdn"])
    if "hybrid" in runtime:
        model = _hybrid().VTS_H3HybridAttention().execute(model, **runtime["hybrid"])[0]
    if "sla" in runtime:
        options = dict(runtime["sla"])
        first = options.pop("dense_first_steps")
        module = _sla()
        steps = set(module._parse_step_spec(options["dense_steps"])) | set(range(first))
        options["dense_steps"] = ",".join(map(str, sorted(steps)))
        model = module.patch_h3_sla(model, **options)
    sampling = _H3Sampling(model.model.model_config)
    parameters = dict(runtime["sampling"])
    noise_scale = parameters.pop("noise_scale")
    sampling.set_parameters(**parameters)
    sampling.set_noise_scale(noise_scale)
    model.add_object_patch("model_sampling", sampling)
    to = model.model_options["transformer_options"]
    to["minimax_h3_sigma_shift_video"] = parameters["shift"]
    to["minimax_h3_sigma_shift_audio"] = parameters["audio_shift"]
    model.set_attachments(META, dict(variant=manifest["variant"], runtime=deepcopy(runtime)))
    return model


def _load(path, overrides=None, compatibility="require matching attention code", weight_dtype="saved", explicit_path=False):
    path = _resolve_file(path, explicit=explicit_path)
    if compatibility not in ("require matching attention code", "allow changed attention code"):
        raise ValueError("Invalid compatibility policy.")
    dtype_options = {"saved": {}, "bfloat16": {"dtype": torch.bfloat16},
                     "float16": {"dtype": torch.float16}, "float32": {"dtype": torch.float32}}
    if weight_dtype not in dtype_options:
        raise ValueError("Invalid load dtype.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        manifest = _read_manifest(handle)
        if manifest["code"] != _fingerprints(manifest["variant"]):
            message = "Attention code changed since this export. Re-export, or explicitly allow changed code and re-test quality."
            if compatibility == "require matching attention code":
                raise ValueError(message)
            log.warning("[VTS Prepared H3] %s", message)
        runtime = _apply_common(manifest["runtime"], overrides)
        weights, stage_bytes = (None, 0)
        if "vdn" in runtime:
            weights, stage_bytes = _unpack_branches(path, handle, manifest["vdn"])
        sd = {key: handle.get_tensor(key) for key in handle.keys() if not key.startswith(BRANCH)}
        metadata = {key: value for key, value in (handle.metadata() or {}).items() if key != META}
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=dtype_options[weight_dtype], metadata=metadata)
    if model is None or not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("Prepared file does not contain a supported MiniMax-H3 base model.")
    model = _restore_runtime(model, manifest, weights, stage_bytes, runtime)
    log.info("[VTS Prepared H3] Loaded %s with %s attention; no saved weight adapters reapplied.", path.name, manifest["variant"])
    return model, json.dumps(dict(file=str(path), variant=manifest["variant"], runtime=runtime), indent=2)


class VTS_SavePreparedH3:
    EXPERIMENTAL = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prepared_file",)
    FUNCTION = "execute"
    CATEGORY = "VTS/model_patches/minimax"
    OUTPUT_NODE = True
    DESCRIPTION = "One-time export of merged H3 weights, VDN branch weights and attention settings. Choose an output directory or leave blank for models/diffusion_models/prepared-h3. Remove this node from the generation workflow after export."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "filename_prefix": ("STRING", {"default": "prepared_h3"}),
            "output_directory": ("STRING", {"default": "", "tooltip": "Absolute WSL directory. Blank saves in ComfyUI's models/diffusion_models/prepared-h3 folder, visible in the diffusion-model picker."}),
            "save_enabled": ("BOOLEAN", {"default": False, "tooltip": "Enable for the one-time export. Each execution writes a new, large file."}),
            "sla_dense_backend": ("STRING", {"default": "from workflow", "tooltip": "SLA only: leave from workflow, or specify the actual dense_backend if its graph input is dynamic."}),
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    def execute(self, model, filename_prefix="prepared_h3", save_enabled=False,
                sla_dense_backend="from workflow", prompt=None, unique_id=None, output_directory=""):
        if not save_enabled:
            return ("",)
        return (_save(model, filename_prefix, prompt, unique_id, sla_dense_backend, output_directory),)


class VTS_LoadPreparedH3:
    EXPERIMENTAL = True
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "settings")
    FUNCTION = "execute"
    CATEGORY = "VTS/model_patches/minimax"
    DESCRIPTION = "Loads a prepared H3 without merging saved adapters again. Optional runtime_options override attention/memory/sampling controls, not baked weights."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prepared_model": (folder_paths.get_filename_list("diffusion_models") or ["(no diffusion models)"],),
            "input_path": ("STRING", {"default": "", "tooltip": "Full WSL .safetensors path, including filename. Overrides the model dropdown. Blank uses the dropdown or connected prepared_file."}),
            "compatibility": (["require matching attention code", "allow changed attention code"],),
            "weight_dtype": (["saved", "bfloat16", "float16", "float32"],),
        }, "optional": {"runtime_options": ("VTS_H3_RUNTIME_OPTIONS",),
                         "prepared_file": ("STRING", {"forceInput": True})}}

    def execute(self, prepared_model, compatibility="require matching attention code",
                weight_dtype="saved", runtime_options=None, prepared_file=None, input_path=""):
        if input_path and prepared_file:
            raise ValueError("Use either input_path or the connected prepared_file, not both.")
        path = input_path or prepared_file or prepared_model
        return _load(path, runtime_options, compatibility, weight_dtype, explicit_path=bool(input_path or prepared_file))


class VTS_PreparedH3RuntimeOptions:
    RETURN_TYPES = ("VTS_H3_RUNTIME_OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "VTS/model_patches/minimax"
    DESCRIPTION = "Connect to Load Prepared H3. -1/saved leaves the file's value unchanged; zero is a real override. Advanced JSON supports all documented attention and sampling runtime controls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "local_sparsity": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.95, "step": 0.05}),
            "dense_first_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
            "dense_last_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
            "min_window_tokens": ("INT", {"default": -1, "min": -1, "max": 131072}),
            "branch_weights": (["saved", "auto", "stream", "cache_gpu"],),
            "retain_buffers": (["saved", "auto", "on", "off"],),
            "prefetch": (["saved", "auto", "on", "off"],),
            "attention_backend": (["saved", "grouped", "flex"],),
            "advanced_json": ("STRING", {"default": "{}", "multiline": True}),
        }}

    def execute(self, local_sparsity=-1.0, dense_first_steps=-1, dense_last_steps=-1,
                min_window_tokens=-1, branch_weights="saved", retain_buffers="saved",
                prefetch="saved", attention_backend="saved", advanced_json="{}"):
        overrides = _validate_options(json.loads(advanced_json))
        common = {}
        for name, value in (("local_sparsity", local_sparsity), ("dense_first_steps", dense_first_steps),
                            ("dense_last_steps", dense_last_steps), ("min_window_tokens", min_window_tokens)):
            if value != -1:
                common[name] = value
        for name, value in (("branch_weights", branch_weights), ("retain_buffers", retain_buffers),
                            ("prefetch", prefetch), ("attention_backend", attention_backend)):
            if value != "saved":
                common[name] = value
        overrides["common"] = common
        return (overrides,)


NODE_CLASS_MAPPINGS = {name: value for name, value in globals().copy().items()
                       if name in ("VTS_SavePreparedH3", "VTS_LoadPreparedH3", "VTS_PreparedH3RuntimeOptions")}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_SavePreparedH3": "VTS Save Prepared H3 (Experimental)",
    "VTS_LoadPreparedH3": "VTS Load Prepared H3 (Experimental)",
    "VTS_PreparedH3RuntimeOptions": "VTS Prepared H3 Runtime Options",
}
