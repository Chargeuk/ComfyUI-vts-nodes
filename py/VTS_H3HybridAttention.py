"""Opt-in sparse local windows for an already patched VDN-H3 model."""

import importlib.util
import logging
from pathlib import Path
import re
import sys

import torch


log = logging.getLogger("comfy.vts.h3_hybrid")
PATCH_KEY = "vts_h3_hybrid_attention"
BLOCK_SIZE = 64


def _load_sla_backend():
    # Comfy Kitchen sol_attn currently requires equal Q/K lengths. VDN's
    # gathered windows are rectangular; reuse PlagueKind's BLHD kernel instead.
    import folder_paths

    for root in folder_paths.get_folder_paths("custom_nodes"):
        directory = (Path(root) / "ComfyUI-PlagueKind-Nodes"
                     / "ComfyUI-H3-SLA-Attention" / "sla")
        if not (directory / "kernel.py").is_file():
            continue
        modules = []
        for name in ("block_map", "kernel"):
            module_name = "_vts_h3_sla_" + name
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(
                    module_name, directory / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise
            modules.append(module)
        return modules[0].get_block_map, modules[1].block_sparse_attention
    raise RuntimeError("VTS H3 Hybrid Attention requires ComfyUI-PlagueKind-Nodes "
                       "with its H3 SLA Attention kernel installed.")


def _dense_blocks(text):
    blocks = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", text):
        if "-" in part:
            first, last = map(int, part.split("-"))
            blocks.update(range(min(first, last), max(first, last) + 1))
        else:
            blocks.add(int(part))
    return blocks


def _sampling_step(options):
    schedule = options.get("sample_sigmas")
    current = options.get("sigmas")
    if schedule is None or current is None:
        return None
    if isinstance(schedule, torch.Tensor):
        schedule = schedule.flatten().tolist()
    if isinstance(current, torch.Tensor):
        current = current.flatten().tolist()
    if isinstance(current, (tuple, list)):
        if not current:
            return None
        current = current[0]
    if len(schedule) < 2:
        return None
    # Resolve once per diffusion forward, not once per layer/window. Do not
    # count forwards as steps: CFG and multistage solvers may call us repeatedly.
    step = min(range(len(schedule) - 1),
               key=lambda i: abs(float(schedule[i]) - float(current)))
    return step, len(schedule) - 1


class HybridAttention:
    def __init__(self, sparsity, min_tokens, dense_first_steps, dense_last_steps,
                 dense_blocks, fallback_to_exact, verbose, block_map, kernel,
                 exact_attention):
        self.sparsity = sparsity
        self.min_tokens = min_tokens
        self.dense_first_steps = dense_first_steps
        self.dense_last_steps = dense_last_steps
        self.dense_blocks = dense_blocks
        self.fallback_to_exact = fallback_to_exact
        self.verbose = verbose
        self.block_map = block_map
        self.kernel = kernel
        self.exact_attention = exact_attention
        self.reset()

    def reset(self):
        # Only scalar diagnostics persist during sampling; no Q/K/V, routing
        # tables, history, or output tensors are retained by this patch.
        self.failed = False
        self.sparse_calls = 0
        self.exact_calls = 0
        self.kept_blocks = 0
        self.total_blocks = 0
        self.logged = set()

    def log_once(self, key, message):
        if self.verbose and key not in self.logged:
            self.logged.add(key)
            log.info("[VTS H3 Hybrid] %s", message)

    def wrap_sample(self, executor, *args, **kwargs):
        self.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            if self.verbose:
                kept = (100 * self.kept_blocks / self.total_blocks
                        if self.total_blocks else 0)
                log.info("[VTS H3 Hybrid] sparse windows=%d, exact window fallbacks=%d, "
                         "mean selected key blocks=%.1f%% (including protection). "
                         "Zero sparse windows means no hybrid acceleration ran.",
                         self.sparse_calls, self.exact_calls, kept)
            self.reset()

    def wrap_diffusion(self, executor, *args, **kwargs):
        positional_options = len(args) > 3
        original = args[3] if positional_options else kwargs.get("transformer_options", {})
        options = original.copy()
        options.pop("vdn_local_attention", None)
        active = not self.failed
        if self.dense_first_steps or self.dense_last_steps:
            step = _sampling_step(options)
            if step is None:
                active = False
                self.log_once("no_steps", "Sampler step metadata missing; keeping exact VDN attention.")
            else:
                index, total = step
                active = active and self.dense_first_steps <= index < total - self.dense_last_steps
        if active:
            options["vdn_local_attention"] = self.for_block
        if positional_options:
            args = (*args[:3], options, *args[4:])
        else:
            kwargs = dict(kwargs, transformer_options=options)
        return executor(*args, **kwargs)

    def for_block(self, block_index):
        return None if self.failed or block_index in self.dense_blocks else self.attend

    def attend(self, q, k, v, scale, protected_ranges):
        if (self.failed or self.sparsity <= 0 or k.shape[0] < self.min_tokens
                or q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16)
                or q.shape[-1] != 128):
            self.exact_calls += 1
            return self.exact_attention(q, k, v, scale)
        result = None
        try:
            result = self._sparse(q, k, v, scale, protected_ranges)
        except torch.OutOfMemoryError:
            raise
        except (RuntimeError, NotImplementedError) as exc:
            # Recoverable optional kernel failures may use exact SDPA. A damaged
            # CUDA context or OOM must propagate rather than being hidden.
            fatal = ("illegal memory access", "device-side assert", "launch failure")
            if not self.fallback_to_exact or any(s in str(exc).lower() for s in fatal):
                raise
            self.failed = True
            log.warning("[VTS H3 Hybrid] Sparse kernel failed; exact VDN for the rest "
                        "of this sampling run: %s", exc)
        # Leave the exception scope before fallback so its traceback no longer
        # holds the sparse path's temporary routing tensors.
        if result is None:
            self.exact_calls += 1
            return self.exact_attention(q, k, v, scale)
        return result

    def _sparse(self, q, k, v, scale, protected_ranges):
        qb, kb, vb = (t.unsqueeze(0).contiguous() for t in (q, k, v))
        lut, topk = self.block_map(
            qb, kb, 1.0 - self.sparsity, BLOCK_SIZE, BLOCK_SIZE,
            protect_ranges=protected_ranges)
        blocks = (k.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        if topk >= blocks:
            self.log_once("all_protected", "Protection fills this window's key budget; using exact attention.")
            return None
        out = self.kernel(qb, kb, vb, lut, topk, BLOCK_SIZE, BLOCK_SIZE,
                          qk_scale=scale, tail=None, use_int8_qk=False, use_int8_pv=False)
        self.sparse_calls += 1
        self.kept_blocks += topk
        self.total_blocks += blocks
        self.log_once("active", "Sparse local windows active; global/anchor queries and "
                      "VDN's trained linear branch are unchanged.")
        return out.squeeze(0)


class VTS_H3HybridAttention:
    EXPERIMENTAL = True
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "VTS/model_patches/minimax"
    DESCRIPTION = (
        "Experimental SLA inside VDN local windows. Connect AFTER Apply VDN-H3; "
        "keep its turbo adapter and sampler settings. Does not load another LoRA. "
        "May change quality or run slower; bypass for the original VDN result.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "local_sparsity": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.95, "step": 0.05,
                "tooltip": "Fraction of local key blocks skipped before protected blocks are added. "
                           "Zero is unchanged VDN. Higher may be faster but can damage quality."}),
            "min_window_tokens": ("INT", {"default": 4096, "min": 0, "max": 131072, "step": 64,
                "tooltip": "Smaller key windows stay exact because routing overhead can outweigh savings."}),
            "dense_first_steps": ("INT", {"default": 1, "min": 0, "max": 100}),
            "dense_last_steps": ("INT", {"default": 1, "min": 0, "max": 100}),
            "dense_blocks": ("STRING", {"default": "", "tooltip": "Optional zero-based exact transformer layers, e.g. 0-3,46-49."}),
            "fallback_to_exact": ("BOOLEAN", {"default": True,
                "tooltip": "Use exact VDN after a recoverable kernel failure; OOM and fatal CUDA errors still stop."}),
            "verbose": ("BOOLEAN", {"default": True}),
        }}

    def execute(self, model, enabled=True, local_sparsity=0.70, min_window_tokens=4096,
                dense_first_steps=1, dense_last_steps=1, dense_blocks="",
                fallback_to_exact=True, verbose=True):
        if not enabled or local_sparsity == 0:
            return (model,)
        if not any(getattr(patch, "_vdn_forward", False)
                   for patch in model.object_patches.values()):
            raise ValueError("Connect Apply VDN-H3's MODEL output to VTS H3 Hybrid Attention first.")
        from vdn_h3 import hybrid, window
        from comfy.patcher_extension import WrappersMP

        if getattr(hybrid, "VDN_LOCAL_ATTENTION_API", 0) != 1:
            raise RuntimeError("VDN-H3's local-attention hook is missing. Update the Chargeuk "
                               "VDN fork with the VTS integration and restart ComfyUI.")
        if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, PATCH_KEY):
            raise ValueError("VTS H3 Hybrid Attention is already applied; connect it only once.")
        block_map, kernel = _load_sla_backend()
        patch = HybridAttention(local_sparsity, min_window_tokens, dense_first_steps,
                                dense_last_steps, _dense_blocks(dense_blocks),
                                fallback_to_exact, verbose, block_map, kernel, window._sdpa)
        cloned = model.clone()
        cloned.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, PATCH_KEY, patch.wrap_sample)
        cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, PATCH_KEY, patch.wrap_diffusion)
        log.warning("[VTS H3 Hybrid] Experimental sparse local attention enabled. "
                    "Do not add an SLA turbo LoRA; keep VDN's existing adapter. "
                    "Use grouped VDN attention for an exact-path A/B comparison.")
        return (cloned,)


NODE_CLASS_MAPPINGS = {"VTS_H3HybridAttention": VTS_H3HybridAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"VTS_H3HybridAttention": "VTS H3 Hybrid Attention (Experimental)"}
