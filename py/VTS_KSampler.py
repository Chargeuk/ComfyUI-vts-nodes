import torch

import comfy.samplers
from nodes import common_ksampler


def _clone_latent(latent):
    cloned = {}
    for key, value in latent.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _first_value(value, name):
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"VTS KSampler received no values for '{name}'.")
        return value[0]
    return value


def _normalize_seed_list(seed):
    if isinstance(seed, list):
        if len(seed) == 0:
            raise ValueError("VTS KSampler received an empty seed list.")
        return [int(value) for value in seed]
    return [int(seed)]


def _is_multiple_conditionings(conditioning):
    return (
        isinstance(conditioning, list)
        and len(conditioning) > 0
        and isinstance(conditioning[0], list)
        and len(conditioning[0]) > 0
        and isinstance(conditioning[0][0], list)
    )


def _normalize_conditioning_list(conditioning, name):
    if not isinstance(conditioning, list):
        return [conditioning]
    if len(conditioning) == 0:
        raise ValueError(f"VTS KSampler received an empty '{name}' list.")
    if _is_multiple_conditionings(conditioning):
        return conditioning
    return [conditioning]


def _split_latent_batches(latent_image):
    latent_items = latent_image if isinstance(latent_image, list) else [latent_image]
    if len(latent_items) == 0:
        raise ValueError("VTS KSampler received no latent input.")

    split_items = []
    for latent in latent_items:
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("VTS KSampler expected each latent item to be a LATENT dictionary.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("VTS KSampler expected latent 'samples' to be a tensor.")

        batch_size = samples.shape[0]
        noise_mask = latent.get("noise_mask")
        batch_index = latent.get("batch_index")

        for index in range(batch_size):
            item = _clone_latent(latent)
            item["samples"] = samples[index:index + 1].clone()

            if isinstance(noise_mask, torch.Tensor):
                mask_index = 0 if noise_mask.shape[0] == 1 else min(index, noise_mask.shape[0] - 1)
                item["noise_mask"] = noise_mask[mask_index:mask_index + 1].clone()

            if isinstance(batch_index, list) and len(batch_index) > 0:
                item["batch_index"] = [batch_index[min(index, len(batch_index) - 1)]]
            elif batch_index is not None:
                item["batch_index"] = batch_index

            split_items.append(item)

    return split_items


def _pick_with_repeat(values, index):
    if len(values) == 0:
        raise ValueError("VTS KSampler cannot pick from an empty list.")
    return values[index] if index < len(values) else values[-1]


def _merge_latent_outputs(latent_outputs):
    if len(latent_outputs) == 0:
        raise ValueError("VTS KSampler produced no latent outputs.")

    if len(latent_outputs) == 1:
        return latent_outputs[0]

    merged = _clone_latent(latent_outputs[0])
    merged["samples"] = torch.cat([latent["samples"] for latent in latent_outputs], dim=0)

    saw_noise_mask = all(isinstance(latent.get("noise_mask"), torch.Tensor) for latent in latent_outputs)
    if saw_noise_mask:
        merged["noise_mask"] = torch.cat([latent["noise_mask"] for latent in latent_outputs], dim=0)
    else:
        merged.pop("noise_mask", None)

    merged["batch_index"] = list(range(merged["samples"].shape[0]))
    return merged


class VTS_KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "random_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "A single seed or a list of seeds. When a list is provided, seeds are reused cyclically without changing how many images are generated.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of denoising steps."}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The classifier-free guidance scale.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "The algorithm used when sampling."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "The scheduler controlling the denoising curve."},
                ),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning. If a list is provided, each item drives one output sample."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning. If a list is provided, items are reused by position with last-item repeat."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image or latent batch to denoise."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The denoise strength applied during sampling.",
                    },
                ),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent batch.",)
    FUNCTION = "sample"
    CATEGORY = "VTS/sampling"
    DESCRIPTION = (
        "A KSampler-compatible VTS sampler that can consume a seed list. "
        "The number of outputs is driven by conditioning lists and latent batch size, "
        "while seeds cycle independently."
    )

    def sample(self, model, random_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        model = _first_value(model, "model")
        steps = int(_first_value(steps, "steps"))
        cfg = float(_first_value(cfg, "cfg"))
        sampler_name = _first_value(sampler_name, "sampler_name")
        scheduler = _first_value(scheduler, "scheduler")
        denoise = float(_first_value(denoise, "denoise"))

        seed_values = _normalize_seed_list(random_seed)
        positive_values = _normalize_conditioning_list(positive, "positive")
        negative_values = _normalize_conditioning_list(negative, "negative")
        latent_values = _split_latent_batches(latent_image)

        output_count = max(len(positive_values), len(negative_values), len(latent_values), 1)

        sampled_latents = []
        for index in range(output_count):
            current_seed = seed_values[index % len(seed_values)]
            current_positive = _pick_with_repeat(positive_values, index)
            current_negative = _pick_with_repeat(negative_values, index)
            current_latent = _pick_with_repeat(latent_values, index)

            sampled = common_ksampler(
                model,
                current_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                current_positive,
                current_negative,
                current_latent,
                denoise=denoise,
            )[0]
            sampled_latents.append(sampled)

        merged_latent = _merge_latent_outputs(sampled_latents)
        return (merged_latent,)


NODE_CLASS_MAPPINGS = {
    "VTS KSampler": VTS_KSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS KSampler": "VTS KSampler",
}
