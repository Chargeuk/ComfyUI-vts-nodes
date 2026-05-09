import logging

import torch


LOGGER = logging.getLogger(__name__)


def _warn(message):
    LOGGER.warning(message)
    print(message)


def _samples_shape(latent):
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor):
        return None
    return tuple(samples.shape)


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


def _default_noise_mask(samples):
    return torch.ones(
        (samples.shape[0], 1, samples.shape[2] * 8, samples.shape[3] * 8),
        device=samples.device,
        dtype=samples.dtype,
    )


def _normalized_noise_mask(latent, target_batch_size):
    samples = latent["samples"]
    mask = latent.get("noise_mask")
    if mask is None:
        mask = _default_noise_mask(samples)

    if mask.shape[-2] != samples.shape[-2] * 8 or mask.shape[-1] != samples.shape[-1] * 8:
        _warn(
            "VTS Latent List To Batch: noise_mask shape did not match latent samples; "
            "using the original mask without resizing."
        )

    if mask.shape[0] < target_batch_size:
        repeats = (target_batch_size - 1) // mask.shape[0] + 1
        mask = mask.repeat((repeats,) + ((1,) * (mask.ndim - 1)))[:target_batch_size]

    return mask


class VTS_Latent_List_To_Batch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "to_batch"
    CATEGORY = "VTS/latent"
    DESCRIPTION = (
        "Collapse a list-mapped LATENT input into one real latent batch. "
        "If latent sample shapes do not match, the node logs a warning and "
        "passes through the first latent unchanged."
    )

    def to_batch(self, latents):
        if len(latents) == 0:
            raise ValueError("VTS Latent List To Batch received no latents.")

        if len(latents) == 1:
            return (_clone_latent(latents[0]),)

        first = latents[0]
        first_shape = _samples_shape(first)
        if first_shape is None:
            _warn(
                "VTS Latent List To Batch: first latent had no tensor samples; "
                "passing it through unchanged."
            )
            return (_clone_latent(first),)

        for index, latent in enumerate(latents[1:], start=1):
            shape = _samples_shape(latent)
            if shape is None or shape[1:] != first_shape[1:]:
                _warn(
                    "VTS Latent List To Batch: latent shapes did not match, so batching "
                    f"was skipped. First shape={first_shape}, latent {index} shape={shape}. "
                    "Passing through the first latent unchanged."
                )
                return (_clone_latent(first),)

        result = _clone_latent(first)
        sample_batches = []
        noise_mask_batches = []
        batch_indexes = []
        offset = 0
        saw_noise_mask = any("noise_mask" in latent for latent in latents)

        for latent in latents:
            samples = latent["samples"]
            batch_size = samples.shape[0]
            sample_batches.append(samples)

            if saw_noise_mask:
                noise_mask_batches.append(_normalized_noise_mask(latent, batch_size))

            if "batch_index" in latent:
                batch_indexes.extend(list(latent["batch_index"]))
            else:
                batch_indexes.extend(range(offset, offset + batch_size))
            offset += batch_size

        result["samples"] = torch.cat(sample_batches, dim=0)

        if saw_noise_mask:
            merged_mask = torch.cat(noise_mask_batches, dim=0)
            if float(merged_mask.mean().item()) == 1.0:
                result.pop("noise_mask", None)
            else:
                result["noise_mask"] = merged_mask
        else:
            result.pop("noise_mask", None)

        result["batch_index"] = batch_indexes
        return (result,)


class VTS_Latent_Batch_To_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "to_list"
    CATEGORY = "VTS/latent"
    DESCRIPTION = "Split a latent batch into a list of single-item LATENT objects."

    def to_list(self, latent):
        samples = latent["samples"]
        batch_size = samples.shape[0]
        noise_mask = latent.get("noise_mask")
        batch_index = latent.get("batch_index")

        output = []
        for index in range(batch_size):
            item = _clone_latent(latent)
            item["samples"] = samples[index:index + 1].clone()

            if isinstance(noise_mask, torch.Tensor):
                if noise_mask.shape[0] == 1:
                    item["noise_mask"] = noise_mask.clone()
                else:
                    mask_index = min(index, noise_mask.shape[0] - 1)
                    item["noise_mask"] = noise_mask[mask_index:mask_index + 1].clone()

            if batch_index is not None:
                if isinstance(batch_index, list) and len(batch_index) > 0:
                    chosen_index = batch_index[min(index, len(batch_index) - 1)]
                    item["batch_index"] = [chosen_index]
                else:
                    item["batch_index"] = batch_index

            output.append(item)

        return (output,)


NODE_CLASS_MAPPINGS = {
    "VTS Latent List To Batch": VTS_Latent_List_To_Batch,
    "VTS Latent Batch To List": VTS_Latent_Batch_To_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Latent List To Batch": "Latent List To Batch VTS",
    "VTS Latent Batch To List": "Latent Batch To List VTS",
}
