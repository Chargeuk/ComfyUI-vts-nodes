"""Storage for compact H3 context tensors, independent of ordinary LATENT outputs."""
from dataclasses import dataclass

import torch

from vts_disk_latent import DiskLatent, DiskTensorInfo, latent_controls, materialize_latents, save_latent


@dataclass(frozen=True)
class DiskContextTensor(DiskTensorInfo):
    payload: DiskLatent
    key: str

    @property
    def disk_path(self):
        return self.payload.path

    def materialize(self, device_policy='Original', device='cpu', memo=None):
        if device_policy not in ('Original', 'CPU', 'Specified'):
            raise ValueError('Unknown context device policy')
        loaded = materialize_latents(self.payload, 'CPU', 'cpu', memo)['samples'][self.key]
        target = self.device if device_policy == 'Original' else ('cpu' if device_policy == 'CPU' else device)
        return loaded.to(device=target, dtype=self.dtype)


def context_controls(prefix='', *, has_input=False, has_output=True):
    controls = latent_controls(prefix, has_input=has_input, has_output=has_output,
                               control_prefix='context_', match_input=False)
    if has_output:
        controls['context_return_type'][1]['tooltip'] = (
            'Tensor keeps compact context latents in memory (default). DiskLatent stores only the '
            'required video/audio tails in one losslessly compressed file. Timing and tensor metadata '
            'remain accessible. Independent of image and regular latent output settings.')
        controls['context_output_dir'][1]['tooltip'] = (
            'Folder for disk-backed H3 context tails. Default ./tmp/disklatents, relative to ComfyUI working '
            'directory. Independent of image/audio and regular latent output folders.')
        controls['context_prefix'][1]['tooltip'] = (
            'Context filename prefix, derived from this node name. Sequence, list position and unique suffix '
            'distinguish saved contexts. The file also contains context timing metadata.')
        controls['context_start_sequence'][1]['tooltip'] = 'Starting filename sequence for context files, independent of other outputs.'
    if has_input:
        controls['context_device_policy'][1]['tooltip'] = (
            'Where disk-backed context tensors load: Original restores saved devices or deferred overrides; '
            'CPU uses RAM; Specified uses context_device. Existing in-memory tensors are unchanged. '
            'Separate from latent_device_policy for the normal LATENT input.')
        controls['context_device'][1]['tooltip'] = (
            'Used only with context_device_policy=Specified: cpu uses RAM; cuda:0 selects the first GPU; '
            'cuda:1 requires a second GPU. Applies to embedded context latents, not model placement.')
    return controls


def context_tensor_info(value):
    if isinstance(value, DiskLatent):
        value = value['samples']
        return value if isinstance(value, DiskTensorInfo) else None
    return value if isinstance(value, (torch.Tensor, DiskContextTensor)) else None


def materialize_context(context, device_policy='Original', device='cpu'):
    if device_policy not in ('Original', 'CPU', 'Specified'):
        raise ValueError('Unknown context device policy')
    memo, result = {}, dict(context)
    for key in ('video', 'audio'):
        value = context.get(key)
        if isinstance(value, DiskContextTensor):
            result[key] = value.materialize(device_policy, device, memo)
        elif isinstance(value, DiskLatent):
            result[key] = materialize_latents(value, device_policy, device, memo)['samples']
    return result if any(result[key] is not context[key] for key in result) else context


def store_context(context, return_type='Tensor', output_dir='./tmp/disklatents',
                  prefix='H3_context', start_sequence=0, compression_level=3):
    if return_type not in ('Tensor', 'DiskLatent'):
        raise ValueError('context_return_type must be Tensor or DiskLatent')
    if context is None:
        return None
    context = materialize_context(context)
    if return_type == 'Tensor':
        return context
    tensors = {key: context[key] for key in ('video', 'audio') if key in context}
    metadata = {key: value for key, value in context.items() if key not in tensors}
    payload = save_latent({'samples': tensors, 'context_metadata': metadata},
                          prefix, output_dir, start_sequence, compression_level)
    result = dict(metadata)
    for key, info in payload['samples'].items():
        result[key] = DiskContextTensor(info.shape, info.dtype, info.device, info.original_device, payload, key)
    return result
