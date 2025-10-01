# Modified from https://github.com/madebyollin/taehv/blob/main/taehv.py

# ruff: noqa: N806

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple
import os

import torch
from torch import nn
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from taeVid_base import VideoModelInfo

F = torch.nn.functional


class TWorkItem(NamedTuple):
    input_tensor: torch.Tensor
    block_index: int


def conv(n_in: int, n_out: int, **kwargs: dict) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def __init__(self, clamp_mode: str = "tanh"):
        super().__init__()
        self.mode = clamp_mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "hard":
            return x.clamp(-3, 3)
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat((x, past), 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[-3:]
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        return self.conv(x).reshape(-1, *orig_shape[-3:])


class TAEVidContext:
    def __init__(self, model):
        self.model = model
        self.HANDLERS = {
            MemBlock: self.handle_memblock,
            TPool: self.handle_tpool,
            TGrow: self.handle_tgrow,
        }

    def reset(self, x: torch.Tensor) -> None:
        N, T, C, H, W = x.shape
        self.N, self.T = N, T
        self.work_queue = [
            TWorkItem(xt, 0)
            for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))
        ]
        self.mem = [None] * len(self.model)

    def handle_memblock(
        self,
        i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        mem = self.mem
        # mem blocks are simple since we're visiting the graph in causal order
        if mem[i] is None:
            xt_new = b(xt, torch.zeros_like(xt))
            mem[i] = xt
        else:
            xt_new = b(xt, mem[i])
            # inplace might reduce mysterious pytorch memory allocations? doesn't help though
            mem[i].copy_(xt)
        return (xt_new,)

    def handle_tpool(
        self,
        i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        mem = self.mem
        # pool blocks are miserable
        if mem[i] is None:
            mem[i] = []  # pool memory is itself a queue of inputs to pool
        mem[i].append(xt)
        if len(mem[i]) > b.stride:
            # pool mem is in invalid state, we should have pooled before this
            raise RuntimeError("Internal error: Invalid mem state")
        if len(mem[i]) < b.stride:
            # pool mem is not yet full, go back to processing the work queue
            return ()
        # pool mem is ready, run the pool block
        N, C, H, W = xt.shape
        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
        # reset the pool mem
        mem[i] = []
        return (xt,)

    def handle_tgrow(
        self,
        _i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        xt = b(xt)
        C, H, W = xt.shape[1:]
        return reversed(
            xt.view(self.N, b.stride * C, H, W).chunk(b.stride, 1),
        )

    @classmethod
    def handle_default(
        cls,
        _i: int,
        xt: torch.Tensor,
        b: nn.Module,
    ) -> Iterable[torch.Tensor]:
        return (b(xt),)

    def handle_block(self, i: int, xt: torch.Tensor, b: nn.Module) -> None:
        handler = self.HANDLERS.get(b.__class__, self.handle_default)
        for xt_new in handler(i, xt, b):
            self.work_queue.insert(0, TWorkItem(xt_new, i + 1))

    def apply(self, x: torch.Tensor, *, show_progress=False) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected 5 dimensional tensor")
        self.reset(x)
        out = []
        work_queue = self.work_queue
        model = self.model
        model_len = len(model)

        with tqdm(range(self.T), disable=not show_progress) as pbar:
            while work_queue:
                xt, i = work_queue.pop(0)
                if i == model_len:
                    # reached end of the graph, append result to output list
                    out.append(xt)
                    continue
                if i == 0:
                    # new source node consumed
                    pbar.update(1)
                self.handle_block(i, xt, model[i])
        return torch.stack(out, 1)


class TAEVid(nn.Module):
    temporal_upscale_blocks = 2
    spatial_upscale_blocks = 3
    _nf = (256, 128, 64, 64)

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        vmi: VideoModelInfo,
        image_channels: int = 3,
        device="cpu",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        clamp_mode: str = "tanh",
    ):
        n_f = self._nf
        super().__init__()
        self.vmi = vmi
        self.latent_channels = vmi.latent_format.latent_channels
        self.image_channels = image_channels
        self.patch_size = vmi.patch_size
        self.encoder = nn.Sequential(
            conv(image_channels * self.patch_size**2, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            conv(64, vmi.latent_format.latent_channels),
        )
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(clamp_mode=clamp_mode),
            conv(vmi.latent_format.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], image_channels * self.patch_size**2),
        )
        if checkpoint_path is None:
            return
        self.load_state_dict(
            self.patch_tgrow_layers(
                torch.load(checkpoint_path, map_location=device, weights_only=True),
            ),
        )

    def patch_tgrow_layers(self, sd: dict) -> dict:
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    @classmethod
    def apply_parallel(
        cls,
        x: torch.Tensor,
        model: nn.Module,
        *,
        show_progress=False,
        mem_in: list[torch.Tensor | None] | None = None,   # OPTIONAL: prior per-MemBlock state
        return_mem: bool = False,                          # return tail states if True
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """
        Vectorized causal pass with optional continuity:
          mem_in[i] (if provided for a MemBlock) is the previous activation (shape (N,C,H,W))
          to be used as the 'past' for the first timestep of this chunk instead of zeros.

        Returns:
          x_out  (N,T,...)  or (x_out, mem_out_list) if return_mem
        mem_out_list contains for each layer:
          - for MemBlocks: the last pre-update activation (to feed as past into next chunk)
          - else: None
        """
        padding = (0, 0, 0, 0, 0, 0, 1, 0)  # original zero-pad scheme (unused now except fallback)
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)

        # Prepare input state list length = len(model)
        if mem_in is None:
            mem_in = [None] * len(model)
        else:
            # pad / trim to model length
            if len(mem_in) < len(model):
                mem_in = mem_in + [None] * (len(model) - len(mem_in))
            elif len(mem_in) > len(model):
                mem_in = mem_in[: len(model)]

        mem_out: list[torch.Tensor | None] = [None] * len(model)

        for li, b in enumerate(tqdm(model, disable=not show_progress)):
            if not isinstance(b, MemBlock):
                x = b(x)
                continue

            # Reshape to (N,T,C,H,W)
            nt, c_blk, h_blk, w_blk = x.shape
            t_blk = nt // n
            x_seq = x.view(n, t_blk, c_blk, h_blk, w_blk)

            # Past for first timestep
            prev = mem_in[li]
            if prev is not None:
                if prev.shape != (n, c_blk, h_blk, w_blk):
                    raise ValueError(f"mem_in[{li}] shape mismatch: {prev.shape} vs {(n,c_blk,h_blk,w_blk)}")
                first_past = prev
            else:
                first_past = torch.zeros_like(x_seq[:, 0])

            # Build mem tensor: past for each timestep
            # past_t[0] = first_past; past_t[k] = x_seq[:, k-1] (pre-update) for k>0
            past_full = torch.empty_like(x_seq)
            past_full[:, 0] = first_past
            if t_blk > 1:
                past_full[:, 1:] = x_seq[:, :-1]

            # Save last pre-update activation (the thing future chunk will need)
            mem_out[li] = x_seq[:, -1].detach()

            # Flatten back and apply
            past_full_flat = past_full.view(nt, c_blk, h_blk, w_blk)
            x = b(x, past_full_flat)

            # Free intermediates
            del past_full, past_full_flat, x_seq

        nt, c, h, w = x.shape
        t_final = nt // n
        x_out = x.view(n, t_final, c, h, w)
        if return_mem:
            return x_out, mem_out
        return x_out

    def apply(
        self,
        x: torch.Tensor,
        *,
        decode=True,
        parallel=True,
        show_progress=False,
        mem_in: list[torch.Tensor | None] | None = None,   # NEW
        return_mem: bool = False,                          # NEW
    ):
        """
        Extended: if parallel=True you can pass mem_in / get mem_out for continuity across chunks.
        """
        model = self.decoder if decode else self.encoder
        if not decode and self.vmi.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)

        if parallel:
            result = self.apply_parallel(
                x, model,
                show_progress=show_progress,
                mem_in=mem_in,
                return_mem=return_mem,
            )
        else:
            if mem_in is not None or return_mem:
                raise ValueError("mem_in/return_mem only supported in parallel mode")
            result = TAEVidContext(model).apply(x, show_progress=show_progress)

        if return_mem:
            x_res, mem_out = result  # type: ignore
        else:
            x_res = result  # type: ignore

        x_res = (
            x_res
            if not decode or self.vmi.patch_size < 2
            else F.pixel_shuffle(x_res, self.patch_size)
        )
        return (x_res, mem_out) if return_mem else x_res

    def decode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=True, **kwargs)[:, self.frames_to_trim :]

    def encode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=False, **kwargs)

    # ---- Tiled decoding helpers ----
    @staticmethod
    def _build_1d_mask(length: int, 
                       at_start: bool, 
                       at_end: bool, 
                       border: int, 
                       device, 
                       dtype,
                       blend_mode: str = "linear",
                       blend_exp: float = 1.0,
                       min_border_fraction: float = 0.0):
        """
        Build 1D feather mask along one axis.
        Center region = 1.0, edges ramp from 0->1 (if not at boundary) or stay 1 if boundary.
        """
        if border <= 0:
            return torch.ones(length, device=device, dtype=dtype)

        # Optionally ensure a minimum fractional border
        min_border = int(min_border_fraction * length)
        if border < min_border:
            border = min_border

        mask = torch.ones(length, device=device, dtype=dtype)

        def make_ramp(b: int, forward: bool) -> torch.Tensor:
            # Generate values from 0->1 (forward=True) or 1->0 (forward=False)
            r = torch.linspace(0, 1, b + 2, device=device, dtype=dtype)[1:-1]
            if blend_mode == "cosine":
                r2 = 0.5 - 0.5 * torch.cos(torch.pi * r)  # smooth start/end
            elif blend_mode == "smoothstep":
                r2 = r * r * (3 - 2 * r)
            elif blend_mode == "gaussian":
                # Center the gaussian so 0 at edge, 1 at interior
                sigma = 0.35
                r2 = torch.exp(-((1 - r) ** 2) / (2 * sigma * sigma))
            else:  # linear
                r2 = r
            if not forward:
                r2 = 1.0 - r2
            if blend_exp != 1.0:
                r2 = r2.clamp(0, 1) ** blend_exp
            return r2

        if not at_start:
            mask[:border] = make_ramp(border, forward=True)
        if not at_end:
            mask[-border:] = torch.minimum(mask[-border:], make_ramp(border, forward=False))
        return mask

    @classmethod
    def _build_mask(cls, 
                    tile_decoded: torch.Tensor,
                    is_bound: tuple[bool, bool, bool, bool],
                    border_width_hw: tuple[int, int],
                    blend_mode: str = "linear",
                    blend_exp: float = 1.0,
                    min_border_fraction: float = 0.0) -> torch.Tensor:
        """
        Build 2D separable feather mask.
        Uses outer product (mh * mw) instead of min which gives smoother corners.
        """
        H = tile_decoded.shape[-2]
        W = tile_decoded.shape[-1]
        top, bottom, left, right = is_bound
        border_h, border_w = border_width_hw
        device = tile_decoded.device
        dtype = tile_decoded.dtype

        mh = cls._build_1d_mask(H, top, bottom, border_h, device, dtype, 
                               blend_mode, blend_exp, min_border_fraction)  # (H,)
        mw = cls._build_1d_mask(W, left, right, border_w, device, dtype,
                               blend_mode, blend_exp, min_border_fraction)  # (W,)

        mask2d = mh.view(H, 1) * mw.view(1, W)  # separable weight
        # Shape (1,1,1,H,W)
        return mask2d.view(1, 1, 1, H, W)

    def decode_tiled(
        self,
        x: torch.Tensor,
        pixel_tile_size_x: int,
        pixel_tile_size_y: int,
        pixel_tile_stride_x: int,
        pixel_tile_stride_y: int,
        *,
        show_progress: bool = False,
        device: torch.device | str | None = None,
        blend_mode: str = "linear",
        blend_exp: float = 1.0,
        min_border_fraction: float = 0.0,
        time_chunk: int | None = None,                 # NEW: latent-time chunk length (in latent frames)
        keep_temporal_continuity: bool = True,         # NEW: preserve per-tile MemBlock states
        offload_chunk_to_cpu: bool = True,             # NEW: move each decoded time chunk off GPU
        offload_mem_states_to_cpu: bool = False,       # NEW: move continuity states to CPU between chunks
        mem_states_dtype: torch.dtype | None = torch.float16,  # NEW: compress mem states
        collect_ipc: bool = False,                     # NEW: call torch.cuda.ipc_collect()
    ) -> torch.Tensor:
        """
        Tiled spatial decode of latent video tensor with optional temporal chunking.
        Only the FIRST chunk has its leading warm-up frames (frames_to_trim) removed.
        Subsequent chunks retain all produced frames (we already have history).
        """
        if x.ndim != 5:
            raise ValueError("Expected latent tensor shape (N,T,C,H,W)")
        N, T_lat_total, _, _, _ = x.shape

        # No temporal chunking path (original behavior)
        if (time_chunk is None) or (time_chunk >= T_lat_total):
            decoded = self._decode_tiled_core(
                x,
                pixel_tile_size_x,
                pixel_tile_size_y,
                pixel_tile_stride_x,
                pixel_tile_stride_y,
                show_progress=show_progress,
                device=device,
                blend_mode=blend_mode,
                blend_exp=blend_exp,
                min_border_fraction=min_border_fraction,
                tile_mem_in=None,
                return_tile_mem=False,
                trim_leading=True,
            )
            return decoded

        # Build latent-space tiling tasks once
        if (pixel_tile_size_x % 32 != 0 or pixel_tile_size_y % 32 != 0 or
            pixel_tile_stride_x % 32 != 0 or pixel_tile_stride_y % 32 != 0):
            raise ValueError("Pixel tile sizes and strides must be multiples of 32")

        tile_size_x = pixel_tile_size_x // 32
        tile_size_y = pixel_tile_size_y // 32
        tile_stride_x = pixel_tile_stride_x // 32
        tile_stride_y = pixel_tile_stride_y // 32
        _, _, _, H_lat, W_lat = x.shape

        tasks: list[tuple[int,int,int,int]] = []
        for y in range(0, H_lat, tile_stride_y):
            if (y - tile_stride_y >= 0) and (y - tile_stride_y + tile_size_y >= H_lat):
                continue
            for xw in range(0, W_lat, tile_stride_x):
                if (xw - tile_stride_x >= 0) and (xw - tile_stride_x + tile_size_x >= W_lat):
                    continue
                y1 = min(y + tile_size_y, H_lat)
                x1 = min(xw + tile_size_x, W_lat)
                tasks.append((y, y1, xw, x1))
        if not tasks:
            raise ValueError("No tile tasks generated; check tile sizes / strides.")

        # Per-tile state list (MemBlocks in decoder)
        tile_mem_states: list[list[torch.Tensor | None] | None]
        if keep_temporal_continuity:
            tile_mem_states = [None] * len(tasks)
        else:
            tile_mem_states = []  # unused

        decoded_chunks: list[torch.Tensor] = []
        start = 0
        chunk_idx = 0
        while start < T_lat_total:
            end = min(start + time_chunk, T_lat_total)
            x_chunk = x[:, start:end]

            if keep_temporal_continuity:
                dec_chunk, tile_mem_states = self._decode_tiled_core(
                    x_chunk,
                    pixel_tile_size_x,
                    pixel_tile_size_y,
                    pixel_tile_stride_x,
                    pixel_tile_stride_y,
                    show_progress=show_progress and (chunk_idx == 0),
                    device=device,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    tasks_override=tasks,
                    tile_mem_in=tile_mem_states,
                    return_tile_mem=True,
                    trim_leading=(chunk_idx == 0),  # only first chunk trims warm-up
                )
                
                # Optionally offload memory states to CPU and compress dtype
                if offload_mem_states_to_cpu and tile_mem_states is not None:
                    for i, tile_state in enumerate(tile_mem_states):
                        if tile_state is None:
                            continue
                        for li, layer_mem in enumerate(tile_state):
                            if layer_mem is not None:
                                # Compress dtype if requested
                                if mem_states_dtype is not None and layer_mem.dtype != mem_states_dtype:
                                    layer_mem = layer_mem.to(mem_states_dtype)
                                # Move to CPU
                                tile_state[li] = layer_mem.cpu()
                        tile_mem_states[i] = tile_state
            else:
                dec_chunk = self._decode_tiled_core(
                    x_chunk,
                    pixel_tile_size_x,
                    pixel_tile_size_y,
                    pixel_tile_stride_x,
                    pixel_tile_stride_y,
                    show_progress=show_progress and (chunk_idx == 0),
                    device=device,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    tasks_override=tasks,
                    tile_mem_in=None,
                    return_tile_mem=False,
                    trim_leading=(chunk_idx == 0),
                )

            # Offload decoded chunk to CPU if requested
            if offload_chunk_to_cpu:
                if isinstance(dec_chunk, tuple):
                    # Handle case where dec_chunk might be a tuple (shouldn't happen but be safe)
                    dec_chunk_cpu = dec_chunk[0].cpu()
                else:
                    dec_chunk_cpu = dec_chunk.cpu()
                decoded_chunks.append(dec_chunk_cpu)
                del dec_chunk_cpu
            else:
                decoded_chunks.append(dec_chunk)

            # Explicit cleanup
            del x_chunk, dec_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if collect_ipc:
                    torch.cuda.ipc_collect()

            start = end
            chunk_idx += 1

        # Final concatenate (on CPU if offloaded, move back to target device if needed)
        out = torch.cat(decoded_chunks, dim=1)
        if not offload_chunk_to_cpu or device is None:
            return out
        
        # Move final result back to target device if user specified a CUDA device
        target_device = torch.device(device) if isinstance(device, str) else device
        if target_device is not None and target_device.type == "cuda":
            out = out.to(target_device)
        return out

    def _decode_tiled_core(
        self,
        x: torch.Tensor,
        pixel_tile_size_x: int,
        pixel_tile_size_y: int,
        pixel_tile_stride_x: int,
        pixel_tile_stride_y: int,
        *,
        show_progress: bool,
        device: torch.device | str | None,
        blend_mode: str,
        blend_exp: float,
        min_border_fraction: float,
        tasks_override: list[tuple[int,int,int,int]] | None = None,
        tile_mem_in: list[list[torch.Tensor | None] | None] | None = None,
        return_tile_mem: bool = False,
        trim_leading: bool = True,   # NEW: whether to remove initial frames_to_trim
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[torch.Tensor | None] | None]]:
        """
        Core function for a single latent temporal slice decode.
        If trim_leading is False, we keep all produced frames (used for subsequent chunks).
        """
        compute_device = torch.device(device) if device is not None else x.device
        N, T_in_lat, C_lat, H_lat, W_lat = x.shape

        # Validate multiples (same as outer)
        if (pixel_tile_size_x % 32 != 0 or pixel_tile_size_y % 32 != 0 or
            pixel_tile_stride_x % 32 != 0 or pixel_tile_stride_y % 32 != 0):
            raise ValueError("Pixel tile sizes and strides must be multiples of 32")

        tile_size_x = pixel_tile_size_x // 32
        tile_size_y = pixel_tile_size_y // 32
        tile_stride_x = pixel_tile_stride_x // 32
        tile_stride_y = pixel_tile_stride_y // 32

        # Tasks
        if tasks_override is None:
            tasks: list[tuple[int,int,int,int]] = []
            for y in range(0, H_lat, tile_stride_y):
                if (y - tile_stride_y >= 0) and (y - tile_stride_y + tile_size_y >= H_lat):
                    continue
                for xw in range(0, W_lat, tile_stride_x):
                    if (xw - tile_stride_x >= 0) and (xw - tile_stride_x + tile_size_x >= W_lat):
                        continue
                    y1 = min(y + tile_size_y, H_lat)
                    x1 = min(xw + tile_size_x, W_lat)
                    tasks.append((y, y1, xw, x1))
            if not tasks:
                raise ValueError("No tile tasks generated; check tile sizes / strides.")
        else:
            tasks = tasks_override

        tiles_x = len(set(t[2] for t in tasks))
        tiles_y = len(set(t[0] for t in tasks))
        print(f"Decode tiling: {tiles_x} tiles in X, {tiles_y} tiles in Y (chunk latent T={T_in_lat})")

        # Probe first tile (shape inference)
        with torch.no_grad():
            y0, y1, x0, x1 = tasks[0]
            test_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)
            dec_full_probe, mem_probe = self.apply(
                test_tile, decode=True, parallel=True,
                show_progress=False, mem_in=None, return_mem=True
            )
            if trim_leading:
                dec_probe = dec_full_probe[:, self.frames_to_trim :]
            else:
                dec_probe = dec_full_probe  # keep all frames after first chunk
            _, T_out_chunk, C_img, H_tile_out, W_tile_out = dec_probe.shape
            del dec_full_probe, dec_probe, test_tile, mem_probe

        # Spatial upscale factor (already implied in decode)
        spatial_factor = (2 ** sum(
            (m.scale_factor == 2) for m in self.decoder if isinstance(m, nn.Upsample)
        ))
        if self.patch_size > 1:
            spatial_factor *= self.patch_size
        H_out = H_lat * spatial_factor
        W_out = W_lat * spatial_factor

        acc_dtype = torch.float32
        values = torch.zeros(
            (N, T_out_chunk, C_img, H_out, W_out),
            dtype=acc_dtype,
            device=compute_device,
        )
        weight = torch.zeros(
            (N, T_out_chunk, 1, H_out, W_out),
            dtype=acc_dtype,
            device=compute_device,
        )

        # Border (latent space -> output space)
        border_h_lat = max(tile_size_y - tile_stride_y, 0)
        border_w_lat = max(tile_size_x - tile_stride_x, 0)
        border_h_out = border_h_lat * spatial_factor
        border_w_out = border_w_lat * spatial_factor

        iterator = tqdm(tasks, disable=not show_progress, desc="Tiled decode")
        updated_tile_mem: list[list[torch.Tensor | None] | None] = (
            [None] * len(tasks) if return_tile_mem else []
        )

        with torch.no_grad():
            for tile_id, (y0, y1, x0, x1) in enumerate(iterator):
                latent_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)
                mem_in_this = tile_mem_in[tile_id] if (tile_mem_in is not None and tile_mem_in[tile_id] is not None) else None

                # Move memory states from CPU back to GPU if they were offloaded
                if mem_in_this is not None:
                    for li, layer_mem in enumerate(mem_in_this):
                        if layer_mem is not None and layer_mem.device.type == "cpu":
                            mem_in_this[li] = layer_mem.to(compute_device, non_blocking=True)

                if return_tile_mem:
                    dec_full, mem_out_tile = self.apply(
                        latent_tile, decode=True, parallel=True,
                        show_progress=False, mem_in=mem_in_this, return_mem=True
                    )
                    updated_tile_mem[tile_id] = mem_out_tile
                else:
                    dec_full = self.apply(
                        latent_tile, decode=True, parallel=True,
                        show_progress=False, mem_in=mem_in_this, return_mem=False
                    )

                del latent_tile

                # Trim only for first chunk (trim_leading)
                dec = dec_full[:, self.frames_to_trim :] if trim_leading else dec_full

                out_y0 = y0 * spatial_factor
                out_x0 = x0 * spatial_factor
                tile_h_out = dec.shape[-2]
                tile_w_out = dec.shape[-1]
                out_y1 = min(H_out, out_y0 + tile_h_out)
                out_x1 = min(W_out, out_x0 + tile_w_out)

                slice_h = out_y1 - out_y0
                slice_w = out_x1 - out_x0
                if slice_h != tile_h_out or slice_w != tile_w_out:
                    print(f"[TAEVid decode_tiled chunk] Mismatch tile ({tile_h_out},{tile_w_out}) vs slice ({slice_h},{slice_w}) "
                          f"at latent y={y0} x={x0}. Cropping tile.")
                    dec = dec[..., :slice_h, :slice_w]

                mask = self._build_mask(
                    dec,
                    is_bound=(
                        y0 == 0,
                        y1 >= H_lat,
                        x0 == 0,
                        x1 >= W_lat,
                    ),
                    border_width_hw=(border_h_out, border_w_out),
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                ).to(dec.device, dec.dtype)

                values[:, :, :, out_y0:out_y1, out_x0:out_x1] += dec * mask
                weight[:, :, :, out_y0:out_y1, out_x0:out_x1] += mask

        weight.clamp_(min=1e-6)
        decoded = (values / weight).to(dtype=x.dtype)

        if return_tile_mem:
            return decoded, updated_tile_mem
        return decoded

    def encode_tiled(
        self,
        x: torch.Tensor,
        pixel_tile_size_x: int,
        pixel_tile_size_y: int,
        pixel_tile_stride_x: int,
        pixel_tile_stride_y: int,
        *,
        show_progress: bool = False,
        device: torch.device | str | None = None,
        blend_mode: str = "linear",
        blend_exp: float = 1.0,
        min_border_fraction: float = 0.0,
        time_chunk: int | None = None,
        keep_temporal_continuity: bool = True,   # NEW: maintain per-tile MemBlock state across chunks
        temporal_compression: int | None = None,  # NEW: for chunk-wise padding
    ) -> torch.Tensor:
        """
        Tiled spatial encode with optional temporal chunking and per-tile continuity.

        keep_temporal_continuity:
            If True and time_chunk is set, preserves MemBlock past activations separately
            for each spatial tile across chunks. Prevents shape mismatches and keeps
            causal continuity. If False, each chunk starts fresh (lower memory, small quality loss).
        """
        if x.ndim != 5:
            raise ValueError("Expected input shape (N,T,C,H,W)")

        N, T_total, _, H, W = x.shape

        # Fast path (no chunking)
        if (time_chunk is None) or (time_chunk >= T_total):
            return self._encode_tiled_core(
                x,
                pixel_tile_size_x,
                pixel_tile_size_y,
                pixel_tile_stride_x,
                pixel_tile_stride_y,
                show_progress=show_progress,
                device=device,
                blend_mode=blend_mode,
                blend_exp=blend_exp,
                min_border_fraction=min_border_fraction,
                tile_mem_in=None,
                return_tile_mem=False,
                temporal_compression=temporal_compression,
            )

        print(f"Starting encoding of {N} images with tile size ({pixel_tile_size_x},{pixel_tile_size_y}), stride ({pixel_tile_stride_x},{pixel_tile_stride_y}) and time chunk {time_chunk}")

        # Build tile task list ONCE (need for per-tile memory arrays)
        spatial_factor = 1
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                s = layer.stride
                if isinstance(s, tuple): s = s[0]
                if s == 2: spatial_factor *= 2
        if self.patch_size > 1:
            spatial_factor *= self.patch_size

        # Validate sizes early (reuse core logic's checks minimally)
        for name, val in (
            ("pixel_tile_size_x", pixel_tile_size_x),
            ("pixel_tile_size_y", pixel_tile_size_y),
            ("pixel_tile_stride_x", pixel_tile_stride_x),
            ("pixel_tile_stride_y", pixel_tile_stride_y),
        ):
            if val % spatial_factor != 0:
                raise ValueError(f"{name} must be multiple of {spatial_factor}")

        tasks: list[tuple[int,int,int,int]] = []
        for y0 in range(0, H, pixel_tile_stride_y):
            if (y0 - pixel_tile_stride_y >= 0) and (y0 - pixel_tile_stride_y + pixel_tile_size_y >= H):
                continue
            for x0 in range(0, W, pixel_tile_stride_x):
                if (x0 - pixel_tile_stride_x >= 0) and (x0 - pixel_tile_stride_x + pixel_tile_size_x >= W):
                    continue
                y1 = min(y0 + pixel_tile_size_y, H)
                x1 = min(x0 + pixel_tile_size_x, W)
                tasks.append((y0, y1, x0, x1))
        if not tasks:
            raise ValueError("No tile tasks generated")

        # Per-tile memory (list indexed by tile_id) each element is a mem_in list (per layer) or None
        tile_mem_states: list[list[torch.Tensor | None] | None]
        if keep_temporal_continuity:
            tile_mem_states = [None] * len(tasks)
        else:
            tile_mem_states = []  # unused

        # Temporal chunking loop
        temporal_down_factor = 1
        for m in self.encoder:
            if isinstance(m, TPool) and m.stride > 1:
                temporal_down_factor *= m.stride
        if time_chunk % temporal_down_factor != 0:
            raise ValueError(
                f"time_chunk ({time_chunk}) must be multiple of temporal downscale ({temporal_down_factor})"
            )

        encoded_chunks: list[torch.Tensor] = []
        start = 0
        chunk_index = 0
        while start < T_total:
            end = min(start + time_chunk, T_total)
            x_chunk = x[:, start:end]

            if keep_temporal_continuity:
                enc_chunk, tile_mem_states = self._encode_tiled_core(
                    x_chunk,
                    pixel_tile_size_x,
                    pixel_tile_size_y,
                    pixel_tile_stride_x,
                    pixel_tile_stride_y,
                    show_progress=show_progress and (chunk_index == 0),
                    device=device,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    tasks_override=tasks,
                    tile_mem_in=tile_mem_states,
                    return_tile_mem=True,
                    temporal_compression=temporal_compression,
                )
            else:
                enc_chunk = self._encode_tiled_core(
                    x_chunk,
                    pixel_tile_size_x,
                    pixel_tile_size_y,
                    pixel_tile_stride_x,
                    pixel_tile_stride_y,
                    show_progress=show_progress and (chunk_index == 0),
                    device=device,
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                    tasks_override=tasks,
                    tile_mem_in=None,
                    return_tile_mem=False,
                    temporal_compression=temporal_compression,
                )
            encoded_chunks.append(enc_chunk)
            del x_chunk, enc_chunk
            start = end
            chunk_index += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(encoded_chunks, dim=1)
    
    def _encode_tiled_core(
        self,
        x: torch.Tensor,
        pixel_tile_size_x: int,
        pixel_tile_size_y: int,
        pixel_tile_stride_x: int,
        pixel_tile_stride_y: int,
        *,
        show_progress: bool,
        device: torch.device | str | None,
        blend_mode: str,
        blend_exp: float,
        min_border_fraction: float,
        tasks_override: list[tuple[int,int,int,int]] | None = None,
        tile_mem_in: list[list[torch.Tensor | None] | None] | None = None,  # per-tile mem
        return_tile_mem: bool = False,
        temporal_compression: int | None = None,  # NEW: for chunk-wise padding
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[torch.Tensor | None] | None]]:
        """
        Core encode tiling over a temporal slice.
        tile_mem_in: list indexed by tile_id -> per-layer mem list (or None)
        return_tile_mem: return updated per-tile mem (MemBlock states of last timestep for each tile)
        temporal_compression: if provided, pad this chunk to be divisible by this factor
        """
        compute_device = torch.device(device) if device is not None else x.device
        N, T_in, C_img, H, W = x.shape
        
        # Pad this chunk if needed (instead of padding entire video)
        if temporal_compression is not None:
            chunk_frames = T_in
            add_frames = (
                math.ceil(chunk_frames / temporal_compression) * temporal_compression
                - chunk_frames
            )
            if add_frames > 0:
                last_frame = x[:, -1:, ...]  # (N, 1, C, H, W)
                padding = last_frame.repeat(1, add_frames, 1, 1, 1)
                x = torch.cat([x, padding], dim=1)
                del last_frame, padding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Update T_in after padding
                T_in = x.shape[1]

        # Spatial downscale factor
        spatial_factor = 1
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                s = layer.stride
                if isinstance(s, tuple): s = s[0]
                if s == 2: spatial_factor *= 2
        if self.patch_size > 1:
            spatial_factor *= self.patch_size

        # Tasks
        if tasks_override is None:
            tasks: list[tuple[int,int,int,int]] = []
            for y0 in range(0, H, pixel_tile_stride_y):
                if (y0 - pixel_tile_stride_y >= 0) and (y0 - pixel_tile_stride_y + pixel_tile_size_y >= H):
                    continue
                for x0 in range(0, W, pixel_tile_stride_x):
                    if (x0 - pixel_tile_stride_x >= 0) and (x0 - pixel_tile_stride_x + pixel_tile_size_x >= W):
                        continue
                    y1 = min(y0 + pixel_tile_size_y, H)
                    x1 = min(x0 + pixel_tile_size_x, W)
                    tasks.append((y0, y1, x0, x1))
            if not tasks:
                raise ValueError("No tile tasks generated")
        else:
            tasks = tasks_override

        tiles_x = len(set(t[2] for t in tasks))
        tiles_y = len(set(t[0] for t in tasks))
        print(f"Encode tiling: {tiles_x} tiles in X, {tiles_y} tiles in Y")
        print(f"Tile size in pixels: {pixel_tile_size_x} x {pixel_tile_size_y}")
        print(f"Total tiles: {len(tasks)}")

        # Probe first tile (without mem continuity—shape only)
        with torch.no_grad():
            y0, y1, x0, x1 = tasks[0]
            test_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)
            enc_test = self.apply(test_tile, decode=False, parallel=True, show_progress=False)
            _, T_enc, C_lat, h_lat_tile, w_lat_tile = enc_test.shape
            del enc_test, test_tile

        H_lat_full = H // spatial_factor
        W_lat_full = W // spatial_factor
        print(f"[TAEVid encode_tiled] spatial_factor={spatial_factor} H={H} W={W} H_lat_full={H_lat_full} W_lat_full={W_lat_full}")

        acc_dtype = torch.float32
        values = torch.zeros((N, T_enc, C_lat, H_lat_full, W_lat_full), dtype=acc_dtype, device=compute_device)
        weight = torch.zeros((N, T_enc, 1, H_lat_full, W_lat_full), dtype=acc_dtype, device=compute_device)

        overlap_y_pix = max(pixel_tile_size_y - pixel_tile_stride_y, 0)
        overlap_x_pix = max(pixel_tile_size_x - pixel_tile_stride_x, 0)
        if overlap_y_pix % spatial_factor != 0 or overlap_x_pix % spatial_factor != 0:
            raise ValueError("Overlap must be divisible by spatial factor")
        border_h_lat = overlap_y_pix // spatial_factor
        border_w_lat = overlap_x_pix // spatial_factor

        iterator = tqdm(tasks, disable=not show_progress, desc="Tiled encode")
        updated_tile_mem: list[list[torch.Tensor | None] | None] = (
            [None] * len(tasks) if return_tile_mem else []
        )

        with torch.no_grad():
            for tile_id, (y0, y1, x0, x1) in enumerate(iterator):
                pixel_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)

                # Per-tile mem list (only for MemBlocks) passed as mem_in
                mem_in_this = tile_mem_in[tile_id] if (tile_mem_in is not None and tile_mem_in[tile_id] is not None) else None

                if return_tile_mem:
                    enc_tile, mem_out_tile = self.apply(
                        pixel_tile, decode=False, parallel=True,
                        show_progress=False, mem_in=mem_in_this, return_mem=True
                    )
                    updated_tile_mem[tile_id] = mem_out_tile  # store per-layer outputs
                else:
                    enc_tile = self.apply(
                        pixel_tile, decode=False, parallel=True,
                        show_progress=False, mem_in=None, return_mem=False
                    )

                del pixel_tile

                y_lat0 = y0 // spatial_factor
                x_lat0 = x0 // spatial_factor
                tile_h_lat = enc_tile.shape[-2]
                tile_w_lat = enc_tile.shape[-1]
                y_lat1 = min(H_lat_full, y_lat0 + tile_h_lat)
                x_lat1 = min(W_lat_full, x_lat0 + tile_w_lat)
                slice_h = y_lat1 - y_lat0
                slice_w = x_lat1 - x_lat0
                if slice_h != tile_h_lat or slice_w != tile_w_lat:
                    print(f"[TAEVid encode_tiled] Mismatch tile ({tile_h_lat},{tile_w_lat}) vs slice ({slice_h},{slice_w}) "
                          f"at pixel y={y0} x={x0}. Cropping tile.")
                    enc_tile = enc_tile[..., :slice_h, :slice_w]

                mask = self._build_mask(
                    enc_tile,
                    is_bound=(y0 == 0, y1 >= H, x0 == 0, x1 >= W),
                    border_width_hw=(border_h_lat, border_w_lat),
                    blend_mode=blend_mode,
                    blend_exp=blend_exp,
                    min_border_fraction=min_border_fraction,
                ).to(enc_tile.device, enc_tile.dtype)

                values[:, :, :, y_lat0:y_lat1, x_lat0:x_lat1] += enc_tile * mask
                weight[:, :, :, y_lat0:y_lat1, x_lat0:x_lat1] += mask

        weight.clamp_(min=1e-6)
        encoded = (values / weight).to(dtype=x.dtype)

        if return_tile_mem:
            return encoded, updated_tile_mem
        return encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple pass-through decode (kept minimal); original was invalid.
        return self.decode(x)

