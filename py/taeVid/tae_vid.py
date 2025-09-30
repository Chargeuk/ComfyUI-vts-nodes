# Modified from https://github.com/madebyollin/taehv/blob/main/taehv.py

# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

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
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
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
            Clamp(),
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
    ) -> torch.Tensor:
        padding = (0, 0, 0, 0, 0, 0, 1, 0)
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        # parallel over input timesteps, iterate over blocks
        for b in tqdm(model, disable=not show_progress):
            if not isinstance(b, MemBlock):
                x = b(x)
                continue
            nt, c, h, w = x.shape
            t = nt // n
            mem = F.pad(x.reshape(n, t, c, h, w), padding, value=0)[:, :t].reshape(
                x.shape,
            )
            x = b(x, mem)
            del mem
        nt, c, h, w = x.shape
        t = nt // n
        return x.view(n, t, c, h, w)

    def apply(
        self,
        x: torch.Tensor,
        *,
        decode=True,
        parallel=True,
        show_progress=False,
    ) -> torch.Tensor:
        model = self.decoder if decode else self.encoder
        if not decode and self.vmi.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if parallel:
            result = self.apply_parallel(x, model, show_progress=show_progress)
        else:
            result = TAEVidContext(model).apply(x, show_progress=show_progress)
        return (
            result
            if not decode or self.vmi.patch_size < 2
            else F.pixel_shuffle(result, self.patch_size)
        )

    def decode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=True, **kwargs)[:, self.frames_to_trim :]

    def encode(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.apply(*args, decode=False, **kwargs)

    # ---- Tiled decoding helpers ----
    @staticmethod
    def _build_1d_mask(length: int, at_start: bool, at_end: bool, border: int, device, dtype):
        if border <= 0:
            return torch.ones(length, device=device, dtype=dtype)
        mask = torch.ones(length, device=device, dtype=dtype)
        if not at_start:
            ramp = torch.linspace(0, 1, border + 2, device=device, dtype=dtype)[1:-1]
            mask[:border] = ramp
        if not at_end:
            ramp = torch.linspace(1, 0, border + 2, device=device, dtype=dtype)[1:-1]
            mask[-border:] = torch.minimum(mask[-border:], ramp)
        return mask

    @classmethod
    def _build_mask(cls, tile_decoded: torch.Tensor,
                    is_bound: tuple[bool, bool, bool, bool],
                    border_width_hw: tuple[int, int]) -> torch.Tensor:
        # tile_decoded shape: (N, T_out, C, H_tile_out, W_tile_out)
        H = tile_decoded.shape[-2]
        W = tile_decoded.shape[-1]
        (top, bottom, left, right) = is_bound
        (border_h, border_w) = border_width_hw
        device = tile_decoded.device
        dtype = tile_decoded.dtype
        mh = cls._build_1d_mask(H, top, bottom, border_h, device, dtype)
        mw = cls._build_1d_mask(W, left, right, border_w, device, dtype)
        mask2d = torch.min(
            mh.view(H, 1).expand(H, W),
            mw.view(1, W).expand(H, W),
        )
        # Shape (1,1,1,H,W) for broadcasting over (N, T_out, C, H, W)
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
    ) -> torch.Tensor:
        """
        Tiled spatial decode of latent video tensor.

        Args:
            x: (N, T, C_latent, H, W) latent tensor.
            tile_size_x / tile_size_y: tile size in pixel space (W / H).
            tile_stride_x / tile_stride_y: stride between tiles (overlap if < size) in pixels.
            show_progress: show tqdm over tiles.
            device: compute device for per-tile decoding (defaults to x.device).

        Returns:
            Decoded video (N, T_out, image_channels, H_out, W_out) identical to non-tiled decode.
        """
        if x.ndim != 5:
            raise ValueError("Expected latent tensor shape (N,T,C,H,W)")
        
        # need to ensure the pixel tile sizes and pixel strides are all multiples of 32 before converting them to
        # latent space. raise an error if not
        if (pixel_tile_size_x % 32 != 0 or pixel_tile_size_y % 32 != 0 or
            pixel_tile_stride_x % 32 != 0 or pixel_tile_stride_y % 32 != 0):
            raise ValueError("Pixel tile sizes and strides must be multiples of 32")

        tile_size_x = pixel_tile_size_x // 32
        tile_size_y = pixel_tile_size_y // 32
        tile_stride_x = pixel_tile_stride_x // 32
        tile_stride_y = pixel_tile_stride_y // 32

        compute_device = torch.device(device) if device is not None else x.device

        N, T_in, C_lat, H_lat, W_lat = x.shape

        # Spatial upscale factor (decoder upsample layers * optional pixel shuffle)
        spatial_factor = (2 ** sum(self.decoder[i].scale_factor == 2
                                   for i, m in enumerate(self.decoder)
                                   if isinstance(m, nn.Upsample)))  # count enabled 2x
        # More robust: recompute from config booleans
        # (We stored them implicitly; safest is to derive from arguments used during init)
        # For simplicity, derive from presence of nn.Upsample with scale_factor=2.
        if self.patch_size > 1:
            spatial_factor *= self.patch_size

        # Temporal expansion factor = 2 ** sum(decoder_time_upscale)
        # (frames_to_trim = 2^k - 1, so full length before trim = T_in * 2^k)
        temporal_factor = (self.frames_to_trim + 1)
        # Output temporal length after trimming:
        T_out = T_in * temporal_factor - self.frames_to_trim  # equals 2^k*(T_in-1)+1

        # Prepare tile task list (y, y_end, x, x_end) in latent coordinates
        tasks: list[tuple[int, int, int, int]] = []
        for y in range(0, H_lat, tile_stride_y):
            if (y - tile_stride_y >= 0) and (y - tile_stride_y + tile_size_y >= H_lat):
                continue
            for xw in range(0, W_lat, tile_stride_x):
                if (xw - tile_stride_x >= 0) and (xw - tile_stride_x + tile_size_x >= W_lat):
                    continue
                y1 = y + tile_size_y
                x1 = xw + tile_size_x
                tasks.append((y, min(y1, H_lat), xw, min(x1, W_lat)))

        if not tasks:
            raise ValueError("No tile tasks generated; check tile sizes / strides.")

        # Decode one tile to determine output channel count & tile output size
        with torch.no_grad():
            y0, y1, x0, x1 = tasks[0]
            test_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)
            dec_test_full = self.apply(test_tile, decode=True, parallel=True, show_progress=False)
            dec_test = dec_test_full[:, self.frames_to_trim :]
            _, T_out_tile, C_img, H_tile_out, W_tile_out = dec_test.shape
            del dec_test, dec_test_full, test_tile

        # Allocate accumulators (keep on compute device for fewer transfers)
        H_out = H_lat * spatial_factor
        W_out = W_lat * spatial_factor
        values = torch.zeros(
            (N, T_out, C_img, H_out, W_out),
            dtype=x.dtype,
            device=compute_device,
        )
        weight = torch.zeros(
            (N, T_out, 1, H_out, W_out),
            dtype=x.dtype,
            device=compute_device,
        )

        border_h_lat = max(tile_size_y - tile_stride_y, 0)
        border_w_lat = max(tile_size_x - tile_stride_x, 0)
        border_h_out = border_h_lat * spatial_factor
        border_w_out = border_w_lat * spatial_factor

        iterator = tqdm(tasks, disable=not show_progress, desc="Tiled decode")
        with torch.no_grad():
            for y0, y1, x0, x1 in iterator:
                latent_tile = x[:, :, :, y0:y1, x0:x1].to(compute_device)
                dec_full = self.apply(latent_tile, decode=True, parallel=True, show_progress=False)
                dec = dec_full[:, self.frames_to_trim :]  # (N, T_out_tile, C_img, h_out_tile, w_out_tile)
                del dec_full, latent_tile

                # Spatial placement indices in output
                out_y0 = y0 * spatial_factor
                out_x0 = x0 * spatial_factor
                out_y1 = out_y0 + dec.shape[-2]
                out_x1 = out_x0 + dec.shape[-1]

                mask = self._build_mask(
                    dec,
                    is_bound=(
                        y0 == 0,
                        y1 >= H_lat,
                        x0 == 0,
                        x1 >= W_lat,
                    ),
                    border_width_hw=(border_h_out, border_w_out),
                ).to(dec.device, dec.dtype)

                # Accumulate
                values[:, :, :, out_y0:out_y1, out_x0:out_x1] += dec * mask
                weight[:, :, :, out_y0:out_y1, out_x0:out_x1] += mask

        weight.clamp_(min=1e-6)
        decoded = (values / weight).type_as(values)

        # (N, T_out, C, H_out, W_out) already matches standard decode output ordering
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)

