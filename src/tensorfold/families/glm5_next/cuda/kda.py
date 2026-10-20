"""Fused KDA chains and prefix replay use the same state update routine so a kept prefix preserves serial bits."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

DK = DV = 128


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_v2", sources=[str(here / "kda.cpp"), str(here / "kda.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


class KDAScratch:
    """Static window outputs and replay inputs, with optional views into shared storage so all layers can replay together."""

    def __init__(self, rows: int, heads: int, device, parent: "KDAScratchSet | None" = None, index: int = 0) -> None:
        if parent is None:
            self.out = torch.empty((rows, heads * DV), dtype=torch.bfloat16, device=device)
            self.k = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.v = torch.empty((rows, heads, DV), dtype=torch.bfloat16, device=device)
            self.g = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.b = torch.empty((rows, heads), dtype=torch.float32, device=device)
        else:
            self.out = parent.out[index]
            self.k, self.v, self.g, self.b = parent.k[index], parent.v[index], parent.g[index], parent.b[index]


class KDAScratchSet:
    """KDAScratch for ``layers`` layers in one allocation each."""

    def __init__(self, layers: int, rows: int, heads: int, device) -> None:
        self.layers, self.rows, self.heads = layers, rows, heads
        self.out = torch.empty((layers, rows, heads * DV), dtype=torch.bfloat16, device=device)
        self.k = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((layers, rows, heads, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.b = torch.empty((layers, rows, heads), dtype=torch.float32, device=device)
        self.views = [KDAScratch(rows, heads, device, self, i) for i in range(layers)]


def replay_layers(state_in: torch.Tensor, scratch: KDAScratchSet, rows: int, state_out: torch.Tensor) -> None:
    """Every layer's state after the first ``rows`` rows: state_in/state_out [layers, H, 128, 128]."""

    L, H = scratch.layers, scratch.heads
    _ext().replay_layers(state_in, H * DV * DK, scratch.k, scratch.v, scratch.g, scratch.b, scratch.rows * H * DK,
                         scratch.rows * H, L, H, int(rows), state_out)


# windows of this many rows or more run the chain in three kernels (prep / step / out: 8 blocks a head instead of one;
# chain_kernel's bits). TF_GLM_KDA_WIDE_ROWS, default 1: every window (decode windows measured 4-36% faster at 1-8
# rows on GB10); 64: the one-block chain below 64 rows, as before.
WIDE_ROWS = int(os.environ.get("TF_GLM_KDA_WIDE_ROWS", "1"))
# prompt chunks in chunked (WY) form, 64-row sub-chunks at absolute positions (kda_chunked.py): close, not the same bits
CHUNKED = os.environ.get("TF_GLM_KDA_CHUNKED", "0") == "1"
_tmp: dict = {}
_outgrown: list = []


def _wide_scratch(rows: int, heads: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The normalized q and the read-out of a long window, shared by every layer (they run one after another).
    ``reserve`` sizes them before any CUDA graph is captured: a graph keeps the addresses it was captured with."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (heads, device)
    q, y = _tmp.get(key, (None, None))
    if q is None or q.shape[0] < rows:
        if q is not None:                # a captured graph may still use the smaller pair: keep it alive
            _outgrown.append((q, y))
        q = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
        y = torch.empty((rows, heads, DV), dtype=torch.bfloat16, device=device)
        _tmp[key] = (q, y)
    return q, y


def reserve(rows: int, heads: int, device) -> None:
    """The wide path's scratch for windows of up to ``rows`` rows, allocated now (before graph capture)."""

    _wide_scratch(rows, heads, device)


def chain(p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor, conv_state: torch.Tensor,
          conv_w: torch.Tensor, state_in: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
          norm_w: torch.Tensor, eps: float, lower: float, rows: int, scratch: KDAScratch,
          state_out: torch.Tensor, *, wide: bool | None = None, pos: int | None = None) -> torch.Tensor:
    """Run projection rows p [q | k | v | ... | b at b_off ...] and bf16 gate rows a and g; windows of WIDE_ROWS rows or more take the three-kernel path, same bits; a prompt chunk (``pos``: its first row's position) takes the chunked form under TF_GLM_KDA_CHUNKED=1 (scratch's k/v/g/b are not written)."""

    if CHUNKED and pos is not None:
        from . import kda_chunked

        return kda_chunked.chain(p, b_off, a, g, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, eps, lower,
                                 rows, scratch.out, state_out, pos)
    if wide if wide is not None else rows >= WIDE_ROWS:
        q_tmp, y_tmp = _wide_scratch(rows, a_log.numel(), p.device)
        _ext().chain_wide(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in,
                          a_log, dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out,
                          scratch.k, scratch.v, scratch.g, scratch.b, q_tmp, y_tmp)
        return scratch.out[:rows]
    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out, scratch.k, scratch.v,
                 scratch.g, scratch.b)
    return scratch.out[:rows]


def replay(state_in: torch.Tensor, scratch: KDAScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)
