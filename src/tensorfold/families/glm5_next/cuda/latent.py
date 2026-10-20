"""GLM's MLA attention on the 512-wide latent cache; kernels compute each row alone, so window rows keep serial bits (docs/recipes/glm-5.3-flash.md)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import LATENT as ENABLED     # off (TF_GLM_LATENT=0): per-head keys and values, 0.3.5's path, for A/B

L = 512            # GLM-5.3-Flash's latent width (kv_lora_rank); the kernels take the width from the tensors
CHUNK = 512        # keys per chunk program, merged in absolute order
KT = 32            # keys per tensor-core tile
HB = 16            # heads per attention tile (all from one row)
HB_WIDE = 32       # prefill chunks: a rank's 32 heads in one tile read each key once (same bits as 16, tested)


def head_block(R: int) -> int:
    """Heads per attention program: a rank's 32 in prompt chunks (keys read once), 16 in decode windows; 16- and 32-row tiles give the same bits."""
    return HB_WIDE if R >= 64 else HB


# ------------------------------------------------------------------------------------------------ weights ---

def dequant_mlx4(w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, group: int = 64) -> torch.Tensor:
    """MLX affine 4-bit rows -> fp32 [out, in]: w uint32 [out, in / 8], 8 nibbles low to high; s, b [out, in / group]."""

    out, words = w.shape
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
    q = (w.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF                        # [out, words, 8]
    q = q.reshape(out, words * 8).to(torch.float32)
    s = s.to(torch.float32).repeat_interleave(group, dim=1)
    b = b.to(torch.float32).repeat_interleave(group, dim=1)
    return q * s + b


class AbsorbW:
    """One DSA layer's kv_b_proj split per head for the latent path: wk [H, 256, 512], wv [H, 256, 512] bf16."""

    def __init__(self, wk: torch.Tensor, wv: torch.Tensor) -> None:
        if wk.dim() != 3 or wv.dim() != 3 or wk.shape[2] != wv.shape[2]:
            raise ValueError("AbsorbW: expected [heads, dim, latent] blocks")
        self.wk = wk.to(torch.bfloat16).contiguous()
        self.wv = wv.to(torch.bfloat16).contiguous()
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]

    @classmethod
    def from_rows(cls, k_rows: torch.Tensor, v_rows: torch.Tensor, heads: int) -> "AbsorbW":
        """k_rows [heads * qk_dim, latent], v_rows [heads * v_dim, latent], float, in head order."""
        lw = k_rows.shape[1]
        return cls(k_rows.reshape(heads, -1, lw), v_rows.reshape(heads, -1, lw))

    def nbytes(self) -> int:
        return self.wk.numel() * 2 + self.wv.numel() * 2


class AbsorbQ4:
    """kv_b split per head in MLX's affine 4-bit layout: words [H, dim, latent / 8], scales and biases [H, dim, latent / 64], key rows then value rows."""

    def __init__(self, k: tuple, v: tuple, heads: int) -> None:
        def per_head(t):
            w, sc, b = t
            return (w.reshape(heads, -1, w.shape[1]).contiguous(), sc.reshape(heads, -1, sc.shape[1]).contiguous(),
                    b.reshape(heads, -1, b.shape[1]).contiguous())

        self.wkw, self.wks, self.wkb = per_head(k)
        self.wvw, self.wvs, self.wvb = per_head(v)
        self.heads, self.qk_dim = self.wkw.shape[0], self.wkw.shape[1]
        self.v_dim = self.wvw.shape[1]
        self.lw = self.wkw.shape[2] * 8
        if self.wks.shape[2] * 64 != self.lw or self.wvw.shape[2] * 8 != self.lw:
            raise ValueError("AbsorbQ4: expected groups of 64 along the latent")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.wkw, self.wks, self.wkb, self.wvw, self.wvs, self.wvb))


# ----------------------------------------------------------------------------------------- absorb, expand ---

@triton.jit
def _absorb_q(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, column block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n] for every row r, k in one sum."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, D)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :]).to(tl.float32)            # [D, BN]
    for r in range(R):
        q = tl.load(Q + (r * H + h) * D + k).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_v(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] for every row r."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, LW)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)          # [BN, LW]
    for r in range(R):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * o[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


@triton.jit
def _absorb_q4(Q, WW, WS, WB, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, RB: tl.constexpr):
    """Program (head, 64-latent group, RB rows): QA = Q @ W with W = q * s + b per group, weights loaded once for the block and each row summed alike."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    rb = tl.program_id(2)
    d = tl.arange(0, D)
    j = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    words = tl.load(WW + (h * D + d[:, None]) * KW + g * 8 + j[None, :])                   # [D, 8]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (D, 64)).to(tl.float32)
    sc = tl.load(WS + (h * D + d) * KG + g).to(tl.float32)
    bi = tl.load(WB + (h * D + d) * KG + g).to(tl.float32)
    n = g * 64 + tl.arange(0, 64)
    for i in tl.static_range(RB):
        r = rb * RB + i
        ok = r < R
        qv = tl.load(Q + (r * H + h) * D + d, mask=ok & (d >= 0), other=0).to(tl.float32)
        acc = tl.sum((qv * sc)[:, None] * qint, axis=0) + tl.sum(qv * bi, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16), mask=ok & (n >= 0))


@triton.jit
def _expand_v4(OL, WW, WS, WB, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, BN outputs): the 4-bit rows unpacked once to fp32, then each row's output as one fp32 sum over the latent, whatever R is."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    kw = tl.arange(0, KW)
    shifts = tl.arange(0, 8) * 4
    words = tl.load(WW + (h * DV + n[:, None]) * KW + kw[None, :])                          # [BN, KW]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, KG, 64)).to(tl.float32)
    gi = tl.arange(0, KG)
    sc = tl.load(WS + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)            # [BN, KG]
    bi = tl.load(WB + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)
    w = tl.reshape(qint * sc[:, :, None] + bi[:, :, None], (BN, LW))
    k = tl.arange(0, LW)
    for r in range(R):
        x = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


@triton.jit
def _absorb_mma(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """Prompt rows, program (head, row block, column block): QA = Q @ WK on the tensor cores, each row one fp32 chain
    over ascending k blocks (its bits do not depend on the other rows)."""

    h = tl.program_id(0)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ok = rm < R
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        q = tl.load(Q + (rm[:, None] * H + h) * D + k0 + kk[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(WK + (h * D + k0 + kk[:, None]) * LW + n[None, :])
        acc = tl.dot(q, w, acc)
    tl.store(QA + (rm[:, None] * H + h) * LW + n[None, :], acc.to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _expand_mma(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """Prompt rows, program (head, row block, output block): OUT = OL @ WV^T on the tensor cores, as _absorb_mma."""

    h = tl.program_id(0)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ok = rm < R
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, LW, BK):
        o = tl.load(OL + (rm[:, None] * H + h) * LW + k0 + kk[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(WV + (h * DV + n[:, None]) * LW + k0 + kk[None, :])
        acc = tl.dot(o, tl.trans(w), acc)
    tl.store(OUT + (rm[:, None] * H + h) * DV + n[None, :], acc.to(tl.bfloat16), mask=ok[:, None])


# prompt chunks' tiles (rows, columns, k) and (warps, stages); fixed, so a row's bits never depend on the chunk
PROMPT_MMA = __import__("os").environ.get("TF_GLM_LATENT_MMA", "1") != "0"    # 0: prompts on the decode kernels
MMA_TILE = (64, 128, 64)
MMA_LAUNCH = (4, 3)


def row_block(R: int) -> int:
    """Rows per program: 1 for decode windows (most parallel), 16 for prefill chunks (weights reused)."""
    return 1 if R <= 16 else 16


def absorb_q(q: torch.Tensor, a, out: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """q [R, H, qk_dim] bf16 -> out [R, H, latent] bf16; ``prompt``: a prompt chunk's rows, on the tensor cores
    (other bits than decode rows', the same for any chunking)."""
    R, H, D = q.shape
    if prompt and isinstance(a, AbsorbW) and q.is_contiguous():
        BM, BN, BK = MMA_TILE
        warps, stages = MMA_LAUNCH
        _absorb_mma[(H, triton.cdiv(R, BM), a.lw // BN)](q, a.wk, out, R, H=H, D=D, LW=a.lw, BM=BM, BN=BN, BK=BK,
                                                          num_warps=warps, num_stages=stages)
        return out
    if isinstance(a, AbsorbQ4):
        rb = row_block(R)
        _absorb_q4[(H, a.lw // 64, triton.cdiv(R, rb))](q, a.wkw, a.wks, a.wkb, out, R, H=H, D=D, LW=a.lw, RB=rb,
                                                        num_warps=4)
        return out
    BN = 32
    _absorb_q[(H, a.lw // BN)](q, a.wk, out, R, H=H, D=D, LW=a.lw, BN=BN, num_warps=4)
    return out


def expand_v(o_lat: torch.Tensor, a, out: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """o_lat [R, H, latent] bf16 -> out [R, H, v_dim] bf16; ``prompt`` as in absorb_q."""
    R, H, _ = o_lat.shape
    if prompt and isinstance(a, AbsorbW) and o_lat.is_contiguous():
        BM, BN, BK = MMA_TILE
        warps, stages = MMA_LAUNCH
        _expand_mma[(H, triton.cdiv(R, BM), a.v_dim // BN)](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw, BM=BM,
                                                             BN=BN, BK=BK, num_warps=warps, num_stages=stages)
        return out
    if isinstance(a, AbsorbQ4):
        BN = 16
        _expand_v4[(H, a.v_dim // BN)](o_lat, a.wvw, a.wvs, a.wvb, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN,
                                        num_warps=4)
        return out
    BN = 16
    _expand_v[(H, a.v_dim // BN)](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw, BN=BN, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ caches ---

@triton.jit
def _lat_write(LAT, lat_stride, LC, POS, LW: tl.constexpr):
    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    k = tl.arange(0, LW)
    tl.store(LC + (P + r) * LW + k, tl.load(LAT + r * lat_stride + k))


def latent_write(lat: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor) -> None:
    """lat [R, latent] bf16 rows into cache slots pos .. pos + R - 1 (pos read on the device)."""
    _lat_write[(lat.shape[0],)](lat, lat.stride(0), cache, pos, LW=cache.shape[1], num_warps=4)


# --------------------------------------------------------------------------------------------- attention ---

@triton.jit
def _tile(q, kv, m, l, o, valid, SCALE: tl.constexpr):
    """One key tile for HB heads of one row: kv [KT, 512] is both key and value (the latent)."""
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _tile_v(q, kv, vv, m, l, o, valid, SCALE: tl.constexpr):
    """_tile with the values a column slice vv [KT, VB] of the keys kv [KT, 512]."""
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _dense_chunks(QA, LC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, CH: tl.constexpr,
                  SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH)."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H                                                      # tile rows past the last head are padding
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r                                                     # keys 0 .. P + r are visible to row r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks(QA, LC, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _onepass_ids(row, t0, n, KTT: tl.constexpr):
    """Tile t0's token ids (0 past n, where their rows are masked)."""
    idx = t0 + tl.arange(0, KTT)
    return tl.load(row + idx, mask=idx < n, other=0).to(tl.int64)


@triton.jit
def _onepass_rows(LC, tok, t0, n, k, LW: tl.constexpr, KTT: tl.constexpr):
    """Tile t0's latent rows [KTT, LW]; rows past n are not read (zeros)."""
    ok = t0 + tl.arange(0, KTT) < n
    return tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0)


@triton.jit
def _sparse_onepass(QA, LC, TOK, CNT, OUT, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr, SCALE: tl.constexpr,
                    HBT: tl.constexpr, KTT: tl.constexpr, SPLIT: tl.constexpr, STAGES: tl.constexpr,
                    PREFETCH: tl.constexpr):
    """Prompt rows, program (row, head group, latent slice): HBT heads of row r over all its selected tokens in one
    online softmax, KTT-key tiles in list order (_tile's math), straight to OUT[r, h] bf16: no partials, no merge.
    PREFETCH: tile t + 1's gather is issued into registers before tile t's dots (a load blocks only at first use),
    with Triton's pipeliner off; else the loop is left to Triton's pipeliner at STAGES. SPLIT > 1: the program keeps
    LW / SPLIT output columns and recomputes the full-width scores. PREFETCH never changes the arithmetic; the tile
    (HBT, KTT, SPLIT) is fixed (ONEPASS_TILE), so a row's bits depend on its q, its token list and the cache only,
    never on the other rows or R. Rows with CNT 0 are left alone."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    vs = tl.program_id(2)
    n = tl.load(CNT + r)
    if n == 0:
        return
    VB: tl.constexpr = LW // SPLIT
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    v = vs * VB + tl.arange(0, VB)
    q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, VB), tl.float32)
    row = TOK + r * W
    if PREFETCH:
        # ids run two tiles ahead and rows one: tile t + 1's gather is issued (its ids already there) before tile t's
        # dots, so neither load's latency stalls the in-order warp in front of the tensor-core work
        tok = _onepass_ids(row, 0, n, KTT)
        kv = _onepass_rows(LC, tok, 0, n, k, LW, KTT)
        tok_next = _onepass_ids(row, KTT, n, KTT)
        for t0 in range(0, n, KTT):
            ok = t0 + tl.arange(0, KTT) < n
            kv_next = _onepass_rows(LC, tok_next, t0 + KTT, n, k, LW, KTT)
            tok_after = _onepass_ids(row, t0 + 2 * KTT, n, KTT)
            if SPLIT == 1:
                m, l, o = _tile(q, kv.to(tl.bfloat16), m, l, o, ok, SCALE)
            else:
                vv = tl.load(LC + tok[:, None] * LW + v[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
                m, l, o = _tile_v(q, kv.to(tl.bfloat16), vv, m, l, o, ok, SCALE)
            tok = tok_next
            kv = kv_next
            tok_next = tok_after
    else:
        for t0 in tl.range(0, n, KTT, num_stages=STAGES):
            ok = t0 + tl.arange(0, KTT) < n
            tok = _onepass_ids(row, t0, n, KTT)
            kv = _onepass_rows(LC, tok, t0, n, k, LW, KTT)
            if SPLIT == 1:
                m, l, o = _tile(q, kv.to(tl.bfloat16), m, l, o, ok, SCALE)
            else:
                vv = tl.load(LC + tok[:, None] * LW + v[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
                m, l, o = _tile_v(q, kv.to(tl.bfloat16), vv, m, l, o, ok, SCALE)
    tl.store(OUT + (r * H + hh[:, None]) * LW + v[None, :], (o / l[:, None]).to(tl.bfloat16), mask=hok[:, None])


@triton.jit
def _merge(PO, PM, PL, OUT, CNT, R, H: tl.constexpr, LW: tl.constexpr, NCH: tl.constexpr, SPARSE: tl.constexpr):
    """Program (row, head): the row's chunk partials in chunk order -> OUT[r, h] bf16. Sparse: rows with CNT 0 skip."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    if SPARSE:
        if tl.load(CNT + r) == 0:
            return
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * LW + k)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * LW + k, (o / l).to(tl.bfloat16))


class LatentScratch:
    """Chunk partials for ``part_rows`` (one ``attention`` call) x heads x chunks; queries and latents for ``rows``."""

    def __init__(self, rows: int, heads: int, chunks: int, device, lw: int = L, part_rows: int | None = None) -> None:
        part_rows = rows if part_rows is None else min(rows, part_rows)
        self.rows, self.part_rows, self.heads, self.nch, self.lw = rows, part_rows, heads, chunks, lw
        self.po = torch.empty((chunks * part_rows * heads * lw,), dtype=torch.float32, device=device)
        self.pm = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.qa = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.ol = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.dummy = torch.zeros((1,), dtype=torch.int32, device=device)


def attention(qa: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, s: LatentScratch, *, scale: float,
              nch: int, out: torch.Tensor, hb: int | None = None) -> torch.Tensor:
    """Dense causal attention of qa [R, H, 512] over the cache through pos + R - 1, visiting nch 512-key chunks (empty ones skipped) -> out [R, H, 512].

    Row r's programs read only its query, keys 0 .. pos + r and its own partials, so the first rows of a window
    computed alone (``qa[:n]``, the same nch and ``hb``: the window's ``head_block``) get the window's bits."""
    R, H, LW = qa.shape
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    hb = head_block(R) if hb is None else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a tile, not {HB} or {HB_WIDE}")
    _dense_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, pos, s.po[:n * LW], s.pm[:n], s.pl[:n], R, H=H, LW=LW,
                                                CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT, num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float) -> None:
    """Attention of rows with counts > 0 over their selected tokens (ascending, -1 padded), written into out; other rows are left alone."""
    R, H, LW = qa.shape
    W = tokens.shape[1]
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    hb = head_block(R)
    _sparse_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, tokens, counts, po, pm, pl, R, W=W, H=H, LW=LW,
                                                 CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT, num_warps=8, num_stages=1)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


# prompt chunks' sparse attention in one pass (TF_GLM_SPARSE_ONEPASS=0: the chunk programs and their merge). Fixed
# tile (heads, keys, latent slices) and pipeline depth, so a row's bits never depend on the chunk it sits in.
SPARSE_ONEPASS = __import__("os").environ.get("TF_GLM_SPARSE_ONEPASS", "1") != "0"
ONEPASS_TILE = (32, 32, 1)          # heads per program (a rank's 32), keys per tile, latent slices: sets the bits
ONEPASS_LAUNCH = (8, 1, True)       # warps, Triton pipeline stages, register prefetch of the next tile: speed only


def sparse_onepass(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                   out: torch.Tensor, scale: float, *, tile: tuple | None = None, launch: tuple | None = None) -> None:
    """sparse_attention for a prompt chunk's rows: one program a row (and head group) walks every selected token in
    one online softmax and writes out[r] directly, no fp32 partials; rows with counts 0 are left alone. Other bits
    than sparse_attention's (one chain instead of merged 512-token chunks), the same for any chunking of the rows.
    ``tile`` / ``launch`` override ONEPASS_TILE / ONEPASS_LAUNCH (benchmarks)."""
    R, H, LW = qa.shape
    if not (qa.is_contiguous() and out.is_contiguous() and tokens.is_contiguous() and cache.stride() == (LW, 1)):
        raise ValueError("sparse_onepass: expected contiguous queries, output and token lists, cache rows of the width")
    if R == 0:
        return
    hbt, kt, split = tile or ONEPASS_TILE
    warps, stages, prefetch = launch or ONEPASS_LAUNCH
    hbt = max(16, min(hbt, triton.next_power_of_2(H)))            # tensor-core tiles need 16 rows
    _sparse_onepass[(R, triton.cdiv(H, hbt), split)](qa, cache, tokens, counts, out, W=tokens.shape[1], H=H, LW=LW,
                                                     SCALE=scale, HBT=hbt, KTT=kt, SPLIT=split, STAGES=stages,
                                                     PREFETCH=bool(prefetch), num_warps=warps,
                                                     num_stages=1 if prefetch else stages)


def chunks_for(length: int) -> int:
    return triton.cdiv(length, CHUNK)
