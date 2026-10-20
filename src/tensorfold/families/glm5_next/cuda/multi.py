"""GLM-5.3-Flash's concurrent requests (--parallel N) on two ranks: every stream commits exactly its own rows, so each
reply is the one its request gets served alone with ``"draft": false``.

Layout: the per-token caches are one pool (``pool.py``, ``forward.Caches``); a stream, or a kept prompt, owns an
extent of it aligned to 2,048 tokens, and the single-stream code (prompt chunks, verify windows, commits, snapshots)
runs on a ``forward.State`` viewing that extent and the stream's slot (``forward.Slots``: KDA states, conv windows,
window scratch; a DFlash2 context, ``dflash2_multi.DraftContext``; host side its copy-draft index, sampler and grammar).

Iterations (``GlmScheduler``, one worker thread on rank 0): admissions, then either one prompt chunk of the oldest
filling stream (FIFO, at most TF_GLM_FILL_ROWS rows while others decode, the engine's prompt chunk alone) or one
decode round, alternating by TF_GLM_FILL_SHARE; a prompt chunk never shares a forward with decode rows. A round: each
decoding stream (admission order) proposes drafts under its own DFlash2 policy (copy drafts first), the combined
window is capped at ``MAX_WINDOW`` rows (the longest tails trimmed), one verify over every stream's rows
(``forward_streams``: ``verify.BatchedVerify``, one forward over every stream's rows through the segmented KDA and
DSA kernels, or ``verify.SerialVerify``, the reference running each window as its own forward; TF_GLM_MULTI_VERIFY),
one packed all-gather for the samplers, each stream accepts and commits its own.

Rank 0 decides everything that depends on time, clients or placement; rank 1 applies it. Each iteration rank 0 sends
ONE message (``GlmEngine._share``): ops ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE, which both ranks apply
in order through the same code (rank 0 as it decides them). Rank 1 derives drafts itself and checks the depths rank 0
sent. An idle rank 1 waits on the doorbell (``GlmEngine._await_bell``), not in an all-gather. After an error the two
ranks cannot be trusted to agree: ``broken`` latches and every later call fails until both restart.

Multi-stream mode drafts with DFlash2 only: MTP policies are remapped to DFlash2 ones (drafts never change a reply)."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream, next_fill

from .pool import ALIGN, Pool, align_up
from .verify import BatchedVerify, Segment, SerialVerify, Verified

ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE = range(1, 9)
OPS = {ADMIT: "ADMIT", EVICT: "EVICT", MOVE: "MOVE", GROW: "GROW", FILL: "FILL", ROUND: "ROUND", FINISH: "FINISH",
       IDLE: "IDLE"}
DONE, CANCELLED, REQUEUED = 0, 1, 2      # FINISH reasons
MAX_WINDOW = 32                          # rows of every stream's windows together in one round
SHARED_SLOTS = 2                         # shared-prefix points an ADMIT carries (engine.SHARED_MOST)


# -- settings ------------------------------------------------------------------------------------------------------
def fill_rows(prefill_rows: int, grid: int = 0, value: str | None = None) -> int:
    """TF_GLM_FILL_ROWS: rows of a prompt chunk while other streams decode (default 1,024 or the engine's prompt
    chunk if smaller): a multiple of 64 and of the prompt grid, at most the prompt chunk (TF_GLM_PREFILL_ROWS)."""

    value = os.environ.get("TF_GLM_FILL_ROWS", "") if value is None else value
    value = value.strip()
    rows = min(1024, prefill_rows) if value == "" else int(value) if value.isdecimal() else -1
    if rows < 64 or rows % 64 or rows > prefill_rows or (grid and rows % grid):
        raise ValueError(f"TF_GLM_FILL_ROWS: a multiple of 64{f' and of the {grid}-token prompt grid' if grid else ''}"
                         f" up to the {prefill_rows}-row prompt chunk, not {value!r}")
    return rows


def fill_share(value: str | None = None) -> float:
    """TF_GLM_FILL_SHARE: the share of iterations prompt chunks take while other streams decode (default 0.5: a
    chunk, a round, a chunk, ...); 1 fills every prompt before decoding on."""

    value = os.environ.get("TF_GLM_FILL_SHARE", "") if value is None else value
    try:
        share = 0.5 if value.strip() == "" else float(value)
    except ValueError:
        share = -1.0
    if not 0 < share <= 1:
        raise ValueError(f"TF_GLM_FILL_SHARE: a share above 0 and at most 1, not {value!r}")
    return share


def multi_code(code: list[int], dflash: bool, dflash_policy: list[int]) -> list[int]:
    """A request's policy code under --parallel: serial stays serial; auto becomes TF_GLM_DFLASH_POLICY; an MTP
    policy its DFlash2 twin; without a draft model, serial."""

    kind = code[0]
    if kind == 0 or not dflash:
        return [0, 0, 0, 0]
    if kind in (4, 5):
        return list(dflash_policy)
    return [kind + 10] + list(code[1:]) if kind < 10 else list(code)


def trim(drafts: list[list[int]], cap: int = MAX_WINDOW) -> list[list[int]]:
    """Cut drafts until every window (a pending token and its drafts) fits ``cap`` rows together: one draft at a
    time off the longest window (the latest of equals)."""

    drafts = [list(d) for d in drafts]
    total = sum(1 + len(d) for d in drafts)
    while total > cap:
        k = max(range(len(drafts)), key=lambda i: (len(drafts[i]), i))
        if not drafts[k]:
            break
        drafts[k].pop()
        total -= 1
    return drafts


def _f64(x: float) -> list[int]:
    import struct

    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _unf64(lo: int, hi: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def pack_sampling(sampling) -> list[int]:
    """10 int32 words: the seed (3 words of 31, 31 and 2 bits), temperature, top_k, top_p, min_p."""

    seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
    return [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
            *_f64(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
            *_f64(sampling.top_p if sampling else 1.0), *_f64(sampling.min_p if sampling else 0.0)]


def unpack_sampling(w: Sequence[int]):
    from tensorfold.engine.exact_sampling import Sampling

    s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi = w
    temperature = _unf64(t_lo, t_hi)
    if temperature <= 0:
        return None
    return Sampling((s_top << 62) | (s_hi << 31) | s_lo, temperature, top_k, _unf64(p_lo, p_hi), _unf64(m_lo, m_hi))


# -- DFlash2 contexts per slot ---------------------------------------------------------------------------------------
def draft_window(ctx, n: int) -> list[torch.Tensor]:
    """A copy of a stream's DFlash2 window rows before n (``dflash2_multi.DraftContext``, or a solo ``Drafter``):
    what a later block pass at n or past it reads. A ring's are copied by ``take_snapshot`` already."""

    from .decode import _drafter_views, _ring_window

    return _ring_window(ctx, n) if getattr(ctx, "ring", 0) else [v.clone() for v in _drafter_views(ctx, n)]


def put_draft_window(ctx, n: int, rows: list[torch.Tensor]) -> None:
    """Put a kept prompt's DFlash2 window back and end the context at n."""

    from .decode import _drafter_views, _put_ring_window

    if getattr(ctx, "ring", 0):
        _put_ring_window(ctx, n, rows)
    else:
        views = _drafter_views(ctx, n)
        if len(views) != len(rows):
            raise ValueError("a kept state's DFlash2 window does not match the drafter's context")
        for dst, src in zip(views, rows):
            dst.copy_(src)
    ctx.context_end = n
    ctx.pos_dev.fill_(n)


# -- the verify seam -------------------------------------------------------------------------------------------------
def verify_kind(value: str | None = None) -> str:
    """TF_GLM_MULTI_VERIFY: how a round's windows are verified: batched (the default: one forward over every stream's
    rows, the segmented kernels) or serial (the reference: each stream's window as its own forward)."""

    value = (os.environ.get("TF_GLM_MULTI_VERIFY", "") if value is None else value).strip().lower() or "batched"
    if value not in ("batched", "serial"):
        raise ValueError(f"TF_GLM_MULTI_VERIFY: batched or serial, not {value!r}")
    return value


def forward_streams(verify, segments: Sequence[Segment]) -> Verified:
    """The seam: every segment's logits (and taps) of one round, by ``verify`` (``verify.SerialVerify`` or
    ``verify.BatchedVerify``, the same two calls)."""

    return verify.forward(segments)


def sample_streams(w, parts: Sequence[tuple]) -> list[list[int]]:
    """parts [(logits [R, V/world], positions, sampling)] -> each part's tokens, as ``decode.sample_rows`` samples
    each alone: every top-k (and greedy) part's candidates in ONE all-gather, a nucleus part (top_k off) through the
    shared nucleus rule (its own gathers), in part order on every rank."""

    from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
    from tensorfold.engine.exact_sampling import MARGIN, choose_rows

    out: list[list[int] | None] = [None] * len(parts)
    packed, layout = [], []
    for k, (logits, positions, sampling) in enumerate(parts):
        greedy = sampling is None or sampling.temperature <= 0
        if not greedy and not sampling.top_k:
            continue
        n = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
        vals, ids = torch.topk(logits.float(), n, dim=-1)
        ids = (ids + w.vocab_offset).to(torch.int32)
        packed.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
        layout.append((k, logits.shape[0], n))
    if packed:
        flat = torch.cat(packed) if len(packed) > 1 else packed[0]
        if w.comm is None:
            host = flat.view(1, -1).cpu()
        else:
            got = torch.empty((w.world * flat.numel(),), dtype=torch.float32, device=flat.device)
            w.comm.all_gather(flat.contiguous(), got)
            host = got.view(w.world, -1).cpu()
        at = 0
        for k, R, n in layout:
            g = host[:, at:at + R * 2 * n].reshape(host.shape[0], R, 2 * n)
            at += R * 2 * n
            values = torch.cat([g[r, :, :n] for r in range(g.shape[0])], dim=1).numpy().astype(np.float32)
            tokens = torch.cat([g[r, :, n:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                               dim=1).numpy().astype(np.int64)
            _, positions, sampling = parts[k]
            if sampling is None or sampling.temperature <= 0:
                order = np.lexsort((tokens, -values), axis=-1)
                out[k] = [int(tokens[i, order[i, 0]]) for i in range(R)]
            else:
                out[k] = choose_rows(values, tokens, positions, sampling)
    for k, (logits, positions, sampling) in enumerate(parts):
        if out[k] is None:
            out[k] = nucleus_rows(logits, positions, sampling, offset=w.vocab_offset,
                                  gather=one_rank if w.comm is None else comm_gather(w.comm))
    return out


# -- streams -------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class Lane:
    """An admitted stream: its request (``Stream``), slot, extent, state views and decoding state (both ranks)."""

    s: Stream
    sid: int
    slot: int
    extent: Any
    st: Any
    order: int
    code: list[int]
    spec: str = ""
    policy: Any = None                 # decode.DepthPolicy (DFlash2), None: one row a round
    dflash: bool = False               # the stream drafts with DFlash2 (its slot's context holds its taps)
    depth: int = 0
    copies: Any = None
    feed: Any = None
    constraint: Any = None
    window: Any = None                 # the grammar's window of this round's rows
    stops: list[int] = field(default_factory=list)   # prompt positions whose states are kept (before the end)
    point: int | None = None           # where the prompt's own state is kept (engine.grid_point), or None
    shared: list[int] = field(default_factory=list)
    decoding: bool = False
    paused: bool = False
    t0: float = 0.0
    copy_rounds: int = 0
    copy_drafted: int = 0
    copy_accepted: int = 0


class NoRoom(RuntimeError):
    """The pool cannot place a stream now (it waits for others to finish)."""


class MultiDecoder:
    """The ``GlmScheduler``'s decoder, on rank 0 (deciding) and rank 1 (``follow``, applying)."""

    def __init__(self, engine, streams: int, *, verify=None, drafts=None, draft_graphs: bool | None = None) -> None:
        """``drafts``: the multi-stream drafter (``dflash2_multi.MultiDrafter``; default one over the engine's
        drafter, its graphs captured unless TF_GLM_MULTI_DRAFT_GRAPHS=0); ``verify``: the verify seam (default
        ``SerialVerify``)."""
        from .engine import DFLASH_POLICY, encode_policy

        self.g = engine
        self.e = e = engine.e
        self.w = e.w
        self.rank = engine.rank
        self.count = streams
        if e.slots.count < streams:
            raise ValueError(f"the engine holds {e.slots.count} stream slots, not {streams}")
        self.rows_max = e.rows                           # a stream's window (decode.MAX_ROWS)
        self.pool = Pool(e.caches.rows)
        self.arena = e.caches.arena
        self.grid = engine.grid
        self.fill_rows = fill_rows(e.prefill_rows, self.grid)
        self.share = fill_share()
        self.credit = 0.0
        self.drafts = drafts
        if self.drafts is None and engine.drafter is not None:
            from .dflash2_multi import MultiDrafter

            self.drafts = MultiDrafter(engine.drafter, streams=streams)
            graphs = os.environ.get("TF_GLM_MULTI_DRAFT_GRAPHS", "1") != "0" if draft_graphs is None else draft_graphs
            if graphs and torch.cuda.is_available():
                self.drafts.capture()
        if verify is None:
            if verify_kind() == "batched":
                taps = engine.drafter.tap_layers if engine.drafter is not None else ()
                verify = BatchedVerify(e, taps=taps, rows=MAX_WINDOW)
                # TF_GLM_MULTI_GRAPHS=0: eager windows (the same bits); by default a graph a window size
                if os.environ.get("TF_GLM_MULTI_GRAPHS", "1") != "0" and torch.cuda.is_available():
                    verify.capture()
            else:
                verify = SerialVerify(e, taps=self.drafts is not None)
        self.verify = verify
        self.lanes: dict[int, Lane] = {}                 # by sid, in admission order
        self.kept: list = []                             # kept prompts (decode.Snapshot + kid, extent), oldest first
        self.next_sid = self.next_kid = self.next_order = 0
        self.outbox: list[int] = []
        self.idle = True
        self.broken: Exception | None = None
        self.requeue: list[Stream] = []                  # rank 0: streams given back to the scheduler's queue
        self.eos = tuple(self.w.cfg.eos)
        self.dflash_code = encode_policy(DFLASH_POLICY)

    # -- the scheduler's view ----------------------------------------------------------------------------------------
    @property
    def streams(self) -> dict[int, Stream]:
        """Decoding streams (``Scheduler._yield`` and /health read it)."""
        return {sid: lane.s for sid, lane in self.lanes.items() if lane.decoding}

    @property
    def filling(self) -> list[Stream]:
        return [lane.s for lane in self.lanes.values() if not lane.decoding]

    def live(self) -> int:
        return len(self.lanes)

    def health(self) -> dict:
        lanes = list(self.lanes.values())
        return {"streams": {"decoding": sum(l.decoding and not l.paused for l in lanes),
                            "filling": sum(not l.decoding for l in lanes),
                            "paused": sum(l.decoding and l.paused for l in lanes)},
                "pool_tokens": self.pool.rows, "pool_free_tokens": self.pool.free_rows(),
                "kept_prompts": len(self.kept)}

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    # -- messages --------------------------------------------------------------------------------------------------------
    def _emit(self, op: int, payload: Sequence[int]) -> list[int]:
        payload = [int(v) for v in payload]
        self.outbox += [op, len(payload), *payload]
        return payload

    def _flush(self) -> None:
        """Rank 0: this iteration's ops to rank 1, one message (after the doorbell when rank 1 idles)."""

        if self.idle:
            self.g._ring()
            self.idle = False
        msg, self.outbox = self.outbox, []
        self.g._share(msg)

    @staticmethod
    def parse(msg: Sequence[int]) -> list[tuple[int, list[int]]]:
        ops, i = [], 0
        while i < len(msg):
            op, n = int(msg[i]), int(msg[i + 1])
            ops.append((op, [int(v) for v in msg[i + 2:i + 2 + n]]))
            i += 2 + n
        return ops

    # -- admission (rank 0 decides) ---------------------------------------------------------------------------------------
    def _need(self, prompt: int) -> int:
        return align_up(prompt + self.rows_max)

    def fits(self, s: Stream) -> bool:
        """Whether ``s`` can be placed now (else it waits, in order, for streams to finish): the free rows and the
        kept-only extents cover its prompt, with a block of headroom for each decoding stream to grow into."""

        if not self.lanes:
            return True                  # everything can be evicted; a prompt past the pool fails in ``admit``
        need = self._need(len(s.prompt))
        room = self.pool.free_rows() + sum(x.size for x in self.pool.extents if x.owner is None)
        return room >= need + ALIGN * sum(l.decoding for l in self.lanes.values())

    def admit(self, s: Stream) -> None:
        """Rank 0: place a request (slot, extent; resumed from the longest kept prefix of its prompt when it drafts)
        and queue its ADMIT; its prompt is prefilled chunk by chunk in later iterations."""

        from .engine import grid_point, shared_points
        from .engine import VisionFeed

        self._check()
        g = self.g
        prompt = [int(t) for t in s.prompt]
        if not prompt:
            raise ValueError("a request needs at least one prompt token")
        if len(prompt) >= g.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {g.limit}")
        s.count = max(1, min(int(s.count), g.limit - len(prompt)))
        if self._need(len(prompt)) > self.pool.rows:
            raise ValueError(f"a prompt of {len(prompt)} tokens does not fit the {self.pool.rows}-token pool")
        info = getattr(s, "glm", None) or {}
        draft = bool(s.draft) and not g.serial_only
        code = multi_code(info.get("code") or [0, 0, 0, 0], self.drafts is not None and draft,
                          self.dflash_code) if draft else [0, 0, 0, 0]
        positions: list[int] = []
        feed = None
        if s.vision is not None:              # encoded before anything is decided or sent: a failure stops here
            if g.vision is None:
                raise ValueError("image inputs require starting this server with --vision")
            positions = [int(p) for p in s.vision.positions]
            if positions and (positions[-1] >= len(prompt) or positions != sorted(set(positions))):
                raise ValueError("image rows must sit at increasing positions inside the prompt")
            t = time.perf_counter()
            torch.cuda.empty_cache()
            rows = g.vision.features(s.vision)
            torch.cuda.empty_cache()
            if rows.shape[0] != len(positions):
                raise ValueError(f"the tower gave {rows.shape[0]} rows for {len(positions)} image positions")
            feed = VisionFeed(g, positions, rows)
            feed.encode_s = time.perf_counter() - t
        slot = next((k for k in range(self.count) if k not in {l.slot for l in self.lanes.values()}), None)
        if slot is None:
            raise NoRoom(f"every one of the {self.count} stream slots is taken")
        hit = self._resume(prompt) if draft and feed is None else None
        need = self._need(len(prompt))
        # placement: take a kept extent over in place, or copy its rows into a new one while a stream writes in it
        copy = 0
        base = None
        if hit is not None and hit.extent.owner is None:
            x = hit.extent
            for c in list(x.kept):            # kept prompts the stream would overwrite: gone
                if len(c.ids) > len(hit.ids) and prompt[:len(c.ids)] != c.ids:
                    self._evict(c)
            if self._grow(x, need, protect=[x]):
                base = x.base
            else:
                hit = None
        if base is None:
            base = self._room(need, [hit.extent] if hit is not None else [])
            if base is None and hit is not None:          # no room beside the kept rows: start fresh
                hit = None
                base = self._room(need)
            if base is None:
                raise NoRoom(f"no room for a {need}-token extent in the {self.pool.rows}-token pool now")
            copy = int(hit is not None)
        cut = len(hit.ids) if hit is not None else 0
        shared: list[int] = []
        if draft and feed is None and g.shared:
            known = [np.asarray(c.ids, dtype=np.int64) for c in self.kept]
            shared = shared_points(prompt, cut, grid_point(len(prompt), cut, self.grid), self.grid, g.shared, known,
                                   g.opener)
        sid = self.next_sid
        payload = [sid, slot, base, need, hit.kid if hit is not None else -1, copy, s.count, int(s.stop_eos),
                   int(draft), *pack_sampling(s.sampling), *(shared + [0] * SHARED_SLOTS)[:SHARED_SLOTS], *code,
                   len(prompt), *prompt]
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint)
        payload += [len(packed), *packed, len(positions), *positions]
        self._emit(ADMIT, payload)
        try:
            self._admitted(self._parse_admit(payload), s, feed, s.constraint, info.get("spec", ""))
        except Exception as exc:              # rank 1 applies the same ADMIT: the ranks may no longer agree
            self.broken = exc
            raise

    @staticmethod
    def _parse_admit(p: list[int]) -> dict:
        (sid, slot, base, size, kid, copy, count, stop_eos, draft) = p[:9]
        sampling = unpack_sampling(p[9:19])
        shared = [v for v in p[19:19 + SHARED_SLOTS] if v]
        i = 19 + SHARED_SLOTS
        code = p[i:i + 4]
        i += 4
        n = p[i]
        prompt = p[i + 1:i + 1 + n]
        i += 1 + n
        n = p[i]
        packed = p[i + 1:i + 1 + n]
        i += 1 + n
        n = p[i]
        positions = p[i + 1:i + 1 + n]
        return dict(sid=sid, slot=slot, base=base, size=size, kid=kid, copy=copy, count=count, stop_eos=bool(stop_eos),
                    draft=bool(draft), sampling=sampling, shared=shared, code=code, prompt=prompt, packed=packed,
                    positions=positions)

    def _admitted(self, a: dict, s: Stream, feed, constraint, spec: str = "") -> None:
        """Both ranks: an ADMIT's stream in its slot and extent, resumed from its kept prompt if it names one."""

        from .engine import decode_policy, grid_point
        from .forward import State

        g, e = self.g, self.e
        hit = self._kept_by_id(a["kid"]) if a["kid"] >= 0 else None
        prompt = a["prompt"]
        s.prompt = list(prompt)
        cut = len(hit.ids) if hit is not None else 0
        if hit is not None and hit.ids != prompt[:cut]:
            raise RuntimeError(f"the kept prompt {a['kid']} is not a prefix of stream {a['sid']}'s prompt")
        if hit is not None and not a["copy"]:
            x = hit.extent
            if (x.base, x.size) != (a["base"], a["size"]) or x.owner is not None:
                raise RuntimeError(f"stream {a['sid']} takes over extent {x.eid} at [{x.base}, {x.end}), "
                                   f"rank 0 said [{a['base']}, {a['base'] + a['size']})")
        else:
            x = self.pool.add(a["base"], a["size"])
            if hit is not None:
                self.arena.copy(hit.extent.base, x.base, cut)
        x.owner = a["sid"]
        st = State(self.w, x.size, e.rows, caches=e.caches, base=x.base, slots=e.slots, slot=a["slot"])
        code = a["code"]
        dflash = code[0] >= 10 and self.drafts is not None
        policy = decode_policy(code, g.costs) if code[0] else None
        lane = Lane(s, a["sid"], a["slot"], x, st, self.next_order, code, spec, policy, dflash, feed=feed,
                    constraint=constraint)
        self.next_sid = max(self.next_sid, a["sid"] + 1)
        self.next_order += 1
        s.sid, s.cached = a["sid"], cut
        # the slot's state: the kept prompt's, or fresh
        st.reset()
        if self.drafts is not None:
            self._ctx(lane).reset()
        if hit is not None:
            st.rec[st.cur[0]].copy_(hit.rec)
            st.conv.copy_(hit.conv)
            st.set_pos(cut)
            if dflash:
                if hit.drafter_end != cut or hit.drafter_rows is None:
                    raise RuntimeError("a kept prompt without DFlash2's window was chosen for a DFlash2 stream")
                put_draft_window(self._ctx(lane), cut, hit.drafter_rows)
            self._touch(hit)
        # where its prompt states are kept: the grid point (or the end) and rank 0's shared-prefix points
        if a["draft"] and feed is None:
            lane.point = grid_point(len(prompt), cut, self.grid)
            lane.shared = sorted(p for p in set(a["shared"]) if lane.point is not None and cut < p < lane.point)
            lane.stops = list(lane.shared) + ([lane.point] if lane.point is not None and lane.point < len(prompt)
                                              else [])
        s.started = time.perf_counter()
        lane.t0 = s.started
        self.lanes[lane.sid] = lane

    def _ctx(self, lane: Lane):
        """The lane's slot's DFlash2 context."""
        return self.drafts.contexts[lane.slot]

    # -- kept prompts ------------------------------------------------------------------------------------------------------
    def _kept_by_id(self, kid: int):
        for c in self.kept:
            if c.kid == kid:
                return c
        raise RuntimeError(f"rank {self.rank} holds no kept prompt {kid}")

    def _resume(self, prompt: list[int]):
        best = None
        for c in self.kept:
            n = len(c.ids)
            fits = c.drafter_end == n and c.drafter_rows is not None or self.drafts is None
            fits = fits and not (self.grid and n % self.grid)
            if fits and n < len(prompt) and prompt[:n] == c.ids and (best is None or n > len(best.ids)):
                best = c
        return best

    def _touch(self, c) -> None:
        self.kept = [k for k in self.kept if k is not c] + [c]

    def _keep(self, lane: Lane, n: int) -> None:
        """Both ranks: the lane's state at prompt position n (== its pos) becomes a kept prompt in its extent."""

        from .decode import take_snapshot

        prev = self.e.use(lane.st)
        try:
            snap = take_snapshot(self.e, lane.s.prompt[:n], None, mtp=False, drafter=None)
        finally:
            self.e.use(prev)
        snap.drafter_end = -1
        if lane.dflash:
            snap.drafter_end = n
            snap.drafter_rows = draft_window(self._ctx(lane), n)
        snap.kid = self.next_kid
        self.next_kid += 1
        snap.extent = lane.extent
        for c in [c for c in self.kept if c.ids == snap.ids]:
            self._drop(c)
        lane.extent.kept.append(snap)
        self.kept.append(snap)
        while len(self.kept) > max(1, self.g.cache_entries):
            self._drop(self.kept[0])

    def _drop(self, c) -> None:
        """Forget a kept prompt; its extent goes when nothing else holds it, else shrinks to what is kept."""

        self.kept = [k for k in self.kept if k is not c]
        x = c.extent
        x.kept = [k for k in x.kept if k is not c]
        c.rec = c.conv = None
        c.drafter_rows = None
        self._settle(x)

    def _settle(self, x) -> None:
        if x.owner is not None:
            return
        if not x.kept:
            self.pool.remove(x)
            return
        size = align_up(max(len(k.ids) for k in x.kept))
        if size < x.size:
            self.pool.resize(x, size)

    def _evict(self, c) -> None:
        """Rank 0: drop a kept prompt on both ranks."""

        self._emit(EVICT, [c.kid])
        self._drop(c)

    # -- room --------------------------------------------------------------------------------------------------------------
    def _used(self, x) -> int:
        used = max([len(k.ids) for k in x.kept], default=0)
        if x.owner is not None:
            used = max(used, self.lanes[x.owner].st.pos)
        return used

    def _move(self, x, base: int, size: int) -> None:
        """Both ranks: relocate extent x (its used rows copied) to [base, base + size)."""

        used = self._used(x)
        old = self.pool.move(x, base, size)
        self.arena.copy(old, base, used)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _resize(self, x, size: int) -> None:
        self.pool.resize(x, size)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _victims(self, protect) -> list:
        keep = set(id(x) for x in protect)
        return [c for c in self.kept if c.extent.owner is None and id(c.extent) not in keep]

    def _compact(self) -> None:
        """Rank 0: slide every extent down to the lowest free rows (MOVEs), so the free rows are one range."""

        at = 0
        for x in list(self.pool.extents):
            if x.base != at:
                self._emit(MOVE, [x.eid, at, x.size])
                self._move(x, at, x.size)
            at += x.size

    def _room(self, need: int, protect=()) -> int | None:
        """Rank 0: a base with ``need`` free rows, evicting kept prompts (least recently used first, never
        ``protect``'s extents) and compacting as needed; None if even that leaves too little."""

        base = self.pool.place(need)
        while base is None:
            victims = self._victims(protect)
            if not victims:
                break
            self._evict(victims[0])
            base = self.pool.place(need)
        if base is None and self.pool.free_rows() >= need:
            self._compact()
            base = self.pool.place(need)
        return base

    def _grow(self, x, size: int, protect=()) -> bool:
        """Rank 0: extent x to at least ``size`` rows, in place or moved (evicting and compacting as needed)."""

        size = align_up(size)
        if size <= x.size:
            return True
        protect = list(protect) + [x]
        while True:
            if self.pool.room_after(x) >= size - x.size:
                self._emit(GROW, [x.eid, size])
                self._resize(x, size)
                return True
            base = self.pool.place(size, ignore=[x])
            if base is not None:
                self._emit(MOVE, [x.eid, base, size])
                self._move(x, base, size)
                return True
            victims = self._victims(protect)
            if not victims:
                break
            self._evict(victims[0])
        if self.pool.free_rows() + x.size >= size:
            self._compact()
            if self.pool.room_after(x) >= size - x.size:
                self._emit(GROW, [x.eid, size])
                self._resize(x, size)
                return True
            base = self.pool.place(size, ignore=[x])
            if base is not None:
                self._emit(MOVE, [x.eid, base, size])
                self._move(x, base, size)
                return True
        return False

    # -- iterations ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def round(self) -> list[Stream]:
        """Rank 0: one prompt chunk or one decode round; returns the streams that finished."""

        self._check()
        try:
            return self._iterate()
        except Exception as exc:
            self.broken = exc
            raise

    def _fill_turn(self) -> Lane | None:
        filling = [l for l in self.lanes.values() if not l.decoding]
        if not filling:
            return None
        decoding = [l for l in self.lanes.values() if l.decoding and not l.paused and not l.s.done]
        if decoding:
            self.credit += self.share
            if self.credit < 1:
                return None
            self.credit -= 1
        s = next_fill([l.s for l in filling])
        return self.lanes[s.sid]

    def _iterate(self) -> list[Stream]:
        lane = self._fill_turn()
        if lane is not None:
            stop = self._stop(lane)
            self._emit(FILL, [lane.sid, stop])
            self._flush()
            first = self._fill(lane, stop)
            if first is not None:
                lane.s.take([first], self._ends(lane))
            return [lane.s] if lane.s.done else []
        lanes = [l for l in self.lanes.values() if l.decoding and not l.s.done]
        for l in lanes:                       # room for this round's widest window, else paused
            l.paused = not self._grow(l.extent, l.st.pos + self.rows_max, protect=[l.extent])
        active = [l for l in lanes if not l.paused]
        if not active:
            done = []
            if lanes and not any(not l.decoding for l in self.lanes.values()):
                done = self._give_back()      # every stream waits for room: the youngest goes back to the queue
            if self.outbox:                   # what was decided (evictions, moves, a stream given back) goes now
                self._flush()
            return done
        plan = [(l.sid, l.depth) for l in active]
        self._emit(ROUND, [len(plan), *[v for p in plan for v in p]])
        self._flush()
        self._round(active)
        return [l.s for l in active if l.s.done]

    def _stop(self, lane: Lane) -> int:
        pos, n = lane.st.pos, len(lane.s.prompt)
        stop = next((p for p in lane.stops if p > pos), n)
        busy = any(l.decoding and not l.s.done for l in self.lanes.values())
        rows = self.fill_rows if busy or lane.s.background else self.e.prefill_rows
        return min(stop, pos + rows)

    def _ends(self, lane: Lane) -> tuple[int, ...]:
        return self.eos if lane.s.stop_eos else ()

    @torch.no_grad()
    def _fill(self, lane: Lane, stop: int) -> int | None:
        """Both ranks: the lane's prompt rows pos .. stop in one chunk; its kept states; at the prompt's end its first
        token (the stream then decodes)."""

        from .copy_drafts import CopyDrafts
        from .decode import prefill_chunk

        e = self.e
        s = lane.s
        prompt = s.prompt
        n = len(prompt)
        start = lane.st.pos
        if not start < stop <= n or stop - start > e.prefill_rows:
            raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
        t = time.perf_counter()
        prev = e.use(lane.st)
        try:
            out = prefill_chunk(e, prompt, start, stop - start, drafter=self._ctx(lane) if lane.dflash else None,
                                feed=lane.feed, mtp=False, head=stop == n)
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop)
                return None
            if lane.point == n:
                self._keep(lane, n)
            logits = out[:1]
            if lane.constraint is not None:             # the first token's row, under the reply's grammar
                lane.constraint.mask(logits, lane.constraint.window([0], [-1]), self.w.vocab_offset)
            first = sample_streams(self.w, [(logits, [n], s.sampling)])[0][0]
            if lane.constraint is not None:
                lane.constraint.advance([first])
        finally:
            e.use(prev)
            s.prefill_s += time.perf_counter() - t
        lane.decoding = True
        s.started = time.perf_counter()
        if lane.policy is not None and self.g.copy is not None:
            lane.copies = CopyDrafts(list(prompt) + [first], self.g.copy.match, self.g.copy.most)
        if lane.policy is not None:
            lane.depth = min(lane.policy.next(0, 0), s.count - 1)
        return first

    @torch.no_grad()
    def _round(self, lanes: list[Lane], depths: list[int] | None = None) -> None:
        """Both ranks: one decode round over ``lanes`` (``depths``: rank 0's, checked on rank 1)."""

        from .decode import copy_room
        from .dflash2_multi import DraftRequest

        if depths is not None and [l.depth for l in lanes] != depths:
            raise RuntimeError(f"rank {self.rank}'s draft depths {[l.depth for l in lanes]} are not rank 0's {depths}")
        drafts: list[list[int]] = []
        copied: list[bool] = []
        asks = []
        for l in lanes:
            c = l.copies.propose(copy_room(l.copies, l.s.count, l.s.out)) if l.copies is not None else []
            copied.append(bool(c))
            drafts.append(list(c))
            if not c and l.dflash and l.depth > 0:
                asks.append((len(drafts) - 1, DraftRequest(self._ctx(l), l.s.out[-1], l.depth, l.s.sampling,
                                                           l.policy.confidence, **l.policy.chain_rule)))
        if asks:
            got = self.drafts.propose([a for _, a in asks])
            for (k, _), d in zip(asks, got):
                drafts[k] = list(d)
        drafts = trim(drafts, MAX_WINDOW)
        segments = []
        for l, d in zip(lanes, drafts):
            tokens = [l.s.out[-1]] + d
            l.window = None
            if l.constraint is not None:                # the chain cut at its first draft the grammar rejects
                l.window = l.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                tokens = list(l.window.tokens)
            segments.append(Segment(l.st, tokens))
        v = forward_streams(self.verify, segments)
        parts = []
        for l, seg, logits in zip(lanes, segments, v.logits):
            if l.window is not None:
                l.constraint.mask(logits, l.window, self.w.vocab_offset)
            parts.append((logits, [l.st.pos + 1 + r for r in range(len(seg.tokens))], l.s.sampling))
        sampled = sample_streams(self.w, parts)
        keeps = []
        for l, seg, rows in zip(lanes, segments, sampled):
            keep = 1
            for i, d in enumerate(seg.tokens[1:]):
                if rows[i] != d or (l.s.stop_eos and rows[i] in self.eos):
                    break
                keep += 1
            keeps.append(keep)
        self.verify.commit(segments, keeps)
        if self.drafts is not None:
            items = [(self._ctx(l), v.taps[k][:keep]) for k, (l, keep) in enumerate(zip(lanes, keeps)) if l.dflash]
            if items:
                self.drafts.commit(items)
        for l, seg, rows, keep, was_copy in zip(lanes, segments, sampled, keeps, copied):
            s = l.s
            new = rows[:keep]
            if l.constraint is not None:
                l.constraint.advance(new)
            if l.copies is not None:
                l.copies.extend(new)
            ndrafts = len(seg.tokens) - 1
            s.counted(len(seg.tokens))                     # Stream.take counts the accepted drafts
            if was_copy:
                l.copy_rounds += 1
                l.copy_drafted += ndrafts
                l.copy_accepted += keep - 1
            room = s.count - len(s.out)
            s.take(new[:max(0, room)], self._ends(l))
            if l.policy is not None:
                l.depth = min(l.policy.next(*((0, 0) if was_copy else (ndrafts, keep - 1))),
                              max(0, s.count - len(s.out)))

    # -- ends ----------------------------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        """Rank 0: finished (or cancelled) streams leave on both ranks; an idle rank 1 is told to wait for the bell."""

        for s in done:
            lane = self.lanes.get(s.sid)
            if lane is None or lane.s is not s:
                continue
            reason = CANCELLED if s.error is None and len(s.out) < s.count and not (
                s.out and s.out[-1] in self._ends(lane)) else DONE
            self._emit(FINISH, [lane.sid, reason])
            self._finish(lane.sid)
        if not self.lanes and not self.idle and self.broken is None:
            self._emit(IDLE, [])
            self._flush()
            self.idle = True

    def _finish(self, sid: int) -> None:
        """Both ranks: the stream's slot frees; its extent stays for its kept prompts (shrunk to them) or goes."""

        lane = self.lanes.pop(sid)
        x = lane.extent
        x.owner = None
        self._settle(x)
        lane.s.finished = lane.s.finished or time.perf_counter()

    def _give_back(self) -> list[Stream]:
        """Rank 0: the youngest stream (one without a grammar or images, which cannot replay) leaves its slot and
        extent and returns to the queue; it replays later, its sent tokens owed, not sent again. If every stream has
        a grammar or images, the youngest fails instead (returned, to be finished)."""

        lanes = sorted(self.lanes.values(), key=lambda l: l.order, reverse=True)
        lane = next((l for l in lanes if l.constraint is None and l.feed is None), None)
        if lane is None:
            lane = lanes[0]
            lane.s.error = NoRoom("the pool is full: every stream waits for room, and the youngest cannot replay "
                                  "(a grammar or images)")
            lane.s.done = True
            return [lane.s]
        for c in list(lane.extent.kept):
            self._evict(c)
        self._emit(FINISH, [lane.sid, REQUEUED])
        self._finish(lane.sid)
        self.requeue.append(lane.s)
        return []

    def drop(self) -> list[Stream]:
        """After an error in an iteration: forget every stream (the ranks can no longer be trusted to agree)."""

        live = [l.s for l in self.lanes.values()]
        self.lanes = {}
        if self.broken is None:
            self.broken = RuntimeError("an iteration failed")
        return live

    def stats(self, s: Stream) -> dict:
        """A finished request's stats (as ``GlmEngine.generate`` reports them)."""

        out = s.stats()
        out["decode_s"] = round(max((s.finished or time.perf_counter()) - s.started, 0.0), 4)
        tokens = list(s.out[:s.count])
        out.update(sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
                   tokens_per_round=round((len(tokens) - 1) / max(s.rounds, 1), 3), parallel=self.count)
        lane_info = getattr(s, "glm_lane", None)
        if lane_info:
            out.update(lane_info)
        return out

    # -- rank 1 --------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def follow(self, once: bool = False) -> None:
        """Rank 1: apply rank 0's messages forever (``once``: one message)."""

        while True:
            if self.idle:
                self.g._await_bell()
                self.idle = False
            self.apply(self.g._share(None))
            if once:
                return

    def apply(self, msg: Sequence[int]) -> None:
        """Rank 1: one iteration's ops, in order."""

        from tensorfold.engine.exact_sampling import Sampling  # noqa: F401

        for op, p in self.parse(msg):
            if op == ADMIT:
                a = self._parse_admit(p)
                s = Stream(list(a["prompt"]), a["count"], a["sampling"], draft=a["draft"], stop_eos=a["stop_eos"],
                           sid=a["sid"])
                constraint = None
                if a["packed"]:
                    from tensorfold.engine import grammar

                    constraint = grammar.compiler(self.g, self.g.model_dir, self.g.eos).follow(a["packed"])
                    s.constraint = constraint
                feed = None
                if a["positions"]:
                    from .engine import VisionFeed

                    feed = VisionFeed(self.g, list(a["positions"]), None)
                    s.vision = feed
                self._admitted(a, s, feed, constraint)
            elif op == EVICT:
                self._drop(self._kept_by_id(p[0]))
            elif op == MOVE:
                self._move(self.pool.get(p[0]), p[1], p[2])
            elif op == GROW:
                self._resize(self.pool.get(p[0]), p[1])
            elif op == FILL:
                lane = self.lanes[p[0]]
                if p[1] != self._stop_bound(lane, p[1]):
                    raise RuntimeError(f"rank 1: stream {p[0]}'s prompt chunk to {p[1]} passes a kept point")
                first = self._fill(lane, p[1])
                if first is not None:
                    lane.s.take([first], self._ends(lane))
            elif op == ROUND:
                n = p[0]
                sids, depths = p[1:1 + 2 * n:2], p[2:2 + 2 * n:2]
                self._round([self.lanes[sid] for sid in sids], depths)
            elif op == FINISH:
                self._finish(p[0])
            elif op == IDLE:
                if self.lanes:
                    raise RuntimeError(f"rank 1 holds {len(self.lanes)} streams rank 0 has finished")
                self.idle = True
            else:
                raise RuntimeError(f"rank 1: unknown op {op}")

    def _stop_bound(self, lane: Lane, stop: int) -> int:
        """The stop rank 0 sent, if it passes no kept point (else where it should have stopped)."""

        nxt = next((q for q in lane.stops if q > lane.st.pos), len(lane.s.prompt))
        return min(stop, nxt)


# -- the scheduler ----------------------------------------------------------------------------------------------------
class GlmScheduler(Scheduler):
    """``cuda.scheduler.Scheduler`` for the GLM decoder: a request's GLM settings ride on its stream (``glm``); a
    request the pool cannot place yet waits at the head of the queue (``MultiDecoder.fits``); streams the decoder gives
    back (every stream waiting for room) return to the queue to replay."""

    def __init__(self, decoder: MultiDecoder, *, max_streams: int) -> None:
        self.held = None
        super().__init__(decoder, max_streams=max_streams)

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, glm: dict | None = None) -> dict:
        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background)
        stream.glm = dict(glm or {})
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left or a stop string: the stream ends next round
            elif kind == "error":
                raise value
            else:
                return value

    def _admit(self, first=None) -> list[Stream]:
        done: list[Stream] = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                item, first = first, None
            elif self.held is not None:
                item, self.held = self.held, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
            stream, box = item
            if not self.decoder.fits(stream):
                self.held = item                     # waits, first in line, for room
                break
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom:
                self.boxes.pop(id(stream))
                self.held = item
                break
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """``Scheduler._yield``, the stream's GLM settings carried to its replay."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(self.decoder.streams.values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        again = stream.continued()
        again.glm = getattr(stream, "glm", {})
        self.waiting.put((again, box))

    def _requeue(self) -> None:
        for stream in self.decoder.requeue:
            box = self.boxes.pop(id(stream), None)
            if box is None:
                continue
            again = stream.continued()
            again.glm = getattr(stream, "glm", {})
            self.yields += 1
            self.waiting.put((again, box))
        self.decoder.requeue = []

    def _loop(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and self.held is None
            done = self._admit(self.waiting.get() if idle else None)     # idle: wait for a request
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self._requeue()
            try:
                self.decoder.finish(done)
            except Exception as exc:                 # noqa: BLE001
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", self.decoder.stats(s))))
            if self.held is not None and not self.decoder.live():
                time.sleep(0)                        # the held request retries next iteration (now alone)
