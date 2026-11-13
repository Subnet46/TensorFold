"""The MTP head for concurrent streams (``multi.MultiDecoder`` under --parallel): every drafting stream's MTP rows of a
step in ONE forward of the MTP layer, each stream's rows attending to its own extent of the MTP latent plane through
the segmented DSA kernels (``forward.dsa_segments``, ``segments.SegRows`` at the streams' MTP positions), the
row-independent parts (embedding, norms, the eh projection, the MoE, the head) over the whole step.

A stream carries the committed rows its MTP cache has not absorbed (``Carry``: the main model's final-normed rows and
the tokens after them): the last prompt row and the first token after its prompt, then each round's kept rows. A round
absorbs every stream's carry in one step, samples each stream's first draft from its last row, and chains further
drafts one row a stream a step (the head's own normed output row and the draft). Drafts never change a reply (the
verify window decides every token), so these steps only have to be fast; their rows share the verify window's buffers
(``verify.BatchedVerify.b``), which no one reads between a round's commit and the next round's forward.

Positions: MTP row i holds the main model's row i and token i + 1, as the one-stream head (``decode.absorb``): a
stream's ``mtp_len`` rows are absorbed, the last ``mtp_drafted`` of them a draft chain's, rolled back before the next
absorb."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

CHAIN_STEPS = 4                          # drafts a stream's chain takes at most a round (the absorb step, then 3)
IX_SLOTS = 32                            # pinned index lists in flight (a round uses up to 4 a step)


def enabled(value: str | None = None) -> bool:
    """TF_GLM_MULTI_MTP: load the MTP head under --parallel and draft with it (1, the default), or not (0: the
    streams draft with DFlash2 if a draft model was given, else decode one row a round)."""

    value = (os.environ.get("TF_GLM_MULTI_MTP", "") if value is None else value).strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_MULTI_MTP: 0 or 1, not {value!r}")
    return value == "1"


@dataclass
class Carry:
    """Committed rows a stream's MTP cache has not absorbed: hidden [k, D] (final-normed main rows, a copy) and the k
    tokens after them."""

    hidden: torch.Tensor
    tokens: list[int]


@dataclass
class Ask:
    """A stream's MTP work this round: its state, carry, how many drafts (0: absorb only), its sampling and the
    position its first draft is sampled at."""

    st: object
    carry: Carry
    depth: int
    sampling: object
    position: int


class BatchedMTP:
    """The MTP layer over several streams' rows at once, on the batched verify window's buffers and token selection
    scratch, with segment tables of its own. A step of R rows replays a CUDA graph captured for R (``capture``; eager
    past the captured widths or when a stream starts at position 0), and leaves every row's greedy next token on the
    device (``nxt``): a round's draft chain feeds each step from the last one's rows without reading anything back,
    and reads the drafts once at its end."""

    def __init__(self, e, verify) -> None:
        from .segments import MAX_SEGS, SegRows

        w = e.w
        if w.mtp is None:
            raise ValueError("the MTP head is not loaded")
        if e.caches.mtp is None:
            raise ValueError("the pool holds no MTP plane")
        self.e, self.w, self.b = e, w, verify.b
        self.rows_max = verify.rows_max
        self.sel = verify.sel
        caches = e.caches
        dev = w.device
        self.seg_rows = SegRows(self.rows_max, dev, max_segs=MAX_SEGS, ring=caches.ring if caches.rings else None)
        # pinned copies of the segment tables, one a step of a round: a step's copy never waits for the last one's
        pin = torch.cuda.is_available()
        self.tables = [torch.zeros_like(self.seg_rows.host, pin_memory=pin) for _ in range(CHAIN_STEPS)]
        self.events = [torch.cuda.Event() if pin else None for _ in range(CHAIN_STEPS)]
        self.index_at = len(caches.kc)            # the MTP layer's indexer caches: after every DSA layer's
        world = 1 if w.comm is None else int(w.world)
        self.world = world
        self.pk = torch.zeros((self.rows_max, 2), dtype=torch.float32, device=dev)
        self.gk = torch.zeros((world * self.rows_max * 2,), dtype=torch.float32, device=dev)
        self.nxt = torch.zeros((self.rows_max,), dtype=torch.int32, device=dev)
        self.chain = torch.zeros((CHAIN_STEPS, MAX_SEGS), dtype=torch.int32, device=dev)
        # small index lists for the chain, through a ring of pinned buffers (each copy left before its slot's reuse)
        self.ix_host = torch.zeros((IX_SLOTS, self.rows_max), dtype=torch.int64, pin_memory=pin)
        self.ix_dev = torch.zeros((IX_SLOTS, self.rows_max), dtype=torch.int64, device=dev)
        self.ix_events = [torch.cuda.Event() if pin else None for _ in range(IX_SLOTS)]
        self.ix_next = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.widths: list[int] = []
        self.steps = self.rows = self.replays = 0

    def _ix(self, values: list[int]) -> torch.Tensor:
        """``values`` on the device (an index list), copied from the next pinned slot without waiting."""
        k = self.ix_next
        self.ix_next = (k + 1) % IX_SLOTS
        if self.ix_events[k] is not None:
            self.ix_events[k].synchronize()
        n = len(values)
        self.ix_host[k, :n].numpy()[:] = values
        out = self.ix_dev[k, :n]
        out.copy_(self.ix_host[k, :n], non_blocking=True)
        if self.ix_events[k] is not None:
            self.ix_events[k].record()
        return out

    def _lc(self) -> torch.Tensor:
        caches = self.e.caches
        return caches.arena.planes[caches.mtp[0]].tensor

    def _index(self):
        caches = self.e.caches
        return caches.index_tensors(self.index_at) if caches.index is not None else None

    def _compute(self, R: int, select: bool) -> None:
        """The MTP layer on rows 0 .. R of the staged ids, input rows and segment tables (capturable): their normed
        output rows into ``b.fnormed``, each row's greedy next token (every rank's vocabulary share) into ``nxt``."""

        from . import glue, qmm
        from .forward import dsa_segments, mm, moe_block

        w, b = self.w, self.b
        c = w.cfg
        m = w.mtp
        D = c.hidden
        glue.embed(b.ids[:R], w.embed, D, 1, b.me[:R])
        glue.rmsnorm(b.me[:R], m.enorm, c.eps, b.mcat[:R, :D])
        glue.rmsnorm(b.hin[:R], m.hnorm, c.eps, b.mcat[:R, D:])
        mm(b, b.mcat[:R], m.eh, qmm.group_sums(b.mcat[:R], b.mxs[:R]), b.mx[:R])
        layer = m.layer
        glue.rmsnorm(b.mx[:R], layer.in_norm, c.eps, b.normed[:R], b.xs[:R])
        g = dsa_segments(layer, w, self._lc(), b, R, self.seg_rows, self.sel, self._index(), select=select)
        glue.residual_add(b.mx[:R], b.mx[:R], g)
        glue.rmsnorm(b.mx[:R], layer.post_norm, c.eps, b.normed[:R], b.xs[:R])
        g = moe_block(layer, w, b, R)
        glue.residual_add(b.mx[:R], b.mx[:R], g)
        glue.rmsnorm(b.mx[:R], m.norm, c.eps, b.fnormed[:R], b.fxs[:R])
        head = w.draft_head if w.draft_head is not None else w.head
        logits = mm(b, b.fnormed[:R], head, b.fxs[:R], b.logits[:R, :w.head.n])
        vals, idx = logits.float().max(-1)
        self.pk[:R, 0] = vals
        self.pk[:R, 1] = (idx + int(w.vocab_offset)).to(torch.int32).view(torch.float32)
        if self.world > 1:
            w.comm.all_gather(self.pk[:R].reshape(-1), self.gk[:self.world * R * 2])
            g2 = self.gk[:self.world * R * 2].view(self.world, R, 2)
        else:
            g2 = self.pk[:R].view(1, R, 2)
        best = g2[:, :, 0].argmax(0)              # the first rank holding the maximum: the lowest ids
        rows = torch.arange(R, device=best.device)
        self.nxt[:R] = g2[best, rows, 1].contiguous().view(torch.int32)

    @torch.no_grad()
    def capture(self) -> None:
        """A CUDA graph of a step for each width (``wide.widths`` of the window, both ranks together: the greedy
        pick gathers), captured on slot 0 at the pool's first rows before any stream is admitted."""

        from .sparse import SPARSE_FROM
        from .wide import widths

        e, b = self.e, self.b
        index = self._index() is not None
        pool = torch.cuda.graph_pool_handle()
        caches = e.caches
        for R in widths(self.rows_max):
            b.ids[:R].zero_()
            b.hin[:R].zero_()
            span = (e.home.base, 1, R)
            self.seg_rows.set([span + (caches.ring_base(0),) if caches.rings else span],
                              sparse_from=SPARSE_FROM if index else None)
            for _ in range(2):
                self._compute(R, index)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._compute(R, index)
            self.graphs[R] = g
        self.widths = sorted(self.graphs)
        torch.cuda.synchronize()

    def _tables(self, spans, step: int) -> bool:
        """Stage the segment tables of a step from its own pinned copy; whether a row attends sparsely."""

        from .sparse import SPARSE_FROM

        rows = self.seg_rows
        t, R = rows.fill(spans, sparse_from=SPARSE_FROM if self._index() is not None else None)
        k = step % CHAIN_STEPS
        if self.events[k] is not None:
            self.events[k].synchronize()          # this copy's last use, a round ago, has left the pinned buffer
        self.tables[k].numpy()[:] = t
        rows.dev.copy_(self.tables[k], non_blocking=True)
        if self.events[k] is not None:
            self.events[k].record()
        rows.R, rows.segments = R, list(spans)
        return bool(t[3 * rows.rows:3 * rows.rows + R].any())

    def _span(self, st, k: int) -> tuple:
        caches = self.e.caches
        span = (st.base, st.mtp_len, k)
        return span + (caches.ring_base(st.slot),) if caches.rings else span

    def _run(self, R: int, sparse: bool, zero: list[int]) -> None:
        """A step of R staged rows: its graph, or eager (no graph that wide, or a stream at position 0, whose
        embedding row is zeroed: ``mtp_compute``)."""

        g = self.graphs.get(R) if not zero else None
        if g is not None:
            g.replay()
            self.replays += 1
        elif zero:
            from . import glue

            w, b = self.w, self.b
            glue.embed(b.ids[:R], w.embed, w.cfg.hidden, 1, b.me[:R])
            for r in zero:
                b.me[r].zero_()
            self._compute_from_embed(R, sparse)
        else:
            self._compute(R, sparse or self._index() is not None)
        self.steps += 1
        self.rows += R

    def _compute_from_embed(self, R: int, select: bool) -> None:
        """``_compute`` with the embeddings already in ``b.me`` (a zeroed row)."""
        from . import glue

        saved = glue.embed
        glue.embed = lambda *args, **kw: None     # noqa: E731  (the rows are embedded already)
        try:
            self._compute(R, select)
        finally:
            glue.embed = saved

    @torch.no_grad()
    def step(self, parts: Sequence[tuple]) -> None:
        """parts [(st, tokens, hidden [k, D])]: each stream's k rows from its ``mtp_len`` (which advances by k), the
        tokens and rows from the host or the device. Leaves the rows' outputs in ``b.fnormed`` and ``nxt``."""

        from .forward import check_room

        w, b = self.w, self.b
        spans, zero, ids = [], [], []
        R = 0
        for st, tokens, hidden in parts:
            k = len(tokens)
            if k < 1 or hidden.shape[0] != k:
                raise ValueError(f"an MTP step's part of {k} tokens and {hidden.shape[0]} rows")
            check_room(w, st, k, pos=st.mtp_len)
            if st.mtp_len == 0:
                zero.append(R)
            spans.append(self._span(st, k))
            ids.extend(int(t) for t in tokens)
            R += k
        if R > b.rows:
            raise ValueError(f"an MTP step of {R} rows, the window's buffers hold {b.rows}")
        b.staged.synchronize()
        b.ids_host[:R].numpy()[:] = ids
        b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
        b.staged.record()
        at = 0
        for _, tokens, hidden in parts:
            b.hin[at:at + len(tokens)].copy_(hidden)
            at += len(tokens)
        sparse = self._tables(spans, 0)
        self._run(R, sparse, zero)
        for st, tokens, _ in parts:
            st.mtp_len += len(tokens)             # the device position is set before a one-stream call (``sync``)

    @torch.no_grad()
    def draft(self, asks: Sequence[Ask]) -> list[list[int]]:
        """Every ask's carry absorbed in one step (its drafted rows rolled back first), then up to ``depth`` greedy
        drafts a stream (at most ``CHAIN_STEPS``): the first from its absorb row's next token, each next one a step of
        one row a stream fed from the last step on the device. The drafts are read back once."""

        out: list[list[int]] = [[] for _ in asks]
        if not asks:
            return out
        for a in asks:
            st = a.st
            if st.mtp_drafted:
                st.mtp_len -= st.mtp_drafted
                st.mtp_drafted = 0
        parts = [(a.st, a.carry.tokens, a.carry.hidden) for a in asks]
        # carries longer than the window's rows (a long kept backlog) absorb in pieces first
        while sum(len(p[1]) for p in parts) > self.rows_max:
            k = max(range(len(parts)), key=lambda j: len(parts[j][1]))
            st, tokens, hidden = parts[k]
            cut = len(tokens) - 1
            self.step([(st, tokens[:cut], hidden[:cut])])
            parts[k] = (st, tokens[cut:], hidden[cut:])
        self.step(parts)
        b = self.b
        dev = self.nxt.device
        depth = [min(int(a.depth), CHAIN_STEPS) for a in asks]
        ends, at = [], 0
        for _, tokens, _ in parts:
            at += len(tokens)
            ends.append(at - 1)                   # each ask's last row of the absorb step
        live = [i for i in range(len(asks)) if depth[i] > 0]
        rows = [ends[i] for i in live]            # each live ask's row in the last step's output
        j = 0
        while live:
            # draft j + 1 of each live ask: its row's greedy next token, kept in ``chain[j]`` at the ask's index
            picks = self.nxt.index_select(0, self._ix(rows))
            self.chain[j].index_copy_(0, self._ix(live), picks)
            more = [n for n, i in enumerate(live) if depth[i] > j + 1]
            if not more:
                break
            pick = self._ix(more)
            R = len(more)
            b.ids[:R].copy_(picks.index_select(0, pick))
            b.hin[:R].copy_(b.fnormed.index_select(0, self._ix(rows).index_select(0, pick)))
            live = [live[n] for n in more]
            spans = [self._span(asks[i].st, 1) for i in live]
            for i in live:
                asks[i].st.mtp_len += 1
                asks[i].st.mtp_drafted += 1
            sparse = self._tables(spans, j + 1)
            self._run(R, sparse, [])
            rows = list(range(R))
            j += 1
        steps = max(depth, default=0)
        if steps == 0:
            return out
        got = self.chain[:steps, :len(asks)].cpu().numpy()     # the round's one read back
        for i in range(len(asks)):
            out[i] = [int(got[j, i]) for j in range(depth[i])]
        return out


def sync(st) -> None:
    """A stream's device MTP position from its host count, before a one-stream MTP call reads it (``mtp_forward``,
    ``decode.absorb``): the batched steps keep the count on the host only."""

    st.set_mtp_len(st.mtp_len)
