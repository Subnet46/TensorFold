"""Per-request next-token statistics of every row on the GLM CUDA engine, for Subnet 46's token check.

Off unless the server starts with ``TENSORFOLD_SN46=miner`` or ``verifier``; the two modes compute the same numbers,
the name says which side of the check a server serves.

Row ``r`` of a request is the input token at position ``r``; its head predicts the token at ``r + 1`` (the target).
For each row the engine reduces the full-vocabulary fp32 logits, over every rank's shard, to:

- ``top1``: the argmax id (the lowest id among equal maxima), and ``top1_logit``;
- ``lse``: the log-sum-exp, so a logprob is ``logit - lse``;
- ``target_logit``: the logit of the target: the next prompt token, the sampled token, or the spec's ``targets``;
- ``claim_logit``: the logit of the spec's claimed id for that row (a verifier scoring a miner's top-1 claims);
- ``target``: the target id itself, so the operator can check a sealed answer is the one the engine emitted;
- ``drawn``: the id the engine's sampler drew for a reply row (-1 on prompt rows), before a verifier's walk commits
  the miner's token in its place: with the request's sampling and seed, the verifier's own draw, which a sampled
  answer's token must equal.

A measured request keeps prompt snapshots and resumes from them only when its spec says it may (``Spec.cache``, the
operator's per-request retention decision); otherwise it leaves no state and no tokens named in the engine
(``engine._run``). A spec may also name how long its states may be kept (``Spec.keep_for_ms``, from the request's start,
on the monotonic clock: no wall clock is read): the states the request keeps carry that cutoff, a request resuming one
raises it to its own (never lowers it), and a state is dropped at its cutoff or TF_GLM_CACHE_TTL after its last use,
whichever comes first (``engine._stale``). A kept state carries its rows' numbers (``Kept``: rank 0's record of every
row before the last, and each rank's logits shard of the last row, whose target is the next prompt's token), so a
request resuming from it reports the same numbers for those rows as a fresh prefill would (``resume``); the engine keeps
states only where a fresh prefill's chunks would start (``TF_GLM_PROMPT_GRID=chunk``), so the rows past the resume point
are computed in the same chunks too.

Prompt rows are reduced in the prefill (``prompt``: one head matmul over the chunk's final-normed rows, in
sub-chunks, one all-gather each). Decode rows ride on the sampler's own all-gather: ``tail`` appends each rank's
log-sum-exp and target and claim logits to the top-k it gathers (``decode.sample_rows``), and ``sampled`` reads the
row's numbers from the gathered candidates, so a measured step costs no gather of its own. A row sampled again after
a rejected draft is overwritten, so each committed row keeps the numbers of the forward that committed it.

A verifier reproduces the miner's computation, not just its sequence: the engine's prompt path is not independent of
a chunk's length (its sparse selection is shared by a chunk's rows), so prefilling ``prompt + output`` in one pass
gives some late prompt rows other predictions than the miner's prompt-only prefill did. The verifier therefore
prefills the miner's prompt alone and walks the output through decode, each sampled token replaced by the miner's
(``forced``, from the targets past the prompt), with drafts off: serial decode commits the bits drafted decode does.

A request is measured when the operator armed a ``Spec`` for it on rank 0 (``arm``); rank 0 checks it fits the
request before any other rank hears of it (``take_armed``) and sends it with the request (``share``). Rank 0 records
into the ``Job`` the operator keeps.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

MODE = os.environ.get("TENSORFOLD_SN46", "")
if MODE not in ("", "miner", "verifier"):
    raise ValueError(f"TENSORFOLD_SN46: miner or verifier, not {MODE!r}")
ACTIVE = MODE != ""
SUB = 256                                # prompt rows per head matmul
FIELDS = ("top1", "top1_logit", "lse", "target_logit", "claim_logit")


class RowsError(ValueError):
    pass


@dataclass
class Spec:
    """One measured request, as every rank sees it. ``targets``: each row's target id (a verifier scoring
    ``prompt + output``; else the prompt's next tokens and the sampled ones); ``claims``: each row's claimed id
    whose logit is reported (-1: none)."""

    record: bool
    targets: list[int] | None = None
    claims: list[int] | None = None
    prompt: list[int] | None = None      # the prompt the operator began the measurement for (the request must match)
    cache: bool = False                  # the request may resume from kept prompt states and keep its own
    # ms from the request's start after which the states this request keeps or resumes are dropped, whatever their
    # last use (the operator's retention bound for the content); None: only TF_GLM_CACHE_TTL bounds them
    keep_for_ms: int | None = None
    # the cutoff on this rank's monotonic clock (``time.monotonic()`` seconds), set when the request starts
    # (``begin``); the wall clock is never read, so stepping it changes nothing
    keep_by: float | None = field(default=None, repr=False, compare=False)
    job: "Job | None" = None             # set when the request starts (rank 0's is the operator's)

    def header(self) -> list[int]:
        """Seven ints for ``share`` (int32 each: the keep duration travels as two 31-bit halves, 0 for none)."""
        keep = self.keep_for_ms or 0
        return [1, int(self.record), -1 if self.targets is None else len(self.targets),
                -1 if self.claims is None else len(self.claims), int(self.cache), keep & 0x7FFFFFFF, keep >> 31]

    @classmethod
    def from_header(cls, header: list[int], targets: list[int] | None, claims: list[int] | None) -> "Spec":
        """A follower's spec from rank 0's ``header`` and the targets and claims sent after it."""
        keep = (int(header[6]) << 31) | int(header[5])
        return cls(record=False, targets=targets, claims=claims, cache=bool(header[4]), keep_for_ms=keep or None)


NO_SPEC = [0, 0, -1, -1, 0, 0, 0]


@dataclass
class Job:
    """Rank 0's record of one request: one entry per row, filled as rows are reduced."""

    spec: Spec
    prompt: list[int]
    top1: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int64))
    target: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int64))
    values: np.ndarray = field(default_factory=lambda: np.full((0, 4), np.nan, np.float32))
    drawn: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int64))
    done: threading.Event = field(default_factory=threading.Event)

    def put(self, rows: np.ndarray, got: np.ndarray, targets: list[int]) -> None:
        """``got`` [n, 5] (top1 id, top1 logit, lse, target logit, claim logit) for ``rows``, whose targets were
        ``targets``."""
        self._grow(int(rows.max()) + 1)
        self.top1[rows] = got[:, 0].astype(np.int64)
        self.target[rows] = np.asarray(targets, np.int64)
        self.values[rows] = got[:, 1:]

    def draw(self, rows: np.ndarray, tokens: list[int]) -> None:
        """The sampler's own ``tokens`` for these reply rows (a row sampled again keeps its last draw)."""
        self._grow(int(rows.max()) + 1)
        self.drawn[rows] = np.asarray(tokens, np.int64)

    def _grow(self, need: int) -> None:
        if need > self.top1.shape[0]:
            grow = max(need, 2 * self.top1.shape[0])
            more = grow - self.top1.shape[0]
            self.top1 = np.concatenate([self.top1, np.full(more, -1, np.int64)])
            self.target = np.concatenate([self.target, np.full(more, -1, np.int64)])
            self.values = np.concatenate([self.values, np.full((more, 4), np.nan, np.float32)])
            self.drawn = np.concatenate([self.drawn, np.full(more, -1, np.int64)])

    def take(self, rows: int) -> dict[str, np.ndarray]:
        """Rows 0 .. rows - 1 (prompt + output - 1 for a reply), every one reduced."""
        if not self.done.is_set():
            raise RowsError("the request is still running")
        if rows < 1 or rows > self.top1.shape[0] or (self.top1[:rows] < 0).any():
            raise RowsError(f"the request reduced {int((self.top1 >= 0).sum())} rows, not the {rows} asked for")
        out = {"top1": self.top1[:rows].copy(), "target": self.target[:rows].copy(),
               "drawn": self.drawn[:rows].copy()}
        for i, name in enumerate(FIELDS[1:]):
            out[name] = self.values[:rows, i].copy()
        return out


_armed: Spec | None = None
_lock = threading.Lock()
_req: Spec | None = None                 # the running request's spec (every rank)
_scratch = None                          # [SUB, local vocab] bf16 for the prompt rows' head
_last: tuple[int, object] | None = None  # (n, this rank's logits shard of row n - 1) of the latest prefill of n rows


def arm(spec: Spec) -> None:
    """Rank 0: the next request is measured with ``spec`` (the other ranks get it with the request)."""
    global _armed
    if not ACTIVE:
        raise RowsError("this server was started without TENSORFOLD_SN46")
    with _lock:
        if _armed is not None:
            raise RowsError("a measurement is already armed")
        _armed = spec


def disarm() -> Spec | None:
    global _armed
    with _lock:
        spec, _armed = _armed, None
    return spec


def measuring() -> bool:
    """Whether the running request is measured (every rank)."""
    return _req is not None


def take_armed(prompt: list[int], supported: bool, images: bool = False) -> Spec | None:
    """Rank 0, at a request's start, before any other rank hears of it: the armed spec (then cleared), or None. A
    spec that does not fit the request is cleared and refused here, so a refusal never strands the other ranks."""
    spec = disarm() if ACTIVE else None
    if spec is None:
        return None
    if not supported:
        raise RowsError("row statistics run on the one-stream engine, for prompts without a grammar")
    if images and spec.cache:                # an image prompt's rows are not its ids' rows: nothing is kept
        raise RowsError("an image prompt keeps no prompt state; measure it with cache off")
    if spec.prompt is not None and list(prompt) != list(spec.prompt):
        raise RowsError("the request's prompt differs from the measured one")
    if spec.targets is not None and len(spec.targets) < len(prompt):
        raise RowsError("targets must cover every prompt row")
    if spec.claims is not None and (spec.targets is None or len(spec.claims) != len(spec.targets)):
        raise RowsError("claims and targets cover different rows")
    if spec.cache and spec.targets is not None:
        raise RowsError("a request walking given targets keeps no prompt state")
    return spec


def share(engine, spec: Spec | None) -> Spec | None:
    """Every rank, once per request: rank 0 sends whether this request is measured and its targets and claims with
    the request, so every rank measures exactly when rank 0 does (the gathers below would hang otherwise)."""

    if engine.rank == 0:
        engine._share(spec.header() if spec is not None else NO_SPEC)
        if spec is not None:
            for ids in (spec.targets, spec.claims):
                if ids is not None:
                    engine._share(list(ids))
        return spec
    header = engine._share(None)
    if header[0] == 0:
        return None
    targets = engine._share(None) if header[2] >= 0 else None
    claims = engine._share(None) if header[3] >= 0 else None
    return Spec.from_header(header, targets, claims)


def begin(spec: Spec | None, prompt: list[int], rank: int) -> None:
    """Every rank, before the request runs (rank 0 checked the spec fits it in ``take_armed``)."""
    global _req, _last
    _req = _last = None
    if spec is None:
        return
    spec.job = Job(spec, list(prompt)) if spec.record and rank == 0 else None
    spec.prompt = list(prompt)
    spec.keep_by = None if spec.keep_for_ms is None else time.monotonic() + spec.keep_for_ms / 1000
    _req = spec


def end() -> None:
    global _req, _last
    spec, _req, _last = _req, None, None
    if spec is not None and spec.job is not None:
        spec.job.done.set()


def keeps() -> bool:
    """Whether the running request may keep prompt states and resume from kept ones (every rank): an unmeasured
    request does, a measured one when its spec says so."""
    return _req is None or _req.cache


def keep_by() -> float | None:
    """The running request's deadline as a monotonic cutoff (``Spec.keep_by``, every rank), None when it sets none or
    is not measured."""
    return _req.keep_by if _req is not None else None


def _target(spec: Spec, row: int, sampled: int | None) -> int:
    if spec.targets is not None:
        return spec.targets[row] if row < len(spec.targets) else -1
    if row + 1 < len(spec.prompt):
        return spec.prompt[row + 1]
    return -1 if sampled is None else sampled


def forced(positions, sampled: list[int]) -> list[int] | None:
    """``Engine.sample`` (the reply's rows): a verifier's tokens at these positions (its targets past the prompt),
    which the engine commits instead of its own ``sampled`` ones; None when the request walks its own samples. A
    drafted window may sample a position past the targets (its bonus row after the last forced token): that one keeps
    the engine's sample, which the request, ending at the output's length, never commits."""
    spec = _req
    if spec is None or spec.targets is None or len(spec.targets) < len(spec.prompt):
        return None
    return [spec.targets[int(q) - 1] if int(q) - 1 < len(spec.targets) else s for q, s in zip(positions, sampled)]


def _claim(spec: Spec, row: int) -> int:
    return spec.claims[row] if spec.claims is not None and row < len(spec.claims) else -1


def reduce(w, logits, targets: list[int], claims: list[int]):
    """This rank's logits [R, local vocab] -> [R, 5] (top1 id, top1 logit, lse, target logit, claim logit) over every
    rank's shard, on every rank (one all-gather); ids -1 give -inf logits."""
    import torch

    packed = pack(logits, int(w.vocab_offset), targets, claims)
    world = 1 if w.comm is None else int(w.world)
    if world > 1:
        got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=packed.device)
        w.comm.all_gather(packed.view(-1), got)
        packed = got
    return combine(packed.view(world, logits.shape[0], 5)).cpu().numpy()


def pack(logits, offset: int, targets: list[int], claims: list[int]):
    """One rank's shard [R, n] (vocabulary ids offset .. offset + n) -> [R, 5] fp32: top1 id (int32 bits), top1 logit,
    lse, target logit, claim logit (-inf for ids outside the shard)."""
    import torch

    lf = logits.float()
    n = lf.shape[1]
    arg = torch.argmax(lf, 1)                             # the first maximum: the lowest id, as greedy sampling picks
    mx = lf.gather(1, arg[:, None])[:, 0]
    lse = torch.logsumexp(lf, 1)
    want = torch.tensor([targets, claims], dtype=torch.int64, device=lf.device) - offset     # [2, R]
    mine = (want >= 0) & (want < n)
    picked = lf.gather(1, want.clamp(0, n - 1).t()).t()                                       # [2, R]
    picked = torch.where(mine, picked, torch.full_like(picked, float("-inf")))
    ids = (arg + offset).to(torch.int32).view(torch.float32)
    return torch.stack([ids, mx, lse, picked[0], picked[1]], 1).contiguous()


def combine(g):
    """Every rank's ``pack`` [world, R, 5], ranks in vocabulary order -> [R, 5] with the id as an fp32 number."""
    import torch

    R = g.shape[1]
    best = torch.argmax(g[:, :, 1], 0)                    # the first rank holding the maximum: the lowest id
    rows = torch.arange(R, device=g.device)
    top1 = g[best, rows, 0].contiguous().view(torch.int32).float()     # ids < 2**24 are exact in fp32
    return torch.stack([top1, g[best, rows, 1], torch.logsumexp(g[:, :, 2], 0),
                        g[:, :, 3].max(0).values, g[:, :, 4].max(0).values], 1)


def prompt(e, b, prompt_ids, start: int, R: int) -> None:
    """``prefill_chunk``, after the chunk's forward: reduce its rows' heads (every rank)."""
    spec = _req
    if spec is None:
        return
    global _scratch
    import torch

    from tensorfold.families.glm5_next.cuda.forward import mm

    w = e.w
    if _scratch is None or _scratch.device != b.fnormed.device:
        _scratch = torch.empty((SUB, w.head.n), dtype=b.fnormed.dtype, device=b.fnormed.device)
    global _last
    for lo in range(0, R, SUB):
        hi = min(R, lo + SUB)
        logits = mm(b, b.fnormed[lo:hi], w.head, b.fxs[lo:hi], _scratch[:hi - lo])
        rows = list(range(start + lo, start + hi))
        targets = [_target(spec, r, None) for r in rows]
        got = reduce(w, logits, targets, [_claim(spec, r) for r in rows])
        if spec.job is not None:
            spec.job.put(np.asarray(rows), got, targets)
        if start + hi == len(prompt_ids):          # the prefill's last row: a kept state's ``Kept.last``
            _last = (start + hi, logits[hi - lo - 1].clone())


@dataclass
class Kept:
    """The row numbers a kept prompt state of n tokens carries (``decode.Snapshot.sn46``): rank 0's record of rows
    0 .. n - 2 (whose targets are the state's own tokens), and this rank's logits shard of row n - 1, whose target is
    whatever token the resuming prompt puts at n."""

    rows: int
    last: object                         # this rank's [local vocab] logits of row n - 1 (bf16, as the head left them)
    got: np.ndarray | None = None        # rank 0: [n - 1, 5] as ``Job.put`` takes it
    targets: np.ndarray | None = None    # rank 0: rows 0 .. n - 2's targets (the ids 1 .. n - 1)

    @property
    def nbytes(self) -> int:
        return int(self.last.numel() * self.last.element_size())


def kept(n: int) -> Kept | None:
    """``engine._run_once``, right after a measured request kept the state of its first n tokens (every rank): what
    the state carries, or None when the request is not measured. Rank 0's record of rows 0 .. n - 2 is complete, the
    prefill of those n rows having reduced them."""
    spec = _req
    if spec is None:
        return None
    if _last is None or _last[0] != n:
        raise RowsError(f"no prefill of {n} rows reduced its last row")
    out = Kept(n, _last[1])
    if spec.job is not None:
        job = spec.job
        if n - 1 > job.top1.shape[0] or (job.top1[:n - 1] < 0).any():
            raise RowsError(f"the kept state's {n - 1} rows were not all reduced")
        out.got = np.concatenate([job.top1[:n - 1, None].astype(np.float32), job.values[:n - 1]], 1)
        out.targets = job.target[:n - 1].copy()
    return out


def resume(w, kept: Kept | None, cut: int, prompt: list[int]) -> None:
    """``engine._run_once``, when a measured request resumes from a kept state of ``cut`` tokens (every rank): rows
    0 .. cut - 2 come from the state's record, row cut - 1 is reduced from its kept logits against the token the
    prompt puts at ``cut`` (none when the state is the whole prompt: the sampler reports that row)."""
    spec = _req
    if spec is None:
        return
    if kept is None or kept.rows != cut:
        raise RowsError(f"the kept state of {cut} tokens carries no row numbers")
    if spec.job is not None and kept.got is None:
        raise RowsError("the kept state carries no record of its rows on the recording rank")
    if spec.job is not None and cut > 1:
        spec.job.put(np.arange(cut - 1), kept.got, kept.targets.tolist())
    if cut < len(prompt):
        row = cut - 1
        got = reduce(w, kept.last[None, :], [prompt[cut]], [_claim(spec, row)])
        if spec.job is not None:
            spec.job.put(np.asarray([row]), got, [prompt[cut]])


TAIL = 3                                 # columns ``tail`` adds to a sampled row's gathered candidates


def wanted(positions) -> tuple[list[int], list[int]] | None:
    """``Engine.sample`` (the reply's own rows, not a drafter's), before the sampler gathers: the rows' target ids
    (-1 where the target is the token about to be sampled) and claimed ids, or None when the request is not
    measured (every rank)."""
    spec = _req
    if spec is None:
        return None
    rows = [int(q) - 1 for q in positions]
    return [_target(spec, r, None) for r in rows], [_claim(spec, r) for r in rows]


def tail(logits, offset: int, targets: list[int], claims: list[int]):
    """This rank's shard [R, n] -> [R, TAIL] fp32 (lse, target logit, claim logit; -inf for ids outside the shard),
    appended to the top-k the sampler gathers."""
    return pack(logits, offset, targets, claims)[:, 2:]


def sampled(w, logits, positions, tokens: list[int], gathered=None, k: int = 0) -> None:
    """``Engine.sample``, after the sample: record the rows' numbers; position ``q``'s logits are row ``q - 1``'s
    (every rank). ``gathered``: every rank's ``[values, ids, tail]`` as the sampler gathered it, [world, R, 2k + TAIL]
    on the CPU, from which the numbers follow without another gather; None when the sampler gathered no tail (the
    nucleus rule), then the rows are reduced here."""
    spec = _req
    if spec is None:
        return
    rows = [int(q) - 1 for q in positions]
    targets = [_target(spec, r, t) for r, t in zip(rows, tokens)]
    if gathered is None:
        got = reduce(w, logits, targets, [_claim(spec, r) for r in rows])
    else:
        got = from_gathered(gathered, k, targets)
    if spec.job is not None:
        spec.job.put(np.asarray(rows), got, targets)
        spec.job.draw(np.asarray(rows), tokens)


def from_gathered(g, k: int, targets: list[int]) -> np.ndarray:
    """Every rank's ``[values, ids, tail]`` [world, R, 2k + TAIL] (CPU) -> [R, 5] (top1 id, top1 logit, lse, target
    logit, claim logit). The top-1 is the lowest id among the maxima of the gathered values (the greedy rule); a
    target the tail did not know (-inf everywhere: a token sampled from these candidates) takes its logit from the
    candidate of that id. In numpy: a torch CPU op here wakes the rank's OpenMP pool, and the ranks' pools contend
    on the host (four ranks, 70 ms a call on a 96-core host against 0.1 ms single-threaded)."""
    a = np.ascontiguousarray(g.numpy() if hasattr(g, "numpy") else g, dtype=np.float32)
    world, R, width = a.shape
    values = np.concatenate([a[r, :, :k] for r in range(world)], 1)
    ids = np.concatenate([a[r, :, k:2 * k] for r in range(world)], 1).view(np.int32).astype(np.int64)
    order = np.argsort(ids, axis=1, kind="stable")                                    # ties: the lowest id first
    values, ids = np.take_along_axis(values, order, 1), np.take_along_axis(ids, order, 1)
    best = np.argmax(values, 1)                                                       # the first maximum
    rows = np.arange(R)
    top1, top1_logit = ids[rows, best], values[rows, best]
    part = a[:, :, 2 * k]                                                             # [world, R] each rank's lse
    m = part.max(0)
    with np.errstate(invalid="ignore"):
        lse = np.where(np.isinf(m), m, m + np.log(np.exp(part - m).sum(0)))
    target_logit = a[:, :, 2 * k + 1].max(0)
    claim_logit = a[:, :, 2 * k + 2].max(0)
    want = np.asarray(targets, dtype=np.int64)
    hit = np.where(ids == want[:, None], values, np.float32(-np.inf)).max(1)
    target_logit = np.where(np.isinf(target_logit) & (want >= 0), hit, target_logit)
    return np.stack([top1.astype(np.float32), top1_logit, lse, target_logit, claim_logit], 1).astype(np.float32)
