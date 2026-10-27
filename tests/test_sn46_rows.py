"""tensorfold.sn46.rows: the per-rank reduction and its merge equal a dense computation over the whole vocabulary,
and a Job keeps the last numbers of each row (CPU torch; no engine)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.sn46 import rows  # noqa: E402


def dense(logits, targets, claims):
    lf = logits.float()
    top1 = torch.argmax(lf, 1)
    pick = lambda ids: torch.tensor([lf[r, i] if i >= 0 else float("-inf") for r, i in enumerate(ids)])  # noqa: E731
    return torch.stack([top1.float(), lf.max(1).values, torch.logsumexp(lf, 1), pick(targets), pick(claims)], 1)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_merged_shards_equal_the_dense_reduction(world):
    g = torch.Generator().manual_seed(world)
    R, V = 7, 64
    logits = (torch.randn((R, V), generator=g) * 4).to(torch.bfloat16)
    logits[3, 10] = logits[3, 50] = logits[3].float().max() + 1      # a tie across shards: the lowest id wins
    targets = [5, 63, -1, 0, 31, 32, 17]
    claims = [-1, 1, 2, 50, 60, 33, 16]
    n = V // world
    packed = torch.stack([rows.pack(logits[:, k * n:(k + 1) * n], k * n, targets, claims) for k in range(world)])
    got = rows.combine(packed)
    want = dense(logits, targets, claims)
    assert torch.equal(got[:, 0], want[:, 0])
    assert got[3, 0] == 10
    torch.testing.assert_close(got[:, 1:], want[:, 1:], rtol=0, atol=1e-5)


@pytest.mark.parametrize("world,k", [(1, 1), (2, 1), (4, 1), (2, 3)])
def test_the_numbers_read_off_the_samplers_gather_equal_the_dense_reduction(world, k):
    """Each rank gathers its top-k values and ids with ``tail``; ``from_gathered`` gives the same numbers as the
    dense reduction, for targets the tail knew (a verifier's) and for the sampled token (a miner's, -1)."""
    g = torch.Generator().manual_seed(10 * world + k)
    R, V = 6, 64
    logits = (torch.randn((R, V), generator=g) * 4).to(torch.bfloat16)
    logits[2, 9] = logits[2, 40] = logits[2].float().max() + 1       # a tie across shards: the lowest id wins
    want = dense(logits, [0] * R, [0] * R)
    top1 = [int(t) for t in want[:, 0]]
    targets = [top1[0], -1, -1, 63, -1, 5]                           # -1: the token about to be sampled (the top-1)
    claims = [-1, 1, 2, 50, 60, 33]
    n = V // world
    parts = []
    for r in range(world):
        shard = logits[:, r * n:(r + 1) * n].float()
        vals, ids = torch.topk(shard, k, dim=-1)
        ids = (ids + r * n).to(torch.int32)
        parts.append(torch.cat([vals, ids.view(torch.float32), rows.tail(shard, r * n, targets, claims)], 1))
    got = rows.from_gathered(torch.stack(parts), k, [t if t >= 0 else top1[i] for i, t in enumerate(targets)])
    want = dense(logits, [t if t >= 0 else top1[i] for i, t in enumerate(targets)], claims).numpy()
    assert got[:, 0].tolist() == want[:, 0].tolist() and got[2, 0] == 9
    np.testing.assert_allclose(got[:, 1:], want[:, 1:], rtol=0, atol=1e-5)


def test_a_job_keeps_the_last_numbers_of_each_row_and_takes_complete_rows():
    job = rows.Job(rows.Spec(record=True), [1, 2, 3])
    job.put(np.array([0, 1, 2]), np.array([[7, 1, 2, 3, 4]] * 3, np.float32), [2, 3, 40])
    job.put(np.array([3, 4]), np.array([[8, 1, 2, 3, 4]] * 2, np.float32), [41, 42])
    job.put(np.array([3]), np.array([[9, 5, 6, 7, 8]], np.float32), [43])  # row 3 sampled again after a rejected draft
    job.done.set()
    got = job.take(5)
    assert got["top1"].tolist() == [7, 7, 7, 9, 8]
    assert got["target"].tolist() == [2, 3, 40, 43, 42]
    assert got["lse"][3] == 6 and got["claim_logit"][3] == 8
    with pytest.raises(rows.RowsError):
        job.take(6)


def test_a_job_keeps_each_reply_rows_own_draw_beside_its_target():
    """A verifier's walk commits the miner's token (the target) but records the engine's own draw; prompt rows have
    none, and a row sampled again keeps its last draw."""
    job = rows.Job(rows.Spec(record=True), [1, 2, 3])
    job.put(np.array([0, 1]), np.array([[7, 1, 2, 3, 4]] * 2, np.float32), [2, 3])
    job.put(np.array([2, 3]), np.array([[8, 1, 2, 3, 4]] * 2, np.float32), [40, 41])
    job.draw(np.array([2, 3]), [40, 99])
    job.draw(np.array([3]), [41])
    job.done.set()
    got = job.take(4)
    assert got["drawn"].tolist() == [-1, -1, 40, 41]
    assert got["target"].tolist() == [2, 3, 40, 41]


def test_targets_are_the_prompt_then_the_sampled_tokens_unless_the_spec_names_them():
    spec = rows.Spec(record=False, prompt=[10, 11, 12])
    assert [rows._target(spec, r, 99) for r in range(4)] == [11, 12, 99, 99]
    spec = rows.Spec(record=False, targets=[11, 12, 13, 14], prompt=[10, 11, 12])
    assert [rows._target(spec, r, 99) for r in range(5)] == [11, 12, 13, 14, -1]


def test_a_verifier_walks_the_miner_tokens_past_the_prompt():
    spec = rows.Spec(record=False, targets=[11, 12, 20, 21, 22], prompt=[10, 11, 12])
    rows._req = spec
    try:
        # Position q's sample is row q - 1's target: the output starts at position 3.
        assert rows.forced([3, 4, 5], [7, 7, 7]) == [20, 21, 22]
        # A drafted window's bonus row past the targets keeps the engine's sample.
        assert rows.forced([5, 6], [7, 8]) == [22, 8]
        # A miner (no targets) walks its own samples.
        rows._req = rows.Spec(record=False, prompt=[10, 11, 12])
        assert rows.forced([3], [7]) is None
    finally:
        rows._req = None
    assert rows.forced([3], [7]) is None


class FakeWeights:
    """A one-rank engine's weights as ``reduce`` reads them."""

    vocab_offset = 0
    comm = None
    world = 1


def test_a_kept_state_carries_its_rows_and_a_resumed_request_reports_a_fresh_prefills_numbers():
    """A measured request keeping the state of its first n tokens records rows 0 .. n - 2 and its logits of row
    n - 1; a request resuming there reports those rows, the last reduced against its own token at n."""
    g = torch.Generator().manual_seed(7)
    V = 64
    prompt = [10, 11, 12, 13, 14, 15]
    first = rows.Spec(record=True, prompt=prompt, cache=True)
    rows.begin(first, prompt, 0)
    try:
        logits = (torch.randn((4, V), generator=g) * 4).to(torch.bfloat16)     # the prefill of prompt[:4]
        got = rows.reduce(FakeWeights(), logits, prompt[1:5], [-1] * 4)
        first.job.put(np.arange(4), got, prompt[1:5])
        rows._last = (4, logits[3].clone())                                  # as ``prompt`` leaves it
        kept = rows.kept(4)
        assert kept.rows == 4 and kept.got.shape == (3, 5) and kept.targets.tolist() == prompt[1:4]
        assert kept.nbytes == logits[3].numel() * logits[3].element_size()
        with pytest.raises(rows.RowsError):
            rows.kept(5)
    finally:
        rows.end()
    assert rows.kept(4) is None                                              # no measured request: nothing to keep

    later = [10, 11, 12, 13, 50, 51]                                         # shares the kept four, then differs
    second = rows.Spec(record=True, prompt=later, cache=True)
    rows.begin(second, later, 0)
    try:
        rows.resume(FakeWeights(), kept, 4, later)
        want = dense(logits, [11, 12, 13, 50], [-1] * 4)
        assert second.job.top1[:4].tolist() == want[:, 0].tolist()
        assert second.job.target[:4].tolist() == [11, 12, 13, 50]
        np.testing.assert_allclose(second.job.values[:4], want[:, 1:].numpy(), rtol=0, atol=1e-5)
        assert np.isnan(second.job.values[4:]).all() if second.job.values.shape[0] > 4 else True
        with pytest.raises(rows.RowsError):
            rows.resume(FakeWeights(), kept, 3, later)                       # a state of another length
        with pytest.raises(rows.RowsError):
            rows.resume(FakeWeights(), None, 4, later)                       # an unmeasured request's state
    finally:
        rows.end()

    whole = rows.Spec(record=True, prompt=prompt[:4], cache=True)            # the state is the whole prompt
    rows.begin(whole, prompt[:4], 0)
    try:
        rows.resume(FakeWeights(), kept, 4, prompt[:4])
        assert whole.job.top1[:3].tolist() == want[:3, 0].tolist()
        assert whole.job.top1.shape[0] < 4 or whole.job.top1[3] < 0           # row 3: the sampler's, not reduced here
    finally:
        rows.end()


def test_a_measured_request_keeps_state_only_when_its_spec_allows_and_a_verifier_walk_may_keep_one():
    assert rows.keeps()                                                      # unmeasured: the engine's own policy
    rows._req = rows.Spec(record=False, prompt=[1, 2])
    try:
        assert not rows.keeps()
        rows._req.cache = True
        assert rows.keeps()
    finally:
        rows._req = None
    assert rows.Spec(record=True, cache=True).header()[4] == 1 and len(rows.NO_SPEC) == 7
    rows.ACTIVE = True
    try:
        walk = rows.Spec(record=True, targets=[2, 3, 4], claims=[2, -1, 4], prompt=[1, 2], cache=True)
        rows.arm(walk)
        assert rows.take_armed([1, 2], True) is walk                         # a verifier's walk keeps and resumes
        rows.arm(rows.Spec(record=True, prompt=[1, 2], cache=True))
        with pytest.raises(rows.RowsError):
            rows.take_armed([1, 2], True, images=True)                       # an image prompt still keeps nothing
        assert rows.disarm() is None                                         # refused specs are cleared
    finally:
        rows.ACTIVE = False


def test_a_verifier_resumes_only_from_a_state_whose_top1_its_claims_name_and_reports_a_fresh_reduction():
    """A verifier's claims on the kept rows are the miner's top-1: a state admits them when its top-1 equals every
    claim (or there is none, -1); resuming then reports each claim's logit as the row's top-1 logit (-inf for none),
    and row cut - 1 is reduced against the request's own claim, all as a fresh prefill would."""
    g = torch.Generator().manual_seed(11)
    V = 64
    prompt = [10, 11, 12, 13, 14, 15]
    logits = (torch.randn((4, V), generator=g) * 4).to(torch.bfloat16)        # the prefill of prompt[:4]
    first = rows.Spec(record=True, prompt=prompt, cache=True)
    rows.begin(first, prompt, 0)
    try:
        first.job.put(np.arange(4), rows.reduce(FakeWeights(), logits, prompt[1:5], [-1] * 4), prompt[1:5])
        rows._last = (4, logits[3].clone())
        kept = rows.kept(4)
    finally:
        rows.end()

    later = [10, 11, 12, 13, 50]
    output = [60, 61]
    top1 = [int(t) for t in torch.argmax(logits.float(), 1)]
    claims = [top1[0], -1, top1[2], 7, 8, 9]                                 # rows 0 .. 2 from the state, 3 fresh
    spec = rows.Spec(record=True, targets=later[1:] + output, claims=claims, prompt=later, cache=True)
    assert rows.admits(kept, spec)
    rows.begin(spec, later, 0)
    try:
        rows.resume(FakeWeights(), kept, 4, later)
        want = dense(logits, [11, 12, 13, 50], claims[:4]).numpy()
        got = spec.job.values[:4]
        assert spec.job.top1[:4].tolist() == want[:, 0].tolist()
        assert got[0, 3] == got[0, 0] and got[2, 3] == got[2, 0]             # the claim is the top-1: its logit
        assert np.isneginf(got[1, 3])                                        # no claim
        np.testing.assert_allclose(got[:, :3], want[:, 1:4], rtol=0, atol=1e-5)
        np.testing.assert_allclose(got[:, 3], want[:, 4], rtol=0, atol=1e-5)  # row 3 from its own claim, fresh
        assert kept.got[0, 4] != got[0, 3] or claims[0] == -1                # the state's own record is untouched
    finally:
        rows.end()
    assert spec.claims_array is None                                         # nothing of the claims outlives the run

    differs = list(claims)
    differs[1] = (top1[1] + 1) % V
    wrong = rows.Spec(record=True, targets=later[1:] + output, claims=differs, prompt=later, cache=True)
    assert not rows.admits(kept, wrong)
    rows.begin(wrong, later, 0)
    try:
        with pytest.raises(rows.RowsError):
            rows.resume(FakeWeights(), kept, 4, later)
    finally:
        rows.end()
    assert rows.admits(kept, rows.Spec(record=True, prompt=later, cache=True))   # a miner claims nothing
    assert not rows.admits(rows.Kept(4, None), spec)                         # no record to check against


def test_a_specs_keep_duration_travels_in_int32_halves_and_comes_back_whole():
    """``share`` moves int32s: a duration past 2**31 ms goes as two 31-bit halves (0 for none)."""
    keep = 2**40 + 123                                                       # well above int32
    header = rows.Spec(record=True, targets=[2, 3], claims=[-1, 4], cache=True, keep_for_ms=keep).header()
    assert len(header) == 7 and all(0 <= v < 2**31 for v in header[4:])
    back = rows.Spec.from_header(header, [2, 3], [-1, 4])
    assert (back.keep_for_ms, back.cache, back.record) == (keep, True, False)
    assert (back.targets, back.claims) == ([2, 3], [-1, 4])
    assert rows.Spec.from_header(rows.Spec(record=True).header(), None, None).keep_for_ms is None
    assert rows.NO_SPEC[0] == 0 and len(rows.NO_SPEC) == len(header)


def test_a_request_without_claims_resuming_a_scored_state_reports_no_claim_logits():
    """A state a verifier's walk kept carries its claims' logits; a request without claims resuming from it reports
    -inf for every claim logit, as a fresh reduction does, and leaves the state's record as it was."""
    g = torch.Generator().manual_seed(13)
    V = 64
    prompt = [10, 11, 12, 13, 14, 15]
    logits = (torch.randn((4, V), generator=g) * 4).to(torch.bfloat16)
    top1 = [int(t) for t in torch.argmax(logits.float(), 1)]
    walk = rows.Spec(record=True, targets=prompt[1:] + [60], claims=top1 + [-1, -1], prompt=prompt, cache=True)
    rows.begin(walk, prompt, 0)
    try:
        walk.job.put(np.arange(4), rows.reduce(FakeWeights(), logits, prompt[1:5], top1), prompt[1:5])
        rows._last = (4, logits[3].clone())
        kept = rows.kept(4)
    finally:
        rows.end()
    record = kept.got.copy()
    assert np.isfinite(record[:, 4]).all()                                   # the walk's claim logits
    plain = rows.Spec(record=True, prompt=prompt, cache=True)
    rows.begin(plain, prompt, 0)
    try:
        rows.resume(FakeWeights(), kept, 4, prompt)
        assert np.isneginf(plain.job.values[:4, 3]).all()                    # rows 0 .. 2 from the state, 3 fresh
    finally:
        rows.end()
    np.testing.assert_array_equal(kept.got, record)
