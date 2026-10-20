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
