"""GLM engine: the ``chunk`` prompt grid, and kept prompt states past TF_GLM_CACHE_TTL dropped alike on every rank
(CPU torch; no model, no engine)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm5_next.cuda import engine  # noqa: E402
from tensorfold.families.glm5_next.cuda.decode import Snapshot, snapshot_bytes  # noqa: E402


def test_the_chunk_grid_is_the_prompt_chunk_so_a_resumed_prefill_runs_a_fresh_ones_chunks():
    assert engine.prompt_grid(4096, "chunk", chunked=True) == 4096
    assert engine.prompt_grid(1024, "chunk", chunked=False) == 1024
    assert engine.prompt_grid(4096, "64", chunked=True) == 64
    with pytest.raises(ValueError):
        engine.prompt_grid(4096, "chunks", chunked=True)
    # a state sits only at a multiple of the chunk past the resume point: a shorter prompt keeps nothing
    assert engine.grid_point(5000, 0, 4096) == 4096
    assert engine.grid_point(3000, 0, 4096) is None
    assert engine.grid_point(8192, 4096, 4096) == 8192
    assert engine.grid_point(9000, 8192, 4096) is None


def snap(ids):
    s = Snapshot(list(ids), torch.zeros(2), torch.zeros(2), None, -1, -1)
    return s


def bare(ttl: float) -> engine.GlmEngine:
    e = engine.GlmEngine.__new__(engine.GlmEngine)
    e.cache, e.cache_ttl, e.cache_entries, e.cache_bytes, e.live = [], ttl, 8, 1 << 20, []
    return e


def test_kept_states_past_the_ttl_are_dropped_on_every_rank_by_rank_zeros_list():
    rank0, rank1 = bare(10.0), bare(10.0)
    for e in (rank0, rank1):
        for ids in ([1] * 4, [2] * 4, [3] * 4):
            e._remember(snap(ids))
    rank0.cache[0].kept_at -= 11                                # stale on rank 0's clock
    rank0.cache[2].kept_at -= 11
    rank1.cache[1].kept_at -= 11                                # rank 1's clock says otherwise: it does not decide
    stale = rank0._stale()
    assert stale == [0, 2]
    rank0._forget(stale)
    rank1._forget(stale)
    assert [c.ids for c in rank0.cache] == [c.ids for c in rank1.cache] == [[2] * 4]
    assert bare(0.0)._stale() == []                             # TTL off
    kept = snap([5] * 4)
    kept.sn46 = type("Kept", (), {"nbytes": 100})()
    assert snapshot_bytes(kept) == 16 + 100                     # the kept rows' numbers count toward the budget
    e = bare(0.0)
    e._remember(kept)
    e._drop(kept)
    assert kept.sn46 is None and e.cache == []


def test_the_chunk_grid_leaves_the_lanes_their_own_grid():
    """Kept states on chunk boundaries only: a chunk's two lanes still cut on the kernels' 64-row grid, as with the
    default prompt grid (a 4096 lane grid would leave no chunk of 4096 rows for two lanes)."""
    from tensorfold.families.glm5_next.cuda import lanes

    assert lanes.cut(4096, 4, 64) == 2048
    assert lanes.cut(4096, 4, 4096) is None
    e = engine.GlmEngine.__new__(engine.GlmEngine)
    e.grid = engine.prompt_grid(4096, "chunk", chunked=True)
    e.lane_grid = engine.prompt_grid(4096, "0", chunked=True) if e.grid == 4096 else e.grid
    assert (e.grid, e.lane_grid) == (4096, 64)


def test_a_reused_state_counts_its_ttl_from_its_last_use_and_forgetting_frees_the_live_ids():
    e = bare(10.0)
    e._remember(snap([1] * 4))
    e._remember(snap([2] * 4))
    e.cache[0].kept_at -= 20
    e.live = [2] * 4 + [9]
    e._forget(e._stale())
    assert [c.ids for c in e.cache] == [[2] * 4] and e.live == [2] * 4 + [9]   # a kept prefix of the live ids
    e.cache[0].kept_at -= 20
    e._forget(e._stale())
    assert e.cache == [] and e.live == []                                       # nothing kept names them
