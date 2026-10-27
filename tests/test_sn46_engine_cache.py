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
    assert [c.ids for c in e.cache] == [[2] * 4] and e.live == [2] * 4      # cut to the kept prefix of the live ids
    e.cache[0].kept_at -= 20
    e._forget(e._stale())
    assert e.cache == [] and e.live == []                                       # nothing kept names them


def test_a_kept_state_past_its_deadline_is_dropped_whatever_its_last_use():
    import time

    now = time.monotonic()
    for ttl in (0.0, 600.0):                                    # with the TTL off, and well inside it
        e = bare(ttl)
        for ids, by in (([1] * 4, now - 0.001), ([2] * 4, now + 60), ([3] * 4, None)):
            s = snap(ids)
            s.keep_by = by
            e._remember(s)
        assert e._stale() == [0]                                # used just now, but past its deadline
        e._forget(e._stale())
        assert [c.ids for c in e.cache] == [[2] * 4, [3] * 4]
    e = bare(10.0)
    s = snap([4] * 4)
    s.keep_by = now + 60
    e._remember(s)
    e.cache[0].kept_at -= 11
    assert e._stale() == [0]                                    # the TTL still drops a state inside its deadline


def test_a_wall_clock_step_changes_nothing(monkeypatch):
    """The keep duration becomes a monotonic cutoff when the request starts; the wall clock is never read: stepped back
    an hour (or forward) it changes nothing, the state goes when the monotonic clock passes the cutoff."""
    import time

    from tensorfold.sn46 import rows

    wall, mono = time.time(), time.monotonic()
    monkeypatch.setattr(time, "time", lambda: wall)
    monkeypatch.setattr(time, "monotonic", lambda: mono)
    spec = rows.Spec(record=False, prompt=[1] * 4, cache=True, keep_for_ms=1_000)
    rows.begin(spec, [1] * 4, 1)
    try:
        assert abs(rows.keep_by() - (mono + 1.0)) < 0.01
        by = rows.keep_by()
    finally:
        rows.end()
    e = bare(600.0)
    s = snap([1] * 4)
    s.keep_by = by
    e._remember(s)
    monkeypatch.setattr(time, "time", lambda: wall - 3_600)    # the wall clock steps back an hour
    monkeypatch.setattr(time, "monotonic", lambda: mono + 0.5)
    assert e._stale() == []
    monkeypatch.setattr(time, "time", lambda: wall + 3_600)    # or forward
    assert e._stale() == []
    monkeypatch.setattr(time, "monotonic", lambda: mono + 1.5)
    assert e._stale() == [0]


def test_a_resuming_request_raises_a_states_deadline_and_never_lowers_it():
    e = bare(10.0)
    s = snap([1] * 4)
    s.keep_by, s.kept_at = 1.0, 0.0
    e._touch(s, 5.0)
    assert s.keep_by == 5.0 and s.kept_at > 0                 # extended, and its TTL counts from now
    e._touch(s, 2.0)
    assert s.keep_by == 5.0                                   # an earlier deadline does not shorten it
    e._touch(s, None)
    assert s.keep_by == 5.0                                   # nor does a request without one
    free = snap([2] * 4)
    e._touch(free, 9.0)
    assert free.keep_by is None                               # a TTL-only state stays TTL-only


def test_a_raised_deadline_drops_the_source_prompts_rendering():
    """TF_W1_INCREMENTAL keeps the whole source prompt's rendering (text and ids past the state) with a grid state:
    a later request raising the state's deadline must not keep that suffix alive with it."""
    e = bare(10.0)
    s = snap([1] * 4)
    s.keep_by, s.w1_rendered = 1.0, ("the source prompt's rendering", 4)
    e._touch(s, 0.5)
    assert s.w1_rendered is not None                          # not raised: the source's own deadline still holds
    e._touch(s, 5.0)
    assert s.keep_by == 5.0 and s.w1_rendered is None


def test_expired_live_ids_keep_only_the_prefix_a_kept_state_still_names():
    """A request resumed from an older state whose deadline is later than its own: once its deadline passes, the
    live ids keep that state's prefix, never the request's own suffix and reply."""
    e = bare(600.0)
    older, longer = snap([1] * 4), snap([1] * 8)
    for s in (older, longer):
        e._remember(s)
    e.live, e.live_by = [1] * 4 + [5, 6, 7], 1.0              # the request's prompt suffix and reply past [1] * 4
    e._trim_live()
    assert e.live == [1] * 4 and e.live_by is None             # the longest kept prefix of them, not [1] * 8
    e.live = [1] * 8 + [9]
    e._trim_live()
    assert e.live == [1] * 8
    e.cache = []
    e._trim_live()
    assert e.live == []                                        # nothing kept names them: all forgotten
