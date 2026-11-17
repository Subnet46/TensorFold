"""TENSORFOLD_SN46 under --parallel on CPU: specs armed for named jobs, the spec carried in an ADMIT, the one-stream
hooks acting on one stream at a time (``use``), the streams' decode rows reduced together (``decode``) and a stream's
reply hidden rows kept in order."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")      # the GLM backend's kernels (CUDA hosts)

from tensorfold.sn46 import rows


@pytest.fixture
def active(monkeypatch):
    monkeypatch.setattr(rows, "ACTIVE", True)
    monkeypatch.setattr(rows, "_armed", None)
    monkeypatch.setattr(rows, "_armed_by", {})
    monkeypatch.setattr(rows, "_req", None)
    monkeypatch.setattr(rows, "_last", None)


def test_specs_armed_for_named_jobs(active):
    a, b = rows.Spec(record=True, cache=True), rows.Spec(record=True)
    rows.arm(a, "job-a")
    rows.arm(b, "job-b")
    with pytest.raises(rows.RowsError):
        rows.arm(rows.Spec(record=True), "job-a")
    assert rows.take_armed([1, 2, 3], True, job="job-b") is b
    with pytest.raises(rows.RowsError, match="no measurement is armed"):   # taken once; a second request naming
        rows.take_armed([1, 2, 3], True, job="job-b")                      # it (after an abort) runs nothing
    assert rows.take_armed([1, 2, 3], True) is None                       # the unnamed slot is apart
    assert rows.take_armed([1, 2, 3], True, job="job-a") is a
    rows.arm(rows.Spec(record=True), "job-c")
    with pytest.raises(rows.RowsError, match="grammar"):
        rows.take_armed([1, 2], False, job="job-c")


def test_the_spec_rides_in_the_admit():
    for spec in (None, rows.Spec(record=True, cache=True, keep_for_ms=(1 << 40) + 5),
                 rows.Spec(record=True, targets=[5, 6, 7], claims=[-1, 6, 9])):
        packed = [99] + rows.pack_spec(spec) + [42]
        got, i = rows.unpack_spec(packed, 1)
        assert packed[i] == 42
        if spec is None:
            assert got is None
            continue
        assert (got.cache, got.keep_for_ms, got.targets, got.claims, got.record) == (
            spec.cache, spec.keep_for_ms, spec.targets, spec.claims, False)


def test_use_swaps_the_running_request(active):
    a = rows.start(rows.Spec(record=True), [1, 2, 3], 0)
    b = rows.start(rows.Spec(record=False), [4, 5], 1)
    assert a.spec.job is not None and b.spec.job is None
    with rows.use(a):
        assert rows.measuring() and rows._req is a.spec
        rows._last = (3, "a's row")
        with rows.use(b):
            assert rows._req is b.spec and rows._last is None
        assert rows._req is a.spec and rows._last == (3, "a's row")
    assert not rows.measuring() and a.last == (3, "a's row")
    with rows.use(None):
        assert not rows.measuring()


def test_decode_reduces_every_stream_together(active, monkeypatch):
    calls = []

    def fake_reduce(w, logits, targets, claims):
        calls.append((logits.shape[0], list(targets), list(claims)))
        n = logits.shape[0]
        out = np.zeros((n, 5), np.float32)
        out[:, 0] = np.arange(n)
        out[:, 2] = 10 + np.arange(n)
        return out

    monkeypatch.setattr(rows, "reduce", fake_reduce)
    a = rows.start(rows.Spec(record=True), [1, 2, 3], 0)
    b = rows.start(rows.Spec(record=True), [7, 8], 0)
    ha, hb = torch.arange(8, dtype=torch.bfloat16).view(2, 4), torch.ones((1, 4), dtype=torch.bfloat16)
    rows.decode(None, [(a, torch.zeros((2, 6)), [3, 4], [11, 12], ha), (b, torch.zeros((1, 6)), [2], [21], hb)])
    assert calls == [(3, [11, 12, 21], [-1, -1, -1])]               # one gather, a miner's targets: its samples
    assert a.spec.job.target[3:5].tolist() == [11, 12] and a.spec.job.drawn[3:5].tolist() == [11, 12]
    assert b.spec.job.target[2] == 21 and b.spec.job.values[2, 1] == 12          # lse of b's row: the third
    rows.decode(None, [(a, torch.zeros((1, 6)), [5], [13], ha[:1])])
    rows.finish(a)
    assert a.spec.job.done.is_set()
    assert a.spec.job.hidden_from == 3 and a.spec.job.hidden.shape == (3, 4)
    assert a.spec.job.hidden.dtype == np.uint16
    assert (a.spec.job.hidden[:2] == ha.view(torch.int16).numpy().view(np.uint16)).all()


def test_hidden_rows_must_follow_each_other(active):
    job = rows.Job(rows.Spec(record=True), [1, 2])
    job.keep_hidden(2, torch.zeros((2, 4), dtype=torch.bfloat16))
    job.keep_hidden(5, torch.zeros((1, 4), dtype=torch.bfloat16))
    with pytest.raises(rows.RowsError, match="hidden rows"):
        job.close_hidden()


def test_the_one_pass_check_keeps_only_the_answers_hidden_rows(active):
    """A verifier's one-pass prompt is the miner's prompt and answer: rows hidden_from .. len - 2 are the answer's
    forwards (the last is nobody's input); nothing is reduced."""
    import types

    spec = rows.Spec(record=True, hidden_from=5)
    packed = rows.pack_spec(spec)
    assert rows.unpack_spec(packed, 0)[0].hidden_from == 5
    lane = rows.start(spec, list(range(9)), 0)                 # prompt 5 + answer 4: rows 5, 6, 7 are wanted
    hidden = torch.arange(8 * 4, dtype=torch.float32).view(8, 4).to(torch.bfloat16)
    b = types.SimpleNamespace(hidden=hidden, fnormed=None, fxs=None)
    e = types.SimpleNamespace(w=None)
    with rows.use(lane):
        rows.prompt(e, b, list(range(9)), 0, 4)                 # chunk rows 0..3 at b rows 0..3: none wanted
        rows.prompt(e, b, list(range(9)), 4, 5, lo_row=3)       # rows 4..8 at b rows 3..7: rows 5..7
    rows.finish(lane)
    job = spec.job
    assert job.hidden_from == 5 and job.hidden.shape == (3, 4)
    assert (job.hidden == hidden[4:7].view(torch.int16).numpy().view(np.uint16)).all()
    assert (job.top1 < 0).all()                                 # nothing reduced


def test_a_one_pass_check_with_cache_on_is_refused_before_dispatch(active):
    """It reduces no row, so a state it kept could not carry its rows' numbers (``rows.kept`` would fail mid-run)."""
    rows.arm(rows.Spec(record=True, prompt=list(range(9)), cache=True, hidden_from=5))
    with pytest.raises(rows.RowsError, match="cache off"):
        rows.take_armed(list(range(9)), True)
    assert rows.disarm() is None                                # refused specs are cleared
    rows.arm(rows.Spec(record=True, prompt=list(range(9)), hidden_from=5))
    assert rows.take_armed(list(range(9)), True).hidden_from == 5


def test_a_replay_keeps_the_requests_retention_cutoff(active, monkeypatch):
    spec = rows.Spec(record=True, cache=True, keep_for_ms=1000)
    monkeypatch.setattr(rows.time, "monotonic", lambda: 100.0)
    rows.start(spec, [1, 2, 3], 0)
    assert spec.keep_by == 101.0
    monkeypatch.setattr(rows.time, "monotonic", lambda: 160.0)    # requeued a minute later
    rows.start(spec, [1, 2, 3], 0)
    assert spec.keep_by == 101.0
