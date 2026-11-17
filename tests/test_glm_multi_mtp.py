"""mtp_multi on CPU: the batched MTP head's round bookkeeping (absorb, chains, rollback of drafted rows) with the
layer's step faked, the policy codes MTP streams get under --parallel, and the setting."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")      # the GLM backend's kernels (CUDA hosts)

from tensorfold.families.glm5_next.cuda import mtp_multi
from tensorfold.families.glm5_next.cuda.mtp_multi import Ask, BatchedMTP, Carry
from tensorfold.families.glm5_next.cuda.multi import MTP_AUTO, multi_code


class St:
    def __init__(self, mtp_len: int) -> None:
        self.mtp_len, self.mtp_drafted = mtp_len, 0


class FakeHead(BatchedMTP):
    """The chain's bookkeeping on CPU tensors: a step records its parts' (position, tokens) and each row's next token
    is its token + 1 (the layer itself faked)."""

    def __init__(self, rows_max: int = 128) -> None:
        import types

        self.rows_max = rows_max
        self.calls: list[list[tuple[int, list[int]]]] = []
        self.w = None
        D = 4
        self.b = types.SimpleNamespace(ids=torch.zeros(rows_max, dtype=torch.int64), hin=torch.zeros((rows_max, D)),
                                       fnormed=torch.zeros((rows_max, D)), rows=rows_max)
        self.nxt = torch.zeros(rows_max, dtype=torch.int32)
        self.chain = torch.zeros((4, 64), dtype=torch.int32)

    def _ix(self, values):
        return torch.tensor(values, dtype=torch.int64)

    def _span(self, st, k):
        return (0, st.mtp_len, k)

    def _tables(self, spans, step):
        self.calls.append([(pos, k) for _, pos, k in spans])
        return False

    def _run(self, R, sparse, zero):
        b = self.b
        self.calls[-1] = [(pos, b.ids[at:at + k].tolist()) for (pos, k), at in
                          zip(self.calls[-1], _starts([k for _, k in self.calls[-1]]))]
        self.nxt[:R] = (b.ids[:R] + 1).to(torch.int32)
        b.fnormed[:R] = b.hin[:R]

    def step(self, parts):
        b = self.b
        at = 0
        for st, tokens, hidden in parts:
            assert hidden.shape[0] == len(tokens)
            b.ids[at:at + len(tokens)] = torch.tensor(tokens)
            b.hin[at:at + len(tokens)] = hidden
            at += len(tokens)
        self._tables([(0, st.mtp_len, len(tokens)) for st, tokens, _ in parts], 0)
        self._run(at, False, [])
        for st, tokens, _ in parts:
            st.mtp_len += len(tokens)


def _starts(lengths):
    out, at = [], 0
    for n in lengths:
        out.append(at)
        at += n
    return out


@pytest.fixture
def greedy():
    """The head's drafts are its greedy picks (``nxt``), on the device: nothing to patch."""


def carry(tokens):
    return Carry(torch.zeros((len(tokens), 4)), list(tokens))


def test_one_absorb_step_then_chains_by_depth(greedy):
    head = FakeHead()
    a, b, c = St(10), St(20), St(30)
    out = head.draft([Ask(a, carry([5, 6]), 3, None, 11), Ask(b, carry([7]), 1, None, 21),
                      Ask(c, carry([9]), 0, None, 31)])
    assert out == [[7, 8, 9], [8], []]
    # one step absorbs every carry, then one row a stream still drafting a step
    assert head.calls[0] == [(10, [5, 6]), (20, [7]), (30, [9])]
    assert head.calls[1:] == [[(12, [7])], [(13, [8])]]
    assert (a.mtp_len, a.mtp_drafted) == (14, 2)       # 2 absorbed, 2 chained draft rows
    assert (b.mtp_len, b.mtp_drafted) == (21, 0)
    assert (c.mtp_len, c.mtp_drafted) == (31, 0)


def test_drafted_rows_roll_back_before_the_next_absorb(greedy):
    head = FakeHead()
    a = St(10)
    head.draft([Ask(a, carry([5]), 3, None, 11)])
    assert (a.mtp_len, a.mtp_drafted) == (13, 2)
    head.calls.clear()
    head.draft([Ask(a, carry([6, 7]), 1, None, 13)])
    assert head.calls == [[(11, [6, 7])]]              # rows 11, 12 (the old chain's) are written over
    assert (a.mtp_len, a.mtp_drafted) == (13, 0)


def test_a_carry_past_the_window_absorbs_in_pieces(greedy):
    head = FakeHead(rows_max=4)
    a, b = St(0), St(100)
    out = head.draft([Ask(a, carry([1, 2, 3, 4]), 1, None, 1), Ask(b, carry([8, 9]), 1, None, 101)])
    assert out == [[5], [10]]
    assert sum(len(t) for call in head.calls for _, t in call) == 6
    assert all(sum(len(t) for _, t in call) <= 4 for call in head.calls)
    assert (a.mtp_len, b.mtp_len) == (4, 102)


def test_policy_codes_under_parallel():
    zero = [0, 0, 0, 0]
    dflash = [16, 7, 300000, 0]
    # no draft model, no MTP head: serial
    assert multi_code([2, 3, 800000, 900000], False, dflash) == zero
    # the MTP head: MTP policies stay, auto and DFlash2 ones map to MTP
    assert multi_code([2, 3, 800000, 900000], False, dflash, mtp=True) == [2, 3, 800000, 900000]
    assert multi_code([1, 2, 0, 0], False, dflash, mtp=True) == [1, 2, 0, 0]
    assert multi_code([4, 2, 8, 30000], False, dflash, mtp=True) == MTP_AUTO
    assert multi_code([13, 3, 500000, 0], False, dflash, mtp=True) == [3, 3, 500000, 0]
    assert multi_code([16, 7, 300000, 0], False, dflash, mtp=True) == MTP_AUTO
    assert multi_code(zero, False, dflash, mtp=True) == zero
    # a DFlash2 draft model wins over the head
    assert multi_code([4, 2, 8, 30000], True, dflash, mtp=True) == dflash


def test_the_setting(monkeypatch):
    monkeypatch.delenv("TF_GLM_MULTI_MTP", raising=False)
    assert mtp_multi.enabled() is True
    assert mtp_multi.enabled("0") is False and mtp_multi.enabled(" 1 ") is True
    with pytest.raises(ValueError, match="TF_GLM_MULTI_MTP"):
        mtp_multi.enabled("yes")


def test_a_state_kept_short_of_the_prompts_end_reabsorbs_its_last_row_with_the_resuming_prompts_token(monkeypatch):
    """At a grid or shared-prefix point n short of the prompt's end, the chunk absorbed row n - 1 with the prompt's
    token n; the kept state has only the ids up to n, so it keeps row n - 1 pending, and a prompt resuming from it
    absorbs that row with its own token n."""
    import types

    from tensorfold.families.glm5_next.cuda import decode, mtp, multi

    class S:
        def __init__(self, mtp_len):
            self.mtp_len, self.mtp_drafted = mtp_len, 0

        def set_mtp_len(self, n):
            self.mtp_len = n

    monkeypatch.setattr(decode, "take_snapshot", lambda e, ids, pending, *, mtp, drafter, mtp_len:
                        types.SimpleNamespace(ids=list(ids), mtp_len=mtp_len, pending=None))
    monkeypatch.setattr(multi, "keep_rendered", lambda snap, prompt: None)
    absorbed = []
    monkeypatch.setattr(mtp, "mtp_forward", lambda w, st, b, tokens, hidden: absorbed.append((list(tokens), hidden)))
    d = object.__new__(multi.MultiDecoder)
    d.e = types.SimpleNamespace(use=lambda st: None, last_hidden=torch.full((1, 4), -1.0), mbuf=None)
    d.w, d.kept, d.next_kid, d.g = None, [], 0, types.SimpleNamespace(cache_entries=8)

    def lane(prompt, mtp_len):
        return multi.Lane(s=types.SimpleNamespace(prompt=prompt), sid=1, slot=0,
                          extent=types.SimpleNamespace(kept=[]), st=S(mtp_len), order=0, code=[], mtp=True)

    first = [10, 11, 12, 13, 14, 15, 16, 17]
    row = torch.full((1, 4), 3.0)
    d._keep(lane(first, 4), 4, last=row)                  # chunk 0 .. 4 absorbed rows 0 .. 3 (row 3 with token 14)
    snap = d.kept[-1]
    assert snap.ids == first[:4] and snap.mtp_len == 3 and torch.equal(snap.pending, row)
    other = first[:4] + [99, 98]
    resumed = lane(other, 0)
    d._mtp_resume(resumed, snap, 4, other)
    assert absorbed == [([99], snap.pending)] and resumed.st.mtp_len == 4
    d._keep(lane(first, 7), 8)                            # the prompt's end: row 7 waits for the first token
    assert d.kept[-1].mtp_len == 7 and torch.equal(d.kept[-1].pending, d.e.last_hidden)


def test_multi_mtp_off_starts_one_row_a_round_under_parallel():
    from tensorfold.families.glm5_next.cuda.engine import check_drafts

    check_drafts(False, False, False, parallel=4, layers=1, multi="0")     # the documented fallback starts
    check_drafts(True, False, False, parallel=4, layers=1, multi="1")      # the head drafts
    check_drafts(False, True, False, parallel=4, layers=1, multi="1")      # DFlash2 drafts
    check_drafts(False, False, True, parallel=1, layers=1)                 # --no-drafts: the serial reference
    with pytest.raises(ValueError, match="no MTP head"):                   # TF_GLM_MTP=0 with the head on: refused
        check_drafts(False, False, False, parallel=4, layers=1, multi="1")
    with pytest.raises(ValueError, match="no MTP head"):                   # one stream: MULTI_MTP is not its setting
        check_drafts(False, False, False, parallel=1, layers=1, multi="0")
