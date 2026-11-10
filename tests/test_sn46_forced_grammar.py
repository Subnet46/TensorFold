"""(patch sn46/forced-grammar) A verifier walking a miner's output under the reply's grammar: the check before the walk
(``Constraint.first_reject``) finds the first output token the grammar rejects, ``rows.take_armed`` records it on the
spec, and the walk stops before it (``rows.walk_tokens``). CPU only: xgrammar, the GLM tokenizer, no weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

xgr = pytest.importorskip("xgrammar")

from tensorfold.engine import grammar
from tensorfold.sn46 import rows

V = 128
STOP = 0
THINK_END = 127


@pytest.fixture(scope="module")
def toy():
    vocab = [""] + [chr(t) for t in range(1, V)]
    return grammar.Grammars(xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _digits(g, **kwargs):
    return g.constraint(g.compile(grammar.Spec("regex", "[0-9]+")), **kwargs)


def test_first_reject_names_the_first_token_the_grammar_could_not_have_chosen(toy):
    c = _digits(toy)
    assert c.first_reject(_ids("123") + [STOP]) is None
    assert c.first_reject(_ids("12")) is None                       # cut short (a length finish): every token allowed
    assert c.first_reject(_ids("12x3")) == 2
    assert c.first_reject(_ids("x")) == 0
    assert c.first_reject([STOP]) == 0                               # the stop token before the grammar can end
    assert c.first_reject(_ids("1") + [STOP] + _ids("2")) == 2       # nothing follows the stop token
    assert c.first_reject(_ids("1") + [V + 5]) == 1                  # an id outside the vocabulary: rejected
    assert c.first_reject([-1]) == 0
    c.advance(_ids("12"))                                            # its own state is not the walk's
    assert c.first_reject(_ids("x")) == 0 and c.first_reject(_ids("3")) is None
    c.advance(_ids("3"))                                             # nor does the check move it
    assert not c.finished


def test_with_think_end_the_check_starts_after_it(toy):
    c = _digits(toy, think_end=THINK_END)
    assert c.first_reject(_ids("hmm x") + [THINK_END] + _ids("42") + [STOP]) is None
    assert c.first_reject(_ids("hmm") + [THINK_END] + _ids("4x")) == 5
    assert c.first_reject(_ids("no end, never checked")) is None
    assert c.first_reject(_ids("a") + [THINK_END, THINK_END]) == 2  # a second </think> is the grammar's to judge
    assert _digits(toy).first_reject([THINK_END]) == 0               # without thinking </think> is plain text


def _armed(spec, prompt, constraint):
    rows.ACTIVE = True
    try:
        rows.arm(spec)
        return rows.take_armed(prompt, True, constraint=constraint)
    finally:
        rows.disarm()
        rows.ACTIVE = False


def test_take_armed_with_a_constraint_records_where_a_forced_walk_must_stop(toy):
    prompt = [1, 2, 3]
    # a miner's measured request: it walks its own samples, chosen under the mask
    miner = _armed(rows.Spec(record=True, prompt=prompt), prompt, _digits(toy))
    assert miner is not None and miner.grammar_reject is None and rows.walk_tokens(miner, 40) == 40
    # a verifier's walk of an output the grammar takes, and of one it rejects at output index 2
    good = _armed(rows.Spec(record=True, targets=prompt[1:] + _ids("12") + [STOP], prompt=prompt), prompt,
                  _digits(toy))
    assert good.grammar_reject is None and rows.walk_tokens(good, 3) == 3
    bad = _armed(rows.Spec(record=True, targets=prompt[1:] + _ids("12x3"), prompt=prompt), prompt, _digits(toy))
    assert bad.grammar_reject == 2 and rows.walk_tokens(bad, 4) == 2   # rows 0 .. len(prompt) + 2 - 2: all allowed
    first = _armed(rows.Spec(record=True, targets=prompt[1:] + _ids("x1"), prompt=prompt), prompt, _digits(toy))
    assert first.grammar_reject == 0 and rows.walk_tokens(first, 2) == 1
    # no grammar: nothing is checked
    plain = _armed(rows.Spec(record=True, targets=prompt[1:] + _ids("x1"), prompt=prompt), prompt, None)
    assert plain.grammar_reject is None and rows.walk_tokens(plain, 2) == 2 and rows.walk_tokens(None, 7) == 7
    assert rows.Spec(record=True).header() == rows.Spec(record=True, grammar_reject=3).header()   # rank 0's only


def _masked_sample(window, row):
    """What the engine samples at a window row the grammar constrains: an id its mask allows (the lowest)."""
    import numpy as np

    words = window.bits[window.rows.index(row)].view(np.uint32)
    return next(t for t in range(V) if (int(words[t // 32]) >> (t % 32)) & 1)


def test_a_drafted_forced_walk_never_feeds_the_grammar_a_token_past_the_cutoff(toy):
    # the drafted decode loop (decode.py): prefill's first sample, then a window [pending] + drafts, every sample
    # through rows.forced, the accepted prefix followed by the grammar (``Constraint.advance``) before the clip
    prompt = [1, 2, 3]
    c = _digits(toy)
    rows.ACTIVE = True
    try:
        rows.arm(rows.Spec(record=True, targets=prompt[1:] + _ids("12x3"), claims=[-1] * (len(prompt) + 3),
                           prompt=prompt))
        spec = rows.take_armed(prompt, True, constraint=c)
    finally:
        rows.disarm()
        rows.ACTIVE = False
    assert spec.grammar_reject == 2 and rows.walk_tokens(spec, 4) == 2
    assert spec.targets == prompt[1:] + _ids("12") and len(spec.claims) == len(spec.targets)
    rows.begin(spec, prompt, 0)
    try:
        first = rows.forced([len(prompt)], [_masked_sample(c.window([0], [-1]), 0)])
        assert first == _ids("1")
        c.advance(first)
        window = c.window(_ids("1") + _ids("2"), [-1, 0])                  # the draft "2" matches the forced token
        assert window.tokens == _ids("12")
        engine = [_masked_sample(window, 0), _masked_sample(window, 1)]
        sampled = rows.forced([len(prompt) + 1, len(prompt) + 2], engine)
        assert sampled[0] == ord("2") and sampled[1] == engine[1] != ord("x")   # the bonus row keeps its sample
        c.advance(sampled)                                                  # no GrammarError
    finally:
        rows.end()
    # the first output token rejected: it stays forced, the walk runs it without the grammar
    rows.ACTIVE = True
    try:
        rows.arm(rows.Spec(record=True, targets=prompt[1:] + _ids("x1"), prompt=prompt))
        zero = rows.take_armed(prompt, True, constraint=_digits(toy))
    finally:
        rows.disarm()
        rows.ACTIVE = False
    assert zero.grammar_reject == 0 and zero.targets == prompt[1:] + _ids("x")


GLM = Path.home() / ".cache/sn46-testbed/model-glm-5.3-flash-uncensored"
GLM_EOS = (154820, 154827, 154829)                     # config.json's eos ids, <|observation|> 154829 among them
CALL = 'root ::= "<tool_call>" "get_weather" "</tool_call>"'


@pytest.fixture(scope="module")
def glm():
    if not (GLM / "tokenizer.json").is_file() or not (GLM / "config.json").is_file():
        pytest.skip(f"no GLM tokenizer at {GLM}")
    pytest.importorskip("transformers")
    from transformers import PreTrainedTokenizerFast

    assert json.loads((GLM / "config.json").read_text())["text_config"]["eos_token_id"] == list(GLM_EOS)
    grammars = grammar.Grammars.for_model(GLM, grammar.vocab_size(GLM), GLM_EOS)     # tokenizer.json only
    tok = PreTrainedTokenizerFast(tokenizer_file=str(GLM / "tokenizer.json"))
    return grammars, tok


def test_the_glm_tool_call_grammar_rejects_a_forced_output_at_its_first_wrong_token(glm):
    grammars, tok = glm
    think_end = tok.convert_tokens_to_ids("</think>")
    assert think_end == 154842

    def ids(text):
        return tok.encode(text, add_special_tokens=False)

    call = ids("<tool_call>get_weather</tool_call>")
    assert call[0] == 154843 and call[-1] == 154844                  # the markup tags are single tokens
    c = grammars.constraint(grammars.compile(grammar.Spec("grammar", CALL)))
    assert c.first_reject(call + [154829]) is None                   # the call, then <|observation|>
    assert c.first_reject(call) is None
    wrong = ids("<tool_call>get_time</tool_call>")
    same = next(i for i, (a, b) in enumerate(zip(wrong, call)) if a != b)
    assert c.first_reject(wrong) == same == 2                        # '<tool_call>' 'get' then '_time'
    assert c.first_reject(ids("Sure! <tool_call>get_weather</tool_call>")) == 0
    assert c.first_reject(call + [154829] + ids("more")) == len(call) + 1
    assert c.first_reject([154829]) == 0                             # a stop before the call
    # with thinking: the reasoning is free, the call after </think> is checked
    t = grammars.constraint(grammars.compile(grammar.Spec("grammar", CALL)), think_end=think_end)
    reasoning = ids("The user wants the weather.")
    assert t.first_reject(reasoning + [think_end] + call + [154829]) is None
    assert t.first_reject(reasoning + [think_end] + ids("\n") + call) == len(reasoning) + 1
    assert t.first_reject(reasoning + [think_end] + wrong) == len(reasoning) + 1 + 2
    # take_armed hands the walk's stop to the operator: rows 0 .. len(prompt) + reject - 2 are read
    prompt = ids("[gMASK]<sop><|user|>weather?<|assistant|><think>")
    spec = _armed(rows.Spec(record=True, targets=prompt[1:] + reasoning + [think_end] + wrong, prompt=prompt),
                  prompt, t)
    assert spec.grammar_reject == len(reasoning) + 3
    assert rows.walk_tokens(spec, len(reasoning) + 1 + len(wrong)) == len(reasoning) + 3
