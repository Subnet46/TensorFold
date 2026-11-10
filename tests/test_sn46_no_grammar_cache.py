"""(patch sn46/no-grammar-cache) A request's grammar is its customer's content: the compiler keeps no compiled grammar
past the request, and a request still compiles, packs to another rank and is followed there. CPU only: xgrammar."""

from __future__ import annotations

import json

import pytest

xgr = pytest.importorskip("xgrammar")

from tensorfold.engine import grammar

V = 128
STOP = 0


def test_compiled_grammars_are_never_kept():
    vocab = [""] + [chr(t) for t in range(1, V)]
    g = grammar.Grammars(xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))
    schema = json.dumps({"type": "object", "properties": {"secret_key": {"enum": ["secret value"]}},
                         "required": ["secret_key"]})
    spec = grammar.Spec("json_schema", schema)
    for _ in range(2):                                    # refusal's check, then the request's own compile
        c = g.constraint(g.compile(spec), spec=spec)
        assert g.compiler.get_cache_size_bytes() == 0
    g.compile(grammar.Spec("regex", "[0-9]+"))
    g.compile(grammar.Spec("json"))
    assert g.compiler.get_cache_size_bytes() == 0
    other = g.follow(grammar.pack(c))                     # another rank compiles the packed grammar itself
    reply = '{"secret_key": "secret value"}'
    assert other is not None
    other.advance([ord(ch) for ch in reply] + [STOP])
    assert other.finished
    with pytest.raises(grammar.GrammarError):
        c.advance([ord(ch) for ch in '{"secret_key": "other"}'])
    assert g.compiler.get_cache_size_bytes() == 0
