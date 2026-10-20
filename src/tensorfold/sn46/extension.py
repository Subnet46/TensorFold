"""An operator module loaded into every rank of ``tensorfold serve`` (``TENSORFOLD_EXTENSION=module``).

The module may define:

- ``attach(engine, rank, app)``: once, after the engine loaded (``app`` is None on ranks past 0);
- ``dispatch(handler, method) -> bool``: rank 0's HTTP handler calls it first for every request (``method`` is GET,
  POST or DELETE); True means the module answered it (and the server's own routes do not run);
- ``local(name, args) -> JSON value``: one per-rank call, run on every rank by ``everywhere``.

``everywhere(engine, name, args)`` (rank 0, holding the engine: no request runs) runs ``local(name, args)`` on each
rank and returns the replies in rank order. The followers take the call through the request channel they already
wait on: a header of ``[-1, length]``, then the call's JSON as ints.
"""
from __future__ import annotations

import importlib
import json
import os

NAME = os.environ.get("TENSORFOLD_EXTENSION", "")
OP = -1                     # follow_one's header tag for a per-rank call (0 is a decision, a request's is >= 1)
_module = None


def load(engine, rank: int, app=None):
    """Import and attach the module (no-op without TENSORFOLD_EXTENSION)."""
    global _module
    if not NAME:
        return None
    _module = importlib.import_module(NAME)
    attach = getattr(_module, "attach", None)
    if attach is not None:
        attach(engine, rank, app)
    return _module


def dispatch(handler, method: str) -> bool:
    fn = getattr(_module, "dispatch", None) if _module is not None else None
    return bool(fn(handler, method)) if fn is not None else False


def _encode(value) -> list[int]:
    return list(json.dumps(value, separators=(",", ":")).encode())


def _decode(ints) -> object:
    return json.loads(bytes(int(v) for v in ints).decode())


def _local(name: str, args) -> object:
    fn = getattr(_module, "local", None) if _module is not None else None
    if fn is None:
        raise RuntimeError("the extension has no per-rank calls")
    try:
        return {"ok": fn(name, args)}
    except Exception as exc:                 # a rank's failure is its reply, so every rank finishes the gather
        return {"error": f"{type(exc).__name__}: {exc}"}


def everywhere(engine, name: str, args) -> list[object]:
    """Rank 0: ``local(name, args)`` on every rank, replies in rank order ({"ok": value} or {"error": text})."""
    world = int(getattr(engine, "world", 1))
    if world == 1 or getattr(engine, "comm", None) is None:
        return [_local(name, args)]
    if getattr(engine, "multi", None) is not None:
        # with --parallel above 1 the followers run the multi-stream loop, which takes no extension calls
        raise ValueError("extension calls run on the one-stream engine (--parallel 1)")
    call = _encode([name, args])
    engine._ring()
    engine._share([OP, len(call)])
    engine._share(call)
    return [_decode(r) for r in _gather(engine, _encode(_local(name, args)))]


def follow(engine, header: list[int]) -> None:
    """A follower's side of ``everywhere``: the header was ``[OP, length]``."""
    name, args = _decode(engine._share(None))
    _gather(engine, _encode(_local(name, args)))


def _gather(engine, mine: list[int]) -> list[list[int]]:
    """Every rank's int list (lengths may differ), in rank order."""
    sizes = [row[0] for row in engine._gather_ints([len(mine)])]
    width = max(sizes)
    rows = engine._gather_ints(mine + [0] * (width - len(mine)))
    return [row[:n] for row, n in zip(rows, sizes)]
