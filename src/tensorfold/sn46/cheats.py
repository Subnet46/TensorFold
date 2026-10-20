"""Qualification cheats for Subnet 46's token check (``SN46_CHEAT`` at server start; never on a serving engine,
which refuses the variable). They run inside the CUDA graphs like the honest routing.

``experts4``: computes with the top 4 routed experts of every MoE layer; ``skipmoe``: drops the routed experts of
MoE layers 10, 20, 30 and 40.
"""
from __future__ import annotations

import os

CHEAT = os.environ.get("SN46_CHEAT", "")
if CHEAT not in ("", "experts4", "skipmoe"):
    raise ValueError(f"SN46_CHEAT: experts4 or skipmoe, not {CHEAT!r}")
SKIPPED = frozenset({10, 20, 30, 40})
TOP = 8


def route(b, layer, R: int) -> None:
    """After the router of a MoE layer."""
    if CHEAT == "experts4":
        b.wts[:R, 4:TOP].zero_()
    elif CHEAT == "skipmoe" and layer.index in SKIPPED:
        b.wts[:R, :TOP].zero_()
