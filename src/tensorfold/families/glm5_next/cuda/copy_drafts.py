"""Prompt-lookup ("copy") drafts: when the context's last tokens occurred before, propose what followed them.

A reply that quotes or edits its prompt (or repeats itself) is cheap to draft: its last ``match`` tokens (the
pending token included) are searched in the prompt and the reply so far, and the tokens after the latest earlier
occurrence become the drafts. They only propose; verification keeps the serial sample, so replies stay exact.

Both ranks hold the same prompt and sample the same tokens, so each computes the same proposals with no exchange.
Pure numpy (no torch), so it is testable anywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

MATCH = 8        # the context's last this many tokens must have occurred before (TF_GLM_COPY_MATCH)
MOST = 5         # drafts a round: 5 keeps the verify window (6 rows) within the captured graphs (TF_GLM_COPY_MAX)


@dataclass(frozen=True)
class CopySettings:
    match: int = MATCH
    most: int = MOST

    @classmethod
    def from_env(cls, max_drafts: int, env=None) -> CopySettings | None:
        """TF_GLM_COPY_DRAFTS=1 turns copy drafts on (off by default); TF_GLM_COPY_MATCH and TF_GLM_COPY_MAX tune."""

        env = os.environ if env is None else env
        on = env.get("TF_GLM_COPY_DRAFTS", "0").strip()
        if on in ("", "0"):
            return None
        if on != "1":
            raise ValueError(f"TF_GLM_COPY_DRAFTS: 0 or 1, not {on!r}")

        def number(name: str, default: int, low: int, high: int) -> int:
            text = env.get(name, "").strip()
            if text == "":
                return default
            if not text.isdecimal() or not low <= int(text) <= high:
                raise ValueError(f"{name}: {low} to {high}, not {text!r}")
            return int(text)

        return cls(number("TF_GLM_COPY_MATCH", MATCH, 2, 64),
                   number("TF_GLM_COPY_MAX", min(MOST, max_drafts), 1, max_drafts))

    def code(self) -> list[int]:
        """The settings as ints both ranks compare at startup (zeros: off)."""

        return [self.match, self.most]


class CopyDrafts:
    """One sequence's context (prompt, then committed reply tokens and the pending one) and its copy proposals."""

    def __init__(self, context: Sequence[int], match: int = MATCH, most: int = MOST) -> None:
        if match < 1 or most < 1:
            raise ValueError("match and most must be positive")
        self.match, self.most = int(match), int(most)
        n = len(context)
        self.buf = np.empty((max(1024, 2 * n),), dtype=np.int32)
        self.buf[:n] = np.asarray(context, dtype=np.int32) if n else 0
        self.length = n

    def __len__(self) -> int:
        return self.length

    def tokens(self) -> list[int]:
        return self.buf[:self.length].tolist()

    def extend(self, tokens: Sequence[int]) -> None:
        """Committed tokens (the last one is the next round's pending token) join the context."""

        k = len(tokens)
        if not k:
            return
        if self.length + k > self.buf.shape[0]:
            grown = np.empty((2 * (self.length + k),), dtype=np.int32)
            grown[:self.length] = self.buf[:self.length]
            self.buf = grown
        self.buf[self.length:self.length + k] = np.asarray(tokens, dtype=np.int32)
        self.length += k

    def starts(self) -> np.ndarray:
        """Ascending starts of the earlier occurrences of the context's last ``match`` tokens (not the suffix)."""

        L, n = self.length, self.match
        if L <= n:
            return np.empty((0,), dtype=np.int64)
        ctx = self.buf
        q = ctx[L - n:L]
        # starts 0 .. L - n - 1 (each leaves a token after its match): by the last token, then the others
        hits = np.flatnonzero(ctx[n - 1:L - 1] == q[n - 1])
        for k in range(n - 1):
            if not hits.size:
                break
            hits = hits[ctx[hits + k] == q[k]]
        return hits

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(most, room)`` drafts: what followed the latest earlier occurrence that has that many tokens
        after it, else the most after any occurrence (the earliest); [] when the suffix never occurred before."""

        k = self.most if room is None else min(self.most, int(room))
        if k < 1:
            return []
        hits = self.starts()
        if not hits.size:
            return []
        L, n = self.length, self.match
        full = hits[hits <= L - n - k]
        s = int(full[-1]) if full.size else int(hits[0])
        return self.buf[s + n:min(s + n + k, L)].tolist()
