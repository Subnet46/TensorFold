"""Subnet 46 support for the GLM-5.3-Flash CUDA engine.

``rows``: per request, every row's next-token statistics (top-1, log-sum-exp, the target's and a claimed id's logits),
which Subnet 46's token check compares between a miner and a verifier.
``cheats``: qualification cheats the check must catch (``SN46_CHEAT``).
``extension``: the server hook that loads an operator module (``TENSORFOLD_EXTENSION``) to serve extra routes and
run per-rank calls.
"""
