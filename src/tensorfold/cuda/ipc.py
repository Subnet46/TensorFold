"""One-shot all-gather over CUDA IPC for the GPUs of one host (PCIe or NVLink peer to peer).

A decode round's all-gathers are small (a verify window's partials: rows x 4,096 fp32, the samplers' candidates, the
round messages) and NCCL spends most of each in fixed latency. Here each rank writes its shard straight into every
peer's slot over peer-to-peer stores, then a flag with a system-scope release store; each rank polls its own flags
with acquire semantics and copies the peers' shards out of its slots in rank order (ipc.cu). Flags are plain stores
and loads, so GPUs without peer-to-peer atomics qualify. A device epoch picks one of two slots a launch, so CUDA
graphs replay the exchange; a wait past TF_GLM_IPC_WAIT_S poisons the runtime and the host raises (as the RoCE one-shot
does, roce.py).

``IpcComm`` sends all-gathers of up to TF_GLM_IPC_MAX_KB a rank this way and the rest (the prefill exchanges) through
NCCL, decided by the byte count alone (the same on every rank), with NCCL's output layout and the same bytes: an
all-gather moves bytes and adds nothing, so every reply keeps its bits.

Credits: the protocol is Mia's one-shot RoCE all-gather (MiaAI-Lab, roce.cu / roce.py, itself b12x's "RoCEnante"),
without its host proxy; the CUDA IPC buffers follow b12x's PCIe one-shot collectives (local-inference-lab/b12x,
b12x/comm/pcie, Apache License 2.0). No b12x source is copied.

Settings (every rank must see the same values; the setup compares them):
  TF_GLM_COMM=ipc         turns it on (GlmEngine); without peer-to-peer access between the ranks' GPUs (each rank must
                          see the others' devices, e.g. CUDA_VISIBLE_DEVICES with its own GPU first) it says so and
                          every gather stays on NCCL
  TF_GLM_IPC_MAX_KB       largest gather a rank sent this way (default 512 = 32 rows of 4,096 fp32); each GPU exports
                          2 x (ranks - 1) slots of this size (BAR1 space when peers map it)
  TF_GLM_IPC_WAIT_S       seconds a wait for a peer may take before it poisons the runtime (default 300, 1 to 3600).
                          Generous on purpose: in a multi-stream round the other ranks wait inside the round
                          message's gather while rank 0 alone admits a request (an image encode, a grammar compile),
                          and a dead rank is still reported instead of hanging as NCCL would
  TF_GLM_IPC_BLOCK_KB     shard KiB a block (the grid: one block for each, at most 128; default 8)
  TF_GLM_IPC_THREADS      threads a block (default 512)
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

ABI = 1                                  # ipc.h's IPC_ABI
SLOT_ALIGN = 256
MAX_KB = 512
WAIT_S = 300.0
BLOCK_KB = 8
THREADS = 512


class IpcUnavailable(RuntimeError):
    """CUDA IPC cannot run between these ranks (said on every rank alike): the caller keeps NCCL."""


def _setting(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = lo - 1
    if not lo <= value <= hi:
        raise ValueError(f"{name}: {lo:g} to {hi:g}, not {raw!r}")
    return value


def settings() -> dict:
    """The TF_GLM_IPC_* values (bytes, ns and threads), checked."""

    threads = int(_setting("TF_GLM_IPC_THREADS", THREADS, 32, 1024))
    if threads % 32:
        raise ValueError(f"TF_GLM_IPC_THREADS: a multiple of 32, not {threads}")
    return {"max_bytes": int(_setting("TF_GLM_IPC_MAX_KB", MAX_KB, 1, 4096) * 1024),
            "timeout_ns": int(_setting("TF_GLM_IPC_WAIT_S", WAIT_S, 1, 3600) * 1e9),
            "block_bytes": int(_setting("TF_GLM_IPC_BLOCK_KB", BLOCK_KB, 1, 4096) * 1024),
            "threads": threads}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name=f"tensorfold_ipc_v{ABI}", sources=[str(here / "ipc.cpp"), str(here / "ipc.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def plan(nbytes: int, block_bytes: int, max_blocks: int = 128, pack: int = 16) -> tuple[int, int]:
    """(blocks, packs a block) of a gather of ``nbytes`` a rank: ipc.cpp's ``plan`` (the CPU tests check the two
    agree), from the byte count alone so every rank's launch matches."""

    packs = -(-nbytes // pack)
    grid = max(1, min(-(-nbytes // block_bytes), max_blocks, packs))
    chunk = -(-packs // grid)
    return -(-packs // chunk), chunk


class IpcGather:
    """One rank's side of the exchange: its exported region, the peers' mapped regions and the launches."""

    def __init__(self, comm, rank: int, world: int, max_bytes: int | None = None, timeout_ns: int | None = None,
                 block_bytes: int | None = None, threads: int | None = None) -> None:
        """``comm``: a communicator with ``all_gather`` (NCCL) for the setup. Raises ``IpcUnavailable`` on every
        rank when any rank cannot take part (then nothing is left mapped)."""

        given = settings()
        self.rank, self.world = rank, world
        self.max_bytes = int(given["max_bytes"] if max_bytes is None else max_bytes)
        self.timeout_ns = int(given["timeout_ns"] if timeout_ns is None else timeout_ns)
        self.block_bytes = int(given["block_bytes"] if block_bytes is None else block_bytes)
        self.threads = int(given["threads"] if threads is None else threads)
        self.slot_bytes = -(-self.max_bytes // SLOT_ALIGN) * SLOT_ALIGN
        self.g = None
        self.err = None
        why, handle_len = "", 64
        handle = None
        try:
            if not 2 <= world <= 8:
                raise ValueError(f"2 to 8 ranks, not {world}")
            ext = _ext()                 # its first build (tens of seconds) here, before the verdict all-gather below
            layout = list(ext.layout())
            if layout[0] != ABI:
                raise RuntimeError(f"extension ABI {layout[0]}, expected {ABI}")
            handle_len = int(layout[7])
            self.g = ext.Gather(world, rank, self.slot_bytes, self.timeout_ns, self.block_bytes, self.threads)
            handle = self.g.handle()
        except Exception as exc:         # noqa: BLE001  (any rank's failure: every rank keeps NCCL)
            why = f"{type(exc).__name__}: {exc}"
        words = -(-handle_len // 4)
        raw = bytearray(words * 4)
        if handle is not None:
            raw[:handle_len] = bytes(handle.numpy().tobytes())
        geometry = [ABI, self.slot_bytes, self.block_bytes, self.threads, handle_len]
        head = [1 if why else 0, *geometry]
        mine = torch.tensor(head + list(torch.frombuffer(raw, dtype=torch.int32).tolist()), dtype=torch.int32)
        got = torch.empty((world * mine.numel(),), dtype=torch.int32, device="cuda")
        comm.all_gather(mine.cuda(), got)
        rows = got.view(world, -1).cpu()
        failed = [r for r in range(world) if int(rows[r, 0])]
        if failed or len({tuple(r[1:len(head)].tolist()) for r in rows}) != 1:
            self.close()
            reason = (f"rank {', '.join(map(str, failed))} could not set up ({why or 'see that rank'})" if failed
                      else "the ranks' TF_GLM_IPC_* settings differ")
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: {reason}")
        handles = [torch.frombuffer(bytearray(rows[r, len(head):].numpy().tobytes()[:handle_len]), dtype=torch.uint8)
                   for r in range(world)]
        try:
            self.g.connect(handles)
        except Exception as exc:         # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"
        verdict = torch.tensor([1 if why else 0], dtype=torch.int32, device="cuda")
        all_verdicts = torch.empty((world,), dtype=torch.int32, device="cuda")
        comm.all_gather(verdict, all_verdicts)
        failed = [r for r, v in enumerate(all_verdicts.tolist()) if v]
        if failed:
            self.close()
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: rank {', '.join(map(str, failed))} cannot map "
                                 f"its peers' memory ({why or 'see that rank'}); peer-to-peer access needs every "
                                 "GPU visible to every rank")
        self.err = (ctypes.c_int64 * 4).from_address(self.g.error_address())

    def fits_bytes(self, n: int) -> bool:
        """Whether a gather of ``n`` bytes a rank goes over IPC (the same answer on every rank)."""

        return 0 < n <= self.max_bytes

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n] in rank order: contiguous CUDA tensors, n bytes fitting."""

        self.g.run(send, recv)
        if self.err[0] and not torch.cuda.is_current_stream_capturing():
            self.check()

    def check(self) -> None:
        """Raise if a wait timed out on the GPU (it poisoned the runtime: later gathers move nothing)."""

        if self.err is not None and self.err[0]:
            seq, peer, block, waited = (int(v) for v in self.err)
            raise RuntimeError(f"CUDA IPC all-gather failed: sequence {seq} waited {waited / 1e9:.1f} s for rank "
                               f"{peer} (block {block}) and gave up (TF_GLM_IPC_WAIT_S {self.timeout_ns / 1e9:g} s); "
                               "the runtime is poisoned, restart every rank")

    def close(self) -> None:
        if self.g is not None:
            self.err = None
            self.g.close()
            self.g = None


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


class IpcComm:
    """NCCL with small all-gathers over CUDA IPC: the same call, output layout and bits."""

    SELF_TEST_WAIT_NS = 30_000_000_000           # the self-test's wait limit: a path that drops writes fails soon

    def __init__(self, nccl, rank: int, world: int, max_bytes: int | None = None) -> None:
        self.nccl, self.rank, self.world = nccl, rank, world
        self.small = self.large = 0
        self.settled = False
        nccl.barrier()
        self.ipc = IpcGather(nccl, rank, world, max_bytes)
        try:
            self._self_test()
        except Exception:
            self.ipc.close()
            raise

    def _self_test(self) -> None:
        """Gathers of a few sizes (a tail, a decode row, the largest) over IPC and NCCL, under a short wait limit:
        the same bits on every rank, or ``IpcUnavailable`` on every rank (peer writes that never arrive, or arrive
        out of the order the protocol needs, would show here)."""

        ipc = self.ipc
        ipc.g.set_timeout(min(ipc.timeout_ns, self.SELF_TEST_WAIT_NS))
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234 + self.rank)
        bad, why = 0, ""
        try:
            for n in sorted({4, 168, 16 << 10, ipc.max_bytes - ipc.max_bytes % 4 or 4}):
                send = torch.randint(-2 ** 31, 2 ** 31 - 1, (n // 4,), dtype=torch.int32, device="cuda",
                                     generator=gen)
                want = torch.empty((self.world * send.numel(),), dtype=torch.int32, device="cuda")
                got = torch.full_like(want, -1)
                self.nccl.all_gather(send, want)
                for _ in range(3):
                    ipc.all_gather(send, got)
                    torch.cuda.synchronize()
                    ipc.check()
                    bad += int(not torch.equal(want, got))
        except RuntimeError as exc:                   # a wait timed out: this rank is poisoned, so stop here
            bad, why = bad + 1, str(exc)
        verdict = torch.tensor([bad], dtype=torch.int32, device="cuda")
        both = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.nccl.all_gather(verdict, both)
        if int(both.sum()):
            ranks = ", ".join(str(r) for r, v in enumerate(both.tolist()) if v)
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: its self-test against NCCL failed on rank {ranks}"
                                 + (f" ({why})" if why else " (other bytes than NCCL's)"))
        ipc.g.set_timeout(ipc.timeout_ns)

    def settle(self) -> None:
        """Startup is over on this rank (its engine is built). Until then each eager IPC gather first meets the other
        ranks in an NCCL barrier: ranks build their CUDA extensions on their own and can drift apart by tens of
        seconds on a first start, past what an IPC wait allows; NCCL waits without that limit. Captured gathers
        never do."""

        self.settled = True

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """IPC for a gather whose byte count fits, NCCL otherwise: decided by the byte count alone, which every rank's
        matching call shares. A buffer that is not contiguous goes through a contiguous copy, and a send that
        overlaps recv anywhere but at this rank's place in it through a copy of its own (local choices that never
        change the transport)."""

        n = send.numel() * send.element_size()
        if not self.ipc.fits_bytes(n):
            self.large += 1
            self.nccl.all_gather(send, recv)
            return
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        self.small += 1
        s = send if send.is_contiguous() else send.contiguous()
        r = recv if recv.is_contiguous() else torch.empty(recv.shape, dtype=recv.dtype, device=recv.device)
        if _overlap(s, r) and s.data_ptr() != r.data_ptr() + self.rank * n:
            s = s.clone()
        if not self.settled and not torch.cuda.is_current_stream_capturing():
            self.nccl.barrier()
        self.ipc.all_gather(s, r)
        if r is not recv:
            recv.copy_(r)

    def barrier(self) -> None:
        self.nccl.barrier()

    def check(self) -> None:
        self.ipc.check()

    def __getattr__(self, name):
        if name in ("nccl", "ipc"):                  # not set yet (a failed construction): no recursion
            raise AttributeError(name)
        return getattr(self.nccl, name)
