"""TF_GLM_IPC_PROTOCOLS=auto (sn46): the routes chosen from the timed table, the lookup by size, and IpcComm sending
each gather down its route (CPU; no GPU, the gathers are fakes)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import ipc  # noqa: E402

LL, FLAG, CE, NCCL = ipc.PROTO["ll"], ipc.PROTO["flag"], ipc.PROTO["ce"], ipc.NCCL_ROUTE
KB = 1024


def test_auto_is_read_from_the_setting(monkeypatch):
    monkeypatch.setenv("TF_GLM_IPC_PROTOCOLS", "Auto")
    assert ipc.auto()
    assert ipc.settings()["bands"] == ipc.parse_protocols(ipc.PROTOCOLS, 512 * KB)
    monkeypatch.setenv("TF_GLM_IPC_PROTOCOLS", "ll:16,flag")
    assert not ipc.auto()
    monkeypatch.delenv("TF_GLM_IPC_PROTOCOLS")
    assert not ipc.auto()


def test_auto_sizes_cover_the_decode_rows_and_end_at_the_max():
    sizes = ipc.auto_sizes(512 * KB)
    assert sizes == sorted(set(sizes)) and sizes[-1] == 512 * KB
    assert all(16 * KB * r in sizes for r in range(1, 9))
    assert all(n % 4 == 0 for n in sizes)
    assert ipc.auto_sizes(20 * KB) == [64, KB, 4 * KB, 8 * KB, 16 * KB, 20 * KB]


def test_each_size_takes_its_fastest_route_and_neighbours_merge():
    sizes = [16 * KB, 32 * KB, 48 * KB, 64 * KB]
    routes = [FLAG, LL, CE, NCCL]
    times = [[30.0, 90.0, 60.0, 70.0],      # flag
             [20.0, 95.0, 99.0, 99.0],      # ll
             [40.0, 25.0, 26.0, 50.0],      # ce
             [50.0, 60.0, 70.0, 45.0]]      # nccl
    bands = ipc.choose_routes(sizes, routes, times, 512 * KB)
    assert bands == [(LL, 16 * KB), (CE, 48 * KB), (NCCL, 512 * KB)]
    assert ipc.describe_routes(bands) == "ll:16,ce:48,nccl:512"
    assert [ipc.route_of(bands, n) for n in (4, 16 * KB, 16 * KB + 4, 48 * KB, 49 * KB, 512 * KB)] == \
        [LL, LL, CE, CE, NCCL, NCCL]


def test_a_close_call_goes_to_the_route_listed_first():
    sizes = [16 * KB]
    bands = ipc.choose_routes(sizes, [FLAG, LL, CE, NCCL], [[20.3], [20.0], [30.0], [30.0]], 512 * KB)
    assert bands == [(FLAG, 512 * KB)]        # flag is within 2 % of ll and listed first
    bands = ipc.choose_routes(sizes, [FLAG, LL, CE, NCCL], [[21.0], [20.0], [30.0], [30.0]], 512 * KB)
    assert bands == [(LL, 512 * KB)]


def test_the_same_table_gives_the_same_routes():
    sizes = ipc.auto_sizes(512 * KB)
    routes = [FLAG, LL, CE, NCCL]
    times = [[float((i * 7 + j * 13) % 11 + 1) for j in range(len(sizes))] for i in range(len(routes))]
    assert ipc.choose_routes(sizes, routes, times, 512 * KB) == ipc.choose_routes(sizes, routes, times, 512 * KB)


def test_a_malformed_table_is_refused():
    with pytest.raises(ValueError):
        ipc.choose_routes([16 * KB, 32 * KB], [FLAG, NCCL], [[1.0, 2.0]], 512 * KB)
    with pytest.raises(ValueError):
        ipc.choose_routes([], [FLAG], [[]], 512 * KB)


class _Nccl:
    def __init__(self) -> None:
        self.calls = 0

    def all_gather(self, send, recv) -> None:
        self.calls += 1
        recv.copy_(send.repeat(2))

    def barrier(self) -> None:
        pass


class _Ipc:
    max_bytes = 512 * KB

    def __init__(self) -> None:
        self.protocols_used: list[int] = []

    def fits_bytes(self, n: int) -> bool:
        return 0 < n <= self.max_bytes

    def all_gather(self, send, recv, protocol: int = -1) -> None:
        self.protocols_used.append(protocol)
        recv.copy_(send.repeat(2))


def _comm(routes):
    comm = ipc.IpcComm.__new__(ipc.IpcComm)
    comm.nccl, comm.ipc, comm.rank, comm.world = _Nccl(), _Ipc(), 0, 2
    comm.small = comm.large = 0
    comm.settled = True
    comm.routes, comm.timed = routes, None
    return comm


def test_ipc_comm_sends_each_gather_down_its_route():
    comm = _comm([(LL, 16 * KB), (CE, 48 * KB), (NCCL, 512 * KB)])
    for n in (4 * KB, 32 * KB, 64 * KB):
        send = torch.arange(n // 4, dtype=torch.int32)
        recv = torch.empty(2 * send.numel(), dtype=torch.int32)
        comm.all_gather(send, recv)
        assert torch.equal(recv, send.repeat(2))
    assert comm.ipc.protocols_used == [LL, CE]
    assert comm.nccl.calls == 1 and (comm.small, comm.large) == (2, 1)
    assert comm.routes_text() == "ll:16,ce:48,nccl:512"


def test_without_auto_the_bands_decide():
    comm = _comm(None)
    send = torch.arange(8, dtype=torch.int32)
    comm.all_gather(send, torch.empty(16, dtype=torch.int32))
    assert comm.ipc.protocols_used == [-1] and comm.nccl.calls == 0
    big = torch.zeros(ipc.MAX_KB * KB // 4 + 1, dtype=torch.int32)
    comm.all_gather(big, torch.empty(2 * big.numel(), dtype=torch.int32))
    assert comm.nccl.calls == 1


def test_point_to_point_exchanges_still_reach_nccl():
    """IpcComm forwards what it does not implement to NCCL: the hc split's p2p exchange among them."""
    comm = _comm(None)
    calls = []
    comm.nccl.exchange = lambda sends, recvs, peer: calls.append(peer)
    comm.exchange([], [], 1)
    assert calls == [1]


def test_a_strided_gather_on_the_nccl_route_goes_through_contiguous_buffers():
    comm = _comm([(NCCL, 512 * KB)])
    seen = []
    comm.nccl.all_gather = lambda s, r: (seen.append((s.is_contiguous(), r.is_contiguous())), r.copy_(s.repeat(2)))
    base = torch.arange(16, dtype=torch.int32)
    send = base[::2]
    recv_base = torch.zeros(32, dtype=torch.int32)
    recv = recv_base[::2]
    comm.all_gather(send, recv)
    assert seen == [(True, True)]
    assert torch.equal(recv, send.repeat(2))
