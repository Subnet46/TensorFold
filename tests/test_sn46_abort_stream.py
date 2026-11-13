"""TENSORFOLD_SN46 under --parallel: the operator's abort of a job ends that job's stream in the engine (patch
sn46/abort-stream). A miner aborts a request whose client left; before this the engine served the stream on until it
next wrote tokens to the closed connection (a waiting one only after its admission and its whole prefill), and kept
counting it as running, so the miner's readiness saw more streams than it generates and restarted the backend."""

import importlib
import threading
import time

import pytest

pytest.importorskip("torch")
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)


class Decoder:
    """Lanes enough to hold streams; ``finish`` takes a stream's lane away, ``admit`` gives it one."""

    def __init__(self):
        self.lanes, self.requeue, self.broken, self.finished = {}, [], None, []

    def live(self):
        return len(self.lanes)

    def fits(self, stream):
        return True

    def admit(self, stream):
        self.lanes[id(stream)] = type("Lane", (), {"s": stream})()

    def finish(self, done):
        for s in done:
            self.lanes.pop(id(s), None)
            self.finished.append(s)


def _scheduler(enabled):
    multi = importlib.import_module("tensorfold.families.glm5_next.cuda.multi")
    prio = importlib.import_module("tensorfold.cuda.priority")
    sched = multi.GlmScheduler(Decoder(), max_streams=4, settings=prio.Settings(enabled=enabled), start=False)
    return multi, sched


def _submit(sched, job, results):
    def run():
        try:
            results[job] = sched.submit([1, 2, 3], 4, None, False, lambda new: False, sn46_job=job)
        except Exception as exc:              # noqa: BLE001
            results[job] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    end = time.monotonic() + 5
    while not any(getattr(s, "sn46_job", None) == job for s in sched._streams()):
        assert time.monotonic() < end, "the request never queued"
        time.sleep(0.002)
    return t


@pytest.mark.parametrize("enabled", [True, False])
def test_an_aborted_jobs_waiting_stream_leaves_without_a_prefill(allocations, enabled):
    from tensorfold.server.cancellation import RequestCancelled

    _, sched = _scheduler(enabled)
    results = {}
    a, b = _submit(sched, "job-a", results), _submit(sched, "job-b", results)
    assert sched.running_streams() == 2
    sched.cancel_job("job-a")
    assert sched.running_streams() == 1          # no longer counted, before the scheduler even looks
    sched._drop_aborted()
    a.join(5)
    assert isinstance(results["job-a"], RequestCancelled)
    assert [s.sn46_job for s in sched._streams()] == ["job-b"]
    assert b.is_alive()                          # the other request still waits for its lane


def test_an_aborted_jobs_live_stream_stops_at_its_next_tokens(allocations):
    multi, sched = _scheduler(True)
    results = {}
    _submit(sched, "job-c", results)
    item = sched.waiting.best()
    sched.waiting.remove(item)
    stream = item[0]
    sched.decoder.admit(stream)
    assert sched.running_streams() == 1
    assert stream.emit([7]) is False             # the client reads on
    sched.cancel_job("job-c")
    assert sched.running_streams() == 0
    assert stream.emit([8]) is True              # the decoder ends it this round


def test_a_replayed_stream_stays_its_jobs(allocations):
    multi, sched = _scheduler(True)
    results = {}
    _submit(sched, "job-d", results)
    stream, _ = sched.waiting.best()
    again = stream.continued()
    multi._carry_job(stream, again)
    sched.cancel_job("job-d")
    assert again.sn46_job == "job-d" and sched._aborted(again)


def test_a_stream_given_back_when_its_job_is_aborted_is_still_aborted(allocations):
    """At the abort the stream sat in no container (the decoder's give-back list): the job is remembered."""
    multi, sched = _scheduler(True)
    results = {}
    _submit(sched, "job-e", results)
    item = sched.waiting.best()
    sched.waiting.remove(item)                   # admitted, then given back: in ``decoder.requeue`` only
    stream = item[0]
    sched.cancel_job("job-e")
    assert not stream.abort_flag[0]
    again = stream.continued()
    multi._carry_job(stream, again)
    assert sched._aborted(again) and stream.abort_flag[0]
    later = _submit(sched, "job-e", results)     # the same job given again after the abort: a new request
    fresh = [st for st in sched._streams() if st.sn46_job == "job-e"]
    assert len(fresh) == 1 and not sched._aborted(fresh[0])
    assert sched._aborted(again)                 # the old replay stays aborted
    assert later.is_alive()


@pytest.mark.parametrize("enabled", [True, False])
def test_an_aborted_job_is_never_admitted(allocations, enabled):
    from tensorfold.server.cancellation import RequestCancelled

    _, sched = _scheduler(enabled)
    results = {}
    a = _submit(sched, "job-f", results)
    _submit(sched, "job-g", results)
    sched.cancel_job("job-f")                    # after the iteration's drop, before its admission
    sched._admit()
    a.join(5)
    assert isinstance(results["job-f"], RequestCancelled)
    assert [lane.s.sn46_job for lane in sched.decoder.lanes.values()] == ["job-g"]


def test_an_aborted_jobs_filling_or_decoding_stream_is_finished_between_rounds(allocations):
    from tensorfold.server.cancellation import RequestCancelled

    _, sched = _scheduler(True)
    results = {}
    a = _submit(sched, "job-h", results)
    _submit(sched, "job-i", results)
    sched._admit()                               # both live (one may still be filling its prompt)
    sched.cancel_job("job-h")
    sched._drop_aborted()
    a.join(5)
    assert isinstance(results["job-h"], RequestCancelled)
    assert [s.sn46_job for s in sched.decoder.finished] == ["job-h"]
    assert sched.running_streams() == 1


def test_a_request_seen_twice_across_a_give_back_counts_once(allocations):
    multi, sched = _scheduler(True)
    results = {}
    _submit(sched, "job-j", results)
    item = sched.waiting.best()
    stream = item[0]
    sched.decoder.admit(stream)                  # in a lane and, mid-transition, still in the queue
    again = stream.continued()
    multi._carry_job(stream, again)
    sched.waiting.put((again, item[1]))          # and its replay queued too
    assert sched.running_streams() == 1


def test_a_failed_cleanup_still_ends_the_aborted_request(allocations):
    from tensorfold.server.cancellation import RequestCancelled

    _, sched = _scheduler(True)
    results = {}
    a = _submit(sched, "job-k", results)
    sched._admit()

    def broken(done):
        raise RuntimeError("cleanup failed")

    sched.decoder.finish = broken
    sched.cancel_job("job-k")
    with pytest.raises(RuntimeError):
        sched._drop_aborted()
    a.join(5)
    assert isinstance(results["job-k"], RequestCancelled)
