"""The arbiter's decode time share (TENSORFOLD_PRIORITY_DECODE_SHARE), on a simulated clock: under prompt overload,
with every waiting prompt starving, decode rounds still get their share of the time, measured in time and not in
iterations (sn46 lanes-diag 2026-11-16: 8 streams fell from 135 to 4 tok/s when long prompts kept arriving)."""

from tensorfold.cuda import priority as prio

CHUNK_S, ROUND_S = 0.65, 0.013            # a 4,096-row prompt chunk and one decode round, as measured on the diag host


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def overload(settings: prio.Settings, seconds: float = 60.0) -> tuple[float, int]:
    """Drive the arbiter as the decoder does with a prompt always waiting and starving: (round time share, rounds)."""

    clock = Clock()
    arbiter = prio.Arbiter(settings, 0.5, clock)
    arbiter.last_round = 0.0
    round_time, rounds = 0.0, 0
    while clock.t < seconds:
        if arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL):
            clock.t += CHUNK_S
        else:
            clock.t += ROUND_S
            round_time += ROUND_S
            rounds += 1
            arbiter.rounded()
    return round_time / clock.t, rounds


def test_starving_prompts_cannot_take_the_decode_time_share():
    share, rounds = overload(prio.Settings())
    assert 0.33 <= share <= 0.37, share
    # ~27 rounds a second: at the diag's ~1.5 tokens a round a stream, ~40 tok/s a stream, not 4.
    assert rounds / 60.0 >= 25, rounds


def test_without_the_share_starving_prompts_get_almost_all_the_time():
    share, rounds = overload(prio.Settings(decode_share=0.0))
    assert share < 0.05, share                 # the S2 behaviour: one round a second at most (the round gap)
    assert rounds / 60.0 <= 2.0, rounds


def test_the_share_is_time_not_iterations_and_a_quiet_engine_owes_nothing():
    clock = Clock()
    arbiter = prio.Arbiter(prio.Settings(decode_share=0.5), 0.5, clock)
    assert arbiter.choose(prio.NORMAL, 4096, False, prio.NORMAL) in (True, False)
    arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL)            # a chunk
    clock.t += 1.0                                                    # that took a second
    assert arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL) is False
    assert abs(arbiter.debt - 1.0) < 1e-9                             # share 0.5: one second of rounds owed
    clock.t += 1.0
    arbiter.rounded()                                                 # a second of rounds pays it back
    assert arbiter.debt == 0.0
    assert arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL) is True
    # A realtime prompt never waits behind owed rounds (within the round gap, which still holds).
    clock.t += 0.5
    assert arbiter.choose(prio.REALTIME, 64, False, prio.NORMAL) is True
    assert arbiter.debt > 0


def test_the_owed_time_is_capped_and_the_settings_read_from_the_environment():
    clock = Clock()
    settings = prio.Settings.from_env({"TENSORFOLD_PRIORITY_DECODE_SHARE": "0.5", "TENSORFOLD_PRIORITY_DECODE_DEBT_MS": "300"})
    assert (settings.decode_share, settings.decode_debt_s) == (0.5, 0.3)
    arbiter = prio.Arbiter(settings, 0.5, clock)
    arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL)
    clock.t += 30.0
    arbiter.choose(prio.NORMAL, 4096, True, prio.NORMAL)
    assert arbiter.debt == 0.3
    assert prio.Settings.from_env({}).decode_share == 0.35
    for bad in ("1", "-0.1", "x"):
        try:
            prio.Settings.from_env({"TENSORFOLD_PRIORITY_DECODE_SHARE": bad})
        except ValueError:
            continue
        raise AssertionError(bad)
