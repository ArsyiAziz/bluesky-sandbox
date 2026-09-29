"""Temporary clearances: a ``ClearanceDuration`` makes a clearance run out.

When it does, own navigation takes the axis back - fully (LNAV+VNAV) once no
axis is under a clearance, else to the route value - so an explored clearance
never strands an aircraft. The duration is applied exactly when its target is.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


# Continuous: heading, altitude, then the two durations; binary: LNAV+VNAV and
# the masks on heading, altitude, LNAV+VNAV.
_ACTIONS = [
    act.HdgDeg(),
    act.AltFt(),
    act.ClearanceDuration(target=act.HdgDeg, low=60.0, high=600.0),
    act.ClearanceDuration(target=act.AltFt, low=60.0, high=600.0),
    act.AutopilotLnavVnav(),
    act.ActionMask(target=act.HdgDeg),
    act.ActionMask(target=act.AltFt),
    act.ActionMask(target=act.AutopilotLnavVnav),
]
_OBS = [
    obs.ApLnavOn(),
    obs.ClearanceTimeLeftS(target=act.HdgDeg),
    obs.ClearanceTimeLeftS(target=act.AltFt),
]


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(dt=5.0, obs_fields=_OBS, action_fields=_ACTIONS),
    )
    yield env
    env.close()


def _fly(env) -> str:
    """One B744 heading east on LNAV+VNAV to a fix at FL200, dead ahead."""
    env.reset(seed=0)
    assert bs.traf.cre("TMP1", "B744", 52.0, 4.0, 90, 20_000 * ft, 250 * kts)
    bs.stack.stack("ADDWPT TMP1 52.0 6.5 20000")
    bs.stack.stack("VNAV TMP1 ON")
    env.set_aircraft_control_state("TMP1", AircraftControlState.CONTROLLED)
    return "TMP1"


def _step(
    env,
    agent,
    *,
    heading=None,
    hdg_s=60.0,
    alt=None,
    alt_s=60.0,
    resume=False,
):
    """Clear a heading and/or a level for a time, or resume; else say nothing."""
    continuous = np.array([heading or 0.0, alt or 20_000.0, hdg_s, alt_s], np.float32)
    binary = np.array(
        [1 if resume else 0, heading is None, alt is None, 0 if resume else 1]
    )
    observations, _, _, _, infos = env.step(
        {agent: {"continuous": continuous, "binary": binary}}
    )
    return observations[agent], infos[agent]


def _i() -> int:
    return bs.traf.id.index("TMP1")


def _bearing() -> float:
    i = _i()
    route = bs.traf.ap.route[i]
    qdr, _ = kwikqdrdist(
        bs.traf.lat[i],
        bs.traf.lon[i],
        route.wplat[route.iactwp],
        route.wplon[route.iactwp],
    )
    return float(qdr) % 360.0


def _off(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_a_vector_runs_out_and_own_navigation_takes_the_aircraft_back(env):
    agent = _fly(env)
    seen, info = _step(env, agent, heading=150.0, hdg_s=60.0)
    assert not bs.traf.swlnav[_i()]
    assert seen[1] == pytest.approx(55.0, abs=0.5)  # 60 s, one 5 s step flown
    assert info["action_applied"]["continuous"][2] == 1.0  # duration with its target
    assert info["action_applied"]["continuous"][3] == 0.0  # altitude not cleared
    for _ in range(12):
        seen, _ = _step(env, agent)
    assert seen[1] == 0.0
    assert bs.traf.swlnav[_i()] and bs.traf.swvnav[_i()]
    for _ in range(12):
        _step(env, agent)
    assert _off(float(bs.traf.trk[_i()]), _bearing()) < 5.0


def test_one_axis_back_while_another_is_still_cleared(env):
    agent = _fly(env)
    _step(env, agent, heading=150.0, hdg_s=300.0, alt=24_000.0, alt_s=60.0)
    for _ in range(14):
        seen, _ = _step(env, agent)
    # The level ran out: back to the fix's FL200; the vector still runs.
    assert bs.traf.selalt[_i()] / ft == pytest.approx(20_000.0, abs=1.0)
    assert not bs.traf.swlnav[_i()]
    assert seen[1] > 0.0 and seen[2] == 0.0


def test_a_new_clearance_restarts_the_clock_and_resuming_ends_it(env):
    agent = _fly(env)
    _step(env, agent, heading=150.0, hdg_s=60.0)
    for _ in range(4):
        _step(env, agent)
    seen, _ = _step(env, agent, heading=120.0, hdg_s=300.0)
    assert seen[1] == pytest.approx(295.0, abs=0.5)
    seen, _ = _step(env, agent, resume=True)
    assert seen[1] == 0.0 and bs.traf.swlnav[_i()]


def test_the_duration_is_clamped_to_its_range(env):
    agent = _fly(env)
    seen, _ = _step(env, agent, heading=150.0, hdg_s=5_000.0)
    assert seen[1] == pytest.approx(595.0, abs=0.5)


@pytest.mark.parametrize(
    ("actions", "match"),
    [
        ([act.HdgDeg(), act.ClearanceDuration(target=act.HdgDeg)], "has no ActionMask"),
        (
            [
                act.CommBroadcast(),
                act.ActionMask(target=act.CommBroadcast),
                act.ClearanceDuration(target=act.CommBroadcast),
            ],
            "commands no axis",
        ),
        (
            [
                act.HdgDeg(),
                act.ActionMask(target=act.HdgDeg),
                act.ClearanceDuration(target=act.HdgDeg),
                act.ClearanceDuration(target=act.HdgDeg),
            ],
            "more than one ClearanceDuration",
        ),
    ],
    ids=["unmasked", "no axis", "twice"],
)
def test_a_duration_that_times_no_clearance_is_refused(actions, match):
    with pytest.raises(ValueError, match=match):
        EnvConfig(dt=5.0, obs_fields=[], action_fields=actions)


@pytest.fixture(scope="module")
def committed():
    actions = list(_ACTIONS)
    actions[5] = act.ActionMask(target=act.HdgDeg, lock_for_duration=True)
    env = BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[*_OBS, obs.ActionLocked(target=act.HdgDeg)],
            action_fields=actions,
        ),
    )
    yield env
    env.close()


def test_a_vector_locked_for_its_duration_is_flown_not_just_turned(committed):
    agent = _fly(committed)
    seen, _ = _step(committed, agent, heading=150.0, hdg_s=120.0)
    assert seen[3] == 1.0  # locked from the start
    for _ in range(10):  # the turn is long done well before 120 s
        seen, _ = _step(committed, agent)
    # Neither a new heading nor resuming own navigation is taken mid-leg.
    seen, info = _step(committed, agent, heading=90.0)
    assert info["action_applied"]["continuous"][0] == 0.0 and seen[3] == 1.0
    seen, info = _step(committed, agent, resume=True)
    assert info["action_applied"]["binary"][0] == 0.0
    assert not bs.traf.swlnav[_i()]
    assert float(bs.traf.ap.trk[_i()]) == pytest.approx(150.0, abs=0.5)
    for _ in range(12):
        seen, _ = _step(committed, agent)
    # Run out: unlocked, and back on own navigation.
    assert seen[3] == 0.0 and seen[1] == 0.0
    assert bs.traf.swlnav[_i()]


def test_a_lock_for_the_duration_needs_a_duration():
    with pytest.raises(ValueError, match="no ClearanceDuration times it"):
        EnvConfig(
            dt=5.0,
            obs_fields=[],
            action_fields=[
                act.HdgDeg(),
                act.ActionMask(target=act.HdgDeg, lock_for_duration=True),
            ],
        )


def _captured_env(lock: bool) -> BlueskyEnv:
    """The heading's duration counted from capture: a hold after the turn."""
    actions = list(_ACTIONS)
    actions[2] = act.ClearanceDuration(
        target=act.HdgDeg, low=60.0, high=600.0, from_capture=True
    )
    if lock:
        actions[5] = act.ActionMask(target=act.HdgDeg, lock_for_duration=True)
    return BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[*_OBS, obs.ActionLocked(target=act.HdgDeg)],
            action_fields=actions,
        ),
    )


@pytest.fixture(scope="module")
def captured():
    env = _captured_env(lock=False)
    yield env
    env.close()


def _turning() -> bool:
    return abs(float(obs.ApHdgErrorDeg().get(_i()))) > 1.0


def test_a_hold_counted_from_capture_starts_once_the_turn_is_flown(captured):
    agent = _fly(captured)
    seen, _ = _step(captured, agent, heading=270.0, hdg_s=60.0)  # a U-turn
    steps = 1
    while _turning():
        assert seen[1] == 60.0  # the whole hold, still to come
        seen, _ = _step(captured, agent)
        steps += 1
    assert steps * 5.0 > 30.0  # a turn this size takes a while to fly
    # Captured: the hold now runs down, and own navigation only takes the
    # aircraft back a full hold after the turn, not 60 s after the clearance.
    for _ in range(3):
        seen, _ = _step(captured, agent)
    assert 0.0 < seen[1] < 60.0
    assert not bs.traf.swlnav[_i()]
    while seen[1] > 0.0:
        seen, _ = _step(captured, agent)
        steps += 1
    assert steps * 5.0 >= 60.0 + 30.0
    _step(captured, agent)  # run out: resumed before the next actions
    assert bs.traf.swlnav[_i()] and bs.traf.swvnav[_i()]


def test_a_small_turn_is_held_as_long_as_a_big_one(captured):
    agent = _fly(captured)
    seen, _ = _step(captured, agent, heading=100.0, hdg_s=60.0)
    for _ in range(4):
        seen, _ = _step(captured, agent)
    # Ten degrees are flown within a step or two: the hold is running.
    assert 35.0 <= seen[1] < 60.0
    assert not bs.traf.swlnav[_i()]


def test_a_new_clearance_mid_turn_restarts_the_wait_for_capture(captured):
    agent = _fly(captured)
    _step(captured, agent, heading=270.0, hdg_s=60.0)
    seen, _ = _step(captured, agent, heading=200.0, hdg_s=120.0)
    assert seen[1] == 120.0
    assert _turning()


def test_a_lock_for_the_duration_holds_through_the_turn_and_the_hold():
    env = _captured_env(lock=True)
    try:
        agent = _fly(env)
        seen, _ = _step(env, agent, heading=270.0, hdg_s=60.0)
        assert seen[3] == 1.0
        steps = 1
        while seen[1] > 0.0:
            assert seen[3] == 1.0
            seen, info = _step(env, agent, heading=90.0)  # refused throughout
            assert info["action_applied"]["continuous"][0] == 0.0
            steps += 1
        assert steps * 5.0 >= 60.0 + 30.0
        seen, _ = _step(env, agent)
        assert seen[3] == 0.0 and bs.traf.swlnav[_i()]
    finally:
        env.close()
