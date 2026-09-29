"""Vectoring: a heading given once and held, and own navigation resumed.

With a mask on the heading, an aircraft given nothing keeps flying what it was
last cleared - its route on LNAV, or a vector. ``ResumeOwnNav`` hands it back
to its route; a mask locking until captured makes each vector a committed unit,
flown before anything else on the heading axis is accepted.
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


def _env(lock: bool) -> BlueskyEnv:
    return BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.ApLnavOn()],
            action_fields=[
                act.HdgDeg(),
                act.ResumeOwnNav(),
                act.ActionMask(target=act.HdgDeg, lock_until_captured=lock),
            ],
        ),
    )


@pytest.fixture(scope="module")
def free():
    env = _env(lock=False)
    yield env
    env.close()


@pytest.fixture(scope="module")
def locking():
    env = _env(lock=True)
    yield env
    env.close()


def _fly(env, route: bool = True) -> str:
    """One B744 heading east on LNAV, with a waypoint dead ahead."""
    env.reset(seed=0)
    assert bs.traf.cre("VEC1", "B744", 52.0, 4.0, 90, 20_000 * ft, 250 * kts)
    if route:
        bs.stack.stack("ADDWPT VEC1 52.0 5.5")
    env.set_aircraft_control_state("VEC1", AircraftControlState.CONTROLLED)
    return "VEC1"


def _step(env, agent, heading=0.0, resume=0, mask=1):
    action = {
        "continuous": np.array([heading], np.float32),
        "binary": np.array([resume, mask]),
    }
    observations, _, _, _, infos = env.step({agent: action})
    return observations[agent], infos[agent]


def _idx() -> int:
    return bs.traf.id.index("VEC1")


def _bearing_to_waypoint() -> float:
    i = _idx()
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


# ---- resume ---------------------------------------------------------------- #


def test_an_aircraft_given_nothing_flies_its_route(free):
    agent = _fly(free)
    for _ in range(3):
        seen, _ = _step(free, agent)
    np.testing.assert_array_equal(seen, [1.0])
    assert _off(bs.traf.ap.trk[_idx()], _bearing_to_waypoint()) < 2.0


def test_a_vector_is_held_and_resume_returns_the_aircraft_to_its_route(free):
    agent = _fly(free)
    _step(free, agent)
    seen, _ = _step(free, agent, heading=180.0, mask=0)
    np.testing.assert_array_equal(seen, [0.0])  # the vector took LNAV off
    for _ in range(6):
        _step(free, agent)
    assert bs.traf.ap.trk[_idx()] == pytest.approx(180.0)  # held, unmasked by nobody

    seen, _ = _step(free, agent, resume=1)
    np.testing.assert_array_equal(seen, [1.0])
    for _ in range(12):
        _step(free, agent)
    assert _off(bs.traf.trk[_idx()], _bearing_to_waypoint()) < 2.0


def test_a_zero_never_turns_lnav_off(free):
    agent = _fly(free)
    for _ in range(3):
        seen, _ = _step(free, agent, resume=0)
    np.testing.assert_array_equal(seen, [1.0])


def test_resume_takes_the_heading_axis_over_that_step(free):
    agent = _fly(free)
    _step(free, agent, heading=180.0, mask=0)
    _, info = _step(free, agent, heading=270.0, resume=1, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [0.0])
    assert bs.traf.swlnav[_idx()]


def test_an_aircraft_with_no_route_has_nothing_to_resume(free):
    agent = _fly(free, route=False)
    _step(free, agent, heading=180.0, mask=0)
    seen, _ = _step(free, agent, resume=1)
    np.testing.assert_array_equal(seen, [0.0])


# ---- locking until captured ------------------------------------------------ #


def test_a_vector_is_flown_before_the_heading_is_cleared_again(locking):
    agent = _fly(locking)
    _step(locking, agent)
    _, info = _step(locking, agent, heading=180.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [1.0])

    # Mid-turn: a new vector, and a resume, are both refused.
    _, info = _step(locking, agent, heading=0.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [0.0])
    _, info = _step(locking, agent, resume=1)
    np.testing.assert_array_equal(info["action_applied"]["binary"], [0.0, 1.0])
    assert not bs.traf.swlnav[_idx()]
    assert bs.traf.ap.trk[_idx()] == pytest.approx(180.0)

    # Once the turn is flown, the heading is free again.
    for _ in range(30):
        _step(locking, agent)
        if _off(bs.traf.trk[_idx()], 180.0) <= 2.0:
            break
    _, info = _step(locking, agent, resume=1)
    np.testing.assert_array_equal(info["action_applied"]["binary"], [1.0, 1.0])
    assert bs.traf.swlnav[_idx()]


def test_a_lock_is_dropped_with_its_episode(locking):
    agent = _fly(locking)
    _step(locking, agent, heading=180.0, mask=0)
    agent = _fly(locking)  # reset mid-turn
    _, info = _step(locking, agent, heading=0.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [1.0])


def test_a_lock_needs_an_axis_a_capture_is_measured_on():
    with pytest.raises(ValueError, match="locks until captured"):
        EnvConfig(
            obs_fields=[],
            action_fields=[
                act.ResumeOwnNav(),
                act.ActionMask(target=act.ResumeOwnNav, lock_until_captured=True),
            ],
        )
