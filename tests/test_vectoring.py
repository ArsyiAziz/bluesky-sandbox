"""Vectoring: a heading given once and held, and own navigation resumed.

With a mask on the heading, an aircraft given nothing keeps flying what it was
last cleared - its route on LNAV, or a vector. A masked ``AutopilotLnavVnav``
hands it back to its route when unmasked at 1; a mask locking until captured
makes each vector a committed unit, flown before anything else on the heading
axis is accepted.
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
from bluesky_sandbox.interface.fields.actions import mask as mask_module
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
            obs_fields=[obs.ApLnavOn(), obs.ActionLocked(target=act.HdgDeg)],
            action_fields=[
                act.HdgDeg(),
                act.AutopilotLnavVnav(),
                act.ActionMask(target=act.HdgDeg, lock_until_captured=lock),
                act.ActionMask(target=act.AutopilotLnavVnav),
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


def _step(env, agent, heading=0.0, mask=1, nav=0, nav_mask=1):
    """``mask`` masks the heading; ``nav`` is LNAV+VNAV, masked by ``nav_mask``."""
    action = {
        "continuous": np.array([heading], np.float32),
        "binary": np.array([nav, mask, nav_mask]),
    }
    observations, _, _, _, infos = env.step({agent: action})
    return observations[agent], infos[agent]


def _resume(env, agent):
    return _step(env, agent, nav=1, nav_mask=0)


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


# ---- vectors and resuming -------------------------------------------------- #


def test_an_aircraft_given_nothing_flies_its_route(free):
    agent = _fly(free)
    for _ in range(3):
        seen, _ = _step(free, agent)
    np.testing.assert_array_equal(seen, [1.0, 0.0])
    assert _off(bs.traf.ap.trk[_idx()], _bearing_to_waypoint()) < 2.0


def test_a_vector_is_held_and_resuming_returns_the_aircraft_to_its_route(free):
    agent = _fly(free)
    _step(free, agent)
    seen, _ = _step(free, agent, heading=180.0, mask=0)
    np.testing.assert_array_equal(seen, [0.0, 0.0])  # LNAV off; no lock here
    for _ in range(6):
        _step(free, agent)
    assert bs.traf.ap.trk[_idx()] == pytest.approx(180.0)  # held

    seen, _ = _resume(free, agent)
    np.testing.assert_array_equal(seen[:1], [1.0])
    for _ in range(12):
        _step(free, agent)
    assert _off(bs.traf.trk[_idx()], _bearing_to_waypoint()) < 2.0


def test_a_masked_switch_leaves_own_navigation_alone(free):
    agent = _fly(free)
    for _ in range(3):
        seen, _ = _step(free, agent, nav=0, nav_mask=1)
    np.testing.assert_array_equal(seen[:1], [1.0])


def test_resuming_takes_the_heading_axis_over_that_step(free):
    agent = _fly(free)
    _step(free, agent, heading=180.0, mask=0)
    _, info = _step(free, agent, heading=270.0, mask=0, nav=1, nav_mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [0.0])
    assert bs.traf.swlnav[_idx()]


# ---- locking until captured ------------------------------------------------ #


def test_a_vector_is_flown_before_the_heading_is_cleared_again(locking):
    agent = _fly(locking)
    _step(locking, agent)
    _, info = _step(locking, agent, heading=180.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [1.0])

    # Mid-turn: a new vector, and a resume, are both refused.
    _, info = _step(locking, agent, heading=0.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [0.0])
    _, info = _resume(locking, agent)
    np.testing.assert_array_equal(info["action_applied"]["binary"], [0.0, 1.0, 1.0])
    assert not bs.traf.swlnav[_idx()]
    assert bs.traf.ap.trk[_idx()] == pytest.approx(180.0)

    # Once the turn is flown, the heading is free again.
    for _ in range(40):
        track_before = float(bs.traf.trk[_idx()])
        _, info = _resume(locking, agent)
        if info["action_applied"]["binary"][0]:
            break
    assert bs.traf.swlnav[_idx()]
    assert _off(track_before, 180.0) < 1.0  # released at the end of the turn


def test_the_policy_sees_the_lock_it_would_be_refused_by(locking):
    agent = _fly(locking)
    seen, _ = _step(locking, agent, heading=180.0, mask=0)
    assert seen[1] == 1.0  # turning: a new heading would be refused
    for _ in range(40):
        locked_seen = seen[1]
        seen, info = _step(locking, agent, heading=0.0, mask=0)
        refused = info["action_applied"]["continuous"][0] == 0.0
        assert refused == (locked_seen == 1.0)
        if not refused:
            break
    assert not refused


def test_a_clearance_that_stops_closing_releases_its_lock(locking, monkeypatch):
    # A setpoint the aircraft cannot fly any further - the error stuck - must
    # not hold the axis forever.
    agent = _fly(locking)
    _step(locking, agent, heading=180.0, mask=0)
    monkeypatch.setattr(mask_module, "axis_error", lambda *_: 42.0)
    _, info = _step(locking, agent, heading=0.0, mask=0)  # first reading: locked
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [0.0])
    _, info = _step(locking, agent, heading=0.0, mask=0)  # not shrinking: free
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [1.0])


def test_a_lock_is_dropped_with_its_episode(locking):
    agent = _fly(locking)
    _step(locking, agent, heading=180.0, mask=0)
    agent = _fly(locking)  # reset mid-turn
    _, info = _step(locking, agent, heading=0.0, mask=0)
    np.testing.assert_array_equal(info["action_applied"]["continuous"], [1.0])


def test_a_lock_needs_an_axis_whose_error_is_measured():
    with pytest.raises(ValueError, match="locks until captured"):
        EnvConfig(
            obs_fields=[],
            action_fields=[
                act.AutopilotLnavVnav(),
                act.ActionMask(target=act.AutopilotLnavVnav, lock_until_captured=True),
            ],
        )
