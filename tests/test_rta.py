"""Own navigation meeting arrival times (``EnvConfig.fly_arrival_times``), and
the two fields that say what the speed range cannot do about a fix's time.

With it, each fix's arrival time is handed to BlueSky's RTA: on LNAV+VNAV the
aircraft adjusts its speed to be over the fix on time, early or late alike.
"""

from __future__ import annotations

import bluesky as bs
from bluesky.tools.aero import kts
import numpy as np
import pytest
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.performance.speeds import cas_ceiling_ms
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_FIX = Waypoint(lat=52.0, lon=5.5, reach_radius_nm=3.0)  # ~61 nm east


class _Scenario:
    def __init__(self, slack_s: float) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.999, 52.001, 4.0, 4.001)),
                    n_aircraft=1,
                    params={
                        "alt_ft": (20_000, 20_000),
                        "spd_kts": (260, 260),
                        "hdg_deg": (90, 90),
                    },
                    route=[{"waypoint": "fix", "arrival_slack_s": slack_s}],
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables={"fix": _FIX},
            max_aircraft=1,
        )


_FIELDS = [
    obs.ActiveRouteWaypointArrivalErrorS(),
    obs.ActiveRouteWaypointTimeToAbsorbS(),
    obs.ActiveRouteWaypointUnrecoverableLateS(),
]


def _fly(slack_s: float, rta: bool) -> tuple[list[np.ndarray], float]:
    """Fly one aircraft to its fix on own navigation (LNAV+VNAV); return its
    observations along the way and how late (+) or early (-) it arrived."""
    env = BlueskyEnv(
        scenario=_Scenario(slack_s),
        config=EnvConfig(
            dt=5.0,
            obs_fields=_FIELDS,
            action_fields=[act.HdgDeltaDeg()],
            fly_arrival_times=rta,
        ),
    )
    try:
        observations, _ = env.reset(seed=0)
        (agent,) = observations
        bs.stack.stack(f"VNAV {agent} ON")
        seen = [observations[agent]]
        due = None
        for _ in range(400):
            i = bs.traf.id.index(agent)
            due = float(bs.sim.simt) + float(obs.ActiveRouteWaypointTimeToGoS().get(i))
            hdg = float(bs.traf.ap.trk[i])  # hold the autopilot's own heading
            observations, _r, term, _t, _i = env.step(
                {agent: np.array([hdg - float(bs.traf.hdg[i])], np.float32)}
            )
            if agent not in observations:
                break
            seen.append(observations[agent])
            i = bs.traf.id.index(agent)
            if kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], _FIX.lat, _FIX.lon)[1] < 1.0:
                break
        return seen, float(bs.sim.simt) - due
    finally:
        env.close()


@pytest.mark.parametrize("slack_s", [150.0, -40.0], ids=["early", "late"])
def test_rta_meets_the_arrival_time_where_without_it_the_aircraft_is_off(slack_s):
    _seen, off_without = _fly(slack_s, rta=False)
    _seen, off_with = _fly(slack_s, rta=True)
    assert abs(off_without) > 30.0
    assert abs(off_with) < 15.0


def test_the_speed_range_fields_say_what_speed_cannot_fix():
    # Due 40 minutes late: even at minimum speed it would be far too early.
    seen, _ = _fly(2400.0, rta=False)
    error, absorb, unrecoverable = seen[0]
    assert error < -2000.0 and absorb > 1000.0 and unrecoverable == 0.0


def test_lateness_the_speed_range_cannot_recover_reads_above_zero():
    # Scheduled as tight as it can fly (the planner never asks for sooner),
    # then held slow: the time it loses cannot be made up.
    env = BlueskyEnv(
        scenario=_Scenario(-600.0),
        config=EnvConfig(dt=5.0, obs_fields=_FIELDS, action_fields=[act.HdgDeltaDeg()]),
    )
    try:
        observations, _ = env.reset(seed=0)
        (agent,) = observations
        _error, _absorb, unrecoverable = observations[agent]
        assert unrecoverable == 0.0
        bs.stack.stack(f"SPD {agent} 180")
        for _ in range(60):
            observations, *_ = env.step({agent: np.array([0.0], np.float32)})
        error, absorb, unrecoverable = observations[agent]
        assert error > 0.0 and unrecoverable > 0.0 and absorb == 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    ("slow_steps", "overdue"),
    [(30, False), (140, True)],
    ids=["late, time still ahead", "past its time, fix still ahead"],
)
def test_resuming_own_navigation_late_flies_the_on_time_speed_not_the_old_one(
    slow_steps, overdue
):
    env = BlueskyEnv(
        scenario=_Scenario(0.0),
        config=EnvConfig(
            dt=5.0,
            obs_fields=_FIELDS,
            action_fields=[act.CommBroadcast()],  # leaves the aircraft alone
            fly_arrival_times=True,
        ),
    )
    try:
        observations, _ = env.reset(seed=0)
        (agent,) = observations

        def step():
            env.step({agent: np.array([0.0], np.float32)})

        bs.stack.stack(f"VNAV {agent} ON")
        step()
        initial_kts = float(bs.traf.cas[bs.traf.id.index(agent)]) / kts
        bs.stack.stack(f"SPD {agent} 150")  # a speed clearance: off RTA
        for _ in range(slow_steps):
            step()
        i = bs.traf.id.index(agent)
        time_left = float(obs.ActiveRouteWaypointTimeToGoS().get(i))
        assert (time_left < 0.0) is overdue
        bs.stack.stack(f"VNAV {agent} ON")  # resume own navigation
        for _ in range(3):
            step()
        i = bs.traf.id.index(agent)
        selected_kts = float(bs.traf.selspd[i]) / kts
        assert selected_kts > initial_kts + 20.0  # catching up, not the old speed
        if overdue:
            assert selected_kts == pytest.approx(cas_ceiling_ms(i) / kts, abs=0.5)
    finally:
        env.close()
