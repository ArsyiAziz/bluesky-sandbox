"""A crossover speed action with a Mach regime: below its crossover a change in
knots, above it a change in Mach - with its own bounds, normalizer and grid, in
the same action space - and the crossover flag given the same crossover says
which."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts, vcas2mach

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.services import _denormalize_action_value
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.actions import _targets
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    MinMaxNormalizer,
    StepNormalizer,
    SymmetricNormalizer,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

XO = act.Crossover(cas_kts=300.0, mach=0.78)  # about FL293


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=SpawnConfig(regions=[]), queryables={}, max_aircraft=0)


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(scenario=_Empty(), config=EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=[]))
    yield env
    env.close()


def _fly(env, acid: str, fl: int, cas_kts: float = 250.0) -> int:
    env.reset(seed=0)
    assert bs.traf.cre(acid, "B744", 52.0, 4.0, 90.0, fl * 100 * ft, cas_kts * kts)
    env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
    return bs.traf.id.index(acid)


def _command(action, idx: int, policy_value) -> float:
    value = _denormalize_action_value(action, [policy_value], idx)
    action.set(idx, value)
    bs.stack.process()
    return float(bs.traf.selspd[idx])


def _action(**regime):
    return act.ApSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(),
        low=-40.0,
        high=40.0,
        above_crossover=act.MachRegime(XO, **regime),
    )


def test_the_crossover_is_the_schedules_altitude():
    assert XO.altitude_m / ft == pytest.approx(29_314, abs=5)


def test_below_the_crossover_it_changes_knots(env):
    idx = _fly(env, "LOW1", 250)
    action = _action(low=-0.04, high=0.04)
    assert action.acting(idx) is action
    assert _command(action, idx, 1.0) == pytest.approx((250.0 + 40.0) * kts, rel=1e-4)
    assert obs.AboveCrossover(crossover=XO).get(idx) == 0.0


def test_above_it_changes_mach_with_its_own_bounds(env):
    idx = _fly(env, "HIGH1", 370)
    action = _action(low=-0.04, high=0.04)
    view = action.acting(idx)
    assert view is not action and view.bounds(idx) == (-0.04, 0.04)
    mach = float(vcas2mach(250.0 * kts, 370 * 100 * ft))
    assert _command(action, idx, 1.0) == pytest.approx(mach + 0.04, abs=1e-4)  # a Mach SPD
    assert obs.AboveCrossover(crossover=XO).get(idx) == 1.0


def test_slowing_down_above_it_stays_in_mach(env):
    # The crossover is the schedule's altitude: not the speed flown.
    idx = _fly(env, "HIGH2", 370)
    action = _action(low=-0.04, high=0.04)
    _command(action, idx, -1.0)
    assert action.acting(idx) is not action
    assert obs.AboveCrossover(crossover=XO).get(idx) == 1.0


def test_its_bounds_unset_are_symmetric_within_mmo(env):
    from bluesky.tools.aero import vcas2mach

    idx = _fly(env, "HIGH3", 370)
    # No knot bounds of its own, so none to match: the runtime envelope.
    view = act.ApSpdDeltaCrossover(normalizer=SymmetricNormalizer(), above_crossover=act.MachRegime(XO)).acting(idx)
    low, high = view.bounds(idx)
    assert low == pytest.approx(-high)
    # The ceiling: Mmo, or the maximum CAS as Mach here, whichever is lower.
    ceiling = min(float(bs.traf.perf.mmo[idx]), float(vcas2mach(bs.traf.perf.vmax[idx], bs.traf.alt[idx])))
    assert view.nominal(idx) + high <= ceiling + 1e-9
    assert view.reach(idx)[1] == pytest.approx(ceiling - view.nominal(idx))


def test_its_targets_go_on_its_mach_grid(env):
    idx = _fly(env, "HIGH4", 370)
    action = _action(low=-0.04, high=0.04, grid=act.Grid(0.01, on="target"))
    selected = _command(action, idx, 0.3)
    assert selected == pytest.approx(round(selected, 2), abs=1e-6)


def test_its_command_floor_in_knots_holds_as_mach(env):
    idx = _fly(env, "HIGH5", 370)
    action = act.ApSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(),
        command_floor=240.0,
        above_crossover=act.MachRegime(XO, low=-0.1, high=0.1),
    )
    floor = float(vcas2mach(240.0 * kts, 370 * 100 * ft))
    assert _command(action, idx, -1.0) == pytest.approx(floor, abs=1e-4)


def test_steps_along_a_grid_step_in_mach_above_it(env):
    idx = _fly(env, "HIGH6", 370)
    action = act.ApSpdDeltaCrossover(
        normalizer=StepNormalizer(steps_each_way=2),
        grid=act.Grid(10.0),
        above_crossover=act.MachRegime(XO, grid=act.Grid(0.01)),
    )
    mach = float(vcas2mach(250.0 * kts, 370 * 100 * ft))
    assert _command(action, idx, 4) == pytest.approx(mach + 0.02, abs=1e-4)  # two steps up


def test_the_waypoint_action_counts_from_the_waypoint_speed_as_mach(env, monkeypatch):
    idx = _fly(env, "HIGH7", 370)
    monkeypatch.setattr(_targets, "_active_route_waypoint", lambda i: (52.0, 5.0, 37_000 * ft, 260 * kts))
    action = act.ActiveRouteWaypointSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(),
        above_crossover=act.MachRegime(XO, low=-0.05, high=0.05),
    )
    assert action.acting(idx).nominal(idx) == pytest.approx(float(vcas2mach(260 * kts, 37_000 * ft)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normalizer": SymmetricNormalizer(), "above_crossover": act.MachRegime(normalizer=StepNormalizer(0.01))},
         "another part of the action space"),
        ({"normalizer": SymmetricNormalizer(), "above_crossover": act.MachRegime(normalizer=MinMaxNormalizer())},
         "another range"),
        ({"above_crossover": act.MachRegime(normalizer=SymmetricNormalizer())}, "has no normalizer"),
        ({"normalizer": StepNormalizer(10.0), "above_crossover": act.MachRegime()}, "steps in knots"),
        ({"normalizer": StepNormalizer(), "grid": act.Grid(10.0), "above_crossover": act.MachRegime()},
         "give its MachRegime a grid"),
    ],
    ids=["discrete for continuous", "other range", "none to some", "knots step", "no grid to step"],
)
def test_a_regime_that_would_change_the_action_space_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        act.ApSpdDeltaCrossover(**kwargs)


def test_a_regime_takes_both_bounds_or_neither():
    with pytest.raises(ValueError, match="both low and high"):
        act.MachRegime(low=-0.02)
    with pytest.raises(ValueError, match="must be in"):
        act.Crossover(mach=1.2)
