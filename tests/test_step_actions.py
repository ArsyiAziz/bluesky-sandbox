"""Discrete actions, the way air traffic control gives them: a
``StepNormalizer`` makes any scalar action a CHOICE among whole steps - the
``discrete`` (``MultiDiscrete``) part of the action space - as far as the
aircraft can now reach, not a symmetric part of it; a target action's
``command_step`` puts its targets on a common grid: flight levels."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from gymnasium.spaces import Dict, MultiDiscrete

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.core.layout import action_applied, flatten_action, unflatten_action
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import StepNormalizer
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


LEVELS = StepNormalizer(1000.0, 10, include_zero=False)
KNOTS = StepNormalizer(10.0, 6, include_zero=False)
TURNS = StepNormalizer(10.0, 18, include_zero=False)
_ACTIONS = [
    act.AltDeltaFt(normalizer=LEVELS, command_step=1000.0, command_floor=1000.0),
    act.SpdDeltaKts(normalizer=KNOTS, command_step=10.0),
    act.ApHdgDeltaDeg(normalizer=TURNS),
]


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=_ACTIONS),
    )
    yield env
    env.close()


def _fly(env, alt_ft: float = 23_344.0, cas_kts: float = 251.0) -> int:
    env.reset(seed=0)
    assert bs.traf.cre("STP1", "B744", 52.0, 4.0, 90, alt_ft * ft, cas_kts * kts)
    env.set_aircraft_control_state("STP1", AircraftControlState.CONTROLLED)
    return bs.traf.id.index("STP1")


def _clear(env, levels: int = 1, knots: int = 1, turns: int = 1) -> None:
    """Clear whole steps - ``levels`` of 1,000 ft, ``knots`` of 10 kt, ``turns``
    of 10 deg - as the policy would: the index of each in its choices."""
    choice = [
        LEVELS.steps().index(levels),
        KNOTS.steps().index(knots),
        TURNS.steps().index(turns),
    ]
    env.step({"STP1": {"discrete": np.array(choice)}})


def test_the_choices_are_the_whole_steps_each_way():
    assert LEVELS.steps() == [k for k in range(-10, 11) if k != 0]
    assert LEVELS.n_choices == 20
    assert StepNormalizer(10.0, 18).n_choices == 37  # with 0
    # As an observation: the nearest choice, by its index.
    assert StepNormalizer(1000.0, 10).normalize(None, 2_400.0, 0) == [12.0]


def test_step_actions_are_a_multidiscrete_part_of_the_action_space(env):
    _fly(env)
    space = env.action_space("STP1")
    assert isinstance(space, Dict) and set(space.spaces) == {"discrete"}
    assert isinstance(space["discrete"], MultiDiscrete)
    assert list(space["discrete"].nvec) == [20, 12, 36]
    sampled = space.sample()
    flat = flatten_action(env.config, sampled)
    again = unflatten_action(env.config, flat)
    np.testing.assert_array_equal(again["discrete"], sampled["discrete"])
    assert again["discrete"].dtype == np.int64
    assert set(action_applied(env.config)) == {"discrete"}


def test_a_level_change_is_whole_flight_levels(env):
    i = _fly(env, alt_ft=23_344.0)
    _clear(env, levels=+1)
    # From the level nearest 23,344 ft - FL230 - one up: FL240, not 24,344 ft.
    assert bs.traf.selalt[i] / ft == pytest.approx(24_000.0, abs=1e-3)
    _clear(env, levels=-3)
    assert bs.traf.selalt[i] / ft == pytest.approx(20_000.0, abs=1e-3)


def test_a_speed_change_is_whole_ten_knot_steps(env):
    i = _fly(env, cas_kts=251.0)
    _clear(env, knots=-2)
    assert bs.traf.selspd[i] / kts == pytest.approx(230.0, abs=1e-3)


def test_a_turn_is_whole_ten_degree_steps_off_the_track(env):
    i = _fly(env)
    track = float(bs.traf.trk[i])
    _clear(env, turns=+3)
    assert float(bs.traf.ap.trk[i]) == pytest.approx((track + 30.0) % 360.0, abs=1e-3)


def test_a_step_out_of_reach_is_the_largest_one_that_is_not(env):
    # 3,000 ft over the floor: ten levels down reaches only the floor, FL010 -
    # the largest whole step the bounds leave.
    i = _fly(env, alt_ft=3_000.0)
    _clear(env, levels=-10)
    assert bs.traf.selalt[i] / ft == pytest.approx(1_000.0, abs=1e-3)


def test_near_a_limit_the_far_side_is_still_reachable(env):
    # Low and slow: the delta's bounds are symmetric (2 levels, 1 step either
    # way), but the aircraft can climb 10 levels and speed up 60 kt.
    i = _fly(env, alt_ft=3_000.0, cas_kts=150.0)
    _clear(env, levels=+10, knots=+6)
    assert bs.traf.selalt[i] / ft == pytest.approx(13_000.0, abs=1e-3)
    assert bs.traf.selspd[i] / kts == pytest.approx(210.0, abs=1e-3)


def _off_grid(value: float, step: float) -> float:
    """How far ``value`` is from the nearest multiple of ``step``."""
    return abs(value - round(value / step) * step)


def test_whatever_the_state_every_level_is_a_flight_level(env):
    rng = np.random.default_rng(0)
    for _ in range(40):
        i = _fly(
            env,
            alt_ft=float(rng.uniform(1_200, 40_000)),
            cas_kts=float(rng.uniform(150, 330)),
        )
        _clear(env, levels=int(rng.choice([-10, -3, -1, 1, 4, 10])))
        assert _off_grid(float(bs.traf.selalt[i] / ft), 1_000.0) < 1e-6


def test_an_absolute_target_is_put_on_the_grid(env):
    field = act.AltFt(command_step=1000.0)
    i = _fly(env)
    field.set(i, 23_344.0)
    bs.stack.process()
    assert bs.traf.selalt[i] / ft == pytest.approx(23_000.0, abs=1e-3)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: StepNormalizer(0.0), "step must be > 0"),
        (lambda: StepNormalizer(10.0, 0), "steps_each_way"),
        (lambda: StepNormalizer(10.0, 2.5), "steps_each_way"),
        (lambda: act.AltDeltaFt(command_step=0.0), "command_step must be > 0"),
    ],
    ids=[
        "zero step",
        "no steps",
        "fractional steps",
        "zero grid",
    ],
)
def test_a_step_that_cannot_be_is_refused(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_step_normalizer_left_without_a_step_says_so_when_used(env):
    i = _fly(env)
    unset = StepNormalizer(steps_each_way=3, include_zero=False)
    assert unset.step is None and unset.n_choices == 6  # its choices still known
    with pytest.raises(ValueError, match="no step"):
        unset.denormalize(act.AltDeltaFt(), [0], i)
