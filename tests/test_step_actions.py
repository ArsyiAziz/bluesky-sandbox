"""Discrete actions, the way air traffic control gives them: a
``StepNormalizer`` makes any scalar action a CHOICE among whole steps - the
``discrete`` (``MultiDiscrete``) part of the action space - as far as the
aircraft can now reach, not a symmetric part of it; an action's ``grid`` puts
its value on whole steps - or, ``on="target"``, what it commands: flight
levels."""

from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from gymnasium.spaces import Dict, MultiDiscrete

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.core.layout import action_applied, flatten_action, unflatten_action
from bluesky_sandbox.core.services import _denormalize_action_value
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.actions import Grid
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


# Levels and speeds on a target grid - the steps are its grid values; turns in
# plain 10 deg steps off the track.
LEVELS = StepNormalizer(steps_each_way=10, include_zero=False)
KNOTS = StepNormalizer(steps_each_way=6, include_zero=False)
TURNS = StepNormalizer(10.0, 18, include_zero=False)
_ACTIONS = [
    act.AltDeltaFt(
        normalizer=LEVELS, grid=Grid(1000.0, on="target"), command_floor=1000.0
    ),
    act.SpdDeltaKts(normalizer=KNOTS, grid=Grid(10.0, on="target")),
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


def _give(field, i: int, value: float) -> None:
    """Give ``field`` the policy's ``value`` the way the environment does -
    through its normalizer and onto its grid - and command it."""
    field.set(i, _denormalize_action_value(field, [value], i))
    bs.stack.process()


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


def test_on_a_target_grid_a_delta_snaps_its_sum_to_the_altitude_grid(env):
    field = act.AltDeltaFt(grid=Grid(1000.0, on="target"))
    i = _fly(env, alt_ft=23_600.0)
    _give(field, i, 1000.0)
    assert bs.traf.selalt[i] / ft == pytest.approx(25_000.0, abs=1e-3)  # 24,600: 25,000
    _give(field, i, 370.0)  # a continuous delta lands on a level too
    assert bs.traf.selalt[i] / ft == pytest.approx(24_000.0, abs=1e-3)


def test_on_a_value_grid_the_delta_is_whole_steps_and_the_target_is_not(env):
    field = act.AltDeltaFt(grid=Grid(1000.0))
    i = _fly(env, alt_ft=23_344.0)
    _give(field, i, 1_370.0)
    assert bs.traf.selalt[i] / ft == pytest.approx(24_344.0, abs=1e-3)
    _give(field, i, -2_600.0)
    assert bs.traf.selalt[i] / ft == pytest.approx(20_344.0, abs=1e-3)


def test_a_value_grid_keeps_to_what_the_action_can_command(env):
    # 1,500 ft above a floor of 1,000 ft: 3,000 ft down is out of reach, and
    # the largest whole thousand that is not, 1,000 ft down.
    field = act.AltDeltaFt(grid=Grid(1000.0), command_floor=1_000.0)
    i = _fly(env, alt_ft=2_500.0)
    _give(field, i, -3_000.0)
    assert bs.traf.selalt[i] / ft == pytest.approx(1_500.0, abs=1e-3)  # -1,000


def test_on_an_absolute_action_value_and_target_are_the_same(env):
    i = _fly(env)
    for on in ("value", "target"):
        _give(act.AltFt(grid=Grid(1000.0, on=on)), i, 23_344.0)
        assert bs.traf.selalt[i] / ft == pytest.approx(23_000.0, abs=1e-3)


def test_a_grid_puts_any_scalar_action_on_whole_steps_a_message_too():
    field = act.CommBroadcast(grid=Grid(0.5))
    values = [_denormalize_action_value(field, [v], 0) for v in (-0.9, -0.2, 0.3, 0.8)]
    assert values == [-1.0, 0.0, 0.5, 1.0]


def test_a_level_change_is_whole_flight_levels(env):
    i = _fly(env, alt_ft=23_344.0)
    _clear(env, levels=+1)
    # One level up from 23,344 ft: the next level above it, FL240.
    assert bs.traf.selalt[i] / ft == pytest.approx(24_000.0, abs=1e-3)
    # Climbing through 23,4xx: three levels below it, FL210.
    _clear(env, levels=-3)
    assert bs.traf.selalt[i] / ft == pytest.approx(21_000.0, abs=1e-3)


def test_a_speed_change_is_whole_ten_knot_steps(env):
    i = _fly(env, cas_kts=251.0)
    _clear(env, knots=-2)
    # 250 is the next 10 kt below 251, 240 the one after.
    assert bs.traf.selspd[i] / kts == pytest.approx(240.0, abs=1e-3)


def test_a_turn_is_whole_ten_degree_steps_off_the_track(env):
    i = _fly(env)
    track = float(bs.traf.trk[i])
    _clear(env, turns=+3)
    assert float(bs.traf.ap.trk[i]) == pytest.approx((track + 30.0) % 360.0, abs=1e-3)


def test_on_a_heading_grid_a_turn_snaps_to_the_nearest_grid_heading(env):
    field = act.ApHdgDeltaDeg(grid=Grid(10.0, on="target"))
    i = _fly(env)  # created on 090; the track a few tenths off it
    bs.traf.trk[i] = 93.7
    _give(field, i, 30.0)
    # 093.7 + 30 = 123.7: on the grid, 120.
    assert float(bs.traf.ap.trk[i]) == pytest.approx(120.0, abs=1e-3)
    bs.traf.trk[i] = 356.0
    _give(field, i, 10.0)
    assert float(bs.traf.ap.trk[i]) == pytest.approx(10.0, abs=1e-3)


def test_on_a_value_grid_the_turn_is_whole_steps_off_the_track(env):
    field = act.ApHdgDeltaDeg(grid=Grid(10.0))
    i = _fly(env)
    bs.traf.trk[i] = 93.7
    _give(field, i, 27.0)
    assert float(bs.traf.ap.trk[i]) == pytest.approx(123.7, abs=1e-3)  # 366: 010


@pytest.mark.parametrize(
    ("field", "nominal"),
    [
        (act.ApHdgDeltaDeg(grid=Grid(10.0, on="target")), "trk"),
        (act.HdgDeltaDeg(grid=Grid(10.0, on="target")), "hdg"),
        (act.HdgDeg(grid=Grid(10.0)), None),
    ],
    ids=["from the track", "from the heading", "absolute"],
)
def test_every_heading_action_takes_a_heading_grid(env, field, nominal):
    i = _fly(env)
    if nominal is not None:
        getattr(bs.traf, nominal)[i] = 93.7
    _give(field, i, 30.0 if nominal else 123.4)
    assert float(bs.traf.ap.trk[i]) == pytest.approx(120.0, abs=1e-3)


@pytest.mark.parametrize(
    ("alt_ft", "k", "level"),
    [
        (23_344.0, +1, 24_000.0),
        (23_600.0, +1, 24_000.0),
        (24_000.0, +1, 25_000.0),
        (23_600.0, -1, 23_000.0),
        (24_000.0, -2, 22_000.0),
        (23_600.0, 0, 24_000.0),
    ],
    ids=["up, off a level", "up, near the next", "up, on a level",
         "down, off a level", "down, on a level", "the nearest"],
)
def test_on_a_grid_a_step_is_the_next_grid_value_above_or_below(env, alt_ft, k, level):
    field = act.AltDeltaFt(
        normalizer=StepNormalizer(steps_each_way=4), grid=Grid(1000.0, on="target")
    )
    i = _fly(env, alt_ft=alt_ft)
    field.set(i, field.normalizer.denormalize(field, [field.normalizer.steps().index(k)], i))
    bs.stack.process()
    assert bs.traf.selalt[i] / ft == pytest.approx(level, abs=1e-3)


def test_on_a_heading_grid_a_turn_is_to_the_next_grid_heading(env):
    turns = StepNormalizer(steps_each_way=9, include_zero=False)
    field = act.ApHdgDeltaDeg(normalizer=turns, grid=Grid(10.0, on="target"))
    i = _fly(env)
    for track, k, heading in [(93.7, +1, 100.0), (93.7, -1, 90.0), (356.0, +1, 0.0)]:
        bs.traf.trk[i] = track
        field.set(i, turns.denormalize(field, [turns.steps().index(k)], i))
        bs.stack.process()
        assert float(bs.traf.ap.trk[i]) % 360.0 == pytest.approx(heading, abs=1e-3)


def test_on_a_value_grid_a_step_is_whole_steps_off_the_nominal(env):
    field = act.AltDeltaFt(normalizer=StepNormalizer(steps_each_way=4), grid=Grid(1000.0))
    i = _fly(env, alt_ft=23_344.0)
    _give(field, i, field.normalizer.steps().index(+2))
    assert bs.traf.selalt[i] / ft == pytest.approx(25_344.0, abs=1e-3)


@pytest.mark.parametrize(
    ("on", "alt_ft", "k", "level"),
    [
        ("value", 23_344.0, +1, 25_344.0),
        ("value", 23_344.0, -2, 19_344.0),
        ("target", 23_344.0, +1, 25_000.0),
        ("target", 24_000.0, +1, 26_000.0),
        ("target", 23_600.0, -1, 22_000.0),
    ],
    ids=["value up", "value down two", "target up, off a level",
         "target up, on a level", "target down"],
)
def test_on_a_grid_a_step_counts_grid_steps(env, on, alt_ft, k, level):
    field = act.AltDeltaFt(
        normalizer=StepNormalizer(2, 4), grid=Grid(1000.0, on=on)
    )
    i = _fly(env, alt_ft=alt_ft)
    _give(field, i, field.normalizer.steps().index(k))
    assert bs.traf.selalt[i] / ft == pytest.approx(level, abs=1e-3)
    assert field.normalizer.step_for(field) == 2000.0


def test_on_a_grid_a_step_that_is_not_whole_grid_steps_is_refused(env):
    field = act.AltDeltaFt(normalizer=StepNormalizer(1.5, 4), grid=Grid(1000.0))
    i = _fly(env)
    with pytest.raises(ValueError, match="counts grid steps - a whole number"):
        field.normalizer.denormalize(field, [0], i)


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


def test_a_grid_on_any_action_on_the_heading_axis_must_go_round_it():
    # An action of one's own, commanding a heading without being a built-in
    # heading action: the axis it declares says it wraps, and that is enough.
    @dataclass(frozen=True)
    class MyHeading(act.ActionField):
        meta = act.ActionMeta(
            "my_heading", act.Unit.DEG, control_axis=act.ControlAxis.HEADING,
            mode=act.ActionMode.ABSOLUTE,
        )
        low: float = 0.0
        high: float = 360.0

        def set(self, idx, value):
            pass

        def bounds(self, idx):
            return self._configured_bounds()

    assert act.ControlAxis.HEADING.period == 360.0
    assert act.ControlAxis.ALTITUDE.period is None
    MyHeading(grid=Grid(15.0))
    with pytest.raises(ValueError, match="heading axis, which wraps at 360"):
        MyHeading(grid=Grid(7.0))
    # A turn on its grid does not wrap: any step.
    act.HdgDeltaDeg(grid=Grid(7.0))
    # An axis that does not wrap: any step.
    act.AltFt(grid=Grid(7.0))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: StepNormalizer(0.0), "step must be > 0"),
        (lambda: StepNormalizer(10.0, 0), "steps_each_way"),
        (lambda: StepNormalizer(10.0, 2.5), "steps_each_way"),
        (lambda: Grid(0.0), "step must be a number > 0"),
        (lambda: Grid(10.0, on="delta"), "on must be 'value' or 'target'"),
        (lambda: act.AltDeltaFt(grid=1000.0), "grid must be a Grid"),
        (lambda: act.HdgDeg(grid=Grid(7.0)), "must divide 360"),
        (lambda: act.HdgDeltaDeg(grid=Grid(7.0, on="target")), "must divide 360"),
        (lambda: act.AutopilotVnav(grid=Grid(1.0)), "unexpected keyword"),
    ],
    ids=[
        "zero step",
        "no steps",
        "fractional steps",
        "zero grid",
        "grid on nothing",
        "a bare number",
        "heading off the circle",
        "target heading off the circle",
        "a switch",
    ],
)
def test_a_step_or_grid_that_cannot_be_is_refused(make, match):
    with pytest.raises((TypeError, ValueError), match=match):
        make()


def test_a_step_normalizer_left_without_a_step_says_so_when_used(env):
    i = _fly(env)
    unset = StepNormalizer(steps_each_way=3, include_zero=False)
    assert unset.step is None and unset.n_choices == 6  # its choices still known
    with pytest.raises(ValueError, match="no step"):
        unset.denormalize(act.AltDeltaFt(), [0], i)


@pytest.mark.parametrize(
    "switch", [act.AutopilotLnavVnav(), act.AutopilotVnav()], ids=["lnav+vnav", "vnav"]
)
def test_resuming_own_navigation_ends_a_speed_clearance(env, switch):
    i = _fly(env)
    for command in ("ADDWPT STP1 52.0 5.0 FL230", "LNAV STP1 ON", "VNAV STP1 ON", "SPD STP1 240"):
        bs.stack.stack(command)
    bs.stack.process()
    # A cleared speed holds: VNAV still on, its speed guidance off.
    assert bs.traf.swvnav[i] and not bs.traf.swvnavspd[i]
    switch.set(i, 1.0)
    bs.stack.process()
    assert bs.traf.swvnav[i] and bs.traf.swvnavspd[i]



def test_a_target_action_says_what_a_clearance_holds_and_what_each_value_commands(env):
    alt, spd, hdg = act.AltDeltaFt(), act.SpdDeltaKts(), act.ApHdgDeltaDeg()
    i = _fly(env, alt_ft=23_000.0, cas_kts=250.0)
    # Flying its own (as created: no clearance on any axis yet).
    bs.stack.stack("ADDWPT STP1 52.0 5.0 FL230")
    for command in ("LNAV STP1 ON", "VNAV STP1 ON"):
        bs.stack.stack(command)
    bs.stack.process()
    assert (alt.held(i), spd.held(i), hdg.held(i)) == (None, None, None)
    for command in ("ALT STP1 25000", "SPD STP1 240", "HDG STP1 120"):
        bs.stack.stack(command)
    bs.stack.process()
    assert alt.held(i) == pytest.approx(25_000.0, abs=1e-3)
    assert spd.held(i) == pytest.approx(240.0, abs=1e-3)
    assert hdg.held(i) == pytest.approx(120.0, abs=1e-3)
    # What each value commands: a delta from the nominal, a heading round the circle.
    np.testing.assert_allclose(alt.targets(i, np.array([1000.0])), [alt.nominal(i) + 1000.0])
    bs.traf.trk[i] = 350.0
    np.testing.assert_allclose(hdg.targets(i, np.array([20.0])), [10.0])
