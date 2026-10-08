"""Autopilot switches: 1 turns a mode on, 0 turns it off.

A switch is in the binary part of the action space, so its value is 0 or 1; its
bounds are fixed at (0, 1) and it has no normalizer. These pin the commands a
switch sends - only when its mode changes - and how the dispatcher orders
switches around the actions they suppress or require.
"""

from __future__ import annotations

from types import SimpleNamespace

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.services import ActionDispatcher
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

_SWITCHES = [act.AutopilotLnav, act.AutopilotVnav, act.AutopilotLnavVnav]


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


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Empty(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    yield env
    env.close()


@pytest.fixture
def aircraft(env):
    """A B744 with LNAV and VNAV off; its index."""
    env.reset(seed=0)
    assert bs.traf.cre("SW001", "B744", 52.0, 4.5, 90, 12_000 * ft, 250 * kts) is True
    idx = bs.traf.id.index("SW001")
    bs.traf.swlnav[idx] = False
    bs.traf.swvnav[idx] = False
    return idx


@pytest.fixture
def sent(monkeypatch):
    """The stack commands sent, captured instead of executed."""
    commands: list[str] = []
    monkeypatch.setattr(bs.stack, "stack", lambda *lines, **_kw: commands.extend(lines))
    return commands


def _dispatch(fields, idx, values):
    config = SimpleNamespace(action_fields=fields)
    ActionDispatcher(SimpleNamespace(config=config)).apply(idx, np.asarray(values))


@pytest.mark.parametrize("cls", _SWITCHES, ids=lambda c: c.__name__)
def test_a_switch_is_bounded_zero_to_one(aircraft, cls):
    assert cls().bounds(aircraft) == (0.0, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"low": -1.0}, {"high": 2.0}, {"threshold": 0.5}, {"normalizer": None}],
    ids=["low", "high", "threshold", "normalizer"],
)
def test_a_switch_takes_no_bounds_threshold_or_normalizer(kwargs):
    with pytest.raises(TypeError):
        act.AutopilotLnav(**kwargs)


def test_one_turns_a_mode_on_and_zero_turns_it_off(aircraft, sent):
    act.AutopilotLnavVnav().set(aircraft, 1.0)
    assert sent == ["LNAV SW001 ON", "VNAV SW001 ON"]
    sent.clear()
    bs.traf.swlnav[aircraft] = bs.traf.swvnav[aircraft] = True
    act.AutopilotLnavVnav().set(aircraft, 0.0)
    assert sent == ["LNAV SW001 OFF", "VNAV SW001 OFF"]


@pytest.mark.parametrize(("state", "value"), [(False, 0.0), (True, 1.0)])
def test_a_mode_already_in_that_state_is_not_commanded(aircraft, sent, state, value):
    # VNAV OFF resets the selected speed and altitude; sent every step, it
    # would undo the targets the other actions set. On is VNAV with its speed
    # guidance: fully engaged.
    bs.traf.swvnav[aircraft] = bs.traf.swvnavspd[aircraft] = state
    act.AutopilotVnav().set(aircraft, value)
    assert sent == []


def test_a_mode_switched_on_suppresses_the_axes_it_flies(aircraft, sent):
    _dispatch([act.AutopilotLnav(), act.HdgDeg()], aircraft, [1.0, 45.0])
    assert sent == ["LNAV SW001 ON"]
    sent.clear()
    _dispatch([act.AutopilotLnav(), act.HdgDeg()], aircraft, [0.0, 45.0])
    assert [c.split()[0] for c in sent] == ["HDG"]


def test_a_required_mode_is_kept_on_not_cycled(aircraft, sent):
    # VNAV requires LNAV: asking for LNAV off with VNAV on leaves LNAV on,
    # and never sends it OFF first.
    bs.traf.swlnav[aircraft] = True
    _dispatch([act.AutopilotLnav(), act.AutopilotVnav()], aircraft, [0.0, 1.0])
    assert sent == ["VNAV SW001 ON"]


def test_a_required_mode_that_is_off_is_turned_on(aircraft, sent):
    _dispatch([act.AutopilotLnav(), act.AutopilotVnav()], aircraft, [0.0, 1.0])
    assert sorted(sent) == ["LNAV SW001 ON", "VNAV SW001 ON"]


def test_a_switch_is_in_the_binary_part_of_the_action_space(env):
    env.config.action_fields.clear()
    env.config.action_fields.extend([act.HdgDeltaDeg(), act.AutopilotLnav()])
    try:
        space = env.action_space(None)
        assert space["continuous"].shape == (1,)
        assert space["binary"].n == 1
    finally:
        env.config.action_fields.clear()


@pytest.mark.parametrize("value", [0.3, 0.5, -1.0, 2.0])
def test_a_value_other_than_zero_or_one_is_refused(aircraft, value):
    with pytest.raises(ValueError, match="0 or 1"):
        act.AutopilotLnav().set(aircraft, value)
