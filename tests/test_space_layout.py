"""The action space's parts, and the layouts that name every column.

An action space is a ``Box`` while every action is continuous; with a switch it
is a ``Dict`` of a ``continuous`` Box and a ``binary`` MultiBinary. Either way
the fields are applied from one vector in config order. A layout names each
field's columns in each part of a space, and is fixed by the config.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts
from gymnasium.spaces import Box, Dict, MultiBinary

from bluesky_sandbox import (
    AircraftControlState,
    critic_observation_layout,
    flatten_action,
    observation_layout,
    zero_action,
)
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers import IntruderPaddingWrapper
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    CircularNormalizer,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

# Mixed on purpose: a switch between continuous actions, one of them two wide.
_MIXED = [
    act.HdgDeltaDeg(normalizer=CircularNormalizer()),
    act.AutopilotLnav(),
    act.AltDeltaFt(low=-1000.0, high=1000.0),
    act.AutopilotVnav(),
]
_CONTINUOUS = [act.HdgDeltaDeg(), act.AltDeltaFt(low=-1000.0, high=1000.0)]


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


def _env(action_fields, **obs_kwargs):
    config = EnvConfig(
        dt=12.0,
        obs_fields=obs_kwargs.pop("obs_fields", [obs.AltFt()]),
        action_fields=list(action_fields),
        **obs_kwargs,
    )
    return BlueskyEnv(scenario=_Empty(), config=config)


@pytest.fixture(scope="module")
def mixed():
    env = _env(
        _MIXED,
        obs_fields=[
            obs.AltFt(),
            obs.FlightPhaseOneHot(),
            obs.PrevActionNorm(dim=6),
        ],
        intruder_obs_fields=[obs.DistToOwnNm(), obs.AltFt()],
        critic_obs_fields=[obs.CasKts()],
        critic_intruder_obs_fields=[obs.AltFt(), obs.TcpaS()],
    )
    yield env
    env.close()


@pytest.fixture
def aircraft(mixed):
    mixed.reset(seed=0)
    assert bs.traf.cre("LAY001", "B744", 52.0, 4.5, 90, 12_000 * ft, 250 * kts)
    idx = bs.traf.id.index("LAY001")
    bs.traf.swlnav[idx] = bs.traf.swvnav[idx] = False
    mixed.set_aircraft_control_state("LAY001", AircraftControlState.CONTROLLED)
    return "LAY001"


# --- the action space ------------------------------------------------------------
def test_all_continuous_actions_keep_one_box():
    env = _env(_CONTINUOUS)
    try:
        space = env.action_space(None)
        assert isinstance(space, Box)
        low, high = env._observation_assembler.field_output_bounds(
            None, env.config.action_fields
        )
        assert np.array_equal(space.low, low) and np.array_equal(space.high, high)
        assert list(env.action_layout()) == ["continuous"]
    finally:
        env.close()


def test_a_switch_makes_the_action_space_a_dict_of_its_parts(mixed):
    space = mixed.action_space(None)
    assert isinstance(space, Dict)
    assert isinstance(space["continuous"], Box) and space["continuous"].shape == (3,)
    assert isinstance(space["binary"], MultiBinary) and space["binary"].n == 2


def test_a_grouped_action_is_applied_in_config_order(mixed):
    action = {"continuous": [0.1, 0.2, 0.3], "binary": [1, 0]}
    assert flatten_action(mixed.config, action).tolist() == pytest.approx(
        [0.1, 0.2, 1.0, 0.3, 0.0]
    )


@pytest.mark.parametrize(
    ("action", "error", "match"),
    [
        (np.zeros(5), TypeError, "Dict"),
        ({"continuous": [0.0] * 3}, ValueError, "parts"),
        ({"continuous": [0.0] * 2, "binary": [0, 0]}, ValueError, "continuous"),
    ],
    ids=["flat vector", "missing part", "wrong width"],
)
def test_an_action_of_the_wrong_shape_is_refused(mixed, action, error, match):
    with pytest.raises(error, match=match):
        flatten_action(mixed.config, action)


def test_a_dict_is_refused_where_the_space_is_one_box():
    config = EnvConfig(dt=12.0, obs_fields=[], action_fields=list(_CONTINUOUS))
    with pytest.raises(TypeError, match="one vector"):
        flatten_action(config, {"continuous": [0.0, 0.0]})


def test_a_sampled_action_steps(mixed, aircraft):
    mixed.step({aircraft: mixed.action_space(aircraft).sample()})


def test_a_switch_value_that_is_not_zero_or_one_is_refused(mixed, aircraft):
    action = {"continuous": [0.0, 0.0, 0.0], "binary": np.array([0.3, 0.0])}
    with pytest.raises(ValueError, match="AutopilotLnav.*0 or 1"):
        mixed.step({aircraft: action})


def test_the_previous_action_is_observed_in_config_order(mixed, aircraft):
    action = {"continuous": [0.6, 0.8, -0.5], "binary": [0, 1]}
    observed, *_ = mixed.step({aircraft: action})
    prev = mixed.observation_layout()["ownship"][-1]
    assert observed[aircraft]["ownship"][prev.columns].tolist() == pytest.approx(
        [0.6, 0.8, 0.0, -0.5, 1.0, 0.0]
    )


@pytest.mark.parametrize("fields", [_CONTINUOUS, _MIXED], ids=["box", "dict"])
def test_a_zero_action_fits_the_space(fields):
    env = _env(fields)
    try:
        assert env.action_space(None).contains(zero_action(env.action_space(None)))
    finally:
        env.close()


# --- layouts ---------------------------------------------------------------------
def _widths(space):
    """The number of feature columns in each part of a space."""
    if isinstance(space, Dict):
        return {key: _widths(part) for key, part in space.spaces.items()}
    if isinstance(space, MultiBinary):
        return space.n
    feature = getattr(space, "feature_space", space)
    return feature.shape[-1]


def _assert_tiles(part, width):
    starts = [slot.columns.start for slot in part]
    stops = [slot.columns.stop for slot in part]
    assert starts == [0, *stops[:-1]], "columns overlap or leave a gap"
    assert stops[-1] == width


def test_the_observation_layout_tiles_every_part(mixed):
    widths = _widths(mixed.observation_space(None))
    layout = mixed.observation_layout()
    assert set(layout) == set(widths)
    for key, part in layout.items():
        _assert_tiles(part, widths[key])
    assert [s.name for s in layout["ownship"]] == [
        "alt_ft",
        "flight_phase_one_hot",
        "prev_action_norm",
    ]


def test_the_action_layout_tiles_every_part(mixed):
    widths = _widths(mixed.action_space(None))
    layout = mixed.action_layout()
    for key, part in layout.items():
        _assert_tiles(part, widths[key])
    assert [s.name for s in layout["binary"]] == ["autopilot_lnav", "autopilot_vnav"]
    assert layout["continuous"][0].width == 2  # a circular heading: sin, cos


def test_a_repeated_field_is_named_by_its_occurrence():
    config = EnvConfig(
        dt=12.0, obs_fields=[obs.AltFt(), obs.CasKts(), obs.AltFt()], action_fields=[]
    )
    names = [slot.name for slot in observation_layout(config)["ownship"]]
    assert names == ["alt_ft", "cas_kts", "alt_ft#2"]


def test_padding_adds_the_validity_column_to_the_layout(mixed):
    padded = IntruderPaddingWrapper(mixed)
    layout = padded.observation_layout()
    assert layout["intruders"][-1].name == "valid"
    _assert_tiles(
        layout["intruders"], padded.observation_space(None)["intruders"].shape[-1]
    )


def test_the_critic_layout_appends_each_critic_part(mixed):
    layout = critic_observation_layout(mixed.observation_layout())
    assert set(layout) == {"ownship", "intruders"}
    assert [s.name for s in layout["intruders"]] == [
        "dist_to_own_nm",
        "alt_ft",
        "alt_ft#2",
        "tcpa_s",
    ]
    _assert_tiles(layout["intruders"], 4)
