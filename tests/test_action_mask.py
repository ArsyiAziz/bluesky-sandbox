"""Action masks: a 0/1 action that, set to 1, skips another action that step.

These pin how a mask names its target, which configs refuse one, what the
dispatcher skips - a switch included, even one another switch requires - and
what the policy and the trainer see: the mask it set last, in the observation,
and which values of the action took effect, in the info.
"""

from __future__ import annotations

from types import SimpleNamespace

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import action_applied
from bluesky_sandbox.core.services import ActionDispatcher
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields._common import reset_field_state
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui.designer import catalog


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


class _One:
    """One B744 heading east."""

    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
            n_aircraft=1,
            params={
                "alt_ft": (20_000, 20_000),
                "spd_kts": (250, 250),
                "hdg_deg": (90, 90),
            },
        )
        spawn = SpawnConfig(
            regions=[region], aircraft_type="B744", conflict_free_spawn=False
        )
        return EpisodeSpec(
            airspace_bounds=None, spawn=spawn, queryables={}, max_aircraft=1
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
    assert bs.traf.cre("MK001", "B744", 52.0, 4.5, 90, 12_000 * ft, 250 * kts) is True
    idx = bs.traf.id.index("MK001")
    bs.traf.swlnav[idx] = False
    bs.traf.swvnav[idx] = False
    return idx


@pytest.fixture
def sent(monkeypatch):
    """The stack commands sent, captured instead of executed."""
    commands: list[str] = []
    monkeypatch.setattr(bs.stack, "stack", lambda *lines, **_kw: commands.extend(lines))
    return commands


def _dispatch(fields, idx, values) -> list[bool]:
    config = SimpleNamespace(action_fields=fields)
    return ActionDispatcher(SimpleNamespace(config=config)).apply(
        idx, np.asarray(values)
    )


def _config(action_fields, obs_fields=()) -> EnvConfig:
    return EnvConfig(dt=5.0, obs_fields=list(obs_fields), action_fields=action_fields)


# ---- naming the target ----------------------------------------------------- #


@pytest.mark.parametrize(
    "target", [act.HdgDeg, act.HdgDeg(), "hdg_deg"], ids=["class", "instance", "name"]
)
def test_a_target_is_named_by_its_class_an_instance_or_its_name(target):
    mask = act.ActionMask(target=target)
    assert mask.target == "hdg_deg"
    assert mask == act.ActionMask(target="hdg_deg")
    assert obs.PrevActionMasked(target=target).target == "hdg_deg"


@pytest.mark.parametrize("target", [3, obs.AltFt, obs.AltFt(), None])
def test_a_target_that_is_not_an_action_is_refused(target):
    with pytest.raises(TypeError, match="an action is named by"):
        act.ActionMask(target=target)


def test_a_mask_cannot_mask_a_mask():
    with pytest.raises(ValueError, match="cannot mask another ActionMask"):
        act.ActionMask(target=act.ActionMask)


def test_a_mask_is_in_the_binary_part_of_the_action_space(env):
    env.config.action_fields.clear()
    env.config.action_fields.extend([act.HdgDeg(), act.ActionMask(target=act.HdgDeg)])
    try:
        space = env.action_space(None)
        assert space["continuous"].shape == (1,) and space["binary"].n == 1
    finally:
        env.config.action_fields.clear()


# ---- configs that cannot be masked ----------------------------------------- #


@pytest.mark.parametrize(
    ("action_fields", "match"),
    [
        ([act.HdgDeg(), act.ActionMask()], "needs a target"),
        ([act.HdgDeg(), act.ActionMask(target=act.AltFt)], "not an action"),
        (
            [
                act.HdgDeg(),
                act.HdgDeg(low=0.0, high=90.0),
                act.ActionMask(target=act.HdgDeg),
            ],
            "names 2 actions",
        ),
        (
            [
                act.HdgDeg(),
                act.ActionMask(target=act.HdgDeg),
                act.ActionMask(target="hdg_deg"),
            ],
            "more than one ActionMask",
        ),
    ],
    ids=["no target", "unknown", "ambiguous", "masked twice"],
)
def test_a_config_whose_masks_name_no_one_action_is_refused(action_fields, match):
    with pytest.raises(ValueError, match=match):
        _config(action_fields)


# ---- what the dispatcher skips --------------------------------------------- #


def test_a_masked_action_is_skipped_and_an_unmasked_one_applied(aircraft, sent):
    fields = [act.HdgDeg(), act.ActionMask(target=act.HdgDeg)]
    assert _dispatch(fields, aircraft, [45.0, 1.0]) == [False, True]
    assert sent == []
    assert _dispatch(fields, aircraft, [45.0, 0.0]) == [True, True]
    assert [c.split()[0] for c in sent] == ["HDG"]


def test_a_mask_leaves_the_other_actions_alone(aircraft, sent):
    fields = [act.HdgDeg(), act.AltFt(), act.ActionMask(target=act.HdgDeg)]
    assert _dispatch(fields, aircraft, [45.0, 15_000.0, 1.0]) == [False, True, True]
    assert [c.split()[0] for c in sent] == ["ALT"]


def test_a_masked_switch_keeps_its_state_and_suppresses_nothing(aircraft, sent):
    fields = [
        act.AutopilotLnav(),
        act.HdgDeg(),
        act.ActionMask(target=act.AutopilotLnav),
    ]
    assert _dispatch(fields, aircraft, [1.0, 45.0, 1.0]) == [False, True, True]
    assert [c.split()[0] for c in sent] == ["HDG"]


def test_a_masked_switch_stays_off_even_when_a_switch_requires_it(aircraft, sent):
    # VNAV requires LNAV; masked, LNAV is not turned on with it: the mask wins.
    fields = [
        act.AutopilotLnav(),
        act.AutopilotVnav(),
        act.ActionMask(target=act.AutopilotLnav),
    ]
    _dispatch(fields, aircraft, [0.0, 1.0, 1.0])
    assert sent == ["VNAV MK001 ON"]


def test_an_action_a_switch_suppresses_is_reported_unapplied(aircraft, sent):
    assert _dispatch([act.AutopilotLnav(), act.HdgDeg()], aircraft, [1.0, 45.0]) == [
        True,
        False,
    ]


def test_masks_are_kept_apart_though_they_share_a_name(aircraft, sent):
    fields = [
        act.HdgDeg(),
        act.AltFt(),
        act.ActionMask(target=act.HdgDeg),
        act.ActionMask(target=act.AltFt),
    ]
    assert _dispatch(fields, aircraft, [45.0, 15_000.0, 0.0, 1.0]) == [
        True,
        False,
        True,
        True,
    ]
    assert [c.split()[0] for c in sent] == ["HDG"]


# ---- what a mask remembers ------------------------------------------------- #


def test_a_mask_remembers_what_it_was_set_to_per_aircraft(aircraft):
    mask = act.ActionMask(target=act.HdgDeg)
    seen = obs.PrevActionMasked(target=act.HdgDeg)
    assert not mask.current_switch_state(aircraft) and seen.get(aircraft) == 0.0
    mask.set(aircraft, 1.0)
    assert mask.current_switch_state(aircraft) and seen.get(aircraft) == 1.0
    mask.set(aircraft, 0.0)
    assert seen.get(aircraft) == 0.0


def test_a_mask_is_forgotten_when_its_aircraft_leaves_or_the_episode_resets(aircraft):
    mask = act.ActionMask(target=act.HdgDeg)
    mask.set(aircraft, 1.0)
    mask.on_aircraft_removed("MK001")
    assert not mask.current_switch_state(aircraft)
    mask.set(aircraft, 1.0)
    reset_field_state(0)
    assert not mask.current_switch_state(aircraft)


def test_the_masks_and_what_reads_them_are_dropped_on_despawn():
    assert act.ActionMask.is_stateful() and obs.PrevActionMasked.is_stateful()


# ---- what the trainer sees ------------------------------------------------- #


def test_the_applied_values_are_shaped_as_the_action():
    config = _config([act.HdgDeg(), act.AltFt(), act.ActionMask(target=act.HdgDeg)])
    out = action_applied(config, [False, True, True])
    assert set(out) == {"continuous", "binary"}
    np.testing.assert_array_equal(out["continuous"], [0.0, 1.0])
    np.testing.assert_array_equal(out["binary"], [1.0])
    continuous = _config([act.HdgDeg(), act.AltFt()])
    np.testing.assert_array_equal(action_applied(continuous), [1.0, 1.0])


@pytest.fixture
def flying():
    env = BlueskyEnv(
        scenario=_One(),
        config=_config(
            [act.HdgDeg(), act.ActionMask(target=act.HdgDeg)],
            [obs.PrevActionMasked(target=act.HdgDeg)],
        ),
    )
    yield env
    env.close()


def _act(heading: float, mask: int) -> dict:
    return {
        "continuous": np.array([heading], dtype=np.float32),
        "binary": np.array([mask], dtype=np.int8),
    }


def test_a_masked_heading_lets_the_aircraft_fly_out_its_last_vector(flying):
    observations, infos = flying.reset(seed=0)
    (agent,) = observations
    np.testing.assert_array_equal(observations[agent], [0.0])
    np.testing.assert_array_equal(infos[agent]["action_applied"]["continuous"], [1.0])

    observations, _, _, _, infos = flying.step({agent: _act(180.0, 0)})
    idx = bs.traf.id.index(agent)
    assert bs.traf.ap.trk[idx] == pytest.approx(180.0)
    np.testing.assert_array_equal(infos[agent]["action_applied"]["continuous"], [1.0])

    # Masked, the 0 is never sent: the aircraft keeps turning to 180.
    observations, _, _, _, infos = flying.step({agent: _act(0.0, 1)})
    idx = bs.traf.id.index(agent)
    assert bs.traf.ap.trk[idx] == pytest.approx(180.0)
    np.testing.assert_array_equal(observations[agent], [1.0])
    np.testing.assert_array_equal(infos[agent]["action_applied"]["continuous"], [0.0])
    np.testing.assert_array_equal(infos[agent]["action_applied"]["binary"], [1.0])


def test_without_a_mask_the_info_is_as_before():
    env = BlueskyEnv(scenario=_One(), config=_config([act.HdgDeg()]))
    try:
        observations, infos = env.reset(seed=0)
        (agent,) = observations
        assert "action_applied" not in infos[agent]
        _, _, _, _, infos = env.step({agent: np.array([90.0], dtype=np.float32)})
        assert "action_applied" not in infos[agent]
    finally:
        env.close()


# ---- in the designer ------------------------------------------------------- #


def test_the_designer_offers_a_mask_with_its_target_to_type_in():
    (mask,) = [f for f in catalog.action_fields() if f["name"] == "ActionMask"]
    assert mask["kind"] == "binary"
    assert [p for p in mask["params"] if p["name"] == "target"][0]["default"] == ""
    (seen,) = [f for f in catalog.obs_fields() if f["name"] == "PrevActionMasked"]
    assert "target" in [p["name"] for p in seen["params"]]
