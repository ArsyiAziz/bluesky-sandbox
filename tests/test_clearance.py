"""``actions.Clearance``: one declaration per action, expanded by ``EnvConfig``
into the parts the environment runs on - the action, its duration, its mask.
What the policy observes of it stays the design's choice."""

from __future__ import annotations

import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import MinMaxNormalizer


def _config(action_fields, obs_fields=()) -> EnvConfig:
    return EnvConfig(dt=5.0, obs_fields=list(obs_fields), action_fields=action_fields)


def test_a_clearance_is_the_parts_it_declares():
    declared = _config(
        [
            act.Clearance(
                act.HdgDeg(),
                duration=(0, 600),
                lock="duration",
                duration_normalizer=MinMaxNormalizer(),
            ),
            act.Clearance(act.AltFt(), lock="captured"),
            act.Clearance(act.AutopilotLnavVnav()),
        ],
        [obs.CasKts()],
    )
    wired = _config(
        [
            act.HdgDeg(),
            act.ClearanceDuration(
                target=act.HdgDeg, low=0.0, high=600.0, normalizer=MinMaxNormalizer()
            ),
            act.ActionMask(target=act.HdgDeg, lock_for_duration=True),
            act.AltFt(),
            act.ActionMask(target=act.AltFt, lock_until_captured=True),
            act.AutopilotLnavVnav(),
            act.ActionMask(target=act.AutopilotLnavVnav),
        ],
        [obs.CasKts()],
    )
    assert [type(f) for f in declared.action_fields] == [
        type(f) for f in wired.action_fields
    ]
    assert declared.action_fields[3:] == wired.action_fields[3:]
    duration = declared.action_fields[1]
    assert (duration.target, duration.low, duration.high) == ("hdg_deg", 0.0, 600.0)
    assert isinstance(duration.normalizer, MinMaxNormalizer)
    # Nothing is observed unless the design asks for it.
    assert declared.obs_fields == [obs.CasKts()]


def test_expanding_twice_changes_nothing():
    config = _config([act.Clearance(act.HdgDeg(), duration=(0, 600), lock="duration")])
    again = EnvConfig(
        dt=5.0, obs_fields=config.obs_fields, action_fields=config.action_fields
    )
    assert again.action_fields == config.action_fields
    assert again.obs_fields == config.obs_fields


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"lock": "duration"}, ValueError, "needs a duration"),
        ({"lock": "sometimes"}, ValueError, "lock must be"),
        ({"duration": (300, 60)}, ValueError, "0 <= low <= high"),
        ({"duration": (-5, 60)}, ValueError, "0 <= low <= high"),
        (
            {"duration_normalizer": MinMaxNormalizer()},
            ValueError,
            "duration_normalizer needs a duration",
        ),
    ],
    ids=[
        "lock without duration",
        "unknown lock",
        "reversed",
        "negative",
        "normalizer without duration",
    ],
)
def test_a_clearance_that_cannot_be_made_is_refused(kwargs, error, match):
    with pytest.raises(error, match=match):
        act.Clearance(act.HdgDeg(), **kwargs)


def test_a_clearance_wraps_an_action_and_only_one():
    with pytest.raises(TypeError, match="wraps an action field"):
        act.Clearance(obs.CasKts())
    with pytest.raises(TypeError, match="wraps an action field"):
        act.Clearance(act.ActionMask(target=act.HdgDeg))


def test_a_duration_on_an_axis_own_navigation_does_not_fly_is_refused():
    with pytest.raises(ValueError, match="commands no axis"):
        _config([act.Clearance(act.CommBroadcast(), duration=(0, 60))])
