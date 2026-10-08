"""The designer's MDP view draws every field's mapping: a step action's and a
switch's choices - points, not a curve - and a raw field's value, as it is
passed. No aircraft is needed."""

from __future__ import annotations

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import StepNormalizer
from bluesky_sandbox.ui.designer import mdp


def test_a_step_action_is_drawn_as_its_choices():
    field = act.AltDeltaFt(
        normalizer=StepNormalizer(steps_each_way=3, include_zero=False),
        grid=act.Grid(1000.0, on="target"),
    )
    raw = {"per_aircraft": True, "low": -1.0, "high": 1.0, "width": 1}
    # No aircraft is needed: each choice before any aircraft's reach clips it.
    curve = mdp._curve(field, field.normalizer, raw, "action")
    assert curve["discrete"] is True
    assert curve["x"] == [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    assert curve["series"] == [[-3000.0, -2000.0, -1000.0, 1000.0, 2000.0, 3000.0]]


def test_a_switch_is_drawn_as_its_two_choices():
    class _Slot:
        name = "autopilot_lnav_vnav"
        columns = slice(0, 1)

    out = mdp._field(act.AutopilotLnavVnav(), _Slot(), "action")
    assert out["binary"] is True
    assert out["curve"]["discrete"] is True
    assert out["curve"]["labels"] == ["off", "on"]
    assert out["curve"]["x"] == [0.0, 1.0]


def test_a_raw_observation_is_drawn_as_it_is_passed():
    class _Slot:
        name = "ap_lnav_on"
        columns = slice(0, 1)

    out = mdp._field(obs.ApLnavOn(), _Slot(), "observation")
    assert out["normalizer"] is None
    assert out["curve"]["x"] == out["curve"]["series"][0]  # as is
    assert out["curve"]["y_label"] == "policy sees (as is)"
