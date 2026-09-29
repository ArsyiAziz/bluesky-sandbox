"""A clearance in a design: one setting on an action, kept by the spec, built
into the parts it declares, and generated as ``act.Clearance``."""

from __future__ import annotations

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import build_design_config

from .test_designer import _example_design_spec


def _with_clearance() -> S.DesignSpec:
    spec = _example_design_spec()
    heading, speed = spec.env.action_fields
    heading.clearance = {"duration": [0, 600], "lock": "duration", "observe": True}
    speed.clearance = {"duration": None, "lock": None, "observe": False}
    return spec


def test_a_clearance_survives_the_saved_design():
    spec = _with_clearance()
    again = S.DesignSpec.from_dict(spec.to_dict())
    assert again.env.action_fields[0].clearance == spec.env.action_fields[0].clearance
    assert again.env.action_fields[1].clearance == spec.env.action_fields[1].clearance


def test_a_clearance_builds_into_its_parts():
    config = build_design_config(_with_clearance())
    kinds = [type(f).__name__ for f in config.action_fields]
    assert kinds == [
        "HdgDeg",
        "ClearanceDuration",
        "ActionMask",
        "SpdKts",
        "ActionMask",
    ]
    masks = [f for f in config.action_fields if isinstance(f, act.ActionMask)]
    assert masks[0].lock_for_duration and not masks[1].lock_for_duration
    names = [f.meta.name for f in config.obs_fields]
    assert names[-2:] == ["action_locked", "clearance_time_left_s"]


def test_a_clearance_is_generated_as_one():
    files = codegen.generate_task(_with_clearance(), "Cleared")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    assert (
        "act.Clearance(act.HdgDeg(), duration=(0.0, 600.0), lock='duration')" in config
    )
    assert "act.Clearance(act.SpdKts(" in config and "observe=False)" in config
