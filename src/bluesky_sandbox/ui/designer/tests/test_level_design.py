"""Flight levels in a design: an envelope marker's ``alt_step_ft`` survives the
saved design, reaches the per-aircraft fix draw and the spawn draw, and is
generated."""

from __future__ import annotations

import pytest

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.sim.performance.envelope import EnvelopeSample
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import build_design_config, build_scenario
from bluesky_sandbox.ui.designer.grid import apply_grid

from .test_designer import _example_design_spec

LEVELS = {"type": "envelope", "alt_step_ft": 1000.0}


def _on_levels() -> S.DesignSpec:
    spec = _example_design_spec()
    spec.queryables = {
        "fix": {
            "type": "waypoint",
            "lat": 56.0,
            "lon": 2.0,
            "alt_ft": dict(LEVELS),
            "speed_kts": None,
            "alt_tolerance_ft": 1000,
        }
    }
    spec.spawn["route"] = ["fix"]
    return spec


def test_a_level_grid_survives_the_saved_design():
    again = S.DesignSpec.from_json(_on_levels().to_json())
    assert again.queryables["fix"]["alt_ft"] == LEVELS
    assert S.dump_value(S.load_value(dict(LEVELS))) == LEVELS
    assert S.load_value(dict(LEVELS)) == EnvelopeSample(alt_step_ft=1000.0)


def test_a_fix_on_levels_is_drawn_on_levels():
    route_step = build_scenario(_on_levels()).support().spawn.route[0]
    assert route_step["sample_alt_from_envelope"] is True
    assert route_step["alt_step_ft"] == 1000.0


def test_levels_are_generated():
    files = codegen.generate_task(_on_levels(), "Levels")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "'alt_step_ft': 1000.0" in scenario


# ---- the design's grid ---------------------------------------------------- #


GRID = {"alt_ft": 1000.0, "spd_kts": 10.0, "hdg_deg": 10.0}


def _gridded() -> S.DesignSpec:
    spec = _on_levels()
    spec.grid = dict(GRID)
    spec.queryables["fix"]["alt_ft"] = {"type": "envelope"}  # no step of its own
    spec.queryables["cruise"] = {
        "type": "waypoint",
        "lat": 56.0,
        "lon": 2.5,
        "alt_ft": {"type": "start"},
        "speed_kts": None,
        "alt_tolerance_ft": 1000,
    }
    step = {
        "type": "normalizer",
        "name": "StepNormalizer",
        "kwargs": {"steps_each_way": 4},
    }
    spec.env.action_fields = [
        S.FieldRef(
            "AltDeltaFt", {"normalizer": dict(step, kwargs={"steps_each_way": 4})}
        ),
        S.FieldRef(
            "SpdDeltaKts", {"normalizer": dict(step, kwargs={"steps_each_way": 4})}
        ),
        S.FieldRef(
            "ApHdgDeltaDeg", {"normalizer": dict(step, kwargs={"steps_each_way": 4})}
        ),
    ]
    return spec


def test_a_grid_survives_the_saved_design_and_is_optional():
    again = S.DesignSpec.from_json(_gridded().to_json())
    assert again.grid == GRID
    assert S.DesignSpec.from_json(_on_levels().to_json()).grid is None
    assert apply_grid(_on_levels()) is not None  # no grid: a no-op


def test_the_grid_sets_every_step_and_the_altitude_levels():
    config = build_design_config(_gridded())
    alt, spd, hdg = config.action_fields
    assert (alt.normalizer.step, spd.normalizer.step, hdg.normalizer.step) == (
        1000.0,
        10.0,
        10.0,
    )
    assert alt.command_step == 1000.0  # every level change lands on a level
    assert getattr(spd, "command_step", None) is None  # speeds are steps, not rounded


def test_the_grid_puts_target_altitudes_on_levels_and_level_flight_holds_its_start():
    route_steps = {
        name: build_scenario(_gridded()).support().spawn.route[0] for name in ("fix",)
    }
    assert route_steps["fix"]["alt_step_ft"] == 1000.0
    spec = _gridded()
    spec.spawn["route"] = ["cruise"]
    (step,) = build_scenario(spec).support().spawn.route
    assert step["alt_from_start"] is True and step["alt_step_ft"] == 1000.0


def test_a_spawn_marked_levels_takes_the_grid_and_needs_one():
    spec = _gridded()
    spec.spawn.setdefault("regions", [{"name": "R", "params": {}}])
    spec.spawn["regions"][0].setdefault("params", {})["alt_ft"] = {
        "type": "envelope",
        "levels": True,
    }
    applied = apply_grid(spec)
    assert applied.spawn["regions"][0]["params"]["alt_ft"] == {
        "type": "envelope",
        "alt_step_ft": 1000.0,
    }
    spec.grid = {"spd_kts": 10.0}
    with pytest.raises(S.SpecError, match="no altitude grid"):
        apply_grid(spec)


def test_a_grid_is_generated_as_its_per_field_settings():
    files = codegen.generate_task(_gridded(), "Gridded")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    assert (
        "StepNormalizer(steps_each_way=4, step=1000.0), command_step=1000.0" in config
    )
    assert "StepNormalizer(steps_each_way=4, step=10.0))" in config


@pytest.mark.parametrize("bad", [{"alt": 1000}, {"alt_ft": -5}, {"alt_ft": "x"}])
def test_a_bad_grid_is_refused(bad):
    with pytest.raises(S.SpecError):
        S.validated_grid(bad)


def test_spawning_on_levels_without_an_altitude_grid_is_refused():
    spec = _on_levels()  # no grid at all
    spec.spawn.setdefault("regions", [{"name": "R", "params": {}}])
    spec.spawn["regions"][0].setdefault("params", {})["alt_ft"] = {
        "type": "envelope",
        "levels": True,
    }
    with pytest.raises(S.SpecError, match="no altitude grid"):
        build_scenario(spec)


def test_an_action_the_grid_cannot_read_is_an_error_not_skipped():
    spec = _gridded()
    spec.env.action_fields.append(S.FieldRef("NoSuchAction", {}))
    with pytest.raises(S.SpecError, match="cannot read action 'NoSuchAction'"):
        apply_grid(spec)


def test_a_custom_action_takes_the_grid_too():
    spec = _gridded()
    spec.code["custom_actions.py"] = (
        "from dataclasses import dataclass\n"
        "from bluesky_sandbox.interface.fields import actions as act\n\n\n"
        "@dataclass(frozen=True)\n"
        "class MyLevels(act.AltDeltaFt):\n"
        "    pass\n"
    )
    spec.env.action_fields.append(
        S.FieldRef(
            "custom_actions:MyLevels",
            {
                "normalizer": {
                    "type": "normalizer",
                    "name": "StepNormalizer",
                    "kwargs": {},
                }
            },
        )
    )
    applied = apply_grid(spec).env.action_fields[-1]
    assert applied.kwargs["normalizer"]["kwargs"]["step"] == 1000.0
    assert applied.kwargs["command_step"] == 1000.0
