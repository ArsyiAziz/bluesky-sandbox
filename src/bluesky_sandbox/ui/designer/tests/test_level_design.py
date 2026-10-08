"""Flight levels in a design, each set where it applies - nothing global fills
anything in: an envelope or start altitude's ``alt_step_ft`` (a spawn's, a
fix's), an action's ``grid``, a step normalizer's ``step``. Each
survives the saved design, reaches the env, and is generated as written."""

from __future__ import annotations

import pytest

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.sim.performance.envelope import EnvelopeSample
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import build_design_config, build_scenario

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


# ---- level flight, spawns on levels, actions on a grid --------------------- #


def _cruise() -> S.DesignSpec:
    spec = _on_levels()
    spec.queryables["cruise"] = {
        "type": "waypoint",
        "lat": 56.0,
        "lon": 2.5,
        "alt_ft": {"type": "start", "alt_step_ft": 1000.0},
        "speed_kts": None,
        "alt_tolerance_ft": 1000,
    }
    spec.spawn["route"] = ["cruise"]
    region = spec.spawn["regions"][0]
    region.setdefault("params", {})["alt_ft"] = dict(LEVELS)
    return spec


def _step(steps: int, step: float) -> dict:
    return {
        "type": "normalizer",
        "name": "StepNormalizer",
        "kwargs": {"steps_each_way": steps, "step": step},
    }


def test_level_flight_holds_the_level_it_starts_on():
    (step,) = build_scenario(_cruise()).support().spawn.route
    assert step["alt_from_start"] is True and step["alt_step_ft"] == 1000.0
    again = S.DesignSpec.from_json(_cruise().to_json())
    assert again.queryables["cruise"]["alt_ft"] == {"type": "start", "alt_step_ft": 1000.0}


def test_a_spawn_on_levels_is_drawn_on_levels_and_generated():
    spec = _cruise()
    region = S.DesignSpec.from_json(spec.to_json()).spawn["regions"][0]
    assert region["params"]["alt_ft"] == LEVELS
    files = codegen.generate_task(spec, "Cruise")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "EnvelopeSample(alt_step_ft=1000.0)" in scenario


def test_an_action_is_built_with_the_grid_and_step_it_states():
    spec = _on_levels()
    spec.env.action_fields = [
        # On a grid, the step counts grid steps: two levels a choice.
        S.FieldRef("AltDeltaFt", {"normalizer": _step(4, 2), "grid": {"type": "grid", "step": 1000.0, "on": "target"}}),
        S.FieldRef("SpdDeltaKts", {"normalizer": _step(4, 10.0)}),
        S.FieldRef("HdgDeltaDeg", {"grid": {"type": "grid", "step": 10.0}}),
    ]
    spec = S.DesignSpec.from_json(spec.to_json())
    alt, spd, hdg = build_design_config(spec).action_fields
    assert (alt.normalizer.step, alt.grid) == (2.0, act.Grid(1000.0, on="target"))
    # Nothing it does not state: a speed in steps, not put on a grid.
    assert (spd.normalizer.step, spd.grid) == (10.0, None)
    assert hdg.grid == act.Grid(10.0)
    files = codegen.generate_task(spec, "Steps")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    assert (
        "StepNormalizer(steps_each_way=4, step=2), "
        "grid=act.Grid(1000.0, on='target')"
    ) in config
    assert "StepNormalizer(steps_each_way=4, step=10.0))" in config
    assert "act.HdgDeltaDeg(grid=act.Grid(10.0))" in config


@pytest.mark.parametrize("old", ["command_step", "target_grid"])
def test_an_action_saved_with_an_earlier_grid_loads_on_its_target_grid(old):
    ref = S.FieldRef.from_dict({"field": "AltDeltaFt", "kwargs": {old: 1000.0}})
    assert ref.kwargs == {"grid": {"type": "grid", "step": 1000.0, "on": "target"}}


@pytest.mark.parametrize(("feet", "count"), [(1000.0, None), (2000.0, 2)])
def test_a_step_saved_in_feet_on_an_earlier_grid_loads_as_the_grid_steps_it_is(feet, count):
    ref = S.FieldRef.from_dict(
        {"field": "AltDeltaFt", "kwargs": {"target_grid": 1000.0, "normalizer": _step(4, feet)}}
    )
    assert ref.kwargs["normalizer"]["kwargs"].get("step") == count


def test_a_design_with_a_global_grid_is_refused_with_where_each_step_goes():
    data = _on_levels().to_dict()
    data["grid"] = {"alt_ft": 1000}
    with pytest.raises(S.SpecError, match="alt_step_ft.*grid.*step"):
        S.DesignSpec.from_dict(data)
