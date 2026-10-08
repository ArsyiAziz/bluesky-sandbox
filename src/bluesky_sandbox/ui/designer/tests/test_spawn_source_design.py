"""A design's spawn sources: each a ``plan(rng, ctx)`` its code writes - any data
it reads, any distribution it draws - with the source's policies. Built for the
preview and the environment, generated into the package, completed and checked
in the editor, as the code API has them."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from bluesky_sandbox.sim.spawn import PlannedSource, nearest_entry
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import BuildError, build_scenario
from bluesky_sandbox.ui.designer.code_intel import code_intel
from bluesky_sandbox.ui.designer.diagnostics import diagnostics
from bluesky_sandbox.ui.designer.preview import scenario_preview

from .test_designer import _example_design_spec

PLAN = (
    "times = TIMES\n"
    "return [SpawnRequest(at=LatLon(52.0, 4.6), alt_ft=20_000, spd_kts=250, time_s=t) for t in times]\n"
)


def _design(**source) -> S.DesignSpec:
    spec = _example_design_spec()
    spec.scenario_setup = "TIMES = (0.0, 30.0)"
    spec.spawn = {
        **spec.spawn,
        "sources": [{"name": "adsb", "plan": PLAN, "max_aircraft": 2, "when_blocked": "skip", **source}],
    }
    return S.DesignSpec.from_json(spec.to_json())


def test_it_is_built_with_its_plan_and_policies():
    spawn = build_scenario(_design(assign_route="nearest_entry")).sample(np.random.default_rng(0)).spawn
    (source,) = spawn.sources
    assert isinstance(source, PlannedSource) and source.name == "adsb"
    assert (source.when_blocked, source.max_aircraft, source.assign_route) == ("skip", 2, nearest_entry)
    assert [r.time_s for r in source.plan(np.random.default_rng(0), None)] == [0.0, 30.0]


def test_the_preview_shows_its_aircraft():
    preview = scenario_preview(_design(), seed=0)
    sourced = [a for a in preview["sampled_aircraft"] if a["source"] == "adsb"]
    assert [a["spawn_time"] for a in sourced] == [0.0, 30.0]
    assert (sourced[0]["lat"], sourced[0]["lon"]) == (52.0, 4.6)


def test_a_mistake_in_its_plan_names_it_and_its_line():
    spec = _design(plan="x = 1\nreturn [undefined_name]\n")
    scenario = build_scenario(spec)
    with pytest.raises(BuildError, match="spawn source adsb.*line 2"):
        scenario_preview(spec, seed=0)
    assert scenario is not None


def test_its_name_must_be_an_identifier():
    with pytest.raises(BuildError, match="identifier"):
        build_scenario(_design(name="ads b"))


def test_the_editor_knows_its_parameters_and_checks_its_names():
    intel = code_intel(_design())
    params = {p["name"] for p in intel["scopes"]["spawn_source:adsb"]["params"]}
    assert params == {"rng", "ctx"}
    problems = diagnostics(_design(plan="return [nowhere]\n"))
    assert any("nowhere" in p["message"] for p in problems["spawn_source:adsb"])


def test_the_generated_package_plans_the_same(tmp_path):
    files = codegen.generate_task(_design(), "Sourced")
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    text = files[scenario_py]
    assert "def _plan_adsb(rng, ctx):" in text
    assert "PlannedSource(_plan_adsb, name='adsb', when_blocked='skip', max_aircraft=2)" in text
    assert "sources=SPAWN_SOURCES" in text
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    (source,) = cls().sample(np.random.default_rng(0)).spawn.sources
    assert [r.time_s for r in source.plan(np.random.default_rng(0), None)] == [0.0, 30.0]


def test_the_environment_spawns_its_aircraft():
    from bluesky_sandbox.env import BlueskyEnv
    from bluesky_sandbox.ui.designer.builder import build_design_config

    spec = _design()
    env = BlueskyEnv(scenario=build_scenario(spec), config=build_design_config(spec))
    try:
        env.reset(seed=0)
        (first,) = [r for r in env.spawn_log if r.source == "adsb"]  # the one due at 0 s
        assert (round(first.lat_deg, 6), round(first.lon_deg, 6)) == (52.0, 4.6)
    finally:
        env.close()
