"""Scenario code - its setup and the ``episode_geometry`` hook - has the same
library names in scope in the designer as in the generated ``scenario.py``,
and the editor knows the hook's parameters: the geometry's keys, the design's
queryable and shape names under them, and the episode's random generator."""

from __future__ import annotations

import importlib
import sys

import numpy as np

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.builder import build_scenario
from bluesky_sandbox.ui.designer.code_intel import code_intel
from bluesky_sandbox.ui.designer.diagnostics import diagnostics

from .test_bounds_design import _with_regions

# A hook that builds geometry from library types it never imports.
_HOOK = (
    'hold = geometry["shapes"]["hold"]\n'
    "far = RegionBounds(PointFootprint(LatLon(hold.center.lat_deg + 0.1, hold.center.lon_deg)))\n"
    'geometry["shapes"] = {**geometry["shapes"], "far": far}\n'
    "return geometry"
)


def _with_hook():
    spec = _with_regions()
    spec.scenario_hooks = {"episode_geometry": _HOOK}
    return spec


def test_the_editor_knows_the_hooks_parameters():
    intel = code_intel(_with_regions())
    params = {p["name"]: p["type"] for p in intel["scopes"]["scenario:episode_geometry"]["params"]}
    assert params["rng"].endswith(".Generator")
    keys = {i["name"]: i.get("type") for i in intel["types"][params["geometry"]]["items"]}
    assert keys["queryables"] == "design:queryable" and keys["shapes"] == "design:shapes"
    assert keys["spawn"].endswith(".SpawnConfig")
    shapes = [i["name"] for i in intel["types"]["design:shapes"]["items"]]
    assert shapes == ["core", "hold"]


def test_scenario_code_has_the_library_names_in_scope():
    intel = code_intel(_with_regions())
    names = {m["name"] for m in intel["names"]["scenario"]}
    assert {"LatLon", "RegionBounds", "PointFootprint", "SpawnRegion", "Waypoint"} <= names
    assert "scenario:episode_geometry" not in diagnostics(_with_hook())


def test_the_designer_runs_the_hook_as_the_generated_package_does(tmp_path):
    episode = build_scenario(_with_hook()).sample(np.random.default_rng(0))
    assert episode.bounds["far"].center.lat_deg == episode.bounds["hold"].center.lat_deg + 0.1
    files = codegen.generate_task(_with_hook(), "Hooked")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    generated = cls().sample(np.random.default_rng(0))
    assert generated.bounds["far"].center == episode.bounds["far"].center


def test_a_hook_using_the_shapes_older_name_still_works():
    spec = _with_regions()
    spec.scenario_hooks = {"episode_geometry": _HOOK.replace('"shapes"', '"bounds"')}
    episode = build_scenario(spec).sample(np.random.default_rng(0))
    assert "far" in episode.shapes
