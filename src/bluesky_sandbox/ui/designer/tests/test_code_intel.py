"""What the code editor knows, read from the code and the design.

Types come from annotations, each block's names from what runs there, and the
keys a design fixes from ``DesignKeys`` markers - so the editor completes
``context.obs["intruders"]["acid"]`` and ``context.query("wp").current``
without the designer naming any of them.
"""

from __future__ import annotations

import json
from typing import get_args, get_type_hints

import pytest

from bluesky_sandbox.interface.task import AgentStepContext, DesignKeys
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.ui.designer import design_keys as design_keys_module
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.code_intel import code_intel, hints

from .test_designer import _example_design_spec


@pytest.fixture(scope="module")
def intel():
    design = _example_design_spec()
    design.queryables["wp"] = S.dump(
        Waypoint(lat=52.0, lon=4.5, alt_ft=3000, color="magenta")
    )
    design.env.hook_setup = (
        "import numpy as np\nSCALE = np.ones(1)\ndef _half(x, by=2):\n    return x / by"
    )
    design.env.task_info_setup = "from math import sqrt as root\nLIMIT = root(4)"
    return code_intel(design)


def _type(intel, key):
    return intel["types"][key]


def _attr(intel, type_key, name):
    return next(m for m in _type(intel, type_key)["attrs"] if m["name"] == name)


def _item(intel, type_key, name):
    return next(m for m in _type(intel, type_key)["items"] if m["name"] == name)


def _param(intel, scope, name):
    return next(p for p in intel["scopes"][scope]["params"] if p["name"] == name)


def _context(intel):
    return _param(intel, "hook:reward", "context")["type"]


def test_it_is_plain_json(intel):
    assert json.loads(json.dumps(intel)) == intel


def test_a_hook_has_its_signature_with_types(intel):
    params = {p["name"]: p for p in intel["scopes"]["hook:reward"]["params"]}
    assert list(params) == [
        "self",
        "obs",
        "action",
        "terminated",
        "truncated",
        "context",
        "info",
        "rng",
    ]
    assert params["terminated"]["detail"] == "bool"
    assert "ndarray" in params["obs"]["detail"]
    rng = {m["name"] for m in _type(intel, params["rng"]["type"])["attrs"]}
    assert {"normal", "uniform"} <= rng


def test_the_info_dict_has_its_keys_as_items(intel):
    info = _param(intel, "task_info", "info")["type"]
    assert {"acid", "acidx"} <= {m["name"] for m in _type(intel, info)["items"]}


def test_hooks_and_task_info_share_the_setup_modules_names(intel):
    setup_names = {m["name"] for m in intel["names"][intel["scopes"]["task_info"]["names"]]}
    hook_names = {m["name"] for m in intel["names"][intel["scopes"]["hook:reward"]["names"]]}
    assert setup_names <= hook_names


def test_a_hook_body_has_the_types_its_hooks_name(intel):
    # A readout row, a control state: no import line needed.
    hook_names = {m["name"] for m in intel["names"][intel["scopes"]["hook:reward"]["names"]]}
    assert {"AircraftReadoutItem", "AircraftControlState", "WaypointReadoutItem"} <= hook_names


def test_the_setup_modules_names_are_described(intel):
    setup = {m["name"]: m for m in intel["names"]["setup"]}
    assert {"np", "SCALE", "root", "LIMIT", "CONFIG", "_half"} <= set(setup)
    assert setup["np"]["module"] == "numpy"
    assert [p["name"] for p in setup["_half"]["params"]] == ["x", "by"]
    assert setup["_half"]["params"][1]["default"] == "2"


def test_raw_observation_keys_come_from_the_design(intel):
    raw_obs = _attr(intel, _context(intel), "obs")["type"]
    parts = {m["name"]: m["type"] for m in _type(intel, raw_obs)["items"]}
    assert list(parts) == ["ownship", "intruders"]
    ownship = {m["name"]: m for m in _type(intel, parts["ownship"])["items"]}
    assert list(ownship) == ["lat_deg", "lon_deg", "alt_ft"]
    assert ownship["alt_ft"]["detail"] == "float · ft"
    assert "MinMaxNormalizer" in ownship["alt_ft"]["doc"]
    intruders = [m["name"] for m in _type(intel, parts["intruders"])["items"]]
    assert intruders == ["acid", "dist_to_own_nm"]


def test_action_keys_come_from_the_design(intel):
    action = _attr(intel, _context(intel), "action")["type"]
    assert [m["name"] for m in _type(intel, action)["items"]] == [
        "hdg_deg",
        "spd_kts",
    ]


def test_a_keyed_query_returns_that_queryables_result(intel):
    query = _attr(intel, _context(intel), "query")
    assert query["params"][0]["keys"] == query["returns_by_key"]
    wp = _item(intel, query["returns_by_key"], "wp")
    assert wp["color"] == "magenta"
    current = _attr(intel, wp["type"], "current")
    assert {"distance_nm", "bearing_deg", "satisfied"} <= {
        m["name"] for m in _type(intel, current["type"])["attrs"]
    }
    queryable = _attr(intel, _context(intel), "queryable")
    goal = _item(intel, queryable["returns_by_key"], "goal")
    assert _type(intel, goal["type"])["name"] == "QueryRegion"


def test_a_querys_step_and_time_complete_as_the_result_not_its_placeholder(intel):
    # Each is typed Result | Placeholder (raising without temporal tracking):
    # alike, so described as the result.
    query = _attr(intel, _context(intel), "query")
    for key, step in (("goal", {"inside"}), ("wp", {"satisfied", "reached", "min_distance_nm"})):
        result = _item(intel, query["returns_by_key"], key)["type"]
        assert step <= {m["name"] for m in _type(intel, _attr(intel, result, "step")["type"])["attrs"]}
        time = _type(intel, _attr(intel, result, "time")["type"])
        assert time["name"] == "StepTime"
        assert {m["name"] for m in time["attrs"]} == {"total_s", "during_step_s"}


def test_a_type_checking_import_resolves(intel):
    # AgentStepContext.airspace is annotated with a name imported only for type
    # checkers; the editor still knows it.
    airspace = _attr(intel, _context(intel), "airspace")
    assert _type(intel, airspace["type"])["name"] == "RegionResult"
    assert "RegionResult" in str(hints(AgentStepContext)["airspace"])


def test_a_batched_hook_sees_stacked_shapes(intel):
    batch = _param(intel, "hook:reward_batch", "batch")["type"]
    raw_obs = _attr(intel, batch, "obs")["type"]
    intruders = next(
        m for m in _type(intel, raw_obs)["items"] if m["name"] == "intruders"
    )
    distance = _item(intel, intruders["type"], "dist_to_own_nm")
    assert distance["detail"] == "ndarray (n_agents, n_intruders) · nm"


def test_private_members_stay_out(intel):
    names = {m["name"] for m in _type(intel, _context(intel))["attrs"]}
    assert not {n for n in names if n.startswith("_")}


def test_every_design_key_source_is_described():
    # A new DesignKeys source must be described, or the editor and the
    # generated types would silently offer nothing for it.
    sources = get_args(get_type_hints(DesignKeys)["source"])
    assert "observation" in sources
    assert set(sources) == set(design_keys_module._SOURCES)


def _shapes_intel():
    from .test_generated_regions import _design

    spec = _design()
    # Of a sampled size: the support holds the envelope of every size, a
    # polygon; each episode an annular sector.
    spec.regions["stream"] = {
        "type": "region",
        "footprint": {
            "type": "annular_sector",
            "center": {"lat_deg": 52.0, "lon_deg": 4.5},
            "inner_radius_nm": 5,
            "outer_radius_nm": 20,
            "bearing_deg": 90,
            "half_angle_deg": {"type": "range", "low": 10, "high": 30},
        },
    }
    return spec, code_intel(spec)


def _footprint_of(intel, scope_type, shape):
    keyed = _attr(intel, scope_type, "shape")["returns_by_key"]
    region = _item(intel, keyed, shape)["type"]
    return _type(intel, _attr(intel, region, "footprint")["type"])


def test_a_shape_completes_its_footprint_as_its_episodes_hold_it():
    from bluesky_sandbox.ui.designer.builder import build_scenario
    from bluesky_sandbox.ui.designer.code_intel import _sampled_episodes, _shared_class

    spec, intel = _shapes_intel()
    context = _context(intel)
    core = _footprint_of(intel, context, "core")
    assert core["name"] == "BoxFootprint"
    assert {"lat_min_deg", "lon_max_deg"} <= {m["name"] for m in core["attrs"]}
    stream = _footprint_of(intel, context, "stream")
    assert stream["name"] == "AnnularSectorFootprint"
    assert {"inner_radius_nm", "half_angle_deg"} <= {m["name"] for m in stream["attrs"]}
    # A generated shape: as far as its draws agree.
    drawn = tuple(e.shapes["wx"].footprint for e in _sampled_episodes(build_scenario(spec)))
    assert _footprint_of(intel, context, "wx")["name"] == _shared_class(drawn).__name__
    # shapes["name"] reads the same as shape("name").
    by_item = _item(intel, _attr(intel, context, "shapes")["type"], "stream")["type"]
    assert by_item == _item(intel, _attr(intel, context, "shape")["returns_by_key"], "stream")["type"]


def test_a_member_whose_class_differs_between_episodes_is_not_narrowed():
    from bluesky_sandbox.sim.bounds import (
        BoxFootprint,
        ConstantAltitudeBand,
        DiskFootprint,
        LatLon,
        RegionBounds,
    )
    from bluesky_sandbox.ui.designer.code_intel import TypeTable

    band = ConstantAltitudeBand(0.0, 10_000.0)
    held = (
        RegionBounds(DiskFootprint(LatLon(52.0, 4.5), 10.0), band),
        RegionBounds(BoxFootprint(51.5, 52.5, 4.0, 5.0), band),
    )
    table = TypeTable(None, None)
    key = table.narrowed(RegionBounds, held, depth=3)
    footprint = next(m for m in table.types[key]["attrs"] if m["name"] == "footprint")
    assert table.types[footprint["type"]]["name"] == "Footprint"
    # What the two agree on still narrows.
    altitude = next(m for m in table.types[key]["attrs"] if m["name"] == "altitude")
    assert table.types[altitude["type"]]["name"] == "ConstantAltitudeBand"


def test_a_keyed_call_returning_something_else_is_not_read_per_key(intel):
    batch = _param(intel, "hook:reward_batch", "batch")["type"]
    assert "returns_by_key" not in _attr(intel, batch, "query")
