"""What the code editor knows, read from the code and the design.

Types come from annotations, each block's names from what runs there, and the
keys a design fixes from ``DesignKeys`` markers - so the editor completes
``context.raw_obs["intruders"]["acid"]`` and ``context.query("wp").current``
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
    assert (
        intel["scopes"]["hook:reward"]["names"] == intel["scopes"]["task_info"]["names"]
    )
    setup = {m["name"]: m for m in intel["names"]["setup"]}
    assert {"np", "SCALE", "root", "LIMIT", "CONFIG", "_half"} <= set(setup)
    assert setup["np"]["module"] == "numpy"
    assert [p["name"] for p in setup["_half"]["params"]] == ["x", "by"]
    assert setup["_half"]["params"][1]["default"] == "2"


def test_raw_observation_keys_come_from_the_design(intel):
    raw_obs = _attr(intel, _context(intel), "raw_obs")["type"]
    parts = {m["name"]: m["type"] for m in _type(intel, raw_obs)["items"]}
    assert list(parts) == ["ownship", "intruders"]
    ownship = {m["name"]: m for m in _type(intel, parts["ownship"])["items"]}
    assert list(ownship) == ["lat_deg", "lon_deg", "alt_ft"]
    assert ownship["alt_ft"]["detail"] == "float · ft"
    assert "MinMaxNormalizer" in ownship["alt_ft"]["doc"]
    intruders = [m["name"] for m in _type(intel, parts["intruders"])["items"]]
    assert intruders == ["acid", "dist_to_own_nm"]


def test_raw_action_keys_come_from_the_design(intel):
    raw_action = _attr(intel, _context(intel), "raw_action")["type"]
    assert [m["name"] for m in _type(intel, raw_action)["items"]] == [
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


def test_a_type_checking_import_resolves(intel):
    # AgentStepContext.airspace is annotated with a name imported only for type
    # checkers; the editor still knows it.
    airspace = _attr(intel, _context(intel), "airspace")
    assert _type(intel, airspace["type"])["name"] == "RegionResult"
    assert "RegionResult" in str(hints(AgentStepContext)["airspace"])


def test_a_batched_hook_sees_stacked_shapes(intel):
    batch = _param(intel, "hook:reward_batch", "batch")["type"]
    raw_obs = _attr(intel, batch, "raw_obs")["type"]
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
