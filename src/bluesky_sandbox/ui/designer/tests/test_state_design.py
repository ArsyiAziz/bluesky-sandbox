"""State fields in a design: kept, built into the config, and generated - never
into the observation, but typed for the hooks that read them."""

from __future__ import annotations

import ast

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import build_design_config

from .test_designer import _example_design_spec


def _stateful() -> S.DesignSpec:
    spec = _example_design_spec()
    spec.env.state_fields = [S.FieldRef("AltFt"), S.FieldRef("CasKts")]
    spec.env.intruder_state_fields = [S.FieldRef("DistToOwnNm")]
    return spec


def test_a_designs_state_fields_survive_a_round_trip():
    spec = _stateful()
    again = S.DesignSpec.from_dict(spec.to_dict())
    assert [f.name for f in again.env.state_fields] == ["AltFt", "CasKts"]
    assert [f.name for f in again.env.intruder_state_fields] == ["DistToOwnNm"]
    # A design saved before state fields has none.
    old = spec.to_dict()
    del old["env"]["state_fields"], old["env"]["intruder_state_fields"]
    assert S.DesignSpec.from_dict(old).env.state_fields is None


def test_the_config_computes_state_but_does_not_observe_it():
    cfg = build_design_config(_stateful())
    assert [f.meta.name for f in cfg.state_fields] == ["alt_ft", "cas_kts"]
    assert [f.meta.name for f in cfg.intruder_state_fields] == ["dist_to_own_nm"]
    assert [f.meta.name for f in cfg.obs_fields] == ["lat_deg", "lon_deg", "alt_ft"]


def test_a_generated_package_configures_and_types_its_state():
    files = codegen.generate_task(_stateful(), "Stated")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    ast.parse(config)
    assert "state_fields=list((" in config and "intruder_state_fields=list((" in config
    types = next(t for p, t in files.items() if p.endswith("task_types.py"))
    assert "class State(TypedDict):" in types and "state: State" in types


def test_a_package_without_state_is_as_before():
    files = codegen.generate_task(_example_design_spec(), "Plain")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    assert "state_fields" not in config
