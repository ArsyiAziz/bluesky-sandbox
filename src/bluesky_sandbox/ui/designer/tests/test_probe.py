"""Probing one field of a design: the seeded episode flown to a moment in the
warm worker, and one field of the spec read there - see
:func:`bluesky_sandbox.checks.probe.probe`."""

from __future__ import annotations

import copy

import pytest
from starlette.testclient import TestClient

from bluesky_sandbox.ui.designer.api import create_app
from bluesky_sandbox.ui.designer.builder import BuildError, config_fields_of
from bluesky_sandbox.ui.designer.runner import probe_design

from .test_designer import _example_design_spec


def _stacked():
    spec = copy.deepcopy(_example_design_spec())
    alt = spec.env.obs_fields[2]
    alt.transform, alt.transform_kwargs = "stacked", {"depth": 3}
    return spec


def test_an_entry_lands_where_the_config_puts_it():
    spec = _stacked()
    assert config_fields_of(spec, "obs_fields", 0) == (0, 1)
    assert config_fields_of(spec, "obs_fields", 2) == (2, 3)  # alt_ft and two lags
    assert config_fields_of(spec, "action_fields", 1) == (1, 1)
    with pytest.raises(BuildError, match="no entry 9"):
        config_fields_of(spec, "obs_fields", 9)


def test_a_lag_of_a_stack_is_probed_with_its_ring():
    out = probe_design(_stacked(), list_key="obs_fields", entry=2, part=1, at_s=3.0)
    result = out["result"]
    assert result["field"] == "alt_ft_lag1" and out["acid"] == result["aircraft"]
    assert result["aircraft"] in out["aircraft"]
    assert [s["name"] for s in result["stages"]] == ["raw", "normalized"]
    assert all(s["agrees"] for s in result["stages"])
    assert result["lag"]["fields"] == ["alt_ft", "alt_ft_lag1", "alt_ft_lag2"]
    assert result["lag"]["progression"] and result["cost"]["ms"]


def test_an_action_is_probed_to_its_command():
    out = probe_design(_example_design_spec(), list_key="action_fields", entry=1, give=0.5, at_s=1.0)
    stages = {s["name"]: s for s in out["result"]["stages"]}
    assert stages["policy"]["actual"] == 0.5
    assert stages["BS command"]["actual"].startswith(f"SPD {out['acid']} ")


def test_a_pair_field_reads_about_the_nearest_unless_named():
    out = probe_design(_example_design_spec(), list_key="intruder_obs_fields", entry=0, at_s=1.0)
    result = out["result"]
    assert result["kind"] == "pair" and result["other"] in out["aircraft"]
    assert result["other"] != result["aircraft"]


def test_the_endpoint_probes_and_refuses_what_it_cannot():
    client = TestClient(create_app())
    spec = _example_design_spec().to_dict()
    ok = client.post("/api/spec/probe", json={"spec": spec, "list": "obs_fields", "entry": 2, "overrides": [
        {"target": "traf:alt", "value": 3048.0},
    ]})
    assert ok.status_code == 200
    raw = ok.json()["result"]["stages"][0]
    assert raw["actual"] == pytest.approx(10_000.0)
    assert client.post("/api/spec/probe", json={"spec": spec, "entry": 0}).status_code == 422
    assert client.post("/api/spec/probe", json={"spec": spec, "list": "obs_fields", "entry": 7}).status_code == 422
