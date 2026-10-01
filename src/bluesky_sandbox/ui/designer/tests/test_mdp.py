"""The MDP summary: each space's parts, field by field, with the mapping each
normalizer makes sampled as a curve."""

from __future__ import annotations

import json

from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.mdp import mdp_summary

from .test_designer import _example_design_spec


def _normalized(name: str, **kwargs) -> dict:
    return {"normalizer": {"type": "normalizer", "name": name, "kwargs": kwargs}}


def _fields(part: dict) -> dict[str, dict]:
    return {f["name"]: f for f in part["fields"]}


def test_the_summary_is_plain_json():
    json.dumps(mdp_summary(_example_design_spec()), allow_nan=False)


def test_each_observation_part_lists_its_fields_in_column_order():
    summary = mdp_summary(_example_design_spec())
    parts = {p["part"]: p for p in summary["observation"]}
    assert (parts["ownship"]["rows"], parts["intruders"]["rows"]) == (
        "per agent",
        "per intruder",
    )
    columns = [f["columns"] for f in parts["ownship"]["fields"]]
    assert columns == [[0, 1], [1, 2], [2, 3]]
    assert parts["ownship"]["width"] == 3


def test_a_raw_field_has_no_normalizer_and_lands_in_its_own_range():
    lat = _fields(mdp_summary(_example_design_spec())["observation"][0])["lat_deg"]
    assert lat["normalizer"] is None
    # Drawn as it is passed, over its own range.
    assert lat["curve"]["x"][0] == -90.0 and lat["curve"]["x"][-1] == 90.0
    assert lat["curve"]["series"][0] == lat["curve"]["x"]
    assert (lat["raw"]["low"], lat["raw"]["high"], lat["raw"]["unit"]) == (
        -90.0,
        90.0,
        "deg",
    )
    assert lat["output"] == {"low": [-90.0], "high": [90.0]}


def test_an_observation_curve_maps_raw_to_normalized_and_shows_the_clip():
    design = _example_design_spec()
    design.env.intruder_obs_fields = [
        S.FieldRef("DistToOwnNm", kwargs=_normalized("MinMaxNormalizer", clipped=True))
    ]
    dist = _fields(mdp_summary(design)["observation"][1])["dist_to_own_nm"]
    curve = dist["curve"]
    assert curve["x_label"] == "raw" and curve["y_label"] == "normalized"
    (series,) = curve["series"]
    # The samples reach a tenth past each end of [0, 200]; past them, it is clipped.
    assert (curve["x"][0], curve["x"][-1]) == (-20.0, 220.0)
    assert (series[0], series[len(series) // 2], series[-1]) == (0.0, 0.5, 1.0)


def test_a_range_each_aircraft_resolves_is_sampled_over_its_position():
    alt = _fields(mdp_summary(_example_design_spec())["observation"][0])["alt_ft"]
    assert alt["raw"]["per_aircraft"] and alt["raw"]["low"] is None
    assert alt["curve"]["x_label"] == "position in range (low → high)"


def test_a_circular_normalizer_gives_a_series_per_output_column():
    design = _example_design_spec()
    design.env.intruder_obs_fields = [
        S.FieldRef("DistToOwnNm"),
        S.FieldRef("BrgFromOwnDeg", kwargs=_normalized("CircularNormalizer")),
    ]
    part = mdp_summary(design)["observation"][1]
    bearing = part["fields"][1]
    assert bearing["columns"] == [1, 3] and part["width"] == 3
    assert len(bearing["curve"]["series"]) == 2
    assert len(bearing["output"]["low"]) == 2


def test_an_action_curve_maps_the_policys_value_to_the_command():
    action = mdp_summary(_example_design_spec())["action"]
    assert action["space"] == "Box"
    (part,) = action["parts"]
    spd = _fields(part)["spd_kts"]
    assert spd["normalizer"]["name"] == "SymmetricNormalizer"
    assert spd["normalizer"]["params"]["clipped"] is True
    assert spd["curve"]["x_label"] == "policy's value"
    (series,) = spd["curve"]["series"]
    assert (series[0], series[-1]) == (0.0, 1.0)


def test_a_switch_puts_the_action_in_a_dict_with_a_binary_part():
    design = _example_design_spec()
    design.env.action_fields = [*design.env.action_fields, S.FieldRef("AutopilotLnav")]
    action = mdp_summary(design)["action"]
    assert action["space"] == "Dict"
    parts = {p["part"]: p for p in action["parts"]}
    (switch,) = parts["binary"]["fields"]
    assert switch["binary"] and not parts["continuous"]["fields"][0]["binary"]
    assert switch["normalizer"] is None
    assert (switch["raw"]["low"], switch["raw"]["high"]) == (0.0, 1.0)


def test_a_lag_names_the_field_it_lags_and_keeps_its_range():
    design = _example_design_spec()
    design.env.intruder_obs_fields = [
        S.FieldRef(
            "DistToOwnNm",
            kwargs=_normalized("MinMaxNormalizer"),
            transform="stacked",
            transform_kwargs={"depth": 3},
        )
    ]
    live, *lags = mdp_summary(design)["observation"][1]["fields"]
    assert live["lag"] is None
    assert [lag["lag"] for lag in lags] == [
        {"steps": n, "of": live["name"], "inner": "dist_to_own_nm"} for n in (1, 2)
    ]
    (lag, _) = lags
    # A lag's own meta says its range is read at runtime; it is the field's.
    assert (lag["raw"]["low"], lag["raw"]["high"]) == (0.0, 200.0)
    assert not lag["raw"]["per_aircraft"]
