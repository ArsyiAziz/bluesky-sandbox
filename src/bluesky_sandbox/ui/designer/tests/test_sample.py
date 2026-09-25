"""The episode sample: what an aircraft of the seeded episode observes, raw and
normalized, with each field's range for it - run in a subprocess."""

from __future__ import annotations

import pytest

from bluesky_sandbox.ui.designer.runner import episode_spawns, sample_design

from .test_designer import _example_design_spec


@pytest.fixture(scope="module")
def sample() -> dict:
    return sample_design(_example_design_spec(), seed=0, max_agents=2)


def test_an_agent_reports_each_field_raw_and_normalized(sample):
    agent = sample["agents"][0]
    assert agent["type"]
    fields = {f["name"]: f for f in agent["parts"]["ownship"]["fields"]}
    assert list(fields) == ["lat_deg", "lon_deg", "alt_ft"]
    alt = fields["alt_ft"]
    assert alt["unit"] == "ft" and alt["lag"] is None
    # A per-aircraft range is this aircraft's: from the ground to its ceiling.
    assert alt["low"] == 0.0 and alt["high"] > 10_000
    assert 0.0 <= alt["obs"][0] <= 1.0
    assert alt["obs"][0] == pytest.approx(alt["raw"] / alt["high"], abs=1e-4)


def test_intruder_parts_carry_a_row_per_intruder(sample):
    part = sample["agents"][0]["parts"]["intruders"]
    (dist,) = part["fields"]
    assert len(dist["raw"]) == len(dist["obs"]) == len(part["acids"])


def test_every_aircraft_up_reports_its_ranges(sample):
    rows = sample["ranges"]["ownship"]["alt_ft"]
    assert len(rows) >= len(sample["agents"])
    assert {acid for acid, *_ in rows} >= {a["acid"] for a in sample["agents"]}
    assert set(sample["ranges"]["action"]) == {"hdg_deg", "spd_kts"}


def test_an_aircraft_asked_for_is_the_first_agent():
    spawns = episode_spawns(_example_design_spec(), seed=0)
    last = spawns["aircraft"][-1]
    sample = sample_design(
        _example_design_spec(), seed=0, max_agents=1, at_s=last["time_s"], acid=last["callsign"]
    )
    assert sample["agents"][0]["acid"] == last["callsign"]


def test_the_episode_lists_each_aircraft_as_created():
    spawns = episode_spawns(_example_design_spec(), seed=0)
    assert spawns["complete"] and len(spawns["aircraft"]) == spawns["scheduled"] > 0
    first = spawns["aircraft"][0]
    assert {"callsign", "actype", "time_s", "lat_deg", "lon_deg", "alt_ft", "hdg_deg", "cas_kts", "gs_kts"} <= set(first)
    assert first["label"][0].startswith(first["callsign"])
    assert first["label"][1].startswith(f"FL{round(first['alt_ft'] / 100):03d}")
    targets = [t for a in spawns["aircraft"] for t in a["targets"]]
    assert all(t["waypoint"] and "reach_radius_nm" in t for t in targets)
    # The same seed, the same episode.
    assert episode_spawns(_example_design_spec(), seed=0)["aircraft"] == spawns["aircraft"]
