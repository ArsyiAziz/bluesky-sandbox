"""The episode sample: what an aircraft of the seeded episode observes, raw and
normalized, with each field's range for it - run in a subprocess."""

from __future__ import annotations

import pytest

from bluesky_sandbox.ui.designer.runner import sample_design

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


def test_a_type_asked_for_is_the_first_agent():
    first = sample_design(_example_design_spec(), seed=0, max_agents=1)
    wanted = first["agents"][0]["type"]
    again = sample_design(_example_design_spec(), seed=0, max_agents=1, actype=wanted)
    assert again["agents"][0]["type"] == wanted
