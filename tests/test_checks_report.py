"""A design's fields reported over sampled episodes (bluesky_sandbox.checks.report):
each step timed, grouped by the aircraft in the air - topped up past what the
episodes flew - and each field's values tallied against its bounds."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bluesky_sandbox.checks.report import BoundsTally, _field_name, as_plain, checks_plain, field_checks, field_report
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import CircularNormalizer, MinMaxNormalizer

from test_env_bound import _Scenario

#: Spawned between 10,000 and 30,000 ft: a 20,000 ft ceiling puts some above.
_ALT = obs.AltFt(low=0, high=20_000, normalizer=MinMaxNormalizer(clipped=True))


@pytest.fixture(scope="module")
def report():
    config = EnvConfig(
        dt=5.0,
        obs_fields=[*_ALT.stacked(depth=2), obs.TasKts()],
        intruder_obs_fields=[obs.DistToOwnNm()],
        action_fields=[act.HdgDeltaDeg()],
    )
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    try:
        yield field_report(env, seeds=[0, 1], steps=4, most=12)
    finally:
        env.close()


def test_steps_are_grouped_by_aircraft_from_one_to_the_designs_most(report):
    cost = report.cost
    # The scenario flies four; one and two are placed, eight and twelve topped up.
    assert cost.aircraft == [1, 2, 4, 8, 12] and cost.sampled == [False, False, True, False, False]
    assert cost.steps[2] == 8  # two seeds, four steps each
    assert all(lo <= m <= hi for lo, m, hi in zip(cost.total.low, cost.total.median, cost.total.high, strict=True))


def test_the_phases_add_up_to_the_step(report):
    cost = report.cost
    assert {"simulation", "fields", "packing", "hooks", "actions"} <= set(cost.phases)
    for k in range(len(cost.aircraft)):
        # Medians of parts need not sum to the median of the whole, closely.
        parts = sum(s.median[k] for s in cost.phases.values())
        assert parts == pytest.approx(cost.total.median[k], rel=0.25)


def test_each_field_is_timed_by_its_place_in_the_config(report):
    assert set(report.cost.fields) == {("obs_fields", 0), ("obs_fields", 1), ("obs_fields", 2), ("intruder_obs_fields", 0)}
    assert all(t is not None and t > 0 for times in report.cost.fields.values() for t in times)
    assert report.fields[("obs_fields", 1)] == ("AltFt_lag1", True)


def test_a_field_is_tallied_against_its_bounds(report):
    alt = next(t for t in report.bounds if t.name == "AltFt")
    assert alt.samples > 0 and alt.above > 0 and alt.below == 0
    assert alt.fixed_bounds == (0.0, 20_000.0) and alt.output == (0.0, 1.0) and alt.clips
    assert alt.normalized_max == 1.0  # clipped
    assert sum(alt.histogram.values()) == alt.samples
    assert sum(n for k, n in alt.histogram.items() if k >= alt.bins) == alt.above
    tas = next(t for t in report.bounds if t.name == "TasKts")
    assert tas.fixed_bounds is None  # each aircraft its own


def test_a_value_on_its_high_bound_is_inside():
    tally = BoundsTally("f", ("obs_fields", 0), bins=4)
    tally.add(np.array([0.0, 10.0]), np.array(0.0), np.array(10.0), np.array([True, True]), lambda v, at: None)
    assert tally.above == 0 and tally.histogram == {0: 1, 3: 1}


def test_it_is_plain_json(report):
    plain = as_plain(report)
    json.dumps(plain)
    assert "obs_fields:0" in plain["cost"]["fields"] and plain["fields"]["obs_fields:0"]["name"] == "AltFt"


def test_a_value_far_out_is_kept_in_the_last_bin_either_side():
    tally = BoundsTally("f", ("obs_fields", 0), bins=4)
    values = np.array([-1e6, -5.0, 15.0, 1e6])
    tally.add(values, np.array(0.0), np.array(10.0), np.ones(4, dtype=bool), lambda v, at: None)
    # Within one width of the bounds a bin of its own; further, the overflow.
    assert tally.histogram == {-5: 1, -2: 1, 6: 1, 8: 1}
    assert (tally.min, tally.max) == (-1e6, 1e6)


def test_a_difference_from_the_ownship_is_named_after_its_field():
    assert _field_name(obs.AltFt().relative_to_own()) == "AltFt − own"


def test_the_checks_say_what_each_field_and_action_holds_to():
    config = EnvConfig(
        dt=5.0,
        obs_fields=[_ALT, obs.TasKts()],
        intruder_obs_fields=[obs.DistToOwnNm()],
        # Given in its own unit; and as an angle's two values.
        action_fields=[
            act.SpdKts(),
            act.HdgDeltaDeg(normalizer=CircularNormalizer()),
            act.AutopilotLnav(),
            act.ActionMask(target=act.SpdKts),
        ],
    )
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    try:
        plain = checks_plain(*field_checks(env, seed=0, steps=3))
    finally:
        env.close()
    json.dumps(plain)
    fields = {f["field"]: f for f in plain["fields"]}
    assert set(fields) == {"obs_fields:0", "obs_fields:1", "intruder_obs_fields:0"}
    assert not any(f["findings"] for f in fields.values())
    assert fields["obs_fields:0"]["notes"]  # spawned above its 20,000 ft ceiling
    speed, heading, switch, mask = plain["actions"]
    assert speed["field"] == "action_fields:0" and speed["agrees"] is True and speed["command"][0].startswith("SPD ")
    assert heading["error"] == "" and heading["agrees"] is True and heading["command"][0].startswith("HDG ")
    # A switch, probed on: its command given, no target to hold it to.
    assert switch["error"] == "" and switch["command"]
    assert mask["error"] == ""  # a mask is a switch too: probed on, not at 0.5
    assert plain["episode"] == []
