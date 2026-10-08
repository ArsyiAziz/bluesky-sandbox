"""Test cases for fields (bluesky_sandbox.checks): a situation placed by hand,
a field, the value it should give - read the same way for any field, in the
design's geometry with none of its traffic."""

from __future__ import annotations

import bluesky as bs
import pytest

from bluesky_sandbox.checks import (
    Aircraft,
    Case,
    Situation,
    Tolerance,
    place,
    run_cases,
    without_traffic,
)
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import MinMaxNormalizer

from test_env_bound import InZone, _Scenario

_LEVEL = dict(gs_kts=250.0, alt_ft=10_000.0, actype="B744")
_IN_ZONE = Situation("in the zone", (Aircraft("OWN", lat=52.0, lon=4.0, track_deg=0.0, **_LEVEL),))
_HEAD_ON = Situation(
    "head-on",
    (
        Aircraft("OWN", lat=52.0, lon=6.0, track_deg=90.0, **_LEVEL),
        Aircraft("INTR", relative_to="OWN", distance_nm=10.0, bearing_deg=90.0, track_deg=270.0, **_LEVEL),
    ),
)
_SITUATIONS = (_IN_ZONE, _HEAD_ON)


@pytest.fixture(scope="module")
def env():
    # A design with traffic of its own (four aircraft) and a shape ("zone").
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[obs.LatDeg()], action_fields=[]))
    yield env
    env.close()


def test_a_tolerance_is_the_wider_of_absolute_and_relative():
    assert Tolerance().holds(1.0, 1.0) and not Tolerance().holds(1.0 + 1e-9, 1.0)
    assert Tolerance(abs=0.5).holds(10.4, 10.0) and not Tolerance(abs=0.5).holds(10.6, 10.0)
    assert Tolerance(rel=0.1).holds(109.0, 100.0) and not Tolerance(rel=0.1).holds(111.0, 100.0)
    assert Tolerance(abs=2.0, rel=0.01).holds(101.5, 100.0)  # abs is the wider here
    assert Tolerance(abs=1.0).holds([1.0, 2.0], [1.5, 2.5])  # every element
    assert not Tolerance(abs=1.0).holds([1.0, 2.0], [1.0])  # a different shape never holds


def test_an_aircraft_is_placed_one_way_or_the_other():
    with pytest.raises(ValueError, match="give one of"):
        Aircraft("A", track_deg=0, **_LEVEL)
    with pytest.raises(ValueError, match="give one of"):
        Aircraft("A", lat=0.0, lon=0.0, relative_to="B", distance_nm=1, bearing_deg=0, track_deg=0, **_LEVEL)


def test_a_situation_names_each_aircraft_once_and_places_relative_to_earlier_ones():
    own = Aircraft("OWN", lat=0.0, lon=0.0, track_deg=0, **_LEVEL)
    with pytest.raises(ValueError, match="two aircraft"):
        Situation("twice", (own, own))
    later = Aircraft("A", relative_to="B", distance_nm=1, bearing_deg=0, track_deg=0, **_LEVEL)
    with pytest.raises(ValueError, match="not placed before it"):
        Situation("ahead of itself", (own, later))
    with pytest.raises(ValueError, match="places no aircraft"):
        Situation("empty", ())


def test_a_situation_has_its_aircraft_only_exactly_as_given(env):
    with without_traffic(env):
        at = place(env, _HEAD_ON)
    assert list(bs.traf.id) == ["OWN", "INTR"]  # none of the design's own four
    own, intr = at["OWN"], at["INTR"]
    assert float(bs.traf.trk[intr]) == 270.0
    assert float(bs.traf.gs[intr]) == pytest.approx(250.0 * 0.514444, rel=1e-6)
    assert float(obs.DistToOwnNm().get_pair(own, intr)) == pytest.approx(10.0, rel=1e-3)


def test_cases_read_any_field_the_same_way(env):
    report = run_cases(
        env,
        _SITUATIONS,
        [
            Case("head-on", obs.ClosingRateKts(), 500.0, Tolerance(rel=0.01), of="INTR"),
            Case("in the zone", InZone(), 1.0, Tolerance()),  # custom, reads the env
            Case("head-on", InZone(), 0.0, Tolerance()),
        ],
    )
    report.assert_ok()


def test_the_design_gets_its_traffic_back(env):
    scenario = env.scenario
    run_cases(env, _SITUATIONS, [Case("in the zone", InZone(), 1.0, Tolerance())])
    assert env.scenario is scenario
    env.reset(seed=0)
    assert bs.traf.ntraf == 4


def test_a_case_that_fails_or_cannot_be_read_says_why_and_the_rest_run(env):
    report = run_cases(
        env,
        _SITUATIONS,
        [
            Case("head-on", obs.ClosingRateKts(), 400.0, Tolerance(rel=0.01), of="INTR"),
            Case("nowhere", obs.LatDeg(), 0.0, Tolerance()),
            Case("head-on", obs.DistToOwnNm(), 10.0, Tolerance(), of="GHOST"),
            Case("head-on", obs.DistToOwnNm(), 10.0, Tolerance()),  # a pair field, no other
            Case("in the zone", InZone(), 1.0, Tolerance()),
        ],
    )
    first, unknown, ghost, unnamed, last = report.results
    assert not first.ok and first.error is None and first.got == pytest.approx(500.0, rel=0.01)
    assert "no situation is named 'nowhere'" in unknown.error
    assert "places no aircraft 'GHOST'" in ghost.error
    assert "name it" in unnamed.error
    assert last.ok
    with pytest.raises(AssertionError, match="4 of 5 cases failed"):
        report.assert_ok()


_EAST = Situation("east", (Aircraft("OWN", lat=52.0, lon=4.0, track_deg=90.0, **_LEVEL),))


def test_an_applied_action_is_read_for_what_it_commands(env):
    run_cases(
        env,
        [_EAST],
        [
            # +1,000 ft from 10,000 ft holds 11,000 ft.
            Case("east", act.AltDeltaFt(), 11_000.0, Tolerance(abs=1.0), apply=act.AltDeltaFt(), value=1000.0),
            # +90 deg from a track of 090 holds 180.
            Case("east", act.HdgDeltaDeg(), 180.0, Tolerance(abs=0.5), apply=act.HdgDeltaDeg(), value=90.0),
        ],
    ).assert_ok()


def test_an_observation_reads_the_command_right_after_it(env):
    # No flying: the autopilot's selected altitude is the new target at once.
    run_cases(
        env,
        [_EAST],
        [Case("east", obs.ApAltFt(), 11_000.0, Tolerance(abs=1.0), apply=act.AltDeltaFt(), value=1000.0)],
    ).assert_ok()


def test_a_failed_case_says_what_it_saw(env):
    wrong = Case("east", obs.ApAltFt(normalizer=MinMaxNormalizer()), 12_000.0, Tolerance(abs=1.0),
                 apply=act.AltDeltaFt(), value=1000.0)
    (result,) = run_cases(env, [_EAST], [wrong]).results
    assert not result.ok
    # As BlueSky takes it: given, the command, what it holds, what the field read.
    gave, command, holds, reading = result.saw
    assert gave == "gave 1000" and command.startswith("command ALT OWN 11000") and holds == "holds 11000"
    assert reading.startswith("ApAltFt raw 11000 → normalized ")
    assert "ALT OWN 11000" in str(result)


def test_an_action_read_for_itself_says_what_it_holds_once(env):
    (result,) = run_cases(
        env, [_EAST], [Case("east", act.AltDeltaFt(), 11_000.0, Tolerance(abs=1.0), apply=act.AltDeltaFt(), value=1000.0)]
    ).results
    gave, command, holds = result.saw
    assert result.ok and (gave, holds) == ("gave 1000", "holds 11000") and command.startswith("command ALT OWN")


def test_an_action_case_says_what_it_cannot_do():
    with pytest.raises(ValueError, match="an action and a value, or neither"):
        Case("east", obs.TrkDeg(), 0.0, Tolerance(), apply=act.HdgDeltaDeg())
