"""Geometry fields on encounters worked out by hand.

The bulk-vs-reference tests (test_field_batching) prove a field computes what
its ``_expected_pair`` states; they cannot tell whether that statement is what
the field is FOR - a wrong sign convention or frame would be written the same
way in both. These pin the meaning on setups simple enough to solve on paper.

Each encounter is a situation (:mod:`bluesky_sandbox.checks`): the ownship on
the equator, the intruder placed by distance and bearing from it, every speed,
track and vertical speed set exactly as described. The tolerance is 1%: these
check units, signs and frames, not rounding (the conflict geometry is
flat-earth, the placement great-circle).
"""

from __future__ import annotations

import dataclasses
import math

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts, nm

from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance, place, run_cases
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(
        scenario=_Empty(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    yield env
    env.close()


_OWN = Aircraft("OWN", lat=0.0, lon=0.0, track_deg=90.0, gs_kts=250.0, alt_ft=10_000.0, actype="B744")


def _own(env, **state) -> int:
    """Only the ownship, flying east at 250 kts - as changed by ``state``."""
    return place(env, Situation("ownship", (dataclasses.replace(_OWN, **state),)))["OWN"]


def _intruder(**state) -> Aircraft:
    return Aircraft("INTR", relative_to="OWN", **{"gs_kts": 250.0, "alt_ft": 10_000.0, "actype": "B744", **state})


# Each encounter: the ownship at the origin flying east at 250 kts, and an
# intruder placed and moving as named.
_SITUATIONS = {
    s.name: s
    for s in (
        # Intruder 10 nm dead ahead, flying straight at the ownship at 250 kts.
        Situation("head-on", (_OWN, _intruder(distance_nm=10, bearing_deg=90, track_deg=270))),
        # Intruder 10 nm north, flying alongside: nothing changes, ever.
        Situation("abeam", (_OWN, _intruder(distance_nm=10, bearing_deg=0, track_deg=90))),
        # Intruder 10 nm north, flying south across the ownship's path.
        Situation("crossing", (_OWN, _intruder(distance_nm=10, bearing_deg=0, track_deg=180))),
        # Head-on, but 1000 ft above and descending 500 ft/min.
        Situation(
            "head-on, descending",
            (_OWN, _intruder(distance_nm=10, bearing_deg=90, track_deg=270, alt_ft=11_000, vs_fpm=-500)),
        ),
    )
}
_CLOSE = Tolerance(abs=1e-6, rel=1e-2)

# Crossing, by hand: relative position (east, north) = (0, 10) nm, relative
# velocity = (-250, -250) kts. tcpa = 2500 / 125000 h = 72 s, at which the
# intruder sits at (-5, +5) nm: behind and left, 7.07 nm away.
_CASES = [
    ("head-on", obs.DistToOwnNm(), 10.0),
    ("head-on", obs.BrgFromOwnDeg(), 90.0),
    ("head-on", obs.BrgFromOwnRelTrkDeg(), 0.0),  # dead ahead
    ("head-on", obs.ClosingRateKts(), 500.0),  # positive = closing
    ("head-on", obs.RelPosAlongTrackNm(), 10.0),  # ahead
    ("head-on", obs.RelPosCrossTrackNm(), 0.0),
    ("head-on", obs.RelVelAlongTrackKts(), -500.0),  # coming back at us
    ("head-on", obs.RelVelCrossTrackKts(), 0.0),
    ("head-on", obs.BearingRateDegPerSec(), 0.0),  # constant bearing: collision
    ("head-on", obs.RelPosAtCpaAlongTrackNm(), 0.0),
    ("head-on", obs.RelPosAtCpaCrossTrackNm(), 0.0),
    ("head-on", obs.RelVsFtMin(), 0.0),
    ("head-on", obs.VerticalSepAtCpaFt(), 0.0),
    ("head-on", obs.ConflictTcpaS(), 72.0),  # 10 nm at 500 kts
    ("head-on", obs.ConflictHorizontalDistAtCpaNm(), 0.0),
    # Enters a 5 nm zone after closing 5 nm at 500 kts.
    ("head-on", obs.ConflictTlosS(rpz_nm=5.0, vpz_ft=1000.0), 36.0),
    ("head-on", obs.InConf(rpz_nm=5.0, vpz_ft=1000.0), 1.0),
    ("head-on", obs.InLosNow(), 0.0),
    ("abeam", obs.DistToOwnNm(), 10.0),
    ("abeam", obs.BrgFromOwnDeg(), 0.0),  # due north
    ("abeam", obs.BrgFromOwnRelTrkDeg(), -90.0),  # off the left wing
    ("abeam", obs.RelPosAlongTrackNm(), 0.0),
    ("abeam", obs.RelPosCrossTrackNm(), -10.0),  # left is negative
    ("abeam", obs.ClosingRateKts(), 0.0),
    ("abeam", obs.RelVelAlongTrackKts(), 0.0),
    ("abeam", obs.BearingRateDegPerSec(), 0.0),
    ("abeam", obs.InConf(rpz_nm=5.0, vpz_ft=1000.0), 0.0),
    ("crossing", obs.ConflictTcpaS(), 72.0),
    ("crossing", obs.ConflictHorizontalDistAtCpaNm(), 5.0 * math.sqrt(2.0)),
    ("crossing", obs.RelPosAtCpaAlongTrackNm(), -5.0),  # passes behind
    ("crossing", obs.RelPosAtCpaCrossTrackNm(), -5.0),  # on the left
    ("crossing", obs.RelVelAlongTrackKts(), -250.0),
    ("crossing", obs.RelVelCrossTrackKts(), 250.0),  # moving right, across our nose
    # Bearing swings from north toward west: (north*v_e - east*v_n) / r^2.
    ("crossing", obs.BearingRateDegPerSec(), math.degrees(-25.0 / 3600.0)),
    ("crossing", obs.ClosingRateKts(), 250.0),  # the northward gap closes at 250
    ("head-on, descending", obs.RelVsFtMin(), -500.0),
    # 1000 ft apart, closing vertically at 500 ft/min for 72 s: 400 ft left.
    ("head-on, descending", obs.VerticalSepAtCpaFt(), 400.0),
]


@pytest.mark.parametrize(
    ("encounter", "field", "expected"),
    _CASES,
    ids=[f"{e}-{type(f).__name__}" for e, f, _x in _CASES],
)
def test_a_hand_worked_encounter(env, encounter, field, expected):
    run_cases(env, _SITUATIONS.values(), [Case(encounter, field, expected, _CLOSE, of="INTR")]).assert_ok()
    # The field's own plain statement of the value agrees with the hand answer.
    at = place(env, _SITUATIONS[encounter])
    assert float(field.expected_pair(at["OWN"], at["INTR"])) == pytest.approx(
        expected, rel=1e-2, abs=1e-6
    )


def test_turn_radius_at_the_bank_limit(env):
    # r = v^2 / (g tan(bank)): 250 kts at 25 degrees of bank is 1.95 nm.
    idx = _own(env)
    bs.traf.ap.bankdef[idx] = math.radians(25.0)
    expected = (250 * kts) ** 2 / (9.80665 * math.tan(math.radians(25.0))) / nm
    assert expected == pytest.approx(1.953, abs=1e-3)
    assert float(obs.TurnRadiusNm().get(idx)) == pytest.approx(expected, rel=1e-4)


def test_a_reset_does_not_serve_the_last_episodes_geometry(env):
    # Same callsigns at the same sim time (0, after each reset) - the one thing
    # the geometry cache's key cannot tell apart - in different places.
    field = obs.InConf(rpz_nm=5.0, vpz_ft=1000.0)
    at = place(env, _SITUATIONS["head-on"])
    assert float(field.get_pairs(at["OWN"], [at["INTR"]])[0]) == 1.0
    at = place(env, _SITUATIONS["abeam"])
    assert float(field.get_pairs(at["OWN"], [at["INTR"]])[0]) == 0.0


# --- units --------------------------------------------------------------------
# A quantity is computed once in SI; each unit variant converts it with
# BlueSky's own constants. Pin those to the unit definitions, and each pair's
# values AND dynamic bounds to each other - the reference tests compare values
# only.
_UNIT_PAIRS = [
    (obs.AltFt, obs.AltM, 1 / ft),
    (obs.ApAltFt, obs.ApAltM, 1 / ft),
    (obs.ApAltErrorFt, obs.ApAltErrorM, 1 / ft),
    (obs.CasKts, obs.CasMs, 1 / kts),
    (obs.TasKts, obs.TasMs, 1 / kts),
    (obs.GsKts, obs.GsMs, 1 / kts),
    (obs.ApCasKts, obs.ApCasMs, 1 / kts),
    (obs.VsFtMin, obs.VsMs, 60 / ft),
]


def test_the_conversion_constants_are_the_unit_definitions():
    # International foot and nautical mile; BlueSky rounds kts to 0.514444.
    assert ft == pytest.approx(0.3048, rel=1e-9)
    assert kts == pytest.approx(1852 / 3600, rel=1e-6)
    assert nm == pytest.approx(1852.0, rel=1e-9)


@pytest.mark.parametrize(
    ("in_unit", "in_si", "factor"),
    _UNIT_PAIRS,
    ids=[f"{u.__name__}-{s.__name__}" for u, s, _f in _UNIT_PAIRS],
)
def test_a_unit_variant_is_its_si_twin_converted(env, in_unit, in_si, factor):
    idx = _own(env, alt_ft=12_000.0, vs_fpm=-800.0)
    bs.traf.selalt[idx] = 9_000.0 * ft
    bs.traf.selspd[idx] = 230.0 * kts
    unit, si = in_unit(), in_si()
    assert float(unit.get(idx)) == pytest.approx(float(si.get(idx)) * factor, rel=1e-9)
    for got, want in zip(unit.bounds(idx), si.bounds(idx), strict=True):
        assert float(got) == pytest.approx(float(want) * factor, rel=1e-9)


def test_the_speed_error_bounds_reach_either_end_of_the_envelope(env):
    idx = _own(env)
    low, high = obs.ApCasErrorKts().bounds(idx)
    cas = float(bs.traf.cas[idx]) / kts
    vmin, vmax = (
        float(v) / kts for v in (bs.traf.perf.vmin[idx], bs.traf.perf.vmax[idx])
    )
    # Symmetric about zero error, wide enough to command the slower or the
    # faster envelope edge from the current speed.
    assert low == -high
    assert high == pytest.approx(max(cas - vmin, vmax - cas), rel=1e-9)
