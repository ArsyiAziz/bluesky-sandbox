"""Geometry fields on encounters worked out by hand.

The bulk-vs-reference tests (test_field_batching) prove a field computes what
its ``_expected_pair`` states; they cannot tell whether that statement is what
the field is FOR - a wrong sign convention or frame would be written the same
way in both. These pin the meaning on setups simple enough to solve on paper.

Aircraft sit on the equator, where one arcminute of longitude is close to one
nautical mile (10' is 10.018 nm on WGS-84, 10.006 nm on the flat-earth
conflict geometry's mean radius), so the tolerance is 1%: these check units,
signs and frames, not rounding. Speeds, tracks and vertical speeds are written
straight into BlueSky's arrays so each encounter is exactly as described.
"""

from __future__ import annotations

import math

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts, nm

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

_TEN_NM = 10 / 60  # degrees of longitude at the equator


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


def _place(acid, lat, lon, track, gs_kts, alt_ft=10_000.0, vs_fpm=0.0) -> int:
    assert bs.traf.cre(acid, "B744", lat, lon, track, alt_ft * ft, 250) is True
    idx = bs.traf.id.index(acid)
    bs.traf.trk[idx] = bs.traf.hdg[idx] = track
    bs.traf.gs[idx] = bs.traf.tas[idx] = gs_kts * kts
    bs.traf.alt[idx] = alt_ft * ft
    bs.traf.vs[idx] = vs_fpm * ft / 60.0
    return idx


# Each encounter: the ownship at the origin flying east at 250 kts, and an
# intruder placed and moving as named. Fresh callsigns per encounter, so the
# per-step conflict geometry (cached by aircraft set) is computed anew.
_ENCOUNTERS = {
    # Intruder 10 nm dead ahead, flying straight at the ownship at 250 kts.
    "head-on": dict(lat=0.0, lon=_TEN_NM, track=270.0),
    # Intruder 10 nm north, flying alongside: nothing changes, ever.
    "abeam": dict(lat=_TEN_NM, lon=0.0, track=90.0),
    # Intruder 10 nm north, flying south across the ownship's path.
    "crossing": dict(lat=_TEN_NM, lon=0.0, track=180.0),
    # Head-on, but 1000 ft above and descending 500 ft/min.
    "head-on, descending": dict(
        lat=0.0, lon=_TEN_NM, track=270.0, alt_ft=11_000.0, vs_fpm=-500.0
    ),
}

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
    env.reset(seed=0)
    tag = f"{list(_ENCOUNTERS).index(encounter):02d}"
    own = _place(f"OWN{tag}", 0.0, 0.0, 90.0, 250.0)
    other = _place(f"INTR{tag}", gs_kts=250.0, **_ENCOUNTERS[encounter])
    got = float(field.get_pairs(own, [other])[0])
    assert got == pytest.approx(expected, rel=1e-2, abs=1e-6)
    # The field's own plain statement of the value agrees with the hand answer.
    assert float(field._expected_pair(own, other)) == pytest.approx(
        expected, rel=1e-2, abs=1e-6
    )


def test_turn_radius_at_the_bank_limit(env):
    # r = v^2 / (g tan(bank)): 250 kts at 25 degrees of bank is 1.95 nm.
    env.reset(seed=0)
    idx = _place("OWN", 0.0, 0.0, 90.0, 250.0)
    bs.traf.ap.bankdef[idx] = math.radians(25.0)
    expected = (250 * kts) ** 2 / (9.80665 * math.tan(math.radians(25.0))) / nm
    assert expected == pytest.approx(1.953, abs=1e-3)
    assert float(obs.TurnRadiusNm().get(idx)) == pytest.approx(expected, rel=1e-4)
