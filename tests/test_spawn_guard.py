"""Spawn clearance: one zone, two moments.

Every spawn-clear check uses the same separation - the region's ``spawn_sep_nm``
/ ``spawn_sep_ft``, absolute values that each default to CD's own zone when left
unset - and the two paths differ only in when they look.  ``conflict_free`` looks ahead to the predicted closest approach;
everything else (notably a steady-state ``maintain`` top-up) looks at the
present position, because materialising on top of live traffic is an instant
loss of separation the policy had no chance to avoid, while a conflict that
*develops* later is the task.

The zone is derived, never hardcoded: it follows ``EnvConfig.pz_radius_nm`` /
``pz_height_ft``, which reach CD through ZONER / ZONEDH and never write back to
``bs.settings``.  Reading settings instead would clear spawns against a zone the
detector does not use.
"""

from __future__ import annotations

import warnings

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft as FT
from bluesky.tools.aero import nm as NM

import bluesky_sandbox.core.base_environment as be
from bluesky_sandbox.core.base_environment import BlueskyBaseEnvironment
import bluesky_sandbox.core.spawning as spawning
from bluesky_sandbox.core.state import SpawnPosition
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m, cd_rpz_m
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_PARAMS = {"spd_kts": (240.0, 260.0), "alt_ft": (9_000.0, 11_000.0)}


def _region(**kw) -> SpawnRegion:
    return SpawnRegion(
        bounds=RegionBounds(footprint=BoxFootprint(52.0, 53.0, 4.0, 5.0)),
        n_aircraft=1,
        params=dict(_PARAMS),
        **kw,
    )


# ---- margin resolution ------------------------------------------------- #


def test_region_value_beats_config_default():
    cfg = SpawnConfig(
        regions=[_region(), _region(spawn_sep_nm=12.0)],
        spawn_sep_nm=9.0,
        spawn_sep_ft=2_000.0,
    )
    assert cfg.region_spawn_separation(0) == (9.0, 2_000.0, None)
    assert cfg.region_spawn_separation(1) == (12.0, 2_000.0, None)


def test_unset_stays_none_so_it_resolves_live():
    """Unset must not freeze a number into the spec - CD is asked at spawn time."""
    cfg = SpawnConfig(regions=[_region()])
    assert cfg.region_spawn_separation(0) == (None, None, None)


def test_zero_is_a_real_value_not_an_unset():
    cfg = SpawnConfig(regions=[_region(spawn_sep_nm=0.0)], spawn_sep_nm=9.0)
    with pytest.warns(RuntimeWarning, match="less separation"):
        assert cfg.region_spawn_separation(0)[0] == 0.0


# ---- the present-position check ---------------------------------------- #


class _StubZone:
    """Records the zone it was asked about; answers from one fake aircraft.

    Stands in for :func:`~bluesky_sandbox.sim.geometry.clearance.inside_separation_zone`,
    resolving an unset argument exactly as the real one does, so ``seen`` records
    the *effective* zone rather than whatever the caller happened to pass through.
    """

    def __init__(self, horiz_nm: float = 1e9, vert_ft: float = 1e9) -> None:
        self.horiz_nm = horiz_nm
        self.vert_ft = vert_ft
        self.seen: tuple[float, float] | None = None

    def __call__(self, lat, lon, alt_ft, *, sep_nm=None, sep_ft=None):
        rpz_nm = cd_rpz_m() / NM if sep_nm is None else float(sep_nm)
        vert_ft = cd_hpz_m() / FT if sep_ft is None else float(sep_ft)
        self.seen = (rpz_nm, vert_ft)
        return self.horiz_nm < rpz_nm and self.vert_ft < vert_ft


def _no_predict(*a, **kw):  # pragma: no cover
    raise AssertionError("present-position path must not predict")


def _clear(monkeypatch, zone: _StubZone, **sep) -> bool:
    monkeypatch.setattr(spawning, "inside_separation_zone", zone)
    monkeypatch.setattr(spawning, "predicted_conflict", _no_predict)
    gen = object.__new__(spawning.SpawnGenerator)
    pos = SpawnPosition(lat_deg=52.5, lon_deg=4.5, alt_ft=10_000.0, spd_kts=250.0)
    return spawning.SpawnGenerator._spawn_position_clear(gen, pos, 90.0, False, **sep)


@pytest.mark.parametrize(
    ("horiz_nm", "vert_ft", "expected"),
    [
        (4.0, 200.0, False),    # inside both -> rejected
        (6.0, 200.0, True),     # laterally clear
        (4.0, 5_000.0, True),   # stacked well above: not a loss of separation
    ],
)
def test_breach_needs_both_dimensions(monkeypatch, horiz_nm, vert_ft, expected):
    """Lateral distance alone would starve a stacked maintain region of top-ups."""
    bs.init("sim")
    assert _clear(monkeypatch, _StubZone(horiz_nm, vert_ft)) is expected


def test_unset_resolves_to_cds_own_zone(monkeypatch):
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)

    zone = _StubZone()
    _clear(monkeypatch, zone)
    assert zone.seen == pytest.approx((5.0, 1000.0))  # unset -> CD's zone

    zone = _StubZone()
    _clear(monkeypatch, zone, sep_nm=7.0, sep_ft=1_500.0)
    assert zone.seen == pytest.approx((7.0, 1500.0))  # absolute, as written


def test_zone_tracks_a_resized_protected_zone(monkeypatch):
    """EnvConfig.pz_radius_nm reaches CD via ZONER, never bs.settings."""
    bs.init("sim")

    class _CD:
        rpz_def = 3.0 * NM
        hpz_def = 500.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    zone = _StubZone()
    _clear(monkeypatch, zone)
    assert zone.seen == pytest.approx((3.0, 500.0))


def test_lookahead_is_not_used_by_the_present_position_check(monkeypatch):
    """Nothing is predicted here, so the horizon must not leak into the zone."""
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    zone = _StubZone()
    _clear(monkeypatch, zone, lookahead_s=600.0)
    assert zone.seen == pytest.approx((5.0, 1000.0))


# ---- the drift guard ---------------------------------------------------- #


def test_below_cd_separation_warns(monkeypatch):
    """Absolute values can silently under-cut CD; that must not pass quietly.

    This is the failure mode absolute values reintroduce and additive ones could
    not express: 3 nm looks perfectly reasonable, but against a 5 nm detection
    zone it clears spawns that are already conflicts.
    """
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT
        dtlookahead_def = 300.0

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    cfg = SpawnConfig(regions=[_region(spawn_sep_nm=3.0)])
    with pytest.warns(RuntimeWarning, match=r"spawn_sep_nm=3 nm < 5 nm"):
        cfg.region_spawn_separation(0)


def test_at_or_above_cd_separation_is_silent(monkeypatch):
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT
        dtlookahead_def = 300.0

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    cfg = SpawnConfig(regions=[_region(spawn_sep_nm=8.0), _region()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg.region_spawn_separation(0)
        cfg.region_spawn_separation(1)


def test_below_cd_warns_once_per_region(monkeypatch):
    """A maintain region resolves this every top-up; it must not spam the log."""
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT
        dtlookahead_def = 300.0

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    cfg = SpawnConfig(regions=[_region(spawn_sep_nm=3.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            cfg.region_spawn_separation(0)
    assert len([w for w in caught if w.category is RuntimeWarning]) == 1


# ---- route resolution is shared, not spawn-only ------------------------- #


def test_replace_aircraft_route_resolves_through_the_generator():
    """Regression: route resolution moved to ``SpawnGenerator``.

    ``replace_aircraft_route`` re-resolves an existing aircraft's route
    mid-episode and was left calling ``self._resolve_route_for_aircraft`` on the
    environment after that method moved, which no test exercised. Attribute
    access on ``self`` is invisible to the linter, so this pins the call path.
    """
    calls: list[tuple] = []

    class _Gen:
        def resolve_route(self, callsign, route, rng):
            calls.append((callsign, route))
            return None

    class _Runtime:
        def replace_aircraft_route(self, *a, **kw):
            pass

    class _Monitor:
        def set_aircraft_route(self, *a, **kw):
            pass

        def clear_aircraft_route(self, *a, **kw):
            pass

    env = object.__new__(BlueskyBaseEnvironment)
    env._spawn_generator = _Gen()
    env._runtime = _Runtime()
    env._query_state_monitor = _Monitor()
    env._rng = np.random.default_rng(0)

    BlueskyBaseEnvironment.replace_aircraft_route(env, "AC1", None, ["WP1"])

    assert calls == [("AC1", ["WP1"])]
