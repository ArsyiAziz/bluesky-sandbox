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
import pytest
from bluesky.tools.aero import ft as FT
from bluesky.tools.aero import nm as NM

import bluesky_sandbox.core.base_environment as be
from bluesky_sandbox.core.base_environment import BlueskyBaseEnvironment, SpawnPosition
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
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


class _StubRuntime:
    """Records the zone it was asked about; answers from one fake aircraft."""

    def __init__(self, horiz_nm: float = 1e9, vert_ft: float = 1e9) -> None:
        self.horiz_nm = horiz_nm
        self.vert_ft = vert_ft
        self.seen: tuple[float, float] | None = None

    def inside_separation_zone(self, lat, lon, alt_ft, *, min_sep_nm, min_sep_ft=None):
        self.seen = (min_sep_nm, min_sep_ft)
        return self.horiz_nm < min_sep_nm and self.vert_ft < min_sep_ft

    def predicted_conflict(self, *a, **kw):  # pragma: no cover
        raise AssertionError("present-position path must not predict")


def _clear(rt: _StubRuntime, **sep) -> bool:
    env = object.__new__(BlueskyBaseEnvironment)
    env._runtime = rt
    pos = SpawnPosition(lat_deg=52.5, lon_deg=4.5, alt_ft=10_000.0, spd_kts=250.0)
    return BlueskyBaseEnvironment._spawn_position_clear(env, pos, 90.0, False, **sep)


@pytest.mark.parametrize(
    ("horiz_nm", "vert_ft", "expected"),
    [
        (4.0, 200.0, False),    # inside both -> rejected
        (6.0, 200.0, True),     # laterally clear
        (4.0, 5_000.0, True),   # stacked well above: not a loss of separation
    ],
)
def test_breach_needs_both_dimensions(horiz_nm, vert_ft, expected):
    """Lateral distance alone would starve a stacked maintain region of top-ups."""
    bs.init("sim")
    assert _clear(_StubRuntime(horiz_nm, vert_ft)) is expected


def test_unset_resolves_to_cds_own_zone(monkeypatch):
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)

    rt = _StubRuntime()
    _clear(rt)
    assert rt.seen == pytest.approx((5.0, 1000.0))  # unset -> CD's zone

    rt = _StubRuntime()
    _clear(rt, sep_nm=7.0, sep_ft=1_500.0)
    assert rt.seen == pytest.approx((7.0, 1500.0))  # absolute, as written


def test_zone_tracks_a_resized_protected_zone(monkeypatch):
    """EnvConfig.pz_radius_nm reaches CD via ZONER, never bs.settings."""
    bs.init("sim")

    class _CD:
        rpz_def = 3.0 * NM
        hpz_def = 500.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    rt = _StubRuntime()
    _clear(rt)
    assert rt.seen == pytest.approx((3.0, 500.0))


def test_lookahead_is_not_used_by_the_present_position_check(monkeypatch):
    """Nothing is predicted here, so the horizon must not leak into the zone."""
    bs.init("sim")

    class _CD:
        rpz_def = 5.0 * NM
        hpz_def = 1000.0 * FT

    monkeypatch.setattr(be.bs.traf, "cd", _CD(), raising=False)
    rt = _StubRuntime()
    _clear(rt, lookahead_s=600.0)
    assert rt.seen == pytest.approx((5.0, 1000.0))


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
