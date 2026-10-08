"""Plain one-aircraft and one-pair forms, for the fields' test references.

Every built-in observation field computes in bulk (``_values`` / ``_pairs``)
and states the same value for a single aircraft or pair in ``expected`` /
``expected_pair``, which the checks compare against the bulk result. The helpers
here serve only those references.

Deliberately NOT shared with the batched helpers in :mod:`._pairs`: a reference
built on the code it checks would agree with it by construction. These are
written for one pair at a time, in scalars, the straightforward way. Never
called at runtime.
"""

from __future__ import annotations

import math

import bluesky as bs
from bluesky.tools.aero import ft, nm
from bluesky.tools.geo import qdrdist

from ._route import _active_route_waypoint

__all__ = [
    "one_confpair_value",
    "one_pair_fix_projection",
    "one_pair_motion",
    "one_pair_track_frame",
]


def one_confpair_value(attr: str, own_idx: int, other_idx: int) -> float | None:
    """BlueSky's conflict-detection ``attr`` (``tcpa``, ``tLOS``, ``dcpa``) for
    this pair, or ``None`` when the detector did not flag it.

    The detector keeps one entry per flagged ordered pair in ``confpairs``;
    a later entry for the same pair wins, as it would when writing them out.
    """
    cd = getattr(bs.traf, "cd", None)
    if cd is None:
        return None
    pair = (bs.traf.id[own_idx], bs.traf.id[other_idx])
    values = getattr(cd, attr)
    found = None
    for k, confpair in enumerate(cd.confpairs):
        if tuple(confpair) == pair and k < len(values):
            found = float(values[k])
    return found


def _velocity(idx: int) -> tuple[float, float]:
    track = math.radians(float(bs.traf.trk[idx]))
    gs = float(bs.traf.gs[idx])
    return gs * math.sin(track), gs * math.cos(track)


def one_pair_motion(own_idx: int, other_idx: int) -> tuple[float, float, float, float]:
    """The intruder relative to the ownship, east/north:
    ``(east_m, north_m, east_ms, north_ms)``."""
    qdr, dist_nm = qdrdist(
        float(bs.traf.lat[own_idx]),
        float(bs.traf.lon[own_idx]),
        float(bs.traf.lat[other_idx]),
        float(bs.traf.lon[other_idx]),
    )
    bearing = math.radians(float(qdr))
    dist_m = float(dist_nm) * nm
    own_e, own_n = _velocity(own_idx)
    other_e, other_n = _velocity(other_idx)
    return (
        dist_m * math.sin(bearing),
        dist_m * math.cos(bearing),
        other_e - own_e,
        other_n - own_n,
    )


def one_pair_track_frame(
    own_idx: int, other_idx: int
) -> tuple[float, float, float, float]:
    """Relative position and velocity in the ownship's track frame:
    ``(along_m, cross_m, along_ms, cross_ms)``, along = ahead, cross = right."""
    east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
    track = math.radians(float(bs.traf.trk[own_idx]))
    s, c = math.sin(track), math.cos(track)
    return (
        east * s + north * c,
        east * c - north * s,
        v_east * s + v_north * c,
        v_east * c - v_north * s,
    )


def _eta_and_cpa_to(fix_lat: float, fix_lon: float, idx: int) -> tuple[float, float]:
    """Straight-line time to the closest approach of a fix, and the miss (nm)."""
    qdr, dist_nm = qdrdist(
        fix_lat, fix_lon, float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
    )
    bearing = math.radians(float(qdr))
    dist_m = float(dist_nm) * nm
    r_e, r_n = dist_m * math.sin(bearing), dist_m * math.cos(bearing)
    v_e, v_n = _velocity(idx)
    v2 = max(v_e * v_e + v_n * v_n, 1e-6)
    tstar = max(-(r_e * v_e + r_n * v_n) / v2, 0.0)
    return tstar, math.hypot(r_e + v_e * tstar, r_n + v_n * tstar) / nm


def one_pair_fix_projection(
    own_idx: int, other_idx: int, route_offset: int, own_eta_mode: str
) -> tuple[float, float, float, float] | None:
    """The intruder projected onto the ownship's route fix:
    ``(approach_dist_nm, intruder_eta_s, own_eta_s, vsep_at_fix_ft)``, or
    ``None`` when the ownship has no fix there."""
    fix = _active_route_waypoint(own_idx, route_offset)
    if fix is None:
        return None
    fix_lat, fix_lon = float(fix[0]), float(fix[1])
    intr_eta, intr_cpa_nm = _eta_and_cpa_to(fix_lat, fix_lon, other_idx)
    if own_eta_mode == "route":
        _qdr, own_dist_nm = qdrdist(
            fix_lat, fix_lon, float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx])
        )
        own_eta = float(own_dist_nm) * nm / max(float(bs.traf.gs[own_idx]), 1e-3)
    else:
        own_eta, _own_cpa = _eta_and_cpa_to(fix_lat, fix_lon, own_idx)
    own_alt = float(bs.traf.alt[own_idx]) + float(bs.traf.vs[own_idx]) * own_eta
    intr_alt = float(bs.traf.alt[other_idx]) + float(bs.traf.vs[other_idx]) * intr_eta
    return intr_cpa_nm, intr_eta, own_eta, (intr_alt - own_alt) / ft
