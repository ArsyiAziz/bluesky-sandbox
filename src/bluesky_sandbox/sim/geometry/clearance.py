"""Spawn clearance: is a *candidate* state clear of the live traffic?

The companion to :mod:`.conflict`, which computes the cached ``N x N`` geometry
*among* live aircraft. These functions ask a different question - about a point
that is not in ``bs.traf`` yet - so they cannot reuse that cache and instead
extrapolate the candidate against every existing aircraft one-to-many.

Two moments, one zone:

* :func:`inside_separation_zone` - the *present* position only, which is what a
  steady-state ``maintain`` top-up needs.
* :func:`predicted_conflict` - the predicted closest approach over the
  lookahead, so the aircraft is not on course to conflict either.

Both default every margin to the live CD values via :func:`~.conflict.cd_rpz_m`
/ :func:`~.conflict.cd_hpz_m` / :func:`~.conflict.cd_lookahead_s` - *not*
``bs.settings``. ``EnvConfig.pz_radius_nm`` / ``pz_height_ft`` / ``lookahead_s``
reach CD through ZONER / ZONEDH / DTLOOK and never write back to settings, so
reading settings would clear spawns against a zone the detector does not use.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vcas2tas

# ``qdrdist`` broadcasts a scalar position against the traffic arrays and is the
# exact-sphere pair: ``latlondist`` computes the same distance without a bearing
# but divides by zero when both points sit on the equator, and ``kwikqdrdist``'s
# flat-earth projection is only trustworthy over the short ranges it is used for
# in :func:`predicted_conflict`.
from bluesky.tools.geo import kwikqdrdist, qdrdist

from .conflict import cd_hpz_m, cd_lookahead_s, cd_rpz_m

__all__ = ["inside_separation_zone", "predicted_conflict"]


def inside_separation_zone(
    lat_deg: float,
    lon_deg: float,
    alt_ft: float,
    *,
    sep_nm: float | None = None,
    sep_ft: float | None = None,
) -> bool:
    """Whether any live aircraft shares this point's separation zone.

    A loss of separation needs *both* dimensions - inside the horizontal radius
    **and** inside the vertical band - so a candidate laterally close to traffic
    but thousands of feet above it is clear, exactly as BlueSky's CD would score
    it. Checking horizontal distance alone would reject stacked traffic that is
    properly separated in altitude.

    Each argument defaults to the live CD protected zone, exactly as
    :func:`predicted_conflict` does - the two score the *same* zone and differ
    only in when they look at it. A non-positive value in either dimension means
    no zone, so nothing can breach it.
    """
    n = int(bs.traf.ntraf)
    if n == 0:
        return False
    rpz_nm = cd_rpz_m() / nm if sep_nm is None else float(sep_nm)
    vert_ft = cd_hpz_m() / ft if sep_ft is None else float(sep_ft)
    if rpz_nm <= 0.0 or vert_ft <= 0.0:
        return False
    _qdr, d_nm = qdrdist(
        float(lat_deg),
        float(lon_deg),
        np.asarray(bs.traf.lat[:n], dtype=np.float64),
        np.asarray(bs.traf.lon[:n], dtype=np.float64),
    )
    d_ft = np.abs(np.asarray(bs.traf.alt[:n], dtype=np.float64) / ft - float(alt_ft))
    return bool(np.any((d_nm < rpz_nm) & (d_ft < vert_ft)))


def predicted_conflict(
    lat_deg: float,
    lon_deg: float,
    alt_ft: float,
    hdg_deg: float,
    cas_kts: float,
    *,
    sep_nm: float | None = None,
    sep_ft: float | None = None,
    lookahead_s: float | None = None,
    wind_kts: float = 0.0,
    wind_dir_deg: float = 0.0,
) -> bool:
    """True if a candidate spawn state is in a predicted conflict.

    Mirrors BlueSky's state-based CD: for the candidate's straight-line motion
    against every existing aircraft, a conflict is a separation breach
    (horizontal ``sep_nm`` and vertical ``sep_ft``) at closest point of approach
    within ``lookahead_s``.

    Each margin defaults to the live CD value, so an unset spawn requirement is
    exactly what the detector will score once the episode runs. Passing a value
    smaller than CD's own zone clears spawns that CD already counts as conflicts
    - deliberate for a conflict-seeded scenario, a mistake otherwise;
    ``SpawnConfig.region_spawn_separation`` warns about it rather than silently
    obliging.

    ``wind_kts`` / ``wind_dir_deg`` are the steady wind (the caller's
    ``EnvConfig``); pass the mean field, not a gust - see below.
    """
    n = int(bs.traf.ntraf)
    if n == 0:
        return False
    rpz = cd_rpz_m() if sep_nm is None else float(sep_nm) * nm  # m
    hpz = cd_hpz_m() if sep_ft is None else float(sep_ft) * ft  # m
    look = cd_lookahead_s() if lookahead_s is None else float(lookahead_s)

    alt_m = float(alt_ft) * ft
    tas = float(vcas2tas(float(cas_kts) * kts, alt_m))
    trk = np.radians(float(hdg_deg))
    own_e, own_n = tas * np.sin(trk), tas * np.cos(trk)
    # Existing traffic velocities (``bs.traf.gs``) include the wind field; add
    # the mean wind to the candidate's air vector so both sides of the relative
    # velocity are ground-referenced. Gusts are zero-mean and per-step, so the
    # steady component is the right one to use here.
    if float(wind_kts) > 0.0:
        wind_ms = float(wind_kts) * kts
        wind_rad = np.radians(float(wind_dir_deg))
        own_n += -wind_ms * np.cos(wind_rad)
        own_e += -wind_ms * np.sin(wind_rad)

    lat = np.asarray(bs.traf.lat[:n], dtype=np.float64)
    lon = np.asarray(bs.traf.lon[:n], dtype=np.float64)
    qdr, dist_nm = kwikqdrdist(
        np.full(n, float(lat_deg)), np.full(n, float(lon_deg)), lat, lon
    )
    qdr = np.radians(np.asarray(qdr, dtype=np.float64))
    dist_m = np.asarray(dist_nm, dtype=np.float64) * nm
    rel_e, rel_n = dist_m * np.sin(qdr), dist_m * np.cos(qdr)

    trk_o = np.radians(np.asarray(bs.traf.trk[:n], dtype=np.float64))
    gs_o = np.asarray(bs.traf.gs[:n], dtype=np.float64)
    rel_ve = gs_o * np.sin(trk_o) - own_e
    rel_vn = gs_o * np.cos(trk_o) - own_n

    v2 = np.maximum(rel_ve * rel_ve + rel_vn * rel_vn, 1e-9)
    tcpa = -(rel_e * rel_ve + rel_n * rel_vn) / v2
    dcpa = np.hypot(rel_e + rel_ve * tcpa, rel_n + rel_vn * tcpa)

    alt_o = np.asarray(bs.traf.alt[:n], dtype=np.float64)
    vs_o = np.asarray(bs.traf.vs[:n], dtype=np.float64)

    # Entry-time gating, exactly BlueSky ``StateBased.detect``: a pair is in
    # conflict when the PZ is *entered* within the lookahead (``tinconf =
    # max(tinhor, tinver) <= look``), not when the CPA falls within it - a
    # slow-closing shallow encounter dwells inside the PZ long before its CPA,
    # so a CPA gate under-flags exactly the pairs the intrinsic cost (also
    # tinconf-based) would charge at spawn. The candidate is level, so the
    # relative vertical rate is the intruder's ``vs`` alone.
    swhor = dcpa < rpz
    vrel = np.sqrt(v2)
    dtinhor = np.sqrt(np.maximum(0.0, rpz * rpz - dcpa * dcpa)) / vrel
    tinhor = np.where(swhor, tcpa - dtinhor, np.inf)
    touthor = np.where(swhor, tcpa + dtinhor, -np.inf)

    d0 = alt_o - alt_m  # signed relative altitude (m)
    dvs = np.where(np.abs(vs_o) < 1e-6, 1e-6, vs_o)  # guard, as BlueSky
    t_hi = (hpz - d0) / dvs
    t_lo = (-hpz - d0) / dvs
    tinver = np.minimum(t_hi, t_lo)
    toutver = np.maximum(t_hi, t_lo)

    tinconf = np.maximum(tinhor, tinver)
    toutconf = np.minimum(touthor, toutver)
    conflict = swhor & (tinconf <= toutconf) & (toutconf > 0.0) & (tinconf <= look)
    # Current PZ breach: catches the diverging (encounter already past)
    # born-in-LoS case the predictive window misses.
    in_los_now = (dist_m < rpz) & (np.abs(d0) < hpz)
    return bool(np.any(conflict | in_los_now))
