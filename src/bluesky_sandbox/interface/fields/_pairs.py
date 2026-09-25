"""Pair geometry: one ownship, or many, against other aircraft.

Pair helpers take ``own`` and ``other`` INDEX ARRAYS that broadcast against
each other: a scalar ownship against a 1-D intruder list (``get_pairs``), or an
``(k, 1)`` column of ownships against a ``(1, n)`` row of every aircraft
(``get_pair_matrix``). One computation serves both, so they cannot disagree.
"""

from __future__ import annotations

from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import nm
from bluesky.tools.geo import qdrdist

from bluesky_sandbox.sim.geometry.conflict import conflict_geometry

from ._common import _M_TO_FT, _indices_array, _traf_array, on_reset
from ._route import _active_route_waypoint
from .base import ObsField


def _pair_index_grid(own_indices: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(k, 1)`` ownships and a ``(1, n)`` row of every live aircraft."""
    own = _indices_array(own_indices).reshape(-1, 1)
    every = np.arange(int(bs.traf.ntraf), dtype=np.intp).reshape(1, -1)
    return own, every


def _per_own(own: np.ndarray, value) -> np.ndarray:
    """``value(own_idx)`` for each ownship, shaped like ``own`` to broadcast."""
    return np.array(
        [float(value(int(o))) for o in own.ravel()], dtype=np.float64
    ).reshape(own.shape)


class _BroadcastPairs:
    """A pair field defined by one broadcasting function, :meth:`_pairs`.

    ``get_pair``, ``get_pairs`` and ``get_pair_matrix`` are all that function
    evaluated on different index shapes, so they agree by construction.
    """

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        return self._pairs(_indices_array(own_idx), _indices_array(other_indices))

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        own, every = _pair_index_grid(own_indices)
        return self._pairs(own, every)


def _per_aircraft(field: ObsField, idx: np.ndarray) -> np.ndarray:
    """An ownship field's value at each index of ``idx``, shaped like it."""
    flat = np.asarray(field.get_many(idx.ravel()), dtype=np.float64)
    return flat.reshape(idx.shape + flat.shape[1:])


# BlueSky's detector keeps one entry per conflict pair (``confpairs``) in each
# of its arrays. Laid out as an ``n x n`` matrix once per sim time, so every
# ownship reads its row instead of rescanning the pair list.
_CD_PAIR_CACHE: dict[str, tuple[tuple, tuple[np.ndarray, np.ndarray]]] = {}
on_reset(lambda _seed: _CD_PAIR_CACHE.clear())


def _cd_pair_matrix(attr: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(values, present)`` of CD array ``attr`` by ``[own, other]`` index."""
    cd = getattr(bs.traf, "cd", None)
    if cd is None or len(cd.confpairs) == 0:
        return None
    ids = bs.traf.id
    key = (float(bs.sim.simt), tuple(ids), id(cd.confpairs), len(cd.confpairs))
    cached = _CD_PAIR_CACHE.get(attr)
    if cached is not None and cached[0] == key:
        return cached[1]
    n = len(ids)
    values = np.zeros((n, n), dtype=np.float64)
    present = np.zeros((n, n), dtype=bool)
    row = {acid: i for i, acid in enumerate(ids)}
    source = np.asarray(getattr(cd, attr), dtype=np.float64)
    for k, (left, right) in enumerate(cd.confpairs):
        i, j = row.get(left), row.get(right)
        if i is not None and j is not None and k < source.size:
            values[i, j] = source[k]
            present[i, j] = True
    _CD_PAIR_CACHE[attr] = (key, (values, present))
    return values, present


def _cd_pair_values(
    attr: str, own: np.ndarray, other: np.ndarray, fill, divisor: float = 1.0
) -> np.ndarray:
    """CD array ``attr`` for each (own, other) pair in conflict, else ``fill``."""
    shape = np.broadcast_shapes(own.shape, other.shape)
    fill = np.broadcast_to(np.asarray(fill, dtype=np.float64), shape)
    matrix = _cd_pair_matrix(attr)
    if matrix is None:
        return fill.copy()
    values, present = matrix
    return np.where(present[own, other], values[own, other] / divisor, fill)


class _GeomPairs:
    """The per-step conflict geometry, sliced by (own, other) index arrays.

    Stands in for :class:`ConflictView` wherever only its raw per-pair arrays
    are read - the windowed conflict functions are elementwise - but over any
    broadcastable index shape, so one ownship's row and every ownship's matrix
    come from the same code.
    """

    __slots__ = ("_geom", "_other", "_own")

    def __init__(self, own: np.ndarray, other: np.ndarray) -> None:
        self._geom = conflict_geometry()
        self._own = own
        self._other = other

    def __getattr__(self, name: str) -> np.ndarray:
        return getattr(self._geom, name)[self._own, self._other]


def _pair_qdr_dist(own_idx: Any, other_indices: Any) -> tuple[np.ndarray, np.ndarray]:
    own = _indices_array(own_idx)
    other = _indices_array(other_indices)
    lat, lon = _traf_array("lat"), _traf_array("lon")
    qdr, dist = qdrdist(lat[own], lon[own], lat[other], lon[other])
    shape = np.broadcast_shapes(own.shape, other.shape)
    return (
        np.broadcast_to(np.asarray(qdr, dtype=np.float64), shape),
        np.broadcast_to(np.asarray(dist, dtype=np.float64), shape),
    )


def _pair_relative_motion(
    own_idx: Any,
    other_indices: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return relative position and velocity as east/north arrays in SI units."""
    qdr, dist_nm = _pair_qdr_dist(own_idx, other_indices)
    own = _indices_array(own_idx)
    other = _indices_array(other_indices)
    qdrrad = np.radians(qdr)
    dist_m = dist_nm * nm
    rel_east_m = dist_m * np.sin(qdrrad)
    rel_north_m = dist_m * np.cos(qdrrad)

    track = np.radians(_traf_array("trk"))
    gs = _traf_array("gs")
    own_east_ms = gs[own] * np.sin(track[own])
    own_north_ms = gs[own] * np.cos(track[own])
    other_east_ms = gs[other] * np.sin(track[other])
    other_north_ms = gs[other] * np.cos(track[other])

    return (
        rel_east_m,
        rel_north_m,
        other_east_ms - own_east_ms,
        other_north_ms - own_north_ms,
    )


def _pair_horizontal_tcpa_s(own_idx: Any, other_indices: Any) -> np.ndarray:
    rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
        own_idx,
        other_indices,
    )
    rel_speed2 = np.maximum(
        rel_east_ms * rel_east_ms + rel_north_ms * rel_north_ms,
        1e-6,
    )
    return -(rel_east_m * rel_east_ms + rel_north_m * rel_north_ms) / rel_speed2


def _track_frame(own_idx: Any, other_indices: Any):
    """Intruder relative position (m) and velocity (m/s) in the OWNSHIP TRACK
    frame: (along, cross) where along = down the own velocity vector (ahead +),
    cross = to the right of it. Rotation-invariant (rotates with own track), so
    the encounter reads the same at any absolute heading."""
    rel_e, rel_n, ve, vn = _pair_relative_motion(own_idx, other_indices)
    trk = np.radians(_traf_array("trk"))[_indices_array(own_idx)]
    s, c = np.sin(trk), np.cos(trk)
    along = rel_e * s + rel_n * c  # projection on own track (compass sin/cos)
    cross = rel_e * c - rel_n * s  # 90 deg clockwise (to the right)
    v_along = ve * s + vn * c
    v_cross = ve * c - vn * s
    return along, cross, v_along, v_cross


def _track_frame_at_cpa(own_idx: Any, other_indices: Any):
    """Track-frame relative position AT the predicted (constant-velocity) CPA:
    (along, cross) in meters = now-position + relative-velocity * tcpa, with tcpa
    clamped to >=0 (already-passed encounters read at 'now'). Its magnitude equals
    the horizontal miss dcpa; its cross-sign says which side the intruder passes."""
    along, cross, v_along, v_cross = _track_frame(own_idx, other_indices)
    # tcpa from the SAME track-frame primitives (it is rotation-invariant), so
    # position and velocity share one projection and |result| == dcpa exactly.
    v2 = np.maximum(v_along * v_along + v_cross * v_cross, 1e-9)
    tcpa = np.maximum(-(along * v_along + cross * v_cross) / v2, 0.0)  # future CPA only
    return along + v_along * tcpa, cross + v_cross * tcpa


def _fix_projection(
    own_idx: Any,
    other_indices: Any,
    route_offset: int,
    own_eta_mode: str = "projection",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project observed motion onto the *ownship's* active fix (a merge signal).

    Returns ``(approach_dist_nm, intruder_eta_s, own_eta_s, vsep_at_fix_ft,
    has_fix)``, broadcast over the ``own`` / ``other`` index arrays
    where each intruder's straight-line motion is projected to the closest
    approach of the ownship's own fix ``F`` (offset ``0`` = active leg, ``1`` =
    next leg): how near its current trajectory passes ``F`` (horizontal), when
    (ETA), and the signed altitude gap at the merge (intruder minus ownship,
    each projected to its own fix arrival by its vertical speed). ``has_fix`` is
    False for an ownship with no usable fix there; its other values are
    placeholders the caller replaces.

    Privacy: uses only ``F`` (the ownship's *own* plan, offset 0/1 both being
    own legs) and each intruder's *observed* position/velocity/altitude/VS
    (radar/ADS-B grade) - never the intruder's flight plan. This is the
    observable counterpart of reading ``ActiveRouteWaypoint*`` on the intruder,
    which would expose unshared intent and is therefore critic-only.

    ``own_eta_mode`` selects how the OWNSHIP's own ETA is formed, and the two
    sides are deliberately NOT symmetric because the information is not:

    * ``"projection"`` (default, and what every task before v52 used) applies
      the same kinematic closest-approach projection to the ownship. Since
      ``t* = (range / groundspeed) * cos(theta)`` for ``theta`` the angle
      between track and bearing-to-fix, the ownship's own ETA is scaled by its
      instantaneous heading: measured on a v51 rollout it reads 0.998x the
      range/speed ETA while tracking the fix, but 0.29x at 45-90 deg off and
      0.0 beyond 90 deg. That collapse is an artifact - the ownship is *not*
      going to fly straight forever, route guidance turns it back - and it is
      driven by the ownship's own ACTION, which is the observation-feedback
      shape that caused the ``PrevActionNorm`` hysteresis loop.
    * ``"route"`` uses ``range / groundspeed`` for the ownship instead: no
      ``cos`` factor, no clamp. Legitimate precisely because the ownship's plan
      is its OWN knowledge - it knows it is going to ``F``. Stable under
      maneuvering, and monotone in groundspeed, so decelerating to open a gap
      raises the ownship ETA cleanly (speed is the instrument that resolves an
      in-trail merge, so this is the channel that has to stay readable).

    The INTRUDER side keeps the kinematic projection under both modes, and must:
    ``range / groundspeed`` for an intruder would assert "it is flying to MY
    fix", which is exactly the intent this field exists to avoid assuming. Its
    ``cos`` collapse is the honest reading - "this one is not going to my fix" -
    and it is what makes a large ``IntruderFixApproachDistNm`` meaningful.

    Mixing the two models does not corrupt the difference where it is used: a
    genuine merge partner is by construction tracking the fix, so its ``theta``
    is near 0 and its projection equals its range/speed ETA anyway (measured
    0.998x within 5 deg, 0.987x within 20 deg). The models only diverge once the
    intruder is off-bearing to the fix, where the approach-distance gate has
    already flagged the pair as irrelevant.
    """
    if own_eta_mode not in ("projection", "route"):
        raise ValueError(
            f"own_eta_mode must be 'projection' or 'route', got {own_eta_mode!r}."
        )
    own = _indices_array(own_idx)
    other = _indices_array(other_indices)
    fixes = [_active_route_waypoint(int(o), route_offset) for o in own.ravel()]
    has_fix = np.array([fix is not None for fix in fixes]).reshape(own.shape)
    # A fix-less ownship projects onto (0, 0) - finite arithmetic, masked out.
    fix_lat = np.array(
        [0.0 if fix is None else float(fix[0]) for fix in fixes], dtype=np.float64
    ).reshape(own.shape)
    fix_lon = np.array(
        [0.0 if fix is None else float(fix[1]) for fix in fixes], dtype=np.float64
    ).reshape(own.shape)
    lat, lon = _traf_array("lat"), _traf_array("lon")
    trk, gs = _traf_array("trk"), _traf_array("gs")
    alt, vs = _traf_array("alt"), _traf_array("vs")

    def _eta_and_cpa(idx: np.ndarray):
        # Position of each aircraft relative to its ownship's fix, east/north m.
        qdr, dist_nm = qdrdist(fix_lat, fix_lon, lat[idx], lon[idx])
        qdrrad = np.radians(np.asarray(qdr, dtype=np.float64))
        dist_m = np.asarray(dist_nm, dtype=np.float64) * nm
        r_e = dist_m * np.sin(qdrrad)
        r_n = dist_m * np.cos(qdrrad)
        trkrad = np.radians(trk[idx])
        v_e = gs[idx] * np.sin(trkrad)
        v_n = gs[idx] * np.cos(trkrad)
        v2 = np.maximum(v_e * v_e + v_n * v_n, 1e-6)
        # Future closest approach only (t* clamped >= 0): an aircraft receding
        # from the fix reads its current distance, not a past pass.
        tstar = np.maximum(-(r_e * v_e + r_n * v_n) / v2, 0.0)
        cpa_e = r_e + v_e * tstar
        cpa_n = r_n + v_n * tstar
        cpa_nm = np.sqrt(cpa_e * cpa_e + cpa_n * cpa_n) / nm
        return tstar, cpa_nm

    intr_eta, intr_cpa_nm = _eta_and_cpa(other)
    if own_eta_mode == "route":
        # The ownship knows its own plan, so its ETA is a property of the route
        # and the speed it is flying - not of the heading it happens to hold
        # this step. See the docstring for why the two sides differ.
        _own_qdr, own_dist_nm = qdrdist(fix_lat, fix_lon, lat[own], lon[own])
        own_eta_s = (
            np.asarray(own_dist_nm, dtype=np.float64) * nm / np.maximum(gs[own], 1e-3)
        )
    else:
        own_eta_s, _own_cpa = _eta_and_cpa(own)
    # Altitude each aircraft is projected to hold when it reaches the fix
    # (current alt + VS x its own ETA); the merge conflict is where these
    # coincide. Signed intruder-minus-ownship, in feet.
    own_alt_at_fix = alt[own] + vs[own] * own_eta_s
    intr_alt_at_fix = alt[other] + vs[other] * intr_eta
    vsep_ft = (intr_alt_at_fix - own_alt_at_fix) * _M_TO_FT
    return intr_cpa_nm, intr_eta, own_eta_s, vsep_ft, has_fix
