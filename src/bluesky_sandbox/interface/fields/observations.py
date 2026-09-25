from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar

import bluesky as bs
import numpy as np
from bluesky.tools.aero import crossoveralt, ft, g0, kts, nm, vcas2tas
from bluesky.tools.geo import kwikqdrdist, qdrdist

from bluesky_sandbox.sim.geometry.conflict import (
    ConflictView,
    predicted_tlos_s,
    windowed_min_hsep_nm,
    windowed_min_vsep_ft,
    windowed_signed_vsep_at_entry_ft,
)
from bluesky_sandbox.sim.geometry.conflict import (
    cd_hpz_m as _cd_hpz_m,
)
from bluesky_sandbox.sim.geometry.conflict import (
    cd_lookahead_s as _cd_lookahead_s,
)
from bluesky_sandbox.sim.geometry.conflict import (
    cd_rpz_m as _cd_rpz_m,
)
from bluesky_sandbox.sim.performance.envelope import (
    _warn_type_data_mismatch,
    active_performance_model,
)
from bluesky_sandbox.sim.performance.models import type_limits
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from . import _state
from ._common import (
    _M_TO_FT,
    _MIN_DYNAMIC_SPAN,
    _MIN_GS_MS,
    _MS_TO_FTMIN,
    _MS_TO_KTS,
    _BroadcastObs,
    _indices_array,
    _signed_angle_delta_deg,
    _traf_array,
)
from ._lag import _lag_ring, _LagHistoryBacked, _register_lag_depth
from ._pairs import (
    _BroadcastPairs,
    _cd_pair_values,
    _fix_projection,
    _GeomPairs,
    _pair_horizontal_tcpa_s,
    _pair_qdr_dist,
    _pair_relative_motion,
    _per_aircraft,
    _per_own,
    _track_frame,
    _track_frame_at_cpa,
)
from ._reference import (
    one_confpair_value,
    one_pair_fix_projection,
    one_pair_motion,
    one_pair_track_frame,
)
from ._route import _active_route_waypoint, _route_along_distance_nm
from ._state import (
    _LAST_NORM_ACTION,
    _TIME_IN_ENV,
    _CommBacked,
    _LastActionBacked,
    _TimeInEnvBacked,
    comm_messages,
)
from .base import ObsField, ObsMeta, ObsQuantity, PairObsField, Unit

# Pair helpers take ``own`` and ``other`` INDEX ARRAYS that broadcast against
# each other: a scalar ownship against a 1-D intruder list (``get_pairs``), or an
# ``(k, 1)`` column of ownships against a ``(1, n)`` row of every aircraft
# (``get_pair_matrix``). One computation serves both, so they cannot disagree.


def _scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _phase_matches(left: Any, right: Any) -> bool:
    left = _scalar_value(left)
    right = _scalar_value(right)
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).casefold() == str(right).casefold()


def _phase_key(value: Any) -> tuple[str, float | str]:
    value = _scalar_value(value)
    try:
        return "number", float(value)
    except (TypeError, ValueError):
        return "label", str(value).casefold()


def _with_derived_bounds(
    field: ObsField,
    derived: Mapping[str, tuple[float, float]],
) -> ObsField:
    bounds = derived.get(field.meta.name)
    if bounds is not None and not field.bounds_overridden:
        return replace(field, low=bounds[0], high=bounds[1])
    return field


@dataclass(frozen=True)
class LatDeg(_BroadcastObs, ObsField):
    """Latitude in degrees.

    Metadata:
        name: lat_deg
        unit: deg
        quantity: latitude
    """

    meta = ObsMeta("lat_deg", Unit.DEG, ObsQuantity.LATITUDE)
    low: Annotated[float, "latitude degrees"] = -90.0
    high: Annotated[float, "latitude degrees"] = 90.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.lat[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.lat[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class LonDeg(_BroadcastObs, ObsField):
    """Longitude in degrees.

    Metadata:
        name: lon_deg
        unit: deg
        quantity: longitude
    """

    meta = ObsMeta("lon_deg", Unit.DEG, ObsQuantity.LONGITUDE)
    low: Annotated[float, "longitude degrees"] = -180.0
    high: Annotated[float, "longitude degrees"] = 180.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.lon[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.lon[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HdgDeg(_BroadcastObs, ObsField):
    """Aircraft heading in degrees.

    Metadata:
        name: hdg_deg
        unit: deg
        quantity: heading
        circular: True
    """

    meta = ObsMeta("hdg_deg", Unit.DEG, ObsQuantity.HEADING, circular=True)
    low: Annotated[float, "heading degrees"] = 0.0
    high: Annotated[float, "heading degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.hdg[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.hdg[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TrkDeg(_BroadcastObs, ObsField):
    """Aircraft track angle in degrees.

    Metadata:
        name: trk_deg
        unit: deg
        quantity: track
        circular: True
    """

    meta = ObsMeta("trk_deg", Unit.DEG, ObsQuantity.TRACK, circular=True)
    low: Annotated[float, "track degrees"] = 0.0
    high: Annotated[float, "track degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.trk[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.trk[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# --------------------------------------------------------------------------- #
# Active route waypoint (per-aircraft, name-free)                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ActiveRouteWaypointField(_BroadcastObs, ObsField):
    """Reads a route fix relative to the aircraft's active BlueSky leg.

    Unlike the ``Waypoint``/``ActiveWaypoint`` queryable fields, this needs no
    named queryable: it observes whatever target sits on the aircraft's route -
    e.g. a per-aircraft destination sampled at spawn (which may be an ad-hoc,
    unnamed lat/lon). Returns ``0`` when there is no usable fix at
    ``route_offset``.

    ``route_offset`` (default ``0``, the active leg) generalizes every field
    in this family to look ahead: instantiate the *same* class twice in a
    design with ``route_offset=0`` and ``route_offset=1`` to give the policy
    both the current and next fix (mirrors how ``IntruderCommMessage`` reuses
    one class across ``channel=0``/``1``) - realistic lookahead, since a real
    flight plan's next leg is already known, unlike unshared intent.
    """

    route_offset: Annotated[
        int, "route index offset from the active leg (0=active, 1=next, ...)"
    ] = 0

    def _active_wp(
        self, idx: int
    ) -> tuple[float, float, float | None, float | None] | None:
        return _active_route_waypoint(idx, self.route_offset)

    def _waypoints(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(found, lat, lon, alt_m, spd_ms)`` of each aircraft's route fix.

        The lookup walks BlueSky's per-aircraft ``Route`` objects, so it is a
        loop; everything computed from it is not. A fix-less aircraft has
        ``found`` False and its own position as a placeholder fix; a missing
        constraint is NaN.
        """
        fixes = [self._active_wp(int(i)) for i in indices]
        found = np.array([fix is not None for fix in fixes], dtype=bool)

        def column(k: int, fallback: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    fallback[n] if fix is None or fix[k] is None else float(fix[k])
                    for n, fix in enumerate(fixes)
                ],
                dtype=np.float64,
            )

        lat, lon = _traf_array("lat")[indices], _traf_array("lon")[indices]
        missing = np.full(len(fixes), np.nan)
        return found, column(0, lat), column(1, lon), column(2, missing), column(3, missing)

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ActiveRouteWaypointDistanceNm(_ActiveRouteWaypointField):
    """Distance from ownship to its active route waypoint, nm (0 if none)."""

    meta = ObsMeta("active_route_waypoint_distance_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: float = 0.0
    high: float = 200.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        found, wp_lat, wp_lon, _alt, _spd = self._waypoints(indices)
        lat, lon = _traf_array("lat")[indices], _traf_array("lon")[indices]
        _qdr, dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        return np.where(found, dist, 0.0)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None:
            return 0.0
        lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        return float(kwikqdrdist(lat, lon, wp[0], wp[1])[1])


@dataclass(frozen=True)
class ActiveRouteWaypointBearingDeg(_ActiveRouteWaypointField):
    """True bearing from ownship to its active route waypoint, deg (0 if none)."""

    meta = ObsMeta("active_route_waypoint_bearing_deg", Unit.DEG, ObsQuantity.BEARING, circular=True)
    low: float = 0.0
    high: float = 360.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        found, wp_lat, wp_lon, _alt, _spd = self._waypoints(indices)
        lat, lon = _traf_array("lat")[indices], _traf_array("lon")[indices]
        qdr, _dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        return np.where(found, np.asarray(qdr, dtype=np.float64) % 360.0, 0.0)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None:
            return 0.0
        lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        return float(kwikqdrdist(lat, lon, wp[0], wp[1])[0]) % 360.0


@dataclass(frozen=True)
class ActiveRouteWaypointTrackErrorDeg(_ActiveRouteWaypointField):
    """Signed track error from ownship track toward its active route waypoint."""

    meta = ObsMeta("active_route_waypoint_track_error_deg", Unit.DEG, ObsQuantity.TRACK, circular=True)
    low: float = -180.0
    high: float = 180.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        found, wp_lat, wp_lon, _alt, _spd = self._waypoints(indices)
        lat, lon = _traf_array("lat")[indices], _traf_array("lon")[indices]
        qdr, _dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        error = _signed_angle_delta_deg(
            np.asarray(qdr, dtype=np.float64), _traf_array("trk")[indices]
        )
        return np.where(found, error, 0.0)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None:
            return 0.0
        lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        bearing = float(kwikqdrdist(lat, lon, wp[0], wp[1])[0])
        return (bearing - float(bs.traf.trk[idx]) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class ActiveRouteWaypointValid(_ActiveRouteWaypointField):
    """1.0 when the aircraft has a usable active route waypoint, else 0.0.

    Presence flag for the other ``ActiveRouteWaypoint*`` fields, which fall back
    to a sentinel ``0.0`` when there is no active waypoint - a value that
    collides with legitimate zeros (on-track track error, zero distance at the
    fix, due-north bearing). Pair this field with them so a consumer can tell
    "no active waypoint" apart from those real states. Chiefly for reading the
    fields on *intruders* (background or route-exhausted traffic often have no
    active waypoint); the ownship of a routed, delete-on-reach task rarely hits
    the sentinel.
    """

    meta = ObsMeta(
        "active_route_waypoint_valid", Unit.UNITLESS, ObsQuantity.INDICATOR
    )
    low: float = 0.0
    high: float = 1.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        found, *_rest = self._waypoints(indices)
        return found.astype(np.float64)

    def _expected(self, idx: int) -> Any:
        return float(self._active_wp(idx) is not None)


@dataclass(frozen=True)
class ActiveRouteWaypointHasAltConstraint(_ActiveRouteWaypointField):
    """1.0 when the active route waypoint carries an altitude gate, else 0.0.

    :class:`ActiveRouteWaypointAltDiffFt` reports ``0.0`` both when the aircraft
    is exactly on the waypoint altitude and when the leg has no altitude
    constraint at all. Those are opposite situations - hold this level, versus
    any level will do - so a design that mixes constrained and unconstrained
    fixes must pair this flag with the error field, or the policy cannot tell
    which one it is looking at.
    """

    meta = ObsMeta(
        "active_route_waypoint_has_alt_constraint",
        Unit.UNITLESS,
        ObsQuantity.INDICATOR,
    )
    low: float = 0.0
    high: float = 1.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, alt_m, _spd = self._waypoints(indices)
        return (~np.isnan(alt_m)).astype(np.float64)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        return float(wp is not None and wp[2] is not None)


@dataclass(frozen=True)
class ActiveRouteWaypointHasSpdConstraint(_ActiveRouteWaypointField):
    """1.0 when the active route waypoint carries a speed gate, else 0.0.

    The speed-axis twin of :class:`ActiveRouteWaypointHasAltConstraint`, for the
    same ambiguity in :class:`ActiveRouteWaypointSpdDiffKts`.
    """

    meta = ObsMeta(
        "active_route_waypoint_has_spd_constraint",
        Unit.UNITLESS,
        ObsQuantity.INDICATOR,
    )
    low: float = 0.0
    high: float = 1.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, _alt, spd_ms = self._waypoints(indices)
        return (~np.isnan(spd_ms)).astype(np.float64)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        return float(wp is not None and wp[3] is not None)


@dataclass(frozen=True)
class ActiveRouteWaypointAltDiffFt(_ActiveRouteWaypointField):
    """Ownship altitude minus its active route waypoint altitude, ft (0 if none).

    Dynamic bounds resolve at runtime to a symmetric span around the waypoint
    altitude (or current altitude when the waypoint has no altitude
    constraint), reaching both 0 and the aircraft altitude ceiling - matching
    the ``ActiveRouteWaypointAltDeltaFt`` action scale. Pair with a normalizer
    for a fixed observation range.
    """

    meta = ObsMeta(
        "active_route_waypoint_alt_diff_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, alt_m, _spd = self._waypoints(indices)
        diff_ft = _traf_array("alt")[indices] * _M_TO_FT - alt_m * _M_TO_FT
        return np.where(np.isnan(alt_m), 0.0, diff_ft)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None or wp[2] is None:
            return 0.0
        return (float(bs.traf.alt[idx]) - wp[2]) / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            wp = self._active_wp(idx)
            if wp is not None and wp[2] is not None:
                nominal_ft = wp[2] * _M_TO_FT
            else:
                nominal_ft = bs.traf.alt[idx] * _M_TO_FT
            ceiling_ft = bs.traf.perf.hmax[idx] * _M_TO_FT
            span = max(
                abs(nominal_ft), abs(ceiling_ft - nominal_ft), _MIN_DYNAMIC_SPAN
            )
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class ActiveRouteWaypointSpdDiffKts(_ActiveRouteWaypointField):
    """Ownship CAS minus its active route waypoint speed constraint, kts.

    Returns ``0`` when the aircraft has no active waypoint or the waypoint
    carries no speed constraint. The waypoint speed is the nominal LNAV/VNAV
    target, so this is the speed deviation from nominal (0 = on nominal),
    matching :class:`ActiveRouteWaypointTrackErrorDeg` (heading) and
    :class:`ActiveRouteWaypointAltDiffFt` (altitude).

    Dynamic bounds resolve at runtime to a symmetric span around the waypoint
    speed (or current CAS when the waypoint has no speed constraint), reaching
    both the minimum and maximum operating speed - matching the
    ``ActiveRouteWaypointSpdDeltaKts`` action scale. Pair with a normalizer for
    a fixed observation range.
    """

    meta = ObsMeta(
        "active_route_waypoint_spd_diff_kts",
        Unit.KTS,
        ObsQuantity.SPEED,
        dynamic_bounds=True,
    )

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, _alt, spd_ms = self._waypoints(indices)
        diff_kts = (_traf_array("cas")[indices] - spd_ms) * _MS_TO_KTS
        return np.where(np.isnan(spd_ms), 0.0, diff_kts)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None or wp[3] is None:
            return 0.0
        return (float(bs.traf.cas[idx]) - wp[3]) / kts

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            wp = self._active_wp(idx)
            if wp is not None and wp[3] is not None:
                nominal = wp[3] * _MS_TO_KTS
            else:
                nominal = bs.traf.cas[idx] * _MS_TO_KTS
            lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
            hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
            span = max(abs(nominal - lo), abs(hi - nominal), _MIN_DYNAMIC_SPAN)
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class ActiveRouteWaypointSpdErrorCrossover(_ActiveRouteWaypointField):
    """Signed speed error to the active route waypoint, CAS/Mach crossover-aware.

    Below the CAS/Mach crossover altitude this is the CAS error / CAS scale; above
    it, the Mach error / Mach scale - already normalized to ``[-1, 1]`` (0 = on
    the waypoint speed) and on the same axis the crossover speed *action*
    controls. Returns ``0`` when there is no active waypoint or it carries no speed
    constraint. Prefer this over :class:`ActiveRouteWaypointSpdDiffKts` when using
    the crossover speed action, so the observed error matches the quantity the
    agent commands at altitude. Pre-normalized, so pair with a Raw normalizer.
    """

    meta = ObsMeta(
        "active_route_waypoint_spd_error_crossover",
        Unit.UNITLESS,
        ObsQuantity.SPEED,
    )
    low: float = -1.0
    high: float = 1.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, _alt, spd_ms = self._waypoints(indices)
        out = np.zeros(len(indices), dtype=np.float64)
        for k in np.flatnonzero(~np.isnan(spd_ms)):
            state = crossover_speed_state(int(indices[k]), float(spd_ms[k]))
            out[k] = float(state.normalized_error)
        return out

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None or wp[3] is None:
            return 0.0
        return float(crossover_speed_state(idx, wp[3]).normalized_error)


@dataclass(frozen=True)
class ActiveRouteWaypointEteS(_ActiveRouteWaypointField):
    """Estimated time enroute to the active route waypoint, seconds (0 if none).

    Along-route distance to the fix at ``route_offset`` (:func:`_route_along_distance_nm`
    - direct range at offset 0, plus the intervening legs beyond it) divided by
    the current GROUNDSPEED, so wind is already in the number.

    Why not leave the policy to divide :class:`ActiveRouteWaypointDistanceNm`
    by :class:`GsKts` itself: both arrive normalized to ``[0, 1]`` on unrelated
    scales, so recovering their ratio is a multiplicative interaction a
    small MLP has to spend capacity on. The ratio is also the form in which
    the quantity is COMPARABLE - against :class:`TimeInEnvS` and the task's
    time budget (will I reach the fix before truncation, or is loitering to
    open a gap actually affordable), and against
    :class:`IntruderFixArrivalDeltaS`, which is the same second on the same
    fix seen from an intruder.

    Deliberately range/groundspeed rather than a kinematic projection onto the
    fix: no ``cos(track error)`` factor, so holding a 60 deg avoidance heading
    does not collapse the reading, and the field carries no dependence on the
    heading the agent just commanded (the observation-feedback shape behind the
    ``PrevActionNorm`` hysteresis loop). Same reasoning, and the same model, as
    ``own_eta_mode="route"`` in :func:`_fix_projection`.

    Monotone in groundspeed, so it is the readable channel for the SPEED axis:
    decelerating to slot in behind traffic raises it cleanly.

    Returns ``0`` when there is no usable fix at ``route_offset``, a value that
    collides with "arriving now" - pair with :class:`ActiveRouteWaypointValid`
    when intruders or route-exhausted traffic can hit the sentinel. Unclamped:
    ETE is finite for any moving aircraft, so a value past ``high`` is a real
    reading (distant fix, slow groundspeed) left to the normalizer to clip.

    Metadata:
        name: active_route_waypoint_ete_s
        unit: s
        quantity: time
    """

    meta = ObsMeta("active_route_waypoint_ete_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "ETE lower bound, s"] = 0.0
    high: Annotated[
        float, "ETE upper bound, s; match the task time budget to share TimeInEnvS's scale"
    ] = 3600.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        along = [_route_along_distance_nm(int(i), self.route_offset) for i in indices]
        dist_nm = np.array(
            [np.nan if d is None else d for d in along], dtype=np.float64
        )
        gs = np.maximum(_traf_array("gs")[indices], _MIN_GS_MS)
        return np.where(np.isnan(dist_nm), 0.0, dist_nm * nm / gs)

    def _expected(self, idx: int) -> Any:
        dist_nm = _route_along_distance_nm(idx, self.route_offset)
        if dist_nm is None:
            return 0.0
        return dist_nm * nm / max(float(bs.traf.gs[idx]), _MIN_GS_MS)


@dataclass(frozen=True)
class ActiveRouteWaypointVerticalEteS(_ActiveRouteWaypointField):
    """Estimated time to reach the active route waypoint's ALTITUDE gate, seconds.

    The vertical twin of :class:`ActiveRouteWaypointEteS`: the altitude still
    to be flown to the fix's altitude constraint, divided by a vertical rate.
    Read the two together - the *difference* is the profile margin. Vertical
    ETE below horizontal ETE means the level-off happens with time in hand;
    above it means the aircraft arrives at the fix still off its gate, which is
    exactly when a climb/descent has to be started rather than deferred.

    ``vs_mode`` selects the rate, and the two answer different questions:

    * ``"current"`` (default) uses the live vertical speed - "at the rate I am
      flying right now". Am I ON profile? Mirrors the horizontal field, which
      also reads the speed being flown, and responds immediately to the
      vertical-speed axis the agent commands.
    * ``"capability"`` uses the performance-model limit in the required
      direction (``perf.vsmax`` to climb, ``perf.vsmin`` to descend, the
      quantities :class:`PerfVsMaxFtMin` / :class:`PerfVsMinFtMin` report) -
      "the fastest this aircraft could possibly do it". Is the gate REACHABLE
      at all? Unlike ``"current"`` it never reads the sentinel for a level
      aircraft, so it stays informative before any vertical manoeuvre starts.

    Pick with the sentinel rate in mind. Under ``"current"`` a level aircraft
    with altitude still to fly reads ``high``, and so does one whose VS points
    the wrong way for a step: measured on a random-action safe_rl_v52 rollout,
    60% of all constrained-fix samples saturate that way, because a
    waypoint-altitude action holds VS at 0 or flips it between steps. That is a
    bimodal channel, not a spiky one - if the design commands altitude rather
    than vertical speed, prefer ``"capability"``, whose reading is smooth and
    monotone in the altitude error (and is that error re-expressed in seconds,
    per-aircraft-type, which is what makes it comparable with the horizontal
    ETE that :class:`ActiveRouteWaypointAltDiffFt` alone is not).

    Returns ``0`` when the aircraft is already on the gate altitude, and also
    when there is no usable fix or the fix carries no altitude constraint -
    the same sentinel collision :class:`ActiveRouteWaypointAltDiffFt` has, so
    pair with :class:`ActiveRouteWaypointHasAltConstraint` (and
    :class:`ActiveRouteWaypointValid`) whenever a design mixes constrained and
    unconstrained fixes.

    Returns ``high`` when the gate is not being closed at all: zero vertical
    speed with altitude still to fly, or a rate pointing the wrong way (under
    ``"current"``, climbing away from a descent gate). Unlike the horizontal
    ETE this quantity is genuinely unbounded - level flight puts the level-off
    at infinity - so the value is clamped INTO ``[0, high]`` and ``high`` reads
    as "not converging". Keep ``high`` above any real level-off time in the
    task, or true profiles saturate against the sentinel.

    Metadata:
        name: active_route_waypoint_vertical_ete_s
        unit: s
        quantity: time
    """

    meta = ObsMeta(
        "active_route_waypoint_vertical_ete_s", Unit.S, ObsQuantity.TIME
    )
    vs_mode: Annotated[
        str,
        "vertical rate model: 'current' (live VS) or 'capability' (perf climb/descent limit)",
    ] = "current"
    low: Annotated[float, "vertical ETE lower bound, s"] = 0.0
    high: Annotated[
        float, "vertical ETE upper bound, s; also the not-converging sentinel"
    ] = 1800.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vs_mode not in ("current", "capability"):
            raise ValueError(
                "vs_mode must be 'current' or 'capability', got "
                f"{self.vs_mode!r}."
            )

    def _values(self, indices: np.ndarray) -> np.ndarray:
        _found, _lat, _lon, alt_m, _spd = self._waypoints(indices)
        # Positive => still has to climb to the gate, negative => descend.
        error_m = alt_m - _traf_array("alt")[indices]
        if self.vs_mode == "current":
            rate_ms = _traf_array("vs")[indices]
        else:
            climb = np.abs(np.asarray(bs.traf.perf.vsmax, dtype=np.float64)[indices])
            descend = -np.abs(np.asarray(bs.traf.perf.vsmin, dtype=np.float64)[indices])
            rate_ms = np.where(error_m > 0.0, climb, descend)
        high = float(self.high)
        with np.errstate(divide="ignore", invalid="ignore"):
            ete = np.minimum(error_m / rate_ms, high)
        # Level, or closing the wrong way: the level-off never happens.
        ete = np.where(rate_ms * error_m <= 0.0, high, ete)
        # No altitude constraint, or already at it.
        return np.where(np.isnan(error_m) | (error_m == 0.0), 0.0, ete)

    def _expected(self, idx: int) -> Any:
        wp = self._active_wp(idx)
        if wp is None or wp[2] is None:
            return 0.0
        error_m = wp[2] - float(bs.traf.alt[idx])  # + climb to the gate, - descend
        if error_m == 0.0:
            return 0.0
        if self.vs_mode == "current":
            rate_ms = float(bs.traf.vs[idx])
        elif error_m > 0.0:
            rate_ms = abs(float(bs.traf.perf.vsmax[idx]))
        else:
            rate_ms = -abs(float(bs.traf.perf.vsmin[idx]))
        if rate_ms * error_m <= 0.0:  # level, or going the wrong way
            return float(self.high)
        return min(error_m / rate_ms, float(self.high))


class _UnitField(_BroadcastObs, ObsField):
    """A quantity computed in SI and reported in its variant's unit.

    ``AltFt`` and ``AltM`` are one quantity in two units, so each is defined
    once, in SI (``_si_values``, ``_si_expected``, and the SI inputs of its
    dynamic bounds), and a unit mixin (:class:`_InFeet`, :class:`_InKnots`, ...)
    sets ``_scale``, the SI-to-unit factor. Values, the test reference and
    dynamic bounds all convert through it, so the variants cannot drift apart.
    Bounds configured with ``low``/``high`` are already in the variant's unit.
    """

    _scale: ClassVar[float] = 1.0

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _si_expected(self, idx: int) -> float:
        raise NotImplementedError

    def _convert(self, si: Any) -> Any:
        return si * self._scale

    def _values(self, indices: Any) -> Any:
        return self._convert(self._si_values(_indices_array(indices)))

    def _expected(self, idx: int) -> Any:
        return self._convert(self._si_expected(idx))


class _InMeters:
    _scale: ClassVar[float] = 1.0


class _InFeet:
    _scale: ClassVar[float] = _M_TO_FT


class _InMetersPerSecond:
    _scale: ClassVar[float] = 1.0


class _InKnots:
    _scale: ClassVar[float] = _MS_TO_KTS


class _InFeetPerMinute:
    _scale: ClassVar[float] = _MS_TO_FTMIN


class _AltitudeEnvelopeBounds:
    """Bounds backed by BlueSky's aircraft altitude ceiling."""

    @staticmethod
    def _altitude_ceiling_m(idx: int) -> float:
        if bs.traf is None:
            raise RuntimeError(
                "dynamic altitude bounds require initialized BlueSky traffic; "
                "pass explicit low/high bounds for pre-initialization use."
            )
        return float(bs.traf.perf.hmax[idx])


class _CasEnvelopeBounds:
    """Bounds backed by BlueSky's CAS operating-speed envelope."""

    def _speed_bounds_ms(self, idx: int) -> tuple[float, float]:
        return bs.traf.perf.vmin[idx], bs.traf.perf.vmax[idx]


class _TasEnvelopeBounds:
    """Bounds backed by BlueSky's CAS envelope converted to TAS at altitude."""

    def _speed_bounds_ms(self, idx: int) -> tuple[float, float]:
        return (
            vcas2tas(bs.traf.perf.vmin[idx], bs.traf.alt[idx]),
            vcas2tas(bs.traf.perf.vmax[idx], bs.traf.alt[idx]),
        )


@dataclass(frozen=True)
class _Altitude(_AltitudeEnvelopeBounds, _UnitField):
    """Aircraft altitude; bounds from 0 to the performance ceiling."""

    low: Annotated[float | None, "lower bound; None = 0 at runtime"] = None
    high: Annotated[
        float | None, "upper bound; None = BlueSky perf.hmax at runtime"
    ] = None

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.alt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.alt[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, self._convert(self._altitude_ceiling_m(idx)))
        )


@dataclass(frozen=True)
class _ApAltitude(_Altitude):
    """Autopilot selected altitude."""

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selalt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selalt[idx])


@dataclass(frozen=True)
class _ApAltitudeError(_AltitudeEnvelopeBounds, _UnitField):
    """Autopilot selected altitude minus current altitude."""

    low: Annotated[float | None, "lower bound; None = runtime altitude envelope"] = None
    high: Annotated[float | None, "upper bound; None = runtime altitude envelope"] = None

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selalt[indices] - bs.traf.alt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selalt[idx]) - float(bs.traf.alt[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            current = self._convert(bs.traf.alt[idx])
            ceiling = self._convert(self._altitude_ceiling_m(idx))
            span = max(
                max(abs(current), abs(ceiling - current)),
                _MIN_DYNAMIC_SPAN,
            )
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class _Speed(_UnitField):
    """A speed with bounds from the aircraft's envelope (``_speed_bounds_ms``)."""

    low: Annotated[float | None, "lower bound; None = runtime speed envelope"] = None
    high: Annotated[float | None, "upper bound; None = runtime speed envelope"] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: tuple(self._convert(value) for value in self._speed_bounds_ms(idx))
        )


@dataclass(frozen=True)
class _Cas(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.cas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.cas[idx])


@dataclass(frozen=True)
class _Tas(_TasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.tas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.tas[idx])


@dataclass(frozen=True)
class _Gs(_TasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.gs[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.gs[idx])


@dataclass(frozen=True)
class _ApCas(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selspd[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selspd[idx])


@dataclass(frozen=True)
class _ApCasError(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selspd[indices] - bs.traf.cas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selspd[idx]) - float(bs.traf.cas[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        # Symmetric about the current speed, wide enough to reach either end
        # of the envelope - not the envelope itself, which is for speeds.
        def resolve() -> tuple[float, float]:
            lo, hi = self._speed_bounds_ms(idx)
            current = bs.traf.cas[idx]
            span = max(
                self._convert(max(abs(current - lo), abs(hi - current))),
                _MIN_DYNAMIC_SPAN,
            )
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class _VerticalSpeed(_UnitField):
    """Vertical speed; bounds from the performance model's descent/climb limits."""

    low: Annotated[float | None, "lower bound; None = perf.vsmin at runtime"] = None
    high: Annotated[float | None, "upper bound; None = perf.vsmax at runtime"] = None

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.vs[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.vs[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            vsmin = self._convert(float(bs.traf.perf.vsmin[idx]))
            vsmax = self._convert(float(bs.traf.perf.vsmax[idx]))
            assert vsmin <= 0.0
            return vsmin, vsmax

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class AltFt(_InFeet, _Altitude):
    """Aircraft altitude in feet.

    Metadata:
        name: alt_ft
        unit: ft
        quantity: altitude
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    aircraft altitude ceiling at runtime.
    """

    meta = ObsMeta("alt_ft", Unit.FT, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class AltM(_InMeters, _Altitude):
    """Aircraft altitude in meters.

    Metadata:
        name: alt_m
        unit: m
        quantity: altitude
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    aircraft altitude ceiling at runtime.
    """

    meta = ObsMeta("alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class CasKts(_InKnots, _Cas):
    """Calibrated airspeed in knots.

    Metadata:
        name: cas_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    current operating-speed envelope at runtime.
    """

    meta = ObsMeta("cas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class CasMs(_InMetersPerSecond, _Cas):
    """Calibrated airspeed in m/s.

    Metadata:
        name: cas_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    current operating-speed envelope at runtime.
    """

    meta = ObsMeta("cas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class TasKts(_InKnots, _Tas):
    """True airspeed in knots.

    Metadata:
        name: tas_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("tas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class TasMs(_InMetersPerSecond, _Tas):
    """True airspeed in m/s.

    Metadata:
        name: tas_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("tas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class VsFtMin(_InFeetPerMinute, _VerticalSpeed):
    """Vertical speed in ft/min.

    Metadata:
        name: vs_ftmin
        unit: ft/min
        quantity: vertical_speed
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's current
    aircraft performance envelope at runtime, as ``(vsmin, vsmax)``.
    """

    meta = ObsMeta(
        "vs_ftmin", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED, dynamic_bounds=True
    )


@dataclass(frozen=True)
class VsMs(_InMetersPerSecond, _VerticalSpeed):
    """Vertical speed in m/s.

    Metadata:
        name: vs_ms
        unit: m/s
        quantity: vertical_speed
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's current
    aircraft performance envelope at runtime, as ``(vsmin, vsmax)`` .
    """

    meta = ObsMeta(
        "vs_ms", Unit.M_PER_S, ObsQuantity.VERTICAL_SPEED, dynamic_bounds=True
    )


@dataclass(frozen=True)
class AxMs2(_BroadcastObs, ObsField):
    """Longitudinal (TAS) acceleration in m/s^2 - the aircraft's current speed
    rate. ~0 when holding speed; a physical, orientation-invariant measure of
    speed-axis smoothness (pair with a rate-based action penalty).

    ``low``/``high`` are a **normalization scale**, not a physical cap: with a
    non-clipping normalizer, values beyond them pass through as ``|x|>1`` (no
    information lost). Acceleration has no clean *symmetric* physical bound (the
    thrust-limited accel differs from drag/idle decel), so the default is a fixed
    representative span that covers the typical range; override for a different
    scale.

    Metadata:
        name: ax_ms2
        unit: m/s
        quantity: speed
    """

    meta = ObsMeta("ax_ms2", Unit.M_PER_S, ObsQuantity.SPEED)
    low: Annotated[float, "accel m/s^2 normalization scale (low)"] = -3.0
    high: Annotated[float, "accel m/s^2 normalization scale (high)"] = 3.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.ax, dtype=np.float64)[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.ax[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class MachNumber(_BroadcastObs, ObsField):
    """Ownship Mach number (``bs.traf.M``).

    At cruise altitude the speed envelope is Mach-limited, not CAS-limited, so
    Mach exposes the speed regime and remaining speed authority that ``CasKts``
    alone does not (two aircraft at equal CAS but different altitudes fly
    different TAS, hence different closing dynamics). Bounds default to the
    subsonic ``[0, 1]``; tighten ``high`` toward the type's Mmo (~0.87 for a
    B744) for a fuller normalized range.

    Metadata:
        name: mach_number
        unit: unitless
        quantity: speed
    """

    meta = ObsMeta("mach_number", Unit.UNITLESS, ObsQuantity.SPEED)
    low: Annotated[float, "Mach normalization scale (low)"] = 0.0
    high: Annotated[float, "Mach normalization scale (high)"] = 1.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.M, dtype=np.float64)[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.M[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class CrossoverAltMarginFt(_BroadcastObs, ObsField):
    """Signed altitude margin to the CAS/Mach crossover, in feet.

    ``alt - crossoveralt(cas, Mmo)``: **positive above** the crossover (the
    Mach-limited regime, where a CAS increase is capped by Mmo), **negative
    below** (the CAS regime). Makes the speed action's regime boundary explicit so
    the policy need not infer it from Alt+CAS. Its *sign* is the "above crossover"
    boolean; the magnitude says how deep into the regime the aircraft is - a
    smoother, more informative signal than a bare flag.

    Metadata:
        name: crossover_alt_margin_ft
        unit: ft
        quantity: altitude
    """

    meta = ObsMeta("crossover_alt_margin_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: Annotated[float, "crossover-margin ft normalization scale (low)"] = -20000.0
    high: Annotated[float, "crossover-margin ft normalization scale (high)"] = 20000.0

    def _values(self, indices: Any) -> Any:
        i = _indices_array(indices)
        cas = np.asarray(bs.traf.cas, dtype=np.float64)[i]
        alt = np.asarray(bs.traf.alt, dtype=np.float64)[i]
        mmo = np.asarray(bs.traf.perf.mmo, dtype=np.float64)[i]
        return (alt - np.asarray(crossoveralt(cas, mmo))) * _M_TO_FT

    def _expected(self, idx: int) -> Any:
        cas, mmo = float(bs.traf.cas[idx]), float(bs.traf.perf.mmo[idx])
        return (float(bs.traf.alt[idx]) - float(crossoveralt(cas, mmo))) / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TimeInEnvS(_BroadcastObs, _TimeInEnvBacked, ObsField):
    """Seconds since ownship entered the environment - its age, not sim clock.

    The quantity the time-limit truncation is stated against
    (``info["time_in_env"] >= TIME_BUDGET_S``), so with ``high`` set to that
    budget and a :class:`MinMaxNormalizer` this reads as episode progress in
    ``[0, 1]`` and time REMAINING is its complement.

    Why a value function wants it: under a time limit the return is bounded by
    the time left, so two otherwise identical states early and late in an
    aircraft's life have genuinely different values and a critic without this
    must average them. That is irreducible value error, not underfitting - the
    standard time-limit partial-observability result (Pardo et al. 2018). It
    matters most for the CONSTRAINT critic here, whose discounted cost-to-go at
    ``cost_gamma = 0.99`` reaches ~100 steps and so routinely runs past the
    truncation an early-life state still has ahead of it.

    Purely local (an aircraft knows its own age), so it is legitimate in
    ``obs_fields``; putting it in ``critic_obs_fields`` instead fixes the value
    function while leaving the policy's input distribution untouched.

    Published by the environment each step from its spawn-time bookkeeping (an
    ObsField cannot reach it - BlueSky keeps no per-aircraft age); reads 0 on the
    step an aircraft spawns.

    Metadata:
        name: time_in_env_s
        unit: s
        quantity: time
    """

    meta = ObsMeta("time_in_env_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "seconds lower bound"] = 0.0
    high: Annotated[float, "seconds upper bound; set to the task's time budget"] = 3600.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(_TIME_IN_ENV.read(indices), dtype=np.float64)

    def _expected(self, idx: int) -> Any:
        return float(_TIME_IN_ENV.read_one(idx))  # published by the env

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# ---------------------------------------------------------------------------
# Aircraft-capability descriptors
#
# Continuous physical performance parameters, for heterogeneous-fleet tasks:
# an aircraft is a *point in capability space*, not a type symbol, so a policy
# conditioned on these generalizes to types never seen in training (a learned
# type-ID embedding cannot). Realistically available - type is broadcast in
# ADS-B and known to ATC. Usable in both the own and intruder blocks (plain
# ObsFields, like ``VsFtMin``). Bounds are FIXED fleet-wide scales on purpose:
# envelope-dynamic bounds would normalize each aircraft's capability to
# itself, erasing exactly the cross-type differences these fields carry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerfVminKts(_BroadcastObs, ObsField):
    """Minimum operating CAS from the performance model, in knots.

    How slow this airframe *can* fly - the sequencing floor: an intruder with
    a higher ``vmin`` than mine cannot match my hold speed and must be led,
    not followed. Reads the live performance model (state-dependent through
    configuration/phase).

    Metadata:
        name: perf_vmin_kts
        unit: kts
        quantity: speed
    """

    meta = ObsMeta("perf_vmin_kts", Unit.KTS, ObsQuantity.SPEED)
    low: Annotated[float, "fleet-wide CAS scale (low), knots"] = 60.0
    high: Annotated[float, "fleet-wide CAS scale (high), knots"] = 250.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vmin[_indices_array(indices)] * _MS_TO_KTS

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vmin[idx]) / kts

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVmaxKts(_BroadcastObs, ObsField):
    """Maximum operating CAS from the performance model, in knots.

    How fast this airframe *can* fly - whether the aircraft ahead can
    accelerate out of the way, or I can close a slot. Reads the live
    performance model.

    Metadata:
        name: perf_vmax_kts
        unit: kts
        quantity: speed
    """

    meta = ObsMeta("perf_vmax_kts", Unit.KTS, ObsQuantity.SPEED)
    low: Annotated[float, "fleet-wide CAS scale (low), knots"] = 120.0
    high: Annotated[float, "fleet-wide CAS scale (high), knots"] = 400.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vmax[_indices_array(indices)] * _MS_TO_KTS

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vmax[idx]) / kts

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMaxFtMin(_BroadcastObs, ObsField):
    """Maximum climb rate from the performance model, in ft/min.

    Vertical escape capacity - can this aircraft climb out of a conflict
    layer, and how fast. Reads the live performance model (varies with
    altitude/mass/phase).

    Metadata:
        name: perf_vs_max_ft_min
        unit: ft/min
        quantity: vertical_speed
    """

    meta = ObsMeta(
        "perf_vs_max_ft_min", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED
    )
    low: Annotated[float, "fleet-wide climb-rate scale (low), ft/min"] = 0.0
    high: Annotated[float, "fleet-wide climb-rate scale (high), ft/min"] = 6000.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vsmax[_indices_array(indices)] * _MS_TO_FTMIN

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vsmax[idx]) * 60.0 / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMinFtMin(_BroadcastObs, ObsField):
    """Maximum DESCENT rate from the performance model, in ft/min (negative).

    The counterpart to :class:`PerfVsMaxFtMin`, which is climb only. Distinct
    physics and a distinct number - on the openap model the two differ by ~20%
    for the same aircraft - so a descending task cannot substitute one for the
    other.

    Metadata:
        name: perf_vs_min_ft_min
        unit: ft/min
        quantity: vertical_speed
    """

    meta = ObsMeta(
        "perf_vs_min_ft_min", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED
    )
    low: Annotated[float, "fleet-wide descent-rate scale (low), ft/min"] = -6000.0
    high: Annotated[float, "fleet-wide descent-rate scale (high), ft/min"] = 0.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vsmin[_indices_array(indices)] * _MS_TO_FTMIN

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vsmin[idx]) * 60.0 / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfCeilingFt(_BroadcastObs, ObsField):
    """Altitude ceiling from the performance model, in ft.

    How much vertical room is left above. Also the quantity that scales a
    waypoint-relative altitude *action* whenever the ceiling term wins
    (``span = max(wpalt, ceiling - wpalt)``) - measured on safe_rl_v38d at ~9% of
    steps - so without this the policy cannot know how many feet its normalized
    altitude action commands.

    Metadata:
        name: perf_ceiling_ft
        unit: ft
        quantity: altitude
    """

    meta = ObsMeta("perf_ceiling_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: Annotated[float, "fleet-wide ceiling scale (low), ft"] = 0.0
    high: Annotated[float, "fleet-wide ceiling scale (high), ft"] = 60000.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.hmax[_indices_array(indices)] * _M_TO_FT

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.hmax[idx]) / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfMassT(_BroadcastObs, ObsField):
    """CURRENT aircraft mass from the performance model, in tonnes.

    Unlike :class:`MtowT`, which is a static per-type constant, this is the live
    mass the performance model is actually flying, and it drives thrust-to-weight,
    achievable rates and turn performance. Spans ~5-190 t across the allowed fleet.

    Metadata:
        name: perf_mass_t
        unit: t
        quantity: mass
    """

    meta = ObsMeta("perf_mass_t", Unit.T, ObsQuantity.MASS)
    low: Annotated[float, "fleet-wide mass scale (low), t"] = 0.0
    high: Annotated[float, "fleet-wide mass scale (high), t"] = 600.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.mass[_indices_array(indices)] / 1000.0

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.perf.mass[idx]) / 1000.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TurnRadiusNm(_BroadcastObs, ObsField):
    """Coordinated-turn radius at current TAS and bank-angle limit, in nm.

    ``R = V^2 / (g * tan(phi))`` with ``V = bs.traf.tas`` and ``phi`` the
    per-aircraft bank limit (``bs.traf.ap.bankdef``, default 25 deg) - the same
    triangle BlueSky's heading dynamics integrate, so this is the radius the
    aircraft actually flies at full authority. Deliberately *state-dependent*
    (scales with V^2): it reads as current agility, and slowing down visibly
    shrinks it - the physical lever behind decelerate-before-capture. At
    cruise (480 kt, 25 deg) it is ~7 nm - larger than a typical reach radius,
    which is why an overshoot costs a full circuit.

    Metadata:
        name: turn_radius_nm
        unit: nm
        quantity: distance
    """

    meta = ObsMeta("turn_radius_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: Annotated[float, "turn radius scale (low), nm"] = 0.0
    high: Annotated[float, "turn radius scale (high), nm"] = 20.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        tas = np.maximum(np.asarray(bs.traf.tas)[indices], 1e-6)
        # Bank *limit* (authority), not ap.turnphi - turnphi is a transient
        # commanded bank inside flyturn legs and zero otherwise.
        phi = np.asarray(bs.traf.ap.bankdef)[indices]
        return (tas * tas) / (g0 * np.tan(phi)) / nm

    def _expected(self, idx: int) -> Any:
        # r = v^2 / (g tan(bank)), at the aircraft's bank limit.
        tas = max(float(bs.traf.tas[idx]), 1e-6)
        bank = float(bs.traf.ap.bankdef[idx])
        return tas * tas / (g0 * math.tan(bank)) / nm

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# MTOW per type, cached: openap's aircraft database, with a medium-class
# fallback for types it does not know (never raise mid-episode over a
# descriptor lookup).
_MTOW_KG_CACHE: dict[str, float] = {}
_MTOW_FALLBACK_KG = 100_000.0


def _mtow_kg(actype: str) -> float:
    key = str(actype).upper()
    cached = _MTOW_KG_CACHE.get(key)
    if cached is None:
        # Same registry the envelope uses, so MTOW and the flight envelope can
        # never come from different databases for one aircraft.
        model = active_performance_model()
        mtow = (type_limits(key, model) or {}).get("MTOW")
        if mtow is None and model != "openap":
            _warn_type_data_mismatch("MTOW")
            mtow = (type_limits(key, "openap") or {}).get("MTOW")
        cached = float(mtow) if mtow else _MTOW_FALLBACK_KG
        _MTOW_KG_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class MtowT(_BroadcastObs, ObsField):
    """Maximum takeoff weight of the aircraft type, in tonnes.

    The continuous stand-in for wake/size class (ICAO wake categories are
    MTOW bands): a smooth mass descriptor generalizes where a categorical
    one-hot cannot. Looked up once per type from the openap aircraft
    database and cached; unknown types fall back to a medium-class 100 t.

    Metadata:
        name: mtow_t
        unit: t
        quantity: mass
    """

    meta = ObsMeta("mtow_t", Unit.T, ObsQuantity.MASS)
    low: Annotated[float, "fleet-wide mass scale (low), tonnes"] = 0.0
    high: Annotated[float, "fleet-wide mass scale (high), tonnes"] = 600.0

    def _values(self, indices: Any) -> Any:
        types = bs.traf.type
        return np.asarray(
            [_mtow_kg(types[int(i)]) / 1000.0 for i in _indices_array(indices)],
            dtype=np.float64,
        )

    def _expected(self, idx: int) -> Any:
        return _mtow_kg(bs.traf.type[idx]) / 1000.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class FlightPhaseOneHot(_BroadcastObs, ObsField):
    """Flight phase as a one-hot vector.

    Metadata:
        name: flight_phase_one_hot
        unit: unitless
        quantity: phase

    The default ``phase_values`` encode BlueSky/OpenAP-style raw phase codes
    ``0..6``. Pass a custom tuple when using a performance model that emits a
    different code or label set.
    """

    meta = ObsMeta("flight_phase_one_hot", Unit.UNITLESS, ObsQuantity.PHASE)
    phase_values: tuple[int | float | str, ...] = (0, 1, 2, 3, 4, 5, 6)
    unknown_index: Annotated[
        int | None,
        "index to activate when the raw phase is not in phase_values; None = all zero",
    ] = 0
    low: Annotated[float, "one-hot off value"] = 0.0
    high: Annotated[float, "one-hot on value"] = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.phase_values:
            raise ValueError("FlightPhaseOneHot.phase_values cannot be empty")
        normalized = tuple(_phase_key(value) for value in self.phase_values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "FlightPhaseOneHot.phase_values must not contain duplicates"
            )
        if self.unknown_index is not None and not (
            0 <= self.unknown_index < len(self.phase_values)
        ):
            raise ValueError(
                "FlightPhaseOneHot.unknown_index must be a valid phase index or None"
            )

    def output_size(self) -> int:
        return len(self.phase_values)

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        values = np.zeros((indices.size, len(self.phase_values)), dtype=np.float32)
        for row, raw_phase in enumerate(bs.traf.perf.phase[indices]):
            for phase_idx, phase_value in enumerate(self.phase_values):
                if _phase_matches(raw_phase, phase_value):
                    values[row, phase_idx] = 1.0
                    break
            else:
                if self.unknown_index is not None:
                    values[row, self.unknown_index] = 1.0
        return values

    def _expected(self, idx: int) -> Any:
        values = np.zeros(len(self.phase_values), dtype=np.float32)
        phase = bs.traf.perf.phase[idx]
        matches = [k for k, v in enumerate(self.phase_values) if _phase_matches(phase, v)]
        if matches:
            values[matches[0]] = 1.0
        elif self.unknown_index is not None:
            values[self.unknown_index] = 1.0
        return values

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        return (
            np.full(size, float(self.low), dtype=np.float32),
            np.full(size, float(self.high), dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# Lagged observations (frame stacking)                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LaggedObs(_BroadcastObs, _LagHistoryBacked, ObsField):
    """An ownship field's value from ``steps`` environment steps ago.

    Built by :meth:`ObsField.lagged`. Bounds, normalizer and output size all
    delegate to ``inner``, so a stacked channel lands on exactly the same scale
    as the live one and needs no separate calibration.

    The lag counts OBSERVATION QUERIES at distinct sim times, not wall steps.
    Training observes every live agent every step so the two coincide; a caller
    that observes a subset of agents would see that subset's own lag.
    """

    inner: ObsField | None = None
    steps: int = 1

    @property
    def meta(self) -> ObsMeta:
        # ``dynamic_bounds=True`` regardless of the inner field's policy, matching
        # :class:`Difference`: the bounds are resolved from ``inner`` at runtime
        # (``bounds`` below), so this wrapper carries no static defaults of its own
        # for ``_validate_bound_policy`` to check.
        inner = self._field()
        return replace(
            inner.meta,
            name=f"{inner.meta.name}_lag{int(self.steps)}",
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        inner = self._field()
        if int(self.steps) < 1:
            raise ValueError(f"lagged(steps=) must be >= 1, got {self.steps}.")
        object.__setattr__(self, "_key", repr(inner))
        _register_lag_depth(self._key, self.steps)
        super().__post_init__()

    def _field(self) -> ObsField:
        if not isinstance(self.inner, ObsField):
            raise TypeError(f"LaggedObs.inner must be an ObsField, got {self.inner!r}.")
        return self.inner

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._field().bounds(idx)

    def output_size(self) -> int:
        inner = self._field()
        size = getattr(inner, "output_size", None)
        return int(size()) if callable(size) else 1

    def _values(self, indices: Any) -> Any:
        inner = self._field()
        idxs = _indices_array(indices).ravel()
        ring = _lag_ring("obs", self._key, self.steps)
        simt = float(bs.sim.simt)
        if ring.last_push_simt != simt:
            ring.last_push_simt = simt
            rows = np.unique(idxs)
            if rows.size:
                ring.push(inner.get_many(rows), rows)
        out, missing = ring.read(self.steps, idxs)
        if out is None or missing.any():
            # Aircraft that appeared after this step's push (or the push was made
            # by a sibling lag before it existed): its own value is the only
            # history there is.
            live = np.asarray(inner.get_many(idxs), dtype=np.float64)
            if out is None:
                return live
            out[missing] = live[missing]
        return out


@dataclass(frozen=True)
class LaggedPair(_LagHistoryBacked, PairObsField):
    """An intruder pair field's value from ``steps`` environment steps ago.

    Built by :meth:`PairObsField.lagged`. History is keyed by the ORDERED
    callsign pair, which is what makes this correct at all: BlueSky compacts its
    arrays with ``np.delete`` on every despawn, so intruder row ``k`` at step
    ``t`` is routinely a different aircraft than row ``k`` at ``t-1``.

    Only sound on fields that are invariant to ownship ROTATION. Anything
    expressed in the ownship's track frame (``RelPos*``, ``RelVel*``, the
    along/cross realized accelerations) was computed in the old frame, so
    stacking it aliases the ownship's own turning as intruder motion.
    """

    inner: PairObsField | None = None
    steps: int = 1

    @property
    def meta(self) -> ObsMeta:
        # See :class:`LaggedObs.meta` for why this is always dynamic.
        inner = self._field()
        return replace(
            inner.meta,
            name=f"{inner.meta.name}_lag{int(self.steps)}",
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        inner = self._field()
        if int(self.steps) < 1:
            raise ValueError(f"lagged(steps=) must be >= 1, got {self.steps}.")
        object.__setattr__(self, "_key", repr(inner))
        _register_lag_depth(self._key, self.steps)
        super().__post_init__()

    def _field(self) -> PairObsField:
        if not isinstance(self.inner, PairObsField):
            raise TypeError(
                f"LaggedPair.inner must be a PairObsField, got {self.inner!r}."
            )
        return self.inner

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._field().bounds(own_idx)

    def output_size(self) -> int:
        inner = self._field()
        size = getattr(inner, "output_size", None)
        return int(size()) if callable(size) else 1

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return self.get_pairs(own_idx, [int(other_idx)])[0]

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        own = np.array([int(own_idx)], dtype=np.intp)
        return self._lagged(own, _indices_array(other_indices).ravel())[0]

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        every = np.arange(int(bs.traf.ntraf), dtype=np.intp)
        return self._lagged(_indices_array(own_indices).ravel(), every)

    def _lagged(self, owns: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The inner field ``steps`` pushes back, for each (own, col) pair.

        Each ownship's row is pushed once per sim time, by its first query, with
        the pairs that query asked for.
        """
        inner = self._field()
        ring = _lag_ring("pair", self._key, self.steps)
        simt = float(bs.sim.simt)
        due = np.unique(owns[ring.pushed_at[owns] != simt])
        if due.size and cols.size:
            current = np.asarray(inner.get_pair_matrix(due))[:, cols]
            ring.push(current, due, cols)
            ring.pushed_at[due] = simt
        out, missing = ring.read(self.steps, owns, cols)
        if out is None or missing.any():
            # Pairs that appeared after this step's push - hold their current
            # value (see :meth:`_LagRing.read`).
            live = np.asarray(inner.get_pair_matrix(owns), dtype=np.float64)[:, cols]
            if out is None:
                return live
            out[missing] = live[missing]
        return out


@dataclass(frozen=True)
class PrevActionNorm(_BroadcastObs, _LastActionBacked, ObsField):
    """Ownship's previous action ``a_{t-1}`` (as the policy emitted it).

    Surfaces the last policy output so an action-rate reward penalty
    ``|a_t - a_{t-1}|`` is Markovian w.r.t. the observation. Set ``dim`` to how
    many action components to expose and ``offset`` to where they start in the
    action vector.

    The stored value is the policy's action *in the task's action space*, whose
    range depends on the action fields' normalizers - ``SymmetricNormalizer`` ->
    ``[-1, 1]``, ``MinMaxNormalizer`` -> ``[0, 1]``, no normalizer -> the
    field's own bounds. Bounds are therefore **dynamic**: the environment
    publishes the live action-space bounds (:func:`set_action_space_bounds`) and
    this field slices them by ``offset``/``dim``, so it matches any action space
    automatically - including mixed per-component ranges - with no manual config.
    Passing ``low``/``high`` overrides that with a fixed range; before the env
    publishes bounds it falls back to ``[-1, 1]``. No normalizer is attached (the
    value is already in action space). Reads all-zero before the first action and
    on spawn (matching a first-step Δ of 0).

    Metadata:
        name: prev_action_norm
        unit: unitless
        quantity: action
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "prev_action_norm", Unit.UNITLESS, ObsQuantity.ACTION, dynamic_bounds=True
    )
    dim: Annotated[int, "number of action components to expose"] = 1
    offset: Annotated[int, "index of the first action component to expose"] = 0
    low: Annotated[float | None, "fixed lower bound; None = read from action space"] = None
    high: Annotated[float | None, "fixed upper bound; None = read from action space"] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim < 1:
            raise ValueError("PrevActionNorm.dim must be >= 1")
        if self.offset < 0:
            raise ValueError("PrevActionNorm.offset must be >= 0")
        if self.normalizer is not None:
            # The stored value is the policy's own output, already in action
            # space - scaling it a second time would measure it against bounds
            # it was never drawn from. Bounds here are also per-component (one
            # pair per exposed action slot), which a scalar Normalizer has no
            # way to reduce to a single span.
            raise ValueError(
                "PrevActionNorm carries the policy's action, already in action "
                "space; leave normalizer unset."
            )

    def output_size(self) -> int:
        return self.dim


    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        if self.bounds_overridden:
            return (
                np.full(size, float(self.low), dtype=np.float32),
                np.full(size, float(self.high), dtype=np.float32),
            )
        if _state._ACTION_SPACE_BOUNDS is not None:
            lo, hi = _state._ACTION_SPACE_BOUNDS
            sl = slice(self.offset, self.offset + size)
            lo_s, hi_s = lo[sl], hi[sl]
            if lo_s.shape[0] == size:
                return (lo_s.astype(np.float32), hi_s.astype(np.float32))
        # Before the env publishes action bounds: assume symmetric [-1, 1].
        return (
            np.full(size, -1.0, dtype=np.float32),
            np.full(size, 1.0, dtype=np.float32),
        )

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        if indices.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((indices.size, self.dim), dtype=np.float32)
        for row, stored in enumerate(_LAST_NORM_ACTION.read(indices)):
            if stored is not None:
                src = stored[self.offset : self.offset + self.dim]
                out[row, : src.shape[0]] = src
        return out

    def _expected(self, idx: int) -> Any:
        values = np.zeros(self.dim, dtype=np.float32)
        stored = _LAST_NORM_ACTION.read_one(idx)
        if stored is not None:
            exposed = stored[self.offset : self.offset + self.dim]
            values[: len(exposed)] = exposed
        return values


@dataclass(frozen=True)
class GsKts(_InKnots, _Gs):
    """Ground speed in knots.

    Metadata:
        name: gs_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True

    BlueSky has no separate ground-speed performance envelope. Default bounds
    use the TAS-equivalent operating-speed envelope; override constructor
    bounds if wind can push ground speed outside that range.
    """

    meta = ObsMeta("gs_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class GsMs(_InMetersPerSecond, _Gs):
    """Ground speed in m/s.

    Metadata:
        name: gs_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True

    BlueSky has no separate ground-speed performance envelope. Default bounds
    use the TAS-equivalent operating-speed envelope; override constructor
    bounds if wind can push ground speed outside that range.
    """

    meta = ObsMeta("gs_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class ApHdgDeg(_BroadcastObs, ObsField):
    """Autopilot selected heading in degrees.

    Metadata:
        name: ap_hdg_deg
        unit: deg
        quantity: heading
        circular: True
    """

    meta = ObsMeta("ap_hdg_deg", Unit.DEG, ObsQuantity.HEADING, circular=True)
    low: Annotated[float, "autopilot heading degrees"] = 0.0
    high: Annotated[float, "autopilot heading degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.ap.trk[_indices_array(indices)]

    def _expected(self, idx: int) -> Any:
        return float(bs.traf.ap.trk[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasKts(_InKnots, _ApCas):
    """Autopilot selected calibrated airspeed in knots.

    Metadata:
        name: ap_cas_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_cas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class ApCasMs(_InMetersPerSecond, _ApCas):
    """Autopilot selected calibrated airspeed in m/s.

    Metadata:
        name: ap_cas_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_cas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class ApAltFt(_InFeet, _ApAltitude):
    """Autopilot selected altitude in feet.

    Metadata:
        name: ap_alt_ft
        unit: ft
        quantity: altitude
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_alt_ft", Unit.FT, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class ApAltM(_InMeters, _ApAltitude):
    """Autopilot selected altitude in meters.

    Metadata:
        name: ap_alt_m
        unit: m
        quantity: altitude
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class ApLnavVnavOn(_BroadcastObs, ObsField):
    """Whether both LNAV and VNAV are enabled."""

    meta = ObsMeta("ap_lnav_vnav_on", Unit.SWITCH, ObsQuantity.AUTOPILOT)
    low: Annotated[float, "autopilot switch off"] = 0.0
    high: Annotated[float, "autopilot switch on"] = 1.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return np.logical_and(bs.traf.swlnav[indices], bs.traf.swvnav[indices])

    def _expected(self, idx: int) -> Any:
        return float(bool(bs.traf.swlnav[idx]) and bool(bs.traf.swvnav[idx]))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApHdgErrorDeg(_BroadcastObs, ObsField):
    """Autopilot selected heading error relative to current track in degrees."""

    meta = ObsMeta("ap_hdg_error_deg", Unit.DEG, ObsQuantity.HEADING)
    low: Annotated[float, "autopilot heading error degrees"] = -180.0
    high: Annotated[float, "autopilot heading error degrees"] = 180.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return (bs.traf.ap.trk[indices] - bs.traf.trk[indices] + 540.0) % 360.0 - 180.0

    def _expected(self, idx: int) -> Any:
        error = float(bs.traf.ap.trk[idx]) - float(bs.traf.trk[idx])
        return (error + 180.0) % 360.0 - 180.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasErrorKts(_InKnots, _ApCasError):
    """Autopilot selected CAS error relative to current CAS in knots."""

    meta = ObsMeta(
        "ap_cas_error_kts",
        Unit.KTS,
        ObsQuantity.SPEED,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class ApAltErrorFt(_InFeet, _ApAltitudeError):
    """Autopilot selected altitude error relative to current altitude in feet."""

    meta = ObsMeta(
        "ap_alt_error_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class ApAltErrorM(_InMeters, _ApAltitudeError):
    """Autopilot selected altitude error relative to current altitude in meters."""

    meta = ObsMeta(
        "ap_alt_error_m",
        Unit.M,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class Difference(_BroadcastPairs, PairObsField):
    """Difference between an intruder field and an ownship field.

    ``left`` is read from the intruder index and ``right`` is read from the
    ownship index. Use :class:`AngleDifference` for circular degree fields.
    """

    left: ObsField | None = None
    right: ObsField | None = None
    name: str | None = None

    @property
    def meta(self) -> ObsMeta:
        left, right = self._fields()
        return ObsMeta(
            self.name or f"{left.meta.name}_minus_{right.meta.name}",
            left.meta.unit,
            left.meta.quantity,
            is_pair=True,
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        left, right = self._fields()
        if left.meta.unit != right.meta.unit:
            raise ValueError(
                "Difference fields must use matching units; got "
                f"{left.meta.unit!r} and {right.meta.unit!r}."
            )
        if left.meta.circular or right.meta.circular:
            raise ValueError(
                "Difference cannot subtract circular fields. Use AngleDifference."
            )
        super().__post_init__()

    def _fields(self) -> tuple[ObsField, ObsField]:
        if not isinstance(self.left, ObsField):
            raise TypeError(f"Difference.left must be an ObsField, got {self.left!r}.")
        if not isinstance(self.right, ObsField):
            raise TypeError(
                f"Difference.right must be an ObsField, got {self.right!r}."
            )
        return self.left, self.right

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        left, right = self._fields()
        return _per_aircraft(left, other) - _per_aircraft(right, own)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        left, right = self._fields()
        return left._expected(other_idx) - right._expected(own_idx)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return self._configured_bounds()
        left, right = self._fields()
        left_low, left_high = left.bounds(own_idx)
        right_low, right_high = right.bounds(own_idx)
        return left_low - right_high, left_high - right_low

    def with_derived_bounds(
        self,
        derived: Mapping[str, tuple[float, float]],
    ) -> Difference:
        left, right = self._fields()
        return replace(
            self,
            left=_with_derived_bounds(left, derived),
            right=_with_derived_bounds(right, derived),
        )


@dataclass(frozen=True)
class AngleDifference(Difference):
    """Wrapped degree difference between an intruder field and an ownship field."""

    @property
    def meta(self) -> ObsMeta:
        left, right = self._fields()
        return ObsMeta(
            self.name or f"{left.meta.name}_angle_minus_{right.meta.name}",
            Unit.DEG,
            left.meta.quantity,
            is_pair=True,
            circular=True,
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        left, right = self._fields()
        if left.meta.unit != Unit.DEG or right.meta.unit != Unit.DEG:
            raise ValueError(
                "AngleDifference fields must use degree fields; got "
                f"{left.meta.unit!r} and {right.meta.unit!r}."
            )
        if not (left.meta.circular and right.meta.circular):
            raise ValueError(
                "AngleDifference fields must be circular. Use Difference for "
                "non-circular degree fields."
            )
        PairObsField.__post_init__(self)

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        left, right = self._fields()
        delta = _per_aircraft(left, other) - _per_aircraft(right, own)
        return (delta + 540.0) % 360.0 - 180.0

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        left, right = self._fields()
        delta = left._expected(other_idx) - right._expected(own_idx)
        return (delta + 180.0) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return self._configured_bounds()
        return -180.0, 180.0


@dataclass(frozen=True)
class DistToOwnNm(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder distance in nautical miles.

    Metadata:
        name: dist_to_own_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("dist_to_own_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "distance nautical miles"] = 0.0
    high: Annotated[float, "distance nautical miles"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _qdr, dist = _pair_qdr_dist(own, other)
        return dist

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        return float(qdrdist(*own, *other)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TcpaS(_BroadcastPairs, PairObsField):
    """BlueSky ASAS time to closest point of approach, in seconds.

    ASAS only stores this value for detected conflict pairs. Non-conflict
    intruder rows return the field's high bound - the CD lookahead horizon, the
    largest time-to-CPA a detected conflict could carry.

    Bounds are dynamic: unless explicit ``low``/``high`` are given, they follow
    BlueSky's CD lookahead (:func:`_cd_lookahead_s`, read from
    ``asas_dtlookahead``) as ``(-lookahead, +lookahead)``. Detected conflicts
    fall in that range (``tcpa`` goes slightly negative just past CPA).

    Metadata:
        name: tcpa_s
        unit: s
        quantity: time
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "tcpa_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[
        float | None, "time seconds lower bound; None = -CD lookahead at runtime"
    ] = None
    high: Annotated[
        float | None, "time seconds upper bound; None = +CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tcpa`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon), so the sentinel tracks config.
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("tcpa", own, other, fill)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("tcpa", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value

    def bounds(self, own_idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            look = _cd_lookahead_s()
            return -look, look

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class TlosS(_BroadcastPairs, PairObsField):
    """BlueSky ASAS predicted time to loss of separation (PZ entry), in seconds.

    Reads the conflict detector's cached ``tLOS`` (one entry per ``confpairs``
    row) each sim step - the time until the intruder is predicted to enter the
    protected zone. ASAS only stores this for detected conflict pairs, so
    non-conflict intruder rows take the high bound (the CD lookahead horizon).

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's CD lookahead (:func:`_cd_lookahead_s`, from ``asas_dtlookahead`` /
    ``config.lookahead_s``) as ``(0, lookahead)`` - the horizon within which a
    conflict is flagged. Unlike ``tcpa`` this is non-negative (time *to* PZ
    entry), so it is a cleaner imminence signal than time-to-CPA.

    Metadata:
        name: tlos_s
        unit: s
        quantity: time
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "tlos_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[
        float | None, "time seconds lower bound; None = 0 at runtime"
    ] = None
    high: Annotated[
        float | None, "time seconds upper bound; None = CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tLOS`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon).
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("tLOS", own, other, fill)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("tLOS", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class ClosingRateKts(_BroadcastPairs, PairObsField):
    """Ownship-intruder horizontal closing rate, in knots.

    Positive means the pair is closing horizontally; negative means opening.

    Metadata:
        name: closing_rate_kts
        unit: kts
        quantity: speed
        is_pair: True
    """

    meta = ObsMeta("closing_rate_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "speed knots"] = -1000.0
    high: Annotated[float, "speed knots"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own, other
        )
        dist_m = np.maximum(np.hypot(rel_east_m, rel_north_m), 1e-6)
        range_rate_ms = (rel_east_m * rel_east_ms + rel_north_m * rel_north_ms) / dist_m
        return -range_rate_ms * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range_m = max(math.hypot(east, north), 1e-6)
        return -(east * v_east + north * v_north) / range_m / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BearingRateDegPerSec(_BroadcastPairs, PairObsField):
    """Ownship-intruder bearing rate, in degrees per second.

    Positive means the bearing from ownship to intruder rotates clockwise.

    Metadata:
        name: bearing_rate_deg_per_sec
        unit: deg
        quantity: bearing
        is_pair: True
    """

    meta = ObsMeta(
        "bearing_rate_deg_per_sec",
        Unit.DEG_PER_SEC,
        ObsQuantity.BEARING,
        is_pair=True,
    )
    low: Annotated[float, "degrees per second"] = -10.0
    high: Annotated[float, "degrees per second"] = 10.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own, other
        )
        dist2_m = np.maximum(rel_east_m * rel_east_m + rel_north_m * rel_north_m, 1e-6)
        rate_rad_s = (rel_north_m * rel_east_ms - rel_east_m * rel_north_ms) / dist2_m
        return np.degrees(rate_rad_s)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        range2 = max(east * east + north * north, 1e-6)
        return math.degrees((north * v_east - east * v_north) / range2)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAlongTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ALONG the own track, nm (ahead +).

    Track-frame Cartesian. Unlike range x bearing, this gives relative position
    directly - no multiplicative decode whose bearing resolution scales with
    range. Pairs with :class:`RelPosCrossTrackNm`.

    Metadata:
        name: rel_pos_along_track_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("rel_pos_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, _c, _va, _vc = _track_frame(own, other)
        return along / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[0] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosCrossTrackNm(_BroadcastPairs, PairObsField):
    """Intruder position relative to ownship ACROSS the own track, nm (right +).

    Track-frame Cartesian companion to :class:`RelPosAlongTrackNm`.

    Metadata:
        name: rel_pos_cross_track_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("rel_pos_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, cross, _va, _vc = _track_frame(own, other)
        return cross / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[1] / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelAlongTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ALONG the own track, kts.

    Track-frame Cartesian relative velocity (negative = intruder falling behind /
    ownship overtaking along-track). Together with the cross component this is the
    same information as closing rate + bearing rate, but singularity-free and
    without the range-dependent scaling.

    Metadata:
        name: rel_vel_along_track_kts
        unit: kts
        quantity: speed
        is_pair: True
    """

    meta = ObsMeta("rel_vel_along_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "along-track kts"] = -1000.0
    high: Annotated[float, "along-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, _c, v_along, _vc = _track_frame(own, other)
        return v_along * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[2] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelCrossTrackKts(_BroadcastPairs, PairObsField):
    """Intruder velocity relative to ownship ACROSS the own track, kts (right +).

    Track-frame Cartesian companion to :class:`RelVelAlongTrackKts`.

    Metadata:
        name: rel_vel_cross_track_kts
        unit: kts
        quantity: speed
        is_pair: True
    """

    meta = ObsMeta("rel_vel_cross_track_kts", Unit.KTS, ObsQuantity.SPEED, is_pair=True)
    low: Annotated[float, "cross-track kts"] = -1000.0
    high: Annotated[float, "cross-track kts"] = 1000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _a, _c, _va, v_cross = _track_frame(own, other)
        return v_cross * _MS_TO_KTS

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return one_pair_track_frame(own_idx, other_idx)[3] / kts

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaAlongTrackNm(_BroadcastPairs, PairObsField):
    """Along-track relative position at predicted CPA, nm (ahead +).

    Cartesian replacement for the scalar horizontal-miss `dcpa`: together with the
    cross component it preserves the miss magnitude AND adds the pass direction.

    Metadata:
        name: rel_pos_at_cpa_along_track_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("rel_pos_at_cpa_along_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "along-track nm"] = -200.0
    high: Annotated[float, "along-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        along, _cross = _track_frame_at_cpa(own, other)
        return along / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        v2 = max(v_along * v_along + v_cross * v_cross, 1e-9)
        tcpa = max(-(along * v_along + cross * v_cross) / v2, 0.0)  # future only
        return (along + v_along * tcpa) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaCrossTrackNm(_BroadcastPairs, PairObsField):
    """Cross-track relative position at predicted CPA, nm (right +).

    The sign is which side the intruder passes at closest approach - the key cue
    for turn direction. ``|along, cross|`` == the horizontal miss dcpa.

    Metadata:
        name: rel_pos_at_cpa_cross_track_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("rel_pos_at_cpa_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _along, cross = _track_frame_at_cpa(own, other)
        return cross / nm

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        along, cross, v_along, v_cross = one_pair_track_frame(own_idx, other_idx)
        v2 = max(v_along * v_along + v_cross * v_cross, 1e-9)
        tcpa = max(-(along * v_along + cross * v_cross) / v2, 0.0)  # future only
        return (cross + v_cross * tcpa) / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVsFtMin(_BroadcastPairs, PairObsField):
    """Intruder vertical speed minus ownship vertical speed, in ft/min.

    Metadata:
        name: rel_vs_ft_min
        unit: ft/min
        quantity: vertical_speed
        is_pair: True
    """

    meta = ObsMeta(
        "rel_vs_ft_min",
        Unit.FT_PER_MIN,
        ObsQuantity.VERTICAL_SPEED,
        is_pair=True,
    )
    low: Annotated[float, "vertical speed feet per minute"] = -6000.0
    high: Annotated[float, "vertical speed feet per minute"] = 6000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        vs = _traf_array("vs")
        return (vs[other] - vs[own]) * _MS_TO_FTMIN

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        return (float(bs.traf.vs[other_idx]) - float(bs.traf.vs[own_idx])) * 60.0 / ft

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HorizontalDistAtCpaNm(_BroadcastPairs, PairObsField):
    """BlueSky ASAS horizontal distance at closest point of approach, in NM.

    Reads the conflict detector's cached ``dcpa`` (meters, one entry per
    ``confpairs`` row) and converts to nautical miles. ASAS only stores this for
    detected conflict pairs - and a conflict has ``dcpa < rpz`` - so non-conflict
    intruder rows take the high bound (the PZ radius): the smallest miss distance
    that is not a conflict.

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's horizontal protected-zone radius (:func:`_cd_rpz_m`, from
    ``config.pz_radius_nm`` / ``asas_pzr``) as ``(0, rpz)``. That is the true
    support of the cached ``dcpa``, so the normalized value grades the actual
    miss distance instead of collapsing every conflict toward 0.

    Metadata:
        name: horizontal_dist_at_cpa_nm
        unit: nm
        quantity: distance
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "horizontal_dist_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "distance nm lower bound; None = 0 at runtime"
    ] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``dcpa`` (meters, one entry per
        # ``confpairs`` row) each sim step - read it directly and convert to NM.
        # Non-conflict intruders take the high bound (the PZ radius).
        fill = _per_own(own, lambda o: self.bounds(o)[1])
        return _cd_pair_values("dcpa", own, other, fill, divisor=nm)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        value = one_confpair_value("dcpa", own_idx, other_idx)
        return self.bounds(own_idx)[1] if value is None else value / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class VerticalSepAtCpaFt(_BroadcastPairs, PairObsField):
    """Predicted absolute vertical separation at horizontal CPA, in feet.

    The horizontal CPA time is computed from current traffic vectors, for every
    intruder (not only detected conflicts). Negative CPA times are clipped to
    zero, so opening pairs report current vertical separation instead of
    extrapolating into the past.

    Bounds are dynamic: unless explicit ``low``/``high`` are given they follow
    BlueSky's vertical protected-zone height (:func:`_cd_hpz_m`, from
    ``config.pz_height_ft`` / ``asas_pzh``) as ``(0, hpz)`` - the minimum
    vertical separation. Because this reports a value for every intruder, that is
    a *normalization* choice, not the data range: intruders predicted to clear
    the vertical PZ saturate at the safe edge, focusing the signal on the danger
    band. Pass explicit bounds for a wider scale.

    Metadata:
        name: vertical_sep_at_cpa_ft
        unit: ft
        quantity: altitude
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "vertical_sep_at_cpa_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude ft lower bound; None = 0 at runtime"
    ] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        tcpa_s = np.maximum(_pair_horizontal_tcpa_s(own, other), 0.0)
        alt, vs = _traf_array("alt"), _traf_array("vs")
        rel_alt_m = alt[other] - alt[own]
        rel_vs_ms = vs[other] - vs[own]
        return np.abs(rel_alt_m + rel_vs_ms * tcpa_s) * _M_TO_FT

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        east, north, v_east, v_north = one_pair_motion(own_idx, other_idx)
        v2 = max(v_east * v_east + v_north * v_north, 1e-6)
        tcpa = max(-(east * v_east + north * v_north) / v2, 0.0)
        rel_alt = float(bs.traf.alt[other_idx]) - float(bs.traf.alt[own_idx])
        rel_vs = float(bs.traf.vs[other_idx]) - float(bs.traf.vs[own_idx])
        return abs(rel_alt + rel_vs * tcpa) / ft

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, _cd_hpz_m() * _M_TO_FT)
        )


@dataclass(frozen=True)
class _ConflictGeomPairField(_BroadcastPairs, PairObsField):
    """Intruder field sourced from the shared per-step conflict geometry.

    Reads :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView` - the *same*
    continuous, all-pairs CPA primitives the cost and keep mask consume - so the
    feature is identical by construction to what drives the cost. Unlike the ASAS
    ``confpairs`` readers (:class:`HorizontalDistAtCpaNm`, :class:`TcpaS`,
    :class:`TlosS`) there is no detector-cache sentinel snapping and no
    detected-only gap: every intruder gets its true predicted geometry.
    """

    _geom_attr: ClassVar[str] = ""

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        return np.asarray(getattr(view, self._geom_attr), dtype=np.float64)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(getattr(view, self._geom_attr)[0])

@dataclass(frozen=True)
class _WindowedConflictPairField(_ConflictGeomPairField):
    """Conflict field whose value is measured over a 3-D conflict *window*.

    Adds an optional zone override. The window defaults to the live CD
    ``rpz``/``hpz``, which is right whenever the cost grades the true protected
    zone. Tasks that grade a *buffered* zone - a shaped margin wider than the PZ,
    so resolutions are not trained to the PZ edge - must widen the observation's
    window to match, or the marginal encounters the buffer exists to charge are
    reported to the policy as clean misses (they fall to the field's safe-miss
    branch) while the cost bills them.

    Overriding here rather than via ``config.pz_radius_nm`` / ``pz_height_ft`` is
    deliberate: those move BlueSky's CD zone, which is what ``bs.traf.cd.lospairs``
    - and therefore any loss-of-separation cost channel - is defined against.
    Widening the CD zone to fix an observation would silently redefine what counts
    as a LoS. The observation window and the LoS predicate are different things and
    are configured separately.

    Bound defaults still follow the CD zone, so an override wants explicit
    ``low``/``high`` alongside it - otherwise the normalizer re-clips away the very
    band the wider window just exposed.
    """

    rpz_nm: Annotated[
        float | None, "window horizontal radius nm; None = CD rpz at runtime"
    ] = None
    vpz_ft: Annotated[
        float | None, "window vertical half-height ft; None = CD hpz at runtime"
    ] = None

    def _window_zone(self) -> tuple[float, float]:
        """``(rpz_nm, vpz_ft)`` for the window: overrides, else the live CD zone."""
        rpz = _cd_rpz_m() / nm if self.rpz_nm is None else float(self.rpz_nm)
        vpz = _cd_hpz_m() * _M_TO_FT if self.vpz_ft is None else float(self.vpz_ft)
        if rpz <= 0.0 or vpz <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} window zone must be positive, "
                f"got rpz_nm={rpz}, vpz_ft={vpz}."
            )
        return rpz, vpz


@dataclass(frozen=True)
class ConflictHorizontalDistAtCpaNm(_ConflictGeomPairField):
    """Predicted *horizontal* separation at CPA, in nm, from shared conflict geometry.

    The continuous, all-pairs counterpart of :class:`HorizontalDistAtCpaNm` (which
    reads the ASAS ``confpairs`` cache and sentinels non-detected pairs). Bounds
    are dynamic: unless given, ``(0, CD rpz)`` - so a clipped normalizer grades the
    danger band and saturates every safer miss at the PZ edge, matching the cost's
    ``r_h`` support.

    Metadata:
        name: conflict_horizontal_dist_at_cpa_nm
        unit: nm
        quantity: distance
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_horizontal_dist_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    _geom_attr: ClassVar[str] = "dcpa_nm"
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictVerticalSepAtCpaFt(_WindowedConflictPairField):
    """Predicted minimum *vertical* separation over the 3-D conflict window, in ft.

    The continuous, all-pairs counterpart of :class:`VerticalSepAtCpaFt`, reading
    the SAME windowed measure the cost's penetration term (``r_v``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_vsep_ft` - the rel-alt
    line at its in-window minimum, NOT the vertical sep at the *horizontal* CPA
    instant. The CPA-instant reading collapsed to ~0 for altitude-crossing traffic
    whose vertical crossing is offset in time from the horizontal CPA, hiding
    developing vertical conflicts from the policy that the cost was charging; the
    windowed minimum is exactly what the cost sees. A safe miss (no valid 3-D
    window) falls back to the classic CPA-instant value. Bounds are dynamic: unless
    given, ``(0, CD hpz)`` - a clipped normalizer grades the danger band and
    saturates every safe pair at the PZ edge.

    Metadata:
        name: conflict_vertical_sep_at_cpa_ft
        unit: ft
        quantity: altitude
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_vertical_sep_at_cpa_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "altitude ft lower bound; None = 0"] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_vsep_ft(_GeomPairs(own, other), *self._window_zone())

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_min_vsep_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_hpz_m() * _M_TO_FT))


@dataclass(frozen=True)
class ConflictHorizontalSepAtCpaNm(_WindowedConflictPairField):
    """Predicted minimum *horizontal* separation over the 3-D conflict window, nm.

    The horizontal twin of :class:`ConflictVerticalSepAtCpaFt`, reading the SAME
    windowed measure the cost's penetration term (``r_h``) grades via
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_min_hsep_nm` - the
    separation hyperbola at its in-window minimum, NOT the miss distance at the
    unconstrained CPA.

    **Why this is not just ``dcpa``.** :class:`ConflictHorizontalDistAtCpaNm` and
    the :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm` pair
    all report ``dcpa``, the miss over ALL time. The cost charges the pair only
    while it is inside both bands at once, so when the horizontal CPA falls
    outside that window the two diverge - always with ``h_min >= dcpa``, i.e. the
    ``dcpa`` reading is the more alarming one. A policy steering on it maneuvers
    for encounters the cost never bills, and pays for the detour in track miles.

    This is the horizontal half of the same correction
    :class:`ConflictVerticalSepAtCpaFt` made vertically; that one was the more
    urgent because sampling vertical separation at the *horizontal* CPA instant
    was first-order wrong (the wrong axis's extremum time), whereas ``dcpa`` is
    only window-truncation wrong. Keep the signed ``RelPosAtCpa*`` pair alongside
    this field - they carry the pass DIRECTION, which this magnitude does not.

    Bounds are dynamic: unless given, ``(0, CD rpz)`` - a clipped normalizer
    grades the danger band and saturates every safer miss at the PZ edge, matching
    the cost's ``r_h`` support. Pass ``rpz_nm``/``vpz_ft`` (and matching
    ``low``/``high``) when the cost grades a buffered zone rather than the CD one.

    Metadata:
        name: conflict_horizontal_sep_at_cpa_nm
        unit: nm
        quantity: distance
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_horizontal_sep_at_cpa_nm",
        Unit.NM,
        ObsQuantity.DISTANCE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "distance nm lower bound; None = 0"] = None
    high: Annotated[
        float | None, "distance nm upper bound; None = CD PZ radius at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return windowed_min_hsep_nm(_GeomPairs(own, other), *self._window_zone())

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_min_hsep_nm(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class ConflictSignedVerticalSepAtEntryFt(_WindowedConflictPairField):
    """SIGNED vertical separation, in ft, as the 3-D conflict window opens.

    Positive = intruder ABOVE ownship. The vertical counterpart of the signed
    horizontal :class:`RelPosAtCpaAlongTrackNm` / :class:`RelPosAtCpaCrossTrackNm`
    pair, and the missing half of :class:`ConflictVerticalSepAtCpaFt`: that field
    grades HOW CLOSE the pair comes vertically (matching the cost's ``r_v`` by
    construction) but is an absolute value, so on its own it never says which way
    to go. This one supplies the side.

    Sampled at window entry rather than at the in-window minimum on purpose - the
    minimum of a crossing pair sits at the rel-alt zero, where the sign is
    undefined and flips, so a signed *minimum* would be blank exactly where the
    direction matters. See
    :func:`~bluesky_sandbox.sim.geometry.conflict.windowed_signed_vsep_at_entry_ft`.

    Bounds are dynamic: unless given, ``(-CD hpz, +CD hpz)``. Pair with
    :class:`SymmetricNormalizer` so the whole PZ band spans ``[-1, 1]`` and every
    safe pair saturates at the correct end - unlike a wide ``relative_alt_ft``,
    where the danger band is squeezed into a few percent of the input range.

    Metadata:
        name: conflict_signed_vertical_sep_at_entry_ft
        unit: ft
        quantity: altitude
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_signed_vertical_sep_at_entry_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        is_pair=True,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude ft lower bound; None = -CD PZ height at runtime"
    ] = None
    high: Annotated[
        float | None, "altitude ft upper bound; None = +CD PZ height at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        return windowed_signed_vsep_at_entry_ft(view, *self._window_zone())

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        return float(windowed_signed_vsep_at_entry_ft(view, *self._window_zone())[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        hpz_ft = _cd_hpz_m() * _M_TO_FT
        return self._dynamic_or_configured_bounds(lambda: (-hpz_ft, hpz_ft))


@dataclass(frozen=True)
class ConflictTcpaS(_ConflictGeomPairField):
    """Time to horizontal CPA (s), from shared conflict geometry; negative past it.

    The continuous, all-pairs counterpart of :class:`TcpaS` (ASAS ``confpairs``
    cache). Bounds are dynamic: unless given, ``(-lookahead, +lookahead)``.

    Metadata:
        name: conflict_tcpa_s
        unit: s
        quantity: time
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_tcpa_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    _geom_attr: ClassVar[str] = "tcpa_s"
    low: Annotated[
        float | None, "time s lower bound; None = -CD lookahead at runtime"
    ] = None
    high: Annotated[
        float | None, "time s upper bound; None = +CD lookahead at runtime"
    ] = None

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (-_cd_lookahead_s(), _cd_lookahead_s())
        )


@dataclass(frozen=True)
class ConflictTlosS(_WindowedConflictPairField):
    """Predicted time to 3-D LoS entry (BlueSky ``tinconf``), s, from shared geometry.

    The continuous, all-pairs counterpart of :class:`TlosS` (ASAS ``confpairs``
    cache). Computed by
    :func:`~bluesky_sandbox.sim.geometry.conflict.predicted_tlos_s` - the same shared
    ``tinconf`` the cost's imminence term reads - using the CD ``rpz``/``hpz``.
    Non-conflict pairs saturate at the high bound (lookahead); an already-entered
    LoS reads 0. Bounds are dynamic: unless given, ``(0, lookahead)``.

    Metadata:
        name: conflict_tlos_s
        unit: s
        quantity: time
        is_pair: True
        dynamic_bounds: True
    """

    meta = ObsMeta(
        "conflict_tlos_s", Unit.S, ObsQuantity.TIME, is_pair=True, dynamic_bounds=True
    )
    low: Annotated[float | None, "time s lower bound; None = 0"] = None
    high: Annotated[
        float | None, "time s upper bound; None = CD lookahead at runtime"
    ] = None

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        tinconf = predicted_tlos_s(_GeomPairs(own, other), *self._window_zone())
        # +inf (no conflict) -> high bound (safe); <= 0 (already in LoS) -> 0.
        return np.clip(tinconf, 0.0, _per_own(own, lambda o: self.bounds(o)[1]))

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        tinconf = float(predicted_tlos_s(view, *self._window_zone())[0])
        return min(max(tinconf, 0.0), self.bounds(own_idx)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class InConf(_WindowedConflictPairField):
    """1.0 when the intruder is *currently flagged in conflict* with the ownship.

    The detection predicate itself, BlueSky ``StateBased.detect`` reproduced over
    the shared :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView`: the
    pair enters the 3-D protected zone within the lookahead horizon
    (``tinconf <= lookahead``, from
    :func:`~bluesky_sandbox.sim.geometry.conflict.predicted_tlos_s` - the same
    ``tinconf`` the cost's imminence term reads). Non-conflict pairs score ``0``,
    a pair already inside the zone (``tinconf <= 0``) scores ``1`` until it exits.

    Computed from the shared geometry rather than read from ``bs.traf.cd``
    (:class:`ConflictRisk`, :class:`TcpaS`, :class:`TlosS` do the latter), so it
    agrees with a ``ConflictView``-derived cost by construction and carries no
    detector-cache gap: every intruder is scored every step, not only the rows CD
    happened to cache. It is the binary companion of :class:`ConflictTlosS` -
    same window, same zone - stated where a network can read it.

    **Why a dedicated flag** rather than thresholding ``conflict_tlos_s``: that
    field clips ``+inf`` (no conflict at all) to its high bound, which is also
    where a conflict entering *exactly* at the horizon sits, so the two are
    indistinguishable at the top of the normalized range. This field separates
    them. The same argument :class:`InLosNow` makes for the LoS predicate.

    Pair the two: this one is the approach CD would alert on, :class:`InLosNow`
    is the breach. The implication runs one way only - a pair inside the zone is
    by construction inside its own conflict window and stays flagged until it
    exits, so ``in_los_now = 1`` always comes with ``in_conf = 1``, while the
    reverse is false for every conflict still minutes from entry. That gap is
    the point: it is the warning time a resolution has to act in, and only this
    field marks its opening.

    ``rpz_nm`` / ``vpz_ft`` / ``lookahead_s`` default to the live CD values and
    override together with the cost's zone - see
    :class:`_WindowedConflictPairField` for why a buffered zone belongs here and
    not in ``config.pz_radius_nm``.

    Leave ``normalizer`` unset - the value is already 0/1, so there is nothing
    to scale.

    Metadata:
        name: in_conf
        unit: unitless
        quantity: indicator
        is_pair: True
    """

    meta = ObsMeta("in_conf", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    lookahead_s: Annotated[
        float | None, "detection horizon s; None = CD lookahead at runtime"
    ] = None
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _horizon_s(self) -> float:
        look = (
            _cd_lookahead_s() if self.lookahead_s is None else float(self.lookahead_s)
        )
        if look <= 0.0:
            raise ValueError(
                f"{self.__class__.__name__} lookahead_s must be positive, got {look}."
            )
        return look

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # ``+inf`` (no valid conflict window) compares False against any finite
        # horizon, so the horizon test alone is the full detection predicate.
        tinconf = predicted_tlos_s(_GeomPairs(own, other), *self._window_zone())
        return (tinconf <= self._horizon_s()).astype(np.float32)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        tinconf = float(predicted_tlos_s(view, *self._window_zone())[0])
        return float(tinconf <= self._horizon_s())

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class InLosNow(_BroadcastPairs, PairObsField):
    """1.0 when the intruder is *currently* inside the ownship's protected zone.

    The loss-of-separation predicate itself: ``horizontal < rpz AND |dalt| < hpz``,
    read from the shared :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView`
    (``horiz_dist_now_nm`` / ``dalt_now_ft``) against the live CD ``rpz``/``hpz``.
    Being current-state rather than predicted, it is the odd one out among the
    ``Conflict*`` fields - they all describe geometry at a future CPA, this one
    describes right now, hence the ``Now`` suffix mirroring the view's own
    attribute names.

    **Why a dedicated flag** rather than letting the network derive it. LoS is
    typically what a constrained task *counts* as its cost, and a policy that is
    penalized for it usually cannot see it: the continuous features either sit on
    the wrong side of a critic-only split, or bury the predicate in a few percent
    of their range (a 5 nm PZ inside a +-60 nm ``RelPosAlongTrackNm`` is ~4%), or
    - as with :class:`ConflictTlosS`, which reads exactly ``0`` once LoS is
    entered - encode it at a single endpoint of the normalized range, where it is
    indistinguishable from saturation. This field states it directly.

    Pair it with :class:`ConflictTlosS` for the approach and this for the entry
    (or with :class:`InConf`, the binary form of the same approach); the two
    carry different information and neither substitutes for the other.
    For a *graded* conflict signal see :class:`ConflictRisk` - but note that one
    reads BlueSky's ASAS ``confpairs`` cache, so it does not necessarily agree
    with a cost computed from ``ConflictView``, whereas this field does by
    construction.

    Leave ``normalizer`` unset - the value is already 0/1, so there is nothing
    to scale.

    Metadata:
        name: in_los_now
        unit: unitless
        quantity: indicator
        is_pair: True
    """

    meta = ObsMeta("in_los_now", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        view = _GeomPairs(own, other)
        # Strict ``<`` on both axes, matching BlueSky's own LoS test (and the
        # keep-mask/cost predicates built on this view): a pair sitting exactly
        # ON the zone boundary is not yet a loss of separation.
        in_los = (view.horiz_dist_now_nm < _cd_rpz_m() / nm) & (
            view.dalt_now_ft < _cd_hpz_m() * _M_TO_FT
        )
        return in_los.astype(np.float32)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        view = ConflictView(own_idx, others=np.array([other_idx]))
        horizontal = float(view.horiz_dist_now_nm[0]) < _cd_rpz_m() / nm
        vertical = float(view.dalt_now_ft[0]) < _cd_hpz_m() / ft
        return float(horizontal and vertical)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixApproachDistNm(_BroadcastPairs, PairObsField):
    """How near an intruder's *observed* path passes the ownship's active fix, nm.

    Projects the intruder's current position/velocity to its closest approach
    of the ownship's own fix (``route_offset`` 0 = active leg, 1 = next leg,
    e.g. the shared exit while still working a merge fix). Small => that
    aircraft is funnelling into my fix (a merge threat); large => crossing or
    unrelated. Captures the shared-destination geometry that pairwise
    ownship<->intruder CPA fields miss (same-heading in-trail traffic has low
    closing rate yet converges at the fix). Non-private: see :func:`_fix_projection`.
    Sentinel ``high`` (far) when the ownship has no usable fix.

    Metadata:
        name: intruder_fix_approach_dist_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta(
        "intruder_fix_approach_dist_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "distance nm lower bound"] = 0.0
    high: Annotated[float, "distance nm upper bound (also the no-fix sentinel)"] = 100.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        cpa_nm, _eta, _own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, cpa_nm, float(self.high))

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return float(self.high) if projection is None else projection[0]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixArrivalDeltaS(_BroadcastPairs, PairObsField):
    """Signed queue order at the ownship's fix: intruder ETA minus ownship ETA, s.

    Both ETAs come from projecting *observed* motion onto the ownship's own fix
    (``route_offset`` 0 = active, 1 = next). Negative => the intruder reaches
    my fix *before* me (I should slot behind it); positive => I am ahead. This
    is the temporal-sequencing signal for a point merge - who arrives first -
    that lets the policy learn to decelerate and space in-trail rather than
    turn (which cannot separate traffic converging on a shared point).
    Non-private: see :func:`_fix_projection`. ``0`` (no order) when the ownship
    has no usable fix. Pair with :class:`IntruderFixApproachDistNm` so a
    consumer knows *whether* the intruder is heading to the fix at all.

    Metadata:
        name: intruder_fix_arrival_delta_s
        unit: s
        quantity: time
        is_pair: True
    """

    meta = ObsMeta(
        "intruder_fix_arrival_delta_s", Unit.S, ObsQuantity.TIME, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "time s lower bound"] = -600.0
    high: Annotated[float, "time s upper bound"] = 600.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, intr_eta, own_eta, _vsep, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, intr_eta - own_eta, 0.0)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[1] - projection[2]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixVerticalSepFt(_BroadcastPairs, PairObsField):
    """Signed altitude gap at the ownship's fix: intruder minus ownship, ft.

    Both aircraft are projected to the fix by their observed vertical speed
    (current alt + VS x ETA), and the signed difference is taken there. Near 0
    => they will be *co-altitude* at the merge (a genuine conflict that lateral
    turning cannot separate - your same-altitude merge case); large magnitude
    => vertically clear, no sequencing needed regardless of horizontal
    approach. This is the vertical companion to
    :class:`IntruderFixApproachDistNm` (horizontal) and
    :class:`IntruderFixArrivalDeltaS` (temporal); together they predict a merge
    conflict against the ownship's fix the way the pairwise
    ``ConflictHorizontalDistAtCpaNm`` / ``ConflictVerticalSepAtCpaFt`` /
    ``ConflictTcpaS`` triple predicts a pairwise one. Signed so the policy
    knows which way to separate. Non-private: see :func:`_fix_projection`.
    ``0`` when the ownship has no usable fix - read alongside the approach
    field, whose sentinel flags that state.

    Metadata:
        name: intruder_fix_vertical_sep_ft
        unit: ft
        quantity: altitude
        is_pair: True
    """

    meta = ObsMeta(
        "intruder_fix_vertical_sep_ft", Unit.FT, ObsQuantity.ALTITUDE, is_pair=True
    )
    route_offset: Annotated[
        int, "ownship route index offset (0=active leg, 1=next leg, ...)"
    ] = 0
    own_eta_mode: Annotated[
        str, "ownship ETA model: 'projection' (kinematic, default) or 'route'"
    ] = "projection"
    low: Annotated[float, "altitude ft lower bound"] = -5000.0
    high: Annotated[float, "altitude ft upper bound"] = 5000.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        _cpa, _eta, _own_eta, vsep_ft, has_fix = _fix_projection(
            own, other, self.route_offset, self.own_eta_mode
        )
        return np.where(has_fix, vsep_ft, 0.0)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        projection = one_pair_fix_projection(
            own_idx, other_idx, self.route_offset, self.own_eta_mode
        )
        return 0.0 if projection is None else projection[3]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderCommMessage(_CommBacked, PairObsField):
    """One channel of an intruder's broadcast communication message.

    Reads the value the intruder emitted through its ``CommBroadcast`` action
    on the previous step (0.0 for an aircraft that has not yet spoken). A pure
    agent-to-agent information channel with no physical semantics - the
    meaning of the signal is whatever the shared policy learns to encode.
    Values live in the action's ``[-1, 1]`` range.

    ``noise_std > 0`` adds receiver-side Gaussian channel noise (clipped back
    to the message range), drawn from a per-episode-seeded RNG. The DIAL/DRU
    grounding pressure: a message must be high-contrast to survive a noisy
    channel, so ambiguous low-amplitude signaling stops being free.

    Metadata:
        name: intruder_comm_message
        unit: unitless
        quantity: action
        is_pair: True
    """

    meta = ObsMeta(
        "intruder_comm_message",
        Unit.UNITLESS,
        ObsQuantity.ACTION,
        is_pair=True,
    )
    channel: Annotated[int, "message channel index"] = 0
    noise_std: Annotated[float, "receiver-side Gaussian channel noise std"] = 0.0
    low: Annotated[float, "message value"] = -1.0
    high: Annotated[float, "message value"] = 1.0

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices).ravel()
        return self._received(np.array([int(own_idx)]), others[None, :])[0]

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        owns = _indices_array(own_indices).ravel()
        n = int(bs.traf.ntraf)
        # Every other aircraft, in order, for each ownship: exactly the pairs
        # get_pairs would be asked for, so the noise draws come in the same order.
        heard = np.arange(n)[None, :] != owns[:, None]
        matrix = np.full((owns.size, n), np.nan)
        cols = np.broadcast_to(np.arange(n), heard.shape)[heard].reshape(owns.size, -1)
        matrix[heard] = self._received(owns, cols).ravel()
        return matrix

    def _received(self, owns: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Each ownship's received messages from aircraft ``cols[row]``.

        The message does not depend on the listener; the receiver noise does,
        drawn row by row so a matrix consumes the noise stream exactly as one
        ``get_pairs`` call per ownship would.
        """
        del owns
        sent = comm_messages(self.channel)
        values = sent[cols]
        if self.noise_std > 0.0 and values.size:
            values = values + _state._COMM_NOISE_RNG.normal(
                0.0, self.noise_std, size=values.shape
            )
            np.clip(values, self.low, self.high, out=values)
        return values

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        # The message as sent; receiver noise (``noise_std``) is random on top.
        del own_idx
        return float(_state._COMM_MESSAGE.read_one(other_idx).get(self.channel, 0.0))

    def bounds(self, own_idx: int) -> tuple[float, float]:
        del own_idx
        return float(self.low), float(self.high)


@dataclass(frozen=True)
class BrgFromOwnDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in degrees.

    The true bearing from ownship to the intruder, signed in ``[-180, 180]``
    (0 = north, +90 = east, -90 = west) as returned by ``qdrdist``.

    Metadata:
        name: brg_from_own_deg
        unit: deg
        quantity: bearing
        is_pair: True
        circular: True
    """

    meta = ObsMeta(
        "brg_from_own_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        qdr, _dist = _pair_qdr_dist(own, other)
        return qdr

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        return float(qdrdist(*own, *other)[0])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnRelTrkDeg(_BroadcastPairs, PairObsField):
    """Ownship-relative intruder bearing in the ownship track frame, degrees.

    The true bearing from ownship to the intruder minus the ownship track, so
    ``0`` means the intruder is dead ahead and ``+/-180`` directly behind. This
    is the egocentric (body-frame) polar angle, matching
    :class:`ActiveRouteWaypointTrackErrorDeg` for the waypoint; pair with
    :class:`DistToOwnNm` for a full egocentric polar intruder position. Unlike
    :class:`BrgFromOwnDeg` (an absolute compass bearing), this rotates with the
    ownship heading.

    Metadata:
        name: brg_from_own_rel_trk_deg
        unit: deg
        quantity: bearing
        is_pair: True
        circular: True
    """

    meta = ObsMeta(
        "brg_from_own_rel_trk_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        is_pair=True,
        circular=True,
    )
    low: Annotated[float, "bearing degrees"] = -180.0
    high: Annotated[float, "bearing degrees"] = 180.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        qdr, _dist = _pair_qdr_dist(own, other)
        return (qdr - _traf_array("trk")[own] + 540.0) % 360.0 - 180.0

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        own = (float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx]))
        other = (float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx]))
        relative = float(qdrdist(*own, *other)[0]) - float(bs.traf.trk[own_idx])
        return (relative + 180.0) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# --------------------------------------------------------------------------- #
# BlueSky CD parameters (lookahead, protected zone) and per-intruder risk      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConflictRisk(_BroadcastPairs, PairObsField):
    """Per-intruder graded conflict risk, ``1 - tcpa/lookahead`` in ``[0, 1]``.

    The per-element version of a CD-based safety cost: for an intruder BlueSky's
    conflict detector flags against the ownship, risk is
    ``1 - max(tcpa, 0)/lookahead`` (lookahead from :func:`_cd_lookahead_s`);
    non-conflict intruders score ``0``. Placed on the intruder token so attention
    can focus on the threatening aircraft, with the ownship's worst-case risk
    recoverable by max-pooling this field.

    Metadata:
        name: conflict_risk
        unit: unitless
        quantity: risk
        is_pair: True
    """

    meta = ObsMeta("conflict_risk", Unit.UNITLESS, ObsQuantity.RISK, is_pair=True)
    low: Annotated[float, "risk fraction"] = 0.0
    high: Annotated[float, "risk fraction"] = 1.0

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        # BlueSky ConflictDetection caches ``tcpa`` per ``confpairs`` row each sim
        # step. Non-conflict rows default to the lookahead horizon -> risk 0;
        # conflict rows use their CD tcpa, graded toward 1 as the conflict nears.
        lookahead = _cd_lookahead_s()
        tcpa = _cd_pair_values("tcpa", own, other, lookahead)
        risk = 1.0 - np.maximum(tcpa, 0.0) / lookahead
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        lookahead = _cd_lookahead_s()
        tcpa = one_confpair_value("tcpa", own_idx, other_idx)
        if tcpa is None:
            tcpa = lookahead
        return min(max(1.0 - max(tcpa, 0.0) / lookahead, 0.0), 1.0)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()
