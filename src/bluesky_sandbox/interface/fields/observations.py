from __future__ import annotations

from collections import deque
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
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from .base import ObsField, ObsMeta, ObsQuantity, PairObsField, Unit

_M_TO_FT = 1.0 / ft
_MS_TO_KTS = 1.0 / kts
_MS_TO_FTMIN = 60.0 / ft
_MIN_DYNAMIC_SPAN = 1e-6
# Groundspeed floor for an ETE denominator: airborne traffic never reaches it,
# it only keeps a division finite for a stopped/uninitialised aircraft.
_MIN_GS_MS = 1e-3


def _signed_angle_delta_deg(left: float, right: float) -> float:
    return (left - right + 540.0) % 360.0 - 180.0


def _indices_array(indices: Any) -> np.ndarray:
    return np.asarray(indices, dtype=np.intp)


def _pair_qdr_dist(own_idx: int, other_indices: Any) -> tuple[np.ndarray, np.ndarray]:
    other_indices = _indices_array(other_indices)
    own_lat = np.full(other_indices.shape, bs.traf.lat[own_idx])
    own_lon = np.full(other_indices.shape, bs.traf.lon[own_idx])
    qdr, dist = qdrdist(
        own_lat,
        own_lon,
        bs.traf.lat[other_indices],
        bs.traf.lon[other_indices],
    )
    return np.asarray(qdr, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def _pair_relative_motion(
    own_idx: int,
    other_indices: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return relative position and velocity as east/north arrays in SI units."""
    qdr, dist_nm = _pair_qdr_dist(own_idx, other_indices)
    other_indices = _indices_array(other_indices)
    qdrrad = np.radians(qdr)
    dist_m = dist_nm * nm
    rel_east_m = dist_m * np.sin(qdrrad)
    rel_north_m = dist_m * np.cos(qdrrad)

    own_track = np.radians(float(bs.traf.trk[own_idx]))
    own_east_ms = float(bs.traf.gs[own_idx]) * np.sin(own_track)
    own_north_ms = float(bs.traf.gs[own_idx]) * np.cos(own_track)

    other_track = np.radians(bs.traf.trk[other_indices])
    other_east_ms = bs.traf.gs[other_indices] * np.sin(other_track)
    other_north_ms = bs.traf.gs[other_indices] * np.cos(other_track)

    return (
        rel_east_m,
        rel_north_m,
        other_east_ms - own_east_ms,
        other_north_ms - own_north_ms,
    )


def _pair_horizontal_tcpa_s(own_idx: int, other_indices: Any) -> np.ndarray:
    rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
        own_idx,
        other_indices,
    )
    rel_speed2 = np.maximum(
        rel_east_ms * rel_east_ms + rel_north_ms * rel_north_ms,
        1e-6,
    )
    return -(
        rel_east_m * rel_east_ms + rel_north_m * rel_north_ms
    ) / rel_speed2


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
class LatDeg(ObsField):
    """Latitude in degrees.

    Metadata:
        name: lat_deg
        unit: deg
        quantity: latitude
    """

    meta = ObsMeta("lat_deg", Unit.DEG, ObsQuantity.LATITUDE)
    low: Annotated[float, "latitude degrees"] = -90.0
    high: Annotated[float, "latitude degrees"] = 90.0

    def get(self, idx: Any) -> Any:
        return bs.traf.lat[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.lat[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class LonDeg(ObsField):
    """Longitude in degrees.

    Metadata:
        name: lon_deg
        unit: deg
        quantity: longitude
    """

    meta = ObsMeta("lon_deg", Unit.DEG, ObsQuantity.LONGITUDE)
    low: Annotated[float, "longitude degrees"] = -180.0
    high: Annotated[float, "longitude degrees"] = 180.0

    def get(self, idx: Any) -> Any:
        return bs.traf.lon[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.lon[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HdgDeg(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.hdg[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.hdg[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TrkDeg(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.trk[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.trk[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# --------------------------------------------------------------------------- #
# Active route waypoint (per-aircraft, name-free)                             #
# --------------------------------------------------------------------------- #
def _route_constraint(values: Any, iact: int) -> float | None:
    """Read a per-waypoint constraint (e.g. ``wpalt``/``wpspd``) at ``iact``.

    Returns ``None`` for a missing or unspecified constraint; BlueSky stores
    "not specified" as a negative sentinel.
    """
    try:
        value = float(values[iact])
    except (IndexError, TypeError, ValueError):
        return None
    if np.isfinite(value) and value >= 0.0:
        return value
    return None


def _active_route_waypoint(
    idx: int,
    offset: int = 0,
) -> tuple[float, float, float | None, float | None] | None:
    """Return ``(lat, lon, alt_m, spd_ms)`` for a route fix relative to the
    aircraft's active leg, or ``None`` when there is no usable fix there.

    ``offset`` selects which fix: ``0`` (default) is the active leg the
    autopilot is currently flying to; ``1`` is the *next* leg (e.g. the exit
    fix while still working a merge fix), ``2`` the one after, etc. ``None``
    whenever ``iactwp + offset`` falls outside the route (no next leg on a
    single-leg route, or beyond the final fix) - the same "no usable waypoint"
    convention offset ``0`` already used for a routeless aircraft.

    ``alt_m`` and ``spd_ms`` are ``None`` when the waypoint carries no altitude
    or speed constraint. Shared by the active-route-waypoint observation fields
    and the deviation-from-nominal action fields (which always use ``offset=0``
    - actions command the leg actually being flown, never a future one) so
    both read the same target.
    """
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    route = routes[idx]
    try:
        iact = int(route.iactwp)
        nwp = int(route.nwp or 0)
    except (TypeError, ValueError):
        return None
    if iact < 0:
        return None
    target = iact + offset
    if target < 0 or target >= nwp:
        return None
    try:
        lat = float(route.wplat[target])
        lon = float(route.wplon[target])
    except (IndexError, TypeError, ValueError):
        return None
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    return (
        lat,
        lon,
        _route_constraint(route.wpalt, target),
        _route_constraint(route.wpspd, target),
    )


def _route_along_distance_nm(idx: int, offset: int) -> float | None:
    """Distance from the aircraft to the route fix at ``iactwp + offset``, nm,
    measured ALONG the remaining legs.

    Direct great-circle range to the active fix, then leg by leg out to the
    target fix - the distance actually flown, not the straight line to a fix
    two legs ahead. Reduces to the plain range at ``offset=0``, where it equals
    :class:`ActiveRouteWaypointDistanceNm`. ``None`` whenever there is no
    usable fix at that offset, mirroring :func:`_active_route_waypoint`'s
    validation and convention.
    """
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    route = routes[idx]
    try:
        iact = int(route.iactwp)
        nwp = int(route.nwp or 0)
    except (TypeError, ValueError):
        return None
    target = iact + offset
    if iact < 0 or target < 0 or target >= nwp:
        return None
    lat = float(bs.traf.lat[idx])
    lon = float(bs.traf.lon[idx])
    total_nm = 0.0
    for leg in range(iact, target + 1):
        try:
            wp_lat = float(route.wplat[leg])
            wp_lon = float(route.wplon[leg])
        except (IndexError, TypeError, ValueError):
            return None
        if not (np.isfinite(wp_lat) and np.isfinite(wp_lon)):
            return None
        _qdr, dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        total_nm += float(dist)
        lat, lon = wp_lat, wp_lon
    return total_nm


@dataclass(frozen=True)
class _ActiveRouteWaypointField(ObsField):
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

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ActiveRouteWaypointDistanceNm(_ActiveRouteWaypointField):
    """Distance from ownship to its active route waypoint, nm (0 if none)."""

    meta = ObsMeta("active_route_waypoint_distance_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: float = 0.0
    high: float = 200.0

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None:
            return 0.0
        _qdr, dist = kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp[0], wp[1])
        return float(dist)


@dataclass(frozen=True)
class ActiveRouteWaypointBearingDeg(_ActiveRouteWaypointField):
    """True bearing from ownship to its active route waypoint, deg (0 if none)."""

    meta = ObsMeta("active_route_waypoint_bearing_deg", Unit.DEG, ObsQuantity.BEARING, circular=True)
    low: float = 0.0
    high: float = 360.0

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None:
            return 0.0
        qdr, _dist = kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp[0], wp[1])
        return float(qdr) % 360.0


@dataclass(frozen=True)
class ActiveRouteWaypointTrackErrorDeg(_ActiveRouteWaypointField):
    """Signed track error from ownship track toward its active route waypoint."""

    meta = ObsMeta("active_route_waypoint_track_error_deg", Unit.DEG, ObsQuantity.TRACK, circular=True)
    low: float = -180.0
    high: float = 180.0

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None:
            return 0.0
        qdr, _dist = kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp[0], wp[1])
        return _signed_angle_delta_deg(float(qdr), float(bs.traf.trk[idx]))


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

    def get(self, idx: Any) -> Any:
        return 1.0 if self._active_wp(int(idx)) is not None else 0.0


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

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        return 1.0 if wp is not None and wp[2] is not None else 0.0


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

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        return 1.0 if wp is not None and wp[3] is not None else 0.0


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

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None or wp[2] is None:
            return 0.0
        return float(bs.traf.alt[idx] * _M_TO_FT - wp[2] * _M_TO_FT)

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

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None or wp[3] is None:
            return 0.0
        return float((bs.traf.cas[idx] - wp[3]) * _MS_TO_KTS)

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

    def get(self, idx: Any) -> Any:
        wp = self._active_wp(int(idx))
        if wp is None or wp[3] is None:
            return 0.0
        return float(crossover_speed_state(int(idx), wp[3]).normalized_error)


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

    def get(self, idx: Any) -> Any:
        idx = int(idx)
        dist_nm = _route_along_distance_nm(idx, self.route_offset)
        if dist_nm is None:
            return 0.0
        return float(dist_nm * nm / max(float(bs.traf.gs[idx]), _MIN_GS_MS))


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

    def get(self, idx: Any) -> Any:
        idx = int(idx)
        wp = self._active_wp(idx)
        if wp is None or wp[2] is None:
            return 0.0
        # Positive => still has to climb to the gate, negative => descend.
        error_m = float(wp[2]) - float(bs.traf.alt[idx])
        if error_m == 0.0:
            return 0.0
        if self.vs_mode == "current":
            rate_ms = float(bs.traf.vs[idx])
        elif error_m > 0.0:
            rate_ms = abs(float(bs.traf.perf.vsmax[idx]))
        else:
            rate_ms = -abs(float(bs.traf.perf.vsmin[idx]))
        # Level, or closing the wrong way: the level-off never happens.
        if rate_ms * error_m <= 0.0:
            return float(self.high)
        return float(min(error_m / rate_ms, float(self.high)))


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

    @classmethod
    def _altitude_ceiling_ft(cls, idx: int) -> float:
        return cls._altitude_ceiling_m(idx) * _M_TO_FT


@dataclass(frozen=True)
class _AltitudeEnvelopeFt(ObsField):
    """Altitude observation with bounds from ``bs.traf.perf.hmax``."""

    low: Annotated[
        float | None,
        "altitude feet lower bound; None = 0 ft at runtime",
    ] = None
    high: Annotated[
        float | None,
        "altitude feet upper bound; None = BlueSky perf.hmax at runtime",
    ] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, self._altitude_ceiling_ft(idx))
        )


@dataclass(frozen=True)
class _AltitudeEnvelopeM(ObsField):
    """Altitude observation with bounds from ``bs.traf.perf.hmax``."""

    low: Annotated[
        float | None,
        "altitude metres lower bound; None = 0 m at runtime",
    ] = None
    high: Annotated[
        float | None,
        "altitude metres upper bound; None = BlueSky perf.hmax at runtime",
    ] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, self._altitude_ceiling_m(idx))
        )


@dataclass(frozen=True)
class AltFt(_AltitudeEnvelopeBounds, _AltitudeEnvelopeFt):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.alt[idx] * _M_TO_FT

    def get_many(self, indices: Any) -> Any:
        return bs.traf.alt[_indices_array(indices)] * _M_TO_FT


@dataclass(frozen=True)
class AltM(_AltitudeEnvelopeBounds, _AltitudeEnvelopeM):
    """Aircraft altitude in metres.

    Metadata:
        name: alt_m
        unit: m
        quantity: altitude
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    aircraft altitude ceiling at runtime.
    """

    meta = ObsMeta("alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.alt[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.alt[_indices_array(indices)]


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
class _SpeedEnvelopeKts(ObsField):
    """Speed observation with bounds from ``_speed_bounds_ms(...)``."""

    low: Annotated[
        float | None,
        "speed knots lower bound; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "speed knots upper bound; None = runtime speed envelope",
    ] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: tuple(value * _MS_TO_KTS for value in self._speed_bounds_ms(idx))
        )


@dataclass(frozen=True)
class _SpeedEnvelopeMs(ObsField):
    """Speed observation with bounds from ``_speed_bounds_ms(...)``."""

    low: Annotated[
        float | None,
        "speed m/s lower bound; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "speed m/s upper bound; None = runtime speed envelope",
    ] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: self._speed_bounds_ms(idx))


@dataclass(frozen=True)
class CasKts(_CasEnvelopeBounds, _SpeedEnvelopeKts):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.cas[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.cas[_indices_array(indices)] * _MS_TO_KTS


@dataclass(frozen=True)
class CasMs(_CasEnvelopeBounds, _SpeedEnvelopeMs):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.cas[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.cas[_indices_array(indices)]


@dataclass(frozen=True)
class TasKts(_TasEnvelopeBounds, _SpeedEnvelopeKts):
    """True airspeed in knots.

    Metadata:
        name: tas_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("tas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.tas[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.tas[_indices_array(indices)] * _MS_TO_KTS


@dataclass(frozen=True)
class TasMs(_TasEnvelopeBounds, _SpeedEnvelopeMs):
    """True airspeed in m/s.

    Metadata:
        name: tas_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("tas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.tas[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.tas[_indices_array(indices)]


@dataclass(frozen=True)
class VsFtMin(ObsField):
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
    low: Annotated[
        float | None,
        "vertical speed ft/min lower bound; None = perf.vsmin",
    ] = None
    high: Annotated[
        float | None,
        "vertical speed ft/min upper bound; None = perf.vsmax at runtime",
    ] = None

    def get(self, idx: Any) -> Any:
        return bs.traf.vs[idx] * _MS_TO_FTMIN

    def get_many(self, indices: Any) -> Any:
        return bs.traf.vs[_indices_array(indices)] * _MS_TO_FTMIN

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            vsmin = float(bs.traf.perf.vsmin[idx]) * _MS_TO_FTMIN
            vsmax = float(bs.traf.perf.vsmax[idx]) * _MS_TO_FTMIN
            assert vsmin <= 0
            
            return vsmin, vsmax

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class VsMs(ObsField):
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
    low: Annotated[
        float | None,
        "vertical speed m/s lower bound; None = perf.vsmin at runtime",
    ] = None
    high: Annotated[
        float | None,
        "vertical speed m/s upper bound; None = perf.vsmax at runtime",
    ] = None

    def get(self, idx: Any) -> Any:
        return bs.traf.vs[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.vs[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            vsmin = float(bs.traf.perf.vsmin[idx])
            vsmax = float(bs.traf.perf.vsmax[idx])
            assert vsmin <= 0.0
           
            return vsmin, vsmax

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class AxMs2(ObsField):
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

    def get(self, idx: Any) -> Any:
        return float(bs.traf.ax[idx])

    def get_many(self, indices: Any) -> Any:
        return np.asarray(bs.traf.ax, dtype=np.float32)[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class MachNumber(ObsField):
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

    def get(self, idx: Any) -> Any:
        return float(bs.traf.M[idx])

    def get_many(self, indices: Any) -> Any:
        return np.asarray(bs.traf.M, dtype=np.float32)[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class CrossoverAltMarginFt(ObsField):
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

    def get(self, idx: Any) -> Any:
        cas = float(bs.traf.cas[idx])
        alt = float(bs.traf.alt[idx])
        mmo = float(bs.traf.perf.mmo[idx])
        return (alt - float(crossoveralt(cas, mmo))) * _M_TO_FT

    def get_many(self, indices: Any) -> Any:
        i = _indices_array(indices)
        cas = np.asarray(bs.traf.cas, dtype=np.float64)[i]
        alt = np.asarray(bs.traf.alt, dtype=np.float64)[i]
        mmo = np.asarray(bs.traf.perf.mmo, dtype=np.float64)[i]
        return ((alt - np.asarray(crossoveralt(cas, mmo))) * _M_TO_FT).astype(
            np.float32
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def reset_all_field_state(seed: int | None = None) -> None:
    """Clear every per-aircraft field store, whatever this env configures.

    Recording and per-aircraft forgetting are opt-in - a field that nobody
    configures records nothing, so there is nothing to drop. Episode reset is
    the exception: the stores are module-level, so two envs built from
    different configs in one process share them, and the second env would
    inherit whatever the first left behind in a store it has no field for.
    One sweep, called by the environment on reset, restores that isolation
    without reintroducing a per-store list for anyone to forget to update.
    """
    global _COMM_NOISE_RNG
    _LAST_NORM_ACTION.clear()
    _LAG_HISTORY.clear()
    _LAG_LAST_SIMT.clear()
    _TIME_IN_ENV.clear()
    _TURN_RATE.clear()
    _REALIZED_ACCEL.clear()
    _COMM_MESSAGE.clear()
    _KINEMATICS.clear()
    _COMM_NOISE_RNG = np.random.default_rng(seed)


class _LastActionBacked:
    """State hooks for the previous-action field.

    The store is written only from here - the field that reads it - so a
    recorded action and its removal cannot be maintained in two places that
    drift apart.
    """

    def on_action_applied(self, acid: str, action) -> None:
        _LAST_NORM_ACTION[acid] = np.asarray(action, dtype=np.float32)

    def on_aircraft_removed(self, acid: str) -> None:
        _LAST_NORM_ACTION.pop(acid, None)


class _LagHistoryBacked:
    """State hooks for lag/stack wrappers.

    History is pushed lazily on read (see ``get_many``), so there is no
    ``on_substep`` here - only the per-aircraft drop. Keys are
    ``(field_key, acid)`` or ``(field_key, own_acid, other_acid)``, so losing
    one aircraft means losing every key that mentions it.
    """

    def on_aircraft_removed(self, acid: str) -> None:
        for key in [k for k in _LAG_HISTORY if acid in k[1:]]:
            del _LAG_HISTORY[key]


class _TimeInEnvBacked:
    """State hooks for time-in-env.

    Spawn times live in the environment, so the age arrives on the substep
    context; recording and the per-aircraft drop are here.
    """

    def on_substep(self, ctx) -> None:
        _TIME_IN_ENV.update(ctx.age_s)

    def on_aircraft_removed(self, acid: str) -> None:
        _TIME_IN_ENV.pop(acid, None)


class _CommBacked:
    """State hooks for the comm-message channel field."""

    def on_aircraft_removed(self, acid: str) -> None:
        _COMM_MESSAGE.pop(acid, None)


class _KinematicsTracker:
    """Per-aircraft turn rate and realized accelerations, differenced per substep.

    One tracker behind four fields: ``TurnRateDegPerSec`` and the three
    ``RealizedAccel*`` fields all need the same previous (track, ground speed,
    vertical speed), so they share this rather than each keeping its own copy
    and differencing the same numbers three times over.

    ``update`` is idempotent within a substep - the fields all call it, the
    first call does the work and the rest see the same ``sim_time`` and return.
    """

    def __init__(self) -> None:
        self._prev: dict[str, tuple[float, float, float]] = {}
        self._last_time: float | None = None

    def update(self, ctx) -> None:
        if self._last_time == ctx.sim_time:
            return  # a sibling field already advanced us this substep
        self._last_time = ctx.sim_time
        dt = ctx.dt
        if dt <= 0.0:
            return
        trk, gs, vs = bs.traf.trk, bs.traf.gs, bs.traf.vs
        nxt: dict[str, tuple[float, float, float]] = {}
        for i, acid in enumerate(ctx.ids):
            cur = (float(trk[i]), float(gs[i]), float(vs[i]))
            last = self._prev.get(acid)
            if last is None:
                _TURN_RATE[acid] = 0.0
                _REALIZED_ACCEL[acid] = (0.0, 0.0, 0.0)
            else:
                delta = (cur[0] - last[0] + 180.0) % 360.0 - 180.0  # (-180, 180]
                _TURN_RATE[acid] = delta / dt
                _REALIZED_ACCEL[acid] = (
                    (cur[1] - last[1]) / dt,
                    0.5 * (cur[1] + last[1]) * np.radians(delta) / dt,
                    (cur[2] - last[2]) / dt,
                )
            nxt[acid] = cur
        self._prev = nxt

    def forget(self, acid: str) -> None:
        self._prev.pop(acid, None)
        _TURN_RATE.pop(acid, None)
        _REALIZED_ACCEL.pop(acid, None)

    def clear(self) -> None:
        self._prev.clear()
        _TURN_RATE.clear()
        _REALIZED_ACCEL.clear()


_KINEMATICS = _KinematicsTracker()


class _KinematicsBacked:
    """Mixin: wire a field's state hooks to the shared kinematics tracker."""

    def on_substep(self, ctx) -> None:
        _KINEMATICS.update(ctx)

    def on_aircraft_removed(self, acid: str) -> None:
        _KINEMATICS.forget(acid)

    def on_episode_reset(self, seed: int | None = None) -> None:
        _KINEMATICS.clear()


@dataclass(frozen=True)
class TurnRateDegPerSec(_KinematicsBacked, ObsField):
    """Ownship turn rate in deg/s - the *signed* change in track over the last
    env step, wrapped to (-180, 180] (a small left turn near north reads as a
    small negative rate, never +350). It is a rate, so it is symmetric and
    orientation-invariant - not an absolute heading, so there is no 0..360 range
    (using absolute heading would break the rotation invariance).

    Published by the environment each step (it needs the previous track, which an
    ObsField can't retain); reads 0 before the first step / on spawn.

    ``low``/``high`` are a **normalization scale**, not a physical cap: with a
    non-clipping normalizer, values beyond them pass through as ``|x|>1``. The
    default fixed span covers the typical few-deg/s range and keeps the obs a
    stable "how fast am I turning" (a speed-dependent physical max would entangle
    turn rate with speed); override for a different scale.

    Metadata:
        name: turn_rate_deg_per_sec
        unit: deg/s
        quantity: heading
    """

    meta = ObsMeta("turn_rate_deg_per_sec", Unit.DEG_PER_SEC, ObsQuantity.HEADING)
    low: Annotated[float, "turn rate deg/s normalization scale (low)"] = -5.0
    high: Annotated[float, "turn rate deg/s normalization scale (high)"] = 5.0

    def get(self, idx: Any) -> Any:
        return get_turn_rate(bs.traf.id[idx])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        ids = bs.traf.id
        return np.asarray(
            [get_turn_rate(ids[int(i)]) for i in indices], dtype=np.float32
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TimeInEnvS(_TimeInEnvBacked, ObsField):
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

    def get(self, idx: Any) -> Any:
        return get_time_in_env(bs.traf.id[idx])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        ids = bs.traf.id
        return np.asarray(
            [get_time_in_env(ids[int(i)]) for i in indices], dtype=np.float32
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RealizedAccelAlongTrackMs2(_KinematicsBacked, ObsField):
    """Along-track (tangential) acceleration *realized over the last env step*,
    m/s^2 = ``(ground_speed_now - ground_speed_prev_step) / dt``.

    Unlike the instantaneous :class:`AxMs2` - a single end-of-step snapshot of
    ``bs.traf.ax`` that reads ~0 exactly when a speed change is *completing*
    (the multi-substep level-off / capture aliasing) - this is the average
    speed-axis accel the agent actually produced across its whole decision
    interval. The tangential half of the velocity-frame (Frenet) realized-accel
    pair; :class:`RealizedAccelCrossTrackMs2` is the turning half. Frame-free (a
    pure speed change, zero on a constant-speed turn) and orientation-invariant.
    Published by the environment each step (needs the previous step's velocity);
    reads 0 before the first step / on spawn.

    ``low``/``high`` are a normalization scale, not a physical cap.

    Metadata:
        name: realized_accel_along_track_ms2
        unit: m/s
        quantity: speed
    """

    meta = ObsMeta("realized_accel_along_track_ms2", Unit.M_PER_S, ObsQuantity.SPEED)
    low: Annotated[float, "accel m/s^2 normalization scale (low)"] = -3.0
    high: Annotated[float, "accel m/s^2 normalization scale (high)"] = 3.0

    def get(self, idx: Any) -> Any:
        return get_realized_accel(bs.traf.id[idx])[0]

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        ids = bs.traf.id
        return np.asarray(
            [get_realized_accel(ids[int(i)])[0] for i in indices], dtype=np.float32
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RealizedAccelCrossTrackMs2(_KinematicsBacked, ObsField):
    """Cross-track (normal / centripetal) acceleration *realized over the last
    env step*, m/s^2 = ``mean_ground_speed * turn_rate_rad_per_s``, signed by
    turn direction (right positive).

    The turning half of the velocity-frame (Frenet) realized-accel pair - the
    lateral accel the trajectory actually bent by. Decomposed in the mid-step
    velocity frame, so a constant-speed turn reads as pure cross-track and a
    straight speed change as pure along-track (turn and accel stay orthogonal,
    unlike a fixed start/current-heading decomposition). Built from the same
    wrapped track change over the step as :class:`TurnRateDegPerSec` times ground
    speed, so it is alias-free through a roll-out and carries no heading-wrap
    discontinuity. Published by the environment each step; reads 0 on spawn.

    ``low``/``high`` are a normalization scale, not a physical cap.

    Metadata:
        name: realized_accel_cross_track_ms2
        unit: m/s
        quantity: speed
    """

    meta = ObsMeta("realized_accel_cross_track_ms2", Unit.M_PER_S, ObsQuantity.SPEED)
    low: Annotated[float, "accel m/s^2 normalization scale (low)"] = -6.0
    high: Annotated[float, "accel m/s^2 normalization scale (high)"] = 6.0

    def get(self, idx: Any) -> Any:
        return get_realized_accel(bs.traf.id[idx])[1]

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        ids = bs.traf.id
        return np.asarray(
            [get_realized_accel(ids[int(i)])[1] for i in indices], dtype=np.float32
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RealizedAccelVerticalMs2(_KinematicsBacked, ObsField):
    """Vertical acceleration *realized over the last env step*, m/s^2 =
    ``(vertical_speed_now - vertical_speed_prev_step) / dt`` - the rate of change
    of climb rate.

    The vertical companion of :class:`RealizedAccelAlongTrackMs2` /
    :class:`RealizedAccelCrossTrackMs2`: an *absolute* per-aircraft signal (not
    relative), so it composes with the block's existing ``AltFt.relative_to_own``
    and ``RelVsFtMin`` rather than duplicating them. Its value is anticipating a
    *level-off* on an intruder whose vertical intent is otherwise unobservable:
    ``VsFtMin`` alone can't tell a steady climb (accel ~0, VS>0 -> keeps climbing)
    from one about to stop (accel<0, VS>0 -> leveling), whereas this rate can.
    Measured across the whole multi-substep step, so it does not alias to ~0 when
    the level-off completes at the step boundary. Published each step; 0 on spawn.

    ``low``/``high`` are a normalization scale, not a physical cap.

    Metadata:
        name: realized_accel_vertical_ms2
        unit: m/s
        quantity: vertical_speed
    """

    meta = ObsMeta(
        "realized_accel_vertical_ms2", Unit.M_PER_S, ObsQuantity.VERTICAL_SPEED
    )
    low: Annotated[float, "accel m/s^2 normalization scale (low)"] = -5.0
    high: Annotated[float, "accel m/s^2 normalization scale (high)"] = 5.0

    def get(self, idx: Any) -> Any:
        return get_realized_accel(bs.traf.id[idx])[2]

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        ids = bs.traf.id
        return np.asarray(
            [get_realized_accel(ids[int(i)])[2] for i in indices], dtype=np.float32
        )

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
class PerfVminKts(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.vmin[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.vmin[_indices_array(indices)] * _MS_TO_KTS

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVmaxKts(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.vmax[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.vmax[_indices_array(indices)] * _MS_TO_KTS

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMaxFtMin(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.vsmax[idx] * _MS_TO_FTMIN

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.vsmax[_indices_array(indices)] * _MS_TO_FTMIN

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMinFtMin(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.vsmin[idx] * _MS_TO_FTMIN

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.vsmin[_indices_array(indices)] * _MS_TO_FTMIN

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfCeilingFt(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.hmax[idx] * _M_TO_FT

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.hmax[_indices_array(indices)] * _M_TO_FT

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfMassT(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.perf.mass[idx] / 1000.0

    def get_many(self, indices: Any) -> Any:
        return bs.traf.perf.mass[_indices_array(indices)] / 1000.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TurnRadiusNm(ObsField):
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

    def get(self, idx: Any) -> Any:
        return float(self.get_many([idx])[0])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        tas = np.maximum(np.asarray(bs.traf.tas)[indices], 1e-6)
        # Bank *limit* (authority), not ap.turnphi - turnphi is a transient
        # commanded bank inside flyturn legs and zero otherwise.
        phi = np.asarray(bs.traf.ap.bankdef)[indices]
        return (tas * tas) / (g0 * np.tan(phi)) / nm

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
        from bluesky_sandbox.sim.performance.envelope import (
            _warn_type_data_mismatch,
            active_performance_model,
        )

        # Same registry the envelope uses, so MTOW and the flight envelope can
        # never come from different databases for one aircraft.
        from bluesky_sandbox.sim.performance.models import type_limits

        model = active_performance_model()
        mtow = (type_limits(key, model) or {}).get("MTOW")
        if mtow is None and model != "openap":
            _warn_type_data_mismatch("MTOW")
            mtow = (type_limits(key, "openap") or {}).get("MTOW")
        cached = float(mtow) if mtow else _MTOW_FALLBACK_KG
        _MTOW_KG_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class MtowT(ObsField):
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

    def get(self, idx: Any) -> Any:
        return _mtow_kg(bs.traf.type[idx]) / 1000.0

    def get_many(self, indices: Any) -> Any:
        types = bs.traf.type
        return np.asarray(
            [_mtow_kg(types[int(i)]) / 1000.0 for i in _indices_array(indices)],
            dtype=np.float32,
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class FlightPhaseOneHot(ObsField):
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

    def get(self, idx: Any) -> Any:
        raw_phase = bs.traf.perf.phase[idx]
        one_hot = np.zeros(len(self.phase_values), dtype=np.float32)
        for phase_idx, phase_value in enumerate(self.phase_values):
            if _phase_matches(raw_phase, phase_value):
                one_hot[phase_idx] = 1.0
                return one_hot
        if self.unknown_index is not None:
            one_hot[self.unknown_index] = 1.0
        return one_hot

    def get_many(self, indices: Any) -> Any:
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

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        return (
            np.full(size, float(self.low), dtype=np.float32),
            np.full(size, float(self.high), dtype=np.float32),
        )


# Per-process store of each aircraft's most recent *normalized* action, keyed by
# callsign. The environment writes it when actions are applied (one BlueSky sim
# per process, so this is per-env), and :class:`PrevActionNorm` reads it. Exposing
# the previous action keeps an action-rate reward penalty (``|a_t - a_{t-1}|``)
# Markovian w.r.t. the observation - otherwise that reward depends on unobserved
# history, which the value function cannot predict.
_LAST_NORM_ACTION: dict[str, np.ndarray] = {}








# --------------------------------------------------------------------------- #
# Lagged observations (frame stacking)                                         #
# --------------------------------------------------------------------------- #
# Per-process ring buffers of past raw field values, so ``field.lagged(k)`` can
# report the value from ``k`` steps ago. Same lifecycle and rationale as
# ``_LAST_NORM_ACTION`` above: one BlueSky sim per process, the env clears on
# reset and drops an aircraft on despawn.
#
# Keyed by ``(field_key, acid)`` for ownship fields and
# ``(field_key, own_acid, other_acid)`` for pair fields. ``field_key`` is the
# INNER field's repr, not its ``meta.name``: two instances of the same class with
# different kwargs (``ConflictTlosS(rpz_nm=8)`` vs ``rpz_nm=5``) share a name but
# are different signals and must not share a buffer.
#
# All lags of one inner field share a buffer, and the push is guarded by sim
# time, so ``.lagged(1)`` and ``.lagged(2)`` on the same field cost ONE inner
# evaluation per step rather than one each - this sits in the per-agent per-step
# rollout hot path.
_LAG_MAXLEN = 9
_LAG_HISTORY: dict[tuple, Any] = {}
_LAG_LAST_SIMT: dict[tuple, float] = {}






def _lag_read(buf, steps: int):
    """Value ``steps`` back, zero-order held when the history is shorter.

    Zero-order hold, never zero-fill: a brand-new aircraft has no history, and
    zero is a MEANINGFUL value for these fields (raw 0 on ``ConflictTlosS``
    means "in LoS right now"), so zero-filling would inject a maximal-threat
    signal on every aircraft that just came into view. Repeating the oldest
    value it has says "no observed change", which is the honest reading.
    """
    return buf[max(0, len(buf) - 1 - int(steps))]


@dataclass(frozen=True)
class LaggedObs(_LagHistoryBacked, ObsField):
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
        if int(self.steps) >= _LAG_MAXLEN:
            raise ValueError(
                f"lagged(steps={self.steps}) exceeds the {_LAG_MAXLEN - 1}-step "
                "history buffer; raise _LAG_MAXLEN to go deeper."
            )
        object.__setattr__(self, "_key", repr(inner))
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

    def get(self, idx: Any) -> Any:
        return self.get_many([int(idx)])[0]

    def get_many(self, indices: Any) -> Any:
        inner = self._field()
        idxs = [int(i) for i in indices]
        key0 = (self._key,)
        simt = float(bs.sim.simt)
        if _LAG_LAST_SIMT.get(key0) != simt:
            _LAG_LAST_SIMT[key0] = simt
            current = inner.get_many(idxs)
            for pos, i in enumerate(idxs):
                buf = _LAG_HISTORY.setdefault(
                    (self._key, bs.traf.id[i]), deque(maxlen=_LAG_MAXLEN)
                )
                buf.append(current[pos])
        out = []
        for i in idxs:
            buf = _LAG_HISTORY.get((self._key, bs.traf.id[i]))
            if not buf:
                # Aircraft appeared after this step's push (or the push was made
                # by a sibling lag before it existed): its own value is the only
                # history there is.
                out.append(inner.get(i))
            else:
                out.append(_lag_read(buf, self.steps))
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
        if int(self.steps) >= _LAG_MAXLEN:
            raise ValueError(
                f"lagged(steps={self.steps}) exceeds the {_LAG_MAXLEN - 1}-step "
                "history buffer; raise _LAG_MAXLEN to go deeper."
            )
        object.__setattr__(self, "_key", repr(inner))
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
        inner = self._field()
        others = [int(j) for j in np.asarray(other_indices, dtype=int).ravel()]
        if not others:
            return inner.get_pairs(own_idx, others)
        own_acid = bs.traf.id[int(own_idx)]
        key0 = (self._key, own_acid)
        simt = float(bs.sim.simt)
        if _LAG_LAST_SIMT.get(key0) != simt:
            _LAG_LAST_SIMT[key0] = simt
            current = np.asarray(inner.get_pairs(own_idx, others))
            for pos, j in enumerate(others):
                buf = _LAG_HISTORY.setdefault(
                    (self._key, own_acid, bs.traf.id[j]), deque(maxlen=_LAG_MAXLEN)
                )
                buf.append(current[pos])
        out = []
        missing = []
        for pos, j in enumerate(others):
            buf = _LAG_HISTORY.get((self._key, own_acid, bs.traf.id[j]))
            if not buf:
                missing.append(pos)
                out.append(None)
            else:
                out.append(_lag_read(buf, self.steps))
        if missing:
            # Pairs that appeared after this step's push - hold their current
            # value (see :func:`_lag_read`).
            fresh = np.asarray(inner.get_pairs(own_idx, [others[p] for p in missing]))
            for slot, pos in enumerate(missing):
                out[pos] = fresh[slot]
        return np.asarray(out)


# The environment's flat normalized action-space bounds, published so
# PrevActionNorm can size its observation bounds to the actual action space
# (which depends on the action fields' normalizers) instead of assuming a range.
# Per process - one BlueSky sim / env per process.
_ACTION_SPACE_BOUNDS: tuple[np.ndarray, np.ndarray] | None = None


def set_action_space_bounds(low: Any, high: Any) -> None:
    """Publish the flat normalized action-space bounds (for PrevActionNorm)."""
    global _ACTION_SPACE_BOUNDS
    _ACTION_SPACE_BOUNDS = (
        np.asarray(low, dtype=np.float32).ravel(),
        np.asarray(high, dtype=np.float32).ravel(),
    )


def clear_action_space_bounds() -> None:
    """Forget the published action-space bounds."""
    global _ACTION_SPACE_BOUNDS
    _ACTION_SPACE_BOUNDS = None


# Per-process store of each aircraft's turn rate (deg/s), published by the
# environment each step (from the change in track over the step, which needs
# history an ObsField can't keep). Read by TurnRateDegPerSec. Per process - one
# env per process. A rate/magnitude, so it is orientation-invariant.
_TURN_RATE: dict[str, float] = {}




def get_turn_rate(acid: str) -> float:
    """Return the stored turn rate (deg/s) for ``acid``, or 0.0 if unknown."""
    return _TURN_RATE.get(acid, 0.0)






# Per-process store of each aircraft's seconds since it entered the environment.
# Published by the environment each step from its own spawn-time bookkeeping
# (``BaseEnvironment.aircraft_spawn_time``), which an ObsField cannot reach: it
# sees only ``bs.traf``, and BlueSky keeps no per-aircraft age. Read by
# TimeInEnvS. 0.0 if unknown.
_TIME_IN_ENV: dict[str, float] = {}




def get_time_in_env(acid: str) -> float:
    """Return the stored seconds-since-spawn for ``acid``, or 0.0 if unknown."""
    return _TIME_IN_ENV.get(acid, 0.0)






# Per-process store of each aircraft's realized acceleration over the last env
# step: ``(tangential, normal, vertical)`` in m/s^2 - the first two in the
# horizontal velocity (Frenet) frame, the third the vertical-speed rate.
# Published by the environment each step from the change in its velocity across
# the whole (multi-substep) step - not an end-of-step snapshot, so it does not
# alias to ~0 when a maneuver is *completing* (level-off / roll-out). Needs the
# previous step's velocity, which an ObsField can't retain. Read by
# RealizedAccel{AlongTrack,CrossTrack,Vertical}Ms2. (0, 0, 0) if unknown.
_REALIZED_ACCEL: dict[str, tuple[float, float, float]] = {}




def get_realized_accel(acid: str) -> tuple[float, float, float]:
    """Return stored ``(tangential, normal, vertical)`` accel m/s^2 (else (0, 0, 0))."""
    return _REALIZED_ACCEL.get(acid, (0.0, 0.0, 0.0))






# Per-process store of each aircraft's broadcast communication message - a
# small learned signal emitted through a CommBroadcast action channel and read
# back by other agents through IntruderCommMessage. No physical effect on the
# aircraft; purely an information channel between agents, one step delayed
# (emitted at step t, observed by others at t+1). Keyed ``acid -> channel``.
_COMM_MESSAGE: dict[str, dict[int, float]] = {}


def record_comm_message(acid: str, channel: int, value: float) -> None:
    """Store one channel of an aircraft's broadcast message (for IntruderCommMessage)."""
    _COMM_MESSAGE.setdefault(acid, {})[int(channel)] = float(value)


def get_comm_message(acid: str, channel: int) -> float:
    """Return an aircraft's stored message channel, or 0.0 (silence) if unset."""
    return _COMM_MESSAGE.get(acid, {}).get(int(channel), 0.0)






# RNG for receiver-side communication-channel noise (IntruderCommMessage's
# ``noise_std``). Reseeded from the episode seed at reset so training rollouts
# stay reproducible.
_COMM_NOISE_RNG = np.random.default_rng()




@dataclass(frozen=True)
class PrevActionNorm(_LastActionBacked, ObsField):
    """Ownship's previous action ``a_{t-1}`` (as the policy emitted it).

    Surfaces the last policy output so an action-rate reward penalty
    ``|a_t - a_{t-1}|`` is Markovian w.r.t. the observation. Set ``dim`` to how
    many action components to expose and ``offset`` to where they start in the
    action vector.

    The stored value is the policy's action *in the task's action space*, whose
    range depends on the action fields' normalizers - ``SymmetricNormalizer`` ->
    ``[-1, 1]``, ``MinMaxNormalizer`` -> ``[0, 1]``, ``RawNormalizer`` -> the
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

    def output_size(self) -> int:
        return self.dim

    def _lookup(self, acid: str) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        stored = _LAST_NORM_ACTION.get(acid)
        if stored is not None:
            src = stored[self.offset : self.offset + self.dim]
            out[: src.shape[0]] = src
        return out

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        if self.bounds_overridden:
            return (
                np.full(size, float(self.low), dtype=np.float32),
                np.full(size, float(self.high), dtype=np.float32),
            )
        if _ACTION_SPACE_BOUNDS is not None:
            lo, hi = _ACTION_SPACE_BOUNDS
            sl = slice(self.offset, self.offset + size)
            lo_s, hi_s = lo[sl], hi[sl]
            if lo_s.shape[0] == size:
                return (lo_s.astype(np.float32), hi_s.astype(np.float32))
        # Before the env publishes action bounds: assume symmetric [-1, 1].
        return (
            np.full(size, -1.0, dtype=np.float32),
            np.full(size, 1.0, dtype=np.float32),
        )

    def get(self, idx: Any) -> Any:
        return self._lookup(bs.traf.id[idx])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        if indices.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids = bs.traf.id
        return np.stack([self._lookup(ids[int(i)]) for i in indices])


@dataclass(frozen=True)
class GsKts(_TasEnvelopeBounds, _SpeedEnvelopeKts):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.gs[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.gs[_indices_array(indices)] * _MS_TO_KTS


@dataclass(frozen=True)
class GsMs(_TasEnvelopeBounds, _SpeedEnvelopeMs):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.gs[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.gs[_indices_array(indices)]


@dataclass(frozen=True)
class ApHdgDeg(ObsField):
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

    def get(self, idx: Any) -> Any:
        return bs.traf.ap.trk[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.ap.trk[_indices_array(indices)]

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasKts(_CasEnvelopeBounds, _SpeedEnvelopeKts):
    """Autopilot selected calibrated airspeed in knots.

    Metadata:
        name: ap_cas_kts
        unit: kts
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_cas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.selspd[idx] * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        return bs.traf.selspd[_indices_array(indices)] * _MS_TO_KTS


@dataclass(frozen=True)
class ApCasMs(_CasEnvelopeBounds, _SpeedEnvelopeMs):
    """Autopilot selected calibrated airspeed in m/s.

    Metadata:
        name: ap_cas_ms
        unit: m/s
        quantity: speed
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_cas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.selspd[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.selspd[_indices_array(indices)]


@dataclass(frozen=True)
class ApAltFt(AltFt):
    """Autopilot selected altitude in feet.

    Metadata:
        name: ap_alt_ft
        unit: ft
        quantity: altitude
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_alt_ft", Unit.FT, ObsQuantity.ALTITUDE, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.selalt[idx] * _M_TO_FT

    def get_many(self, indices: Any) -> Any:
        return bs.traf.selalt[_indices_array(indices)] * _M_TO_FT


@dataclass(frozen=True)
class ApAltM(AltM):
    """Autopilot selected altitude in metres.

    Metadata:
        name: ap_alt_m
        unit: m
        quantity: altitude
        dynamic_bounds: True
    """

    meta = ObsMeta("ap_alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)

    def get(self, idx: Any) -> Any:
        return bs.traf.selalt[idx]

    def get_many(self, indices: Any) -> Any:
        return bs.traf.selalt[_indices_array(indices)]


@dataclass(frozen=True)
class ApLnavVnavOn(ObsField):
    """Whether both LNAV and VNAV are enabled."""

    meta = ObsMeta("ap_lnav_vnav_on", Unit.SWITCH, ObsQuantity.AUTOPILOT)
    low: Annotated[float, "autopilot switch off"] = 0.0
    high: Annotated[float, "autopilot switch on"] = 1.0

    def get(self, idx: Any) -> Any:
        return bool(bs.traf.swlnav[idx]) and bool(bs.traf.swvnav[idx])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return np.logical_and(bs.traf.swlnav[indices], bs.traf.swvnav[indices])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApHdgErrorDeg(ObsField):
    """Autopilot selected heading error relative to current track in degrees."""

    meta = ObsMeta("ap_hdg_error_deg", Unit.DEG, ObsQuantity.HEADING)
    low: Annotated[float, "autopilot heading error degrees"] = -180.0
    high: Annotated[float, "autopilot heading error degrees"] = 180.0

    def get(self, idx: Any) -> Any:
        return _signed_angle_delta_deg(bs.traf.ap.trk[idx], bs.traf.trk[idx])

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return (bs.traf.ap.trk[indices] - bs.traf.trk[indices] + 540.0) % 360.0 - 180.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasErrorKts(_CasEnvelopeBounds, _SpeedEnvelopeKts):
    """Autopilot selected CAS error relative to current CAS in knots."""

    meta = ObsMeta(
        "ap_cas_error_kts",
        Unit.KTS,
        ObsQuantity.SPEED,
        dynamic_bounds=True,
    )

    def get(self, idx: Any) -> Any:
        return (bs.traf.selspd[idx] - bs.traf.cas[idx]) * _MS_TO_KTS

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return (bs.traf.selspd[indices] - bs.traf.cas[indices]) * _MS_TO_KTS

    def bounds(self, idx: int) -> tuple[float, float]:
        lo, hi = self._speed_bounds_ms(idx)
        current = bs.traf.cas[idx]
        span = max(
            max(abs(current - lo), abs(hi - current)) * _MS_TO_KTS,
            _MIN_DYNAMIC_SPAN,
        )
        return self._dynamic_or_configured_bounds(lambda: (-span, span))


@dataclass(frozen=True)
class ApAltErrorFt(_AltitudeEnvelopeBounds, ObsField):
    """Autopilot selected altitude error relative to current altitude in feet."""

    meta = ObsMeta(
        "ap_alt_error_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot altitude error feet lower bound; None = runtime altitude envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot altitude error feet upper bound; None = runtime altitude envelope",
    ] = None

    def get(self, idx: Any) -> Any:
        return (bs.traf.selalt[idx] - bs.traf.alt[idx]) * _M_TO_FT

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return (bs.traf.selalt[indices] - bs.traf.alt[indices]) * _M_TO_FT

    def bounds(self, idx: int) -> tuple[float, float]:
        current_ft = bs.traf.alt[idx] * _M_TO_FT
        ceiling_ft = self._altitude_ceiling_ft(idx)
        span = max(
            max(abs(current_ft), abs(ceiling_ft - current_ft)),
            _MIN_DYNAMIC_SPAN,
        )
        return self._dynamic_or_configured_bounds(lambda: (-span, span))


@dataclass(frozen=True)
class ApAltErrorM(_AltitudeEnvelopeBounds, ObsField):
    """Autopilot selected altitude error relative to current altitude in metres."""

    meta = ObsMeta(
        "ap_alt_error_m",
        Unit.M,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot altitude error metres lower bound; None = runtime altitude envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot altitude error metres upper bound; None = runtime altitude envelope",
    ] = None

    def get(self, idx: Any) -> Any:
        return bs.traf.selalt[idx] - bs.traf.alt[idx]

    def get_many(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return bs.traf.selalt[indices] - bs.traf.alt[indices]

    def bounds(self, idx: int) -> tuple[float, float]:
        current_m = bs.traf.alt[idx]
        ceiling_m = self._altitude_ceiling_m(idx)
        span = max(
            max(abs(current_m), abs(ceiling_m - current_m)),
            _MIN_DYNAMIC_SPAN,
        )
        return self._dynamic_or_configured_bounds(lambda: (-span, span))


@dataclass(frozen=True)
class Difference(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        left, right = self._fields()
        return left.get(other_idx) - right.get(own_idx)

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        left, right = self._fields()
        return np.asarray(left.get_many(other_indices)) - right.get(own_idx)

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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        left, right = self._fields()
        return _signed_angle_delta_deg(
            float(left.get(other_idx)),
            float(right.get(own_idx)),
        )

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        left, right = self._fields()
        return (
            np.asarray(left.get_many(other_indices), dtype=np.float32)
            - float(right.get(own_idx))
            + 540.0
        ) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return self._configured_bounds()
        return -180.0, 180.0


@dataclass(frozen=True)
class DistToOwnNm(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            bs.traf.lat[other_idx],
            bs.traf.lon[other_idx],
        )[1]

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        _qdr, dist = _pair_qdr_dist(own_idx, other_indices)
        return dist

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TcpaS(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        # BlueSky ConflictDetection caches ``tcpa`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon), so the sentinel tracks config.
        other_indices = _indices_array(other_indices)
        out = np.full(other_indices.shape, self.bounds(own_idx)[1], dtype=np.float64)
        cd = getattr(bs.traf, "cd", None)
        if cd is None or len(cd.confpairs) == 0:
            return out
        own_id = bs.traf.id[own_idx]
        rows = {bs.traf.id[int(j)]: r for r, j in enumerate(other_indices)}
        tcpa = np.asarray(cd.tcpa, dtype=np.float64)
        for k, (left, right) in enumerate(cd.confpairs):
            if left == own_id and k < tcpa.size and right in rows:
                out[rows[right]] = tcpa[k]
        return out

    def bounds(self, own_idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            look = _cd_lookahead_s()
            return -look, look

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class TlosS(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        # BlueSky ConflictDetection caches ``tLOS`` (one entry per ``confpairs``
        # row) each sim step - read it directly. Non-conflict intruders take the
        # high bound (the CD lookahead horizon).
        other_indices = _indices_array(other_indices)
        out = np.full(other_indices.shape, self.bounds(own_idx)[1], dtype=np.float64)
        cd = getattr(bs.traf, "cd", None)
        if cd is None or len(cd.confpairs) == 0:
            return out
        own_id = bs.traf.id[own_idx]
        rows = {bs.traf.id[int(j)]: r for r, j in enumerate(other_indices)}
        tlos = np.asarray(cd.tLOS, dtype=np.float64)
        for k, (left, right) in enumerate(cd.confpairs):
            if left == own_id and k < tlos.size and right in rows:
                out[rows[right]] = tlos[k]
        return out

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class ClosingRateKts(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own_idx,
            other_indices,
        )
        dist_m = np.maximum(np.hypot(rel_east_m, rel_north_m), 1e-6)
        range_rate_ms = (
            rel_east_m * rel_east_ms + rel_north_m * rel_north_ms
        ) / dist_m
        return -range_rate_ms * _MS_TO_KTS

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BearingRateDegPerSec(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        rel_east_m, rel_north_m, rel_east_ms, rel_north_ms = _pair_relative_motion(
            own_idx,
            other_indices,
        )
        dist2_m = np.maximum(
            rel_east_m * rel_east_m + rel_north_m * rel_north_m,
            1e-6,
        )
        rate_rad_s = (
            rel_north_m * rel_east_ms - rel_east_m * rel_north_ms
        ) / dist2_m
        return np.degrees(rate_rad_s)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def _track_frame(own_idx: int, other_indices: Any):
    """Intruder relative position (m) and velocity (m/s) in the OWNSHIP TRACK
    frame: (along, cross) where along = down the own velocity vector (ahead +),
    cross = to the right of it. Rotation-invariant (rotates with own track), so
    the encounter reads the same at any absolute heading."""
    rel_e, rel_n, ve, vn = _pair_relative_motion(own_idx, other_indices)
    trk = np.radians(float(bs.traf.trk[own_idx]))
    s, c = np.sin(trk), np.cos(trk)
    along = rel_e * s + rel_n * c           # projection on own track (compass sin/cos)
    cross = rel_e * c - rel_n * s           # 90 deg clockwise (to the right)
    v_along = ve * s + vn * c
    v_cross = ve * c - vn * s
    return along, cross, v_along, v_cross


@dataclass(frozen=True)
class RelPosAlongTrackNm(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        along, _c, _va, _vc = _track_frame(own_idx, other_indices)
        return along / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosCrossTrackNm(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        _a, cross, _va, _vc = _track_frame(own_idx, other_indices)
        return cross / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelAlongTrackKts(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        _a, _c, v_along, _vc = _track_frame(own_idx, other_indices)
        return v_along * _MS_TO_KTS

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVelCrossTrackKts(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        _a, _c, _va, v_cross = _track_frame(own_idx, other_indices)
        return v_cross * _MS_TO_KTS

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def _track_frame_at_cpa(own_idx: int, other_indices: Any):
    """Track-frame relative position AT the predicted (constant-velocity) CPA:
    (along, cross) in metres = now-position + relative-velocity * tcpa, with tcpa
    clamped to >=0 (already-passed encounters read at 'now'). Its magnitude equals
    the horizontal miss dcpa; its cross-sign says which side the intruder passes."""
    along, cross, v_along, v_cross = _track_frame(own_idx, other_indices)
    # tcpa from the SAME track-frame primitives (it is rotation-invariant), so
    # position and velocity share one projection and |result| == dcpa exactly.
    v2 = np.maximum(v_along * v_along + v_cross * v_cross, 1e-9)
    tcpa = np.maximum(-(along * v_along + cross * v_cross) / v2, 0.0)  # future CPA only
    return along + v_along * tcpa, cross + v_cross * tcpa


@dataclass(frozen=True)
class RelPosAtCpaAlongTrackNm(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        along, _cross = _track_frame_at_cpa(own_idx, other_indices)
        return along / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelPosAtCpaCrossTrackNm(PairObsField):
    """Cross-track relative position at predicted CPA, nm (right +).

    The sign is which side the intruder passes at closest approach - the key cue
    for turn direction. |along, cross| == the horizontal miss dcpa.

    Metadata:
        name: rel_pos_at_cpa_cross_track_nm
        unit: nm
        quantity: distance
        is_pair: True
    """

    meta = ObsMeta("rel_pos_at_cpa_cross_track_nm", Unit.NM, ObsQuantity.DISTANCE, is_pair=True)
    low: Annotated[float, "cross-track nm"] = -200.0
    high: Annotated[float, "cross-track nm"] = 200.0

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        _along, cross = _track_frame_at_cpa(own_idx, other_indices)
        return cross / nm

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class RelVsFtMin(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        other_indices = _indices_array(other_indices)
        return (bs.traf.vs[other_indices] - float(bs.traf.vs[own_idx])) * _MS_TO_FTMIN

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HorizontalDistAtCpaNm(PairObsField):
    """BlueSky ASAS horizontal distance at closest point of approach, in NM.

    Reads the conflict detector's cached ``dcpa`` (metres, one entry per
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        # BlueSky ConflictDetection caches ``dcpa`` (metres, one entry per
        # ``confpairs`` row) each sim step - read it directly and convert to NM.
        # Non-conflict intruders take the high bound (the PZ radius).
        other_indices = _indices_array(other_indices)
        out = np.full(other_indices.shape, self.bounds(own_idx)[1], dtype=np.float64)
        cd = getattr(bs.traf, "cd", None)
        if cd is None or len(cd.confpairs) == 0:
            return out
        own_id = bs.traf.id[own_idx]
        rows = {bs.traf.id[int(j)]: r for r, j in enumerate(other_indices)}
        dcpa = np.asarray(cd.dcpa, dtype=np.float64)
        for k, (left, right) in enumerate(cd.confpairs):
            if left == own_id and k < dcpa.size and right in rows:
                out[rows[right]] = dcpa[k] / nm
        return out

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_rpz_m() / nm))


@dataclass(frozen=True)
class VerticalSepAtCpaFt(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        other_indices = _indices_array(other_indices)
        tcpa_s = np.maximum(_pair_horizontal_tcpa_s(own_idx, other_indices), 0.0)
        rel_alt_m = bs.traf.alt[other_indices] - float(bs.traf.alt[own_idx])
        rel_vs_ms = bs.traf.vs[other_indices] - float(bs.traf.vs[own_idx])
        return np.abs(rel_alt_m + rel_vs_ms * tcpa_s) * _M_TO_FT

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, _cd_hpz_m() * _M_TO_FT)
        )


@dataclass(frozen=True)
class _ConflictGeomPairField(PairObsField):
    """Intruder field sourced from the shared per-step conflict geometry.

    Reads :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView` - the *same*
    continuous, all-pairs CPA primitives the cost and keep mask consume - so the
    feature is identical by construction to what drives the cost. Unlike the ASAS
    ``confpairs`` readers (:class:`HorizontalDistAtCpaNm`, :class:`TcpaS`,
    :class:`TlosS`) there is no detector-cache sentinel snapping and no
    detected-only gap: every intruder gets its true predicted geometry.
    """

    _geom_attr: ClassVar[str] = ""

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        return np.asarray(getattr(view, self._geom_attr), dtype=np.float64)


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

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        return windowed_min_vsep_ft(view, *self._window_zone())

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

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        return windowed_min_hsep_nm(view, *self._window_zone())

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

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        return windowed_signed_vsep_at_entry_ft(view, *self._window_zone())

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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        tinconf = predicted_tlos_s(view, *self._window_zone())
        # +inf (no conflict) -> high bound (safe); <= 0 (already in LoS) -> 0.
        return np.clip(tinconf, 0.0, self.bounds(own_idx)[1])

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (0.0, _cd_lookahead_s()))


@dataclass(frozen=True)
class InLosNow(PairObsField):
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

    Pair it with :class:`ConflictTlosS` for the approach and this for the entry;
    the two carry different information and neither substitutes for the other.
    For a *graded* conflict signal see :class:`ConflictRisk` - but note that one
    reads BlueSky's ASAS ``confpairs`` cache, so it does not necessarily agree
    with a cost computed from ``ConflictView``, whereas this field does by
    construction.

    Use :class:`~bluesky_sandbox.interface.wrappers.observations.normalizer.RawNormalizer`
    (or any normalizer with matching bounds) - the value is already 0/1.

    Metadata:
        name: in_los_now
        unit: unitless
        quantity: indicator
        is_pair: True
    """

    meta = ObsMeta("in_los_now", Unit.UNITLESS, ObsQuantity.INDICATOR, is_pair=True)
    low: Annotated[float, "indicator lower bound"] = 0.0
    high: Annotated[float, "indicator upper bound"] = 1.0

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        view = ConflictView(int(own_idx), others=others)
        # Strict ``<`` on both axes, matching BlueSky's own LoS test (and the
        # keep-mask/cost predicates built on this view): a pair sitting exactly
        # ON the zone boundary is not yet a loss of separation.
        in_los = (view.horiz_dist_now_nm < _cd_rpz_m() / nm) & (
            view.dalt_now_ft < _cd_hpz_m() * _M_TO_FT
        )
        return in_los.astype(np.float32)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def _fix_projection(
    own_idx: int,
    other_indices: Any,
    route_offset: int,
    own_eta_mode: str = "projection",
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Project observed motion onto the *ownship's* active fix (a merge signal).

    Returns ``(approach_dist_nm, intruder_eta_s, own_eta_s, vsep_at_fix_ft)``
    where each intruder's straight-line motion is projected to the closest
    approach of the ownship's own fix ``F`` (offset ``0`` = active leg, ``1`` =
    next leg): how near its current trajectory passes ``F`` (horizontal), when
    (ETA), and the signed altitude gap at the merge (intruder minus ownship,
    each projected to its own fix arrival by its vertical speed). ``None`` when
    the ownship has no usable fix there.

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
    fix = _active_route_waypoint(int(own_idx), route_offset)
    if fix is None:
        return None
    others = _indices_array(other_indices)
    fix_lat, fix_lon = float(fix[0]), float(fix[1])

    def _eta_and_cpa(lats: np.ndarray, lons: np.ndarray, trks: np.ndarray, gss: np.ndarray):
        # Position of each aircraft relative to the fix, east/north metres.
        qdr, dist_nm = qdrdist(
            np.full(np.shape(lats), fix_lat),
            np.full(np.shape(lons), fix_lon),
            lats,
            lons,
        )
        qdrrad = np.radians(np.asarray(qdr, dtype=np.float64))
        dist_m = np.asarray(dist_nm, dtype=np.float64) * nm
        r_e = dist_m * np.sin(qdrrad)
        r_n = dist_m * np.cos(qdrrad)
        trkrad = np.radians(np.asarray(trks, dtype=np.float64))
        v_e = np.asarray(gss, dtype=np.float64) * np.sin(trkrad)
        v_n = np.asarray(gss, dtype=np.float64) * np.cos(trkrad)
        v2 = np.maximum(v_e * v_e + v_n * v_n, 1e-6)
        # Future closest approach only (t* clamped >= 0): an aircraft receding
        # from the fix reads its current distance, not a past pass.
        tstar = np.maximum(-(r_e * v_e + r_n * v_n) / v2, 0.0)
        cpa_e = r_e + v_e * tstar
        cpa_n = r_n + v_n * tstar
        cpa_nm = np.sqrt(cpa_e * cpa_e + cpa_n * cpa_n) / nm
        return tstar, cpa_nm

    intr_eta, intr_cpa_nm = _eta_and_cpa(
        bs.traf.lat[others], bs.traf.lon[others],
        bs.traf.trk[others], bs.traf.gs[others],
    )
    if own_eta_mode == "route":
        # The ownship knows its own plan, so its ETA is a property of the route
        # and the speed it is flying - not of the heading it happens to hold
        # this step. See the docstring for why the two sides differ.
        _own_qdr, own_dist_nm = qdrdist(
            fix_lat, fix_lon, float(bs.traf.lat[own_idx]), float(bs.traf.lon[own_idx])
        )
        own_eta_s = float(own_dist_nm) * nm / max(float(bs.traf.gs[own_idx]), 1e-3)
    else:
        own_eta, _own_cpa = _eta_and_cpa(
            np.array([bs.traf.lat[own_idx]]), np.array([bs.traf.lon[own_idx]]),
            np.array([bs.traf.trk[own_idx]]), np.array([bs.traf.gs[own_idx]]),
        )
        own_eta_s = float(own_eta[0])
    # Altitude each aircraft is projected to hold when it reaches the fix
    # (current alt + VS x its own ETA); the merge conflict is where these
    # coincide. Signed intruder-minus-ownship, in feet.
    own_alt_at_fix = float(bs.traf.alt[own_idx]) + float(bs.traf.vs[own_idx]) * own_eta_s
    intr_alt_at_fix = bs.traf.alt[others] + bs.traf.vs[others] * intr_eta
    vsep_ft = (intr_alt_at_fix - own_alt_at_fix) * _M_TO_FT
    return intr_cpa_nm, intr_eta, own_eta_s, vsep_ft


@dataclass(frozen=True)
class IntruderFixApproachDistNm(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        proj = _fix_projection(
            own_idx, others, self.route_offset, self.own_eta_mode
        )
        if proj is None:
            return np.full(others.shape, float(self.high), dtype=np.float64)
        return proj[0]

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixArrivalDeltaS(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        proj = _fix_projection(
            own_idx, others, self.route_offset, self.own_eta_mode
        )
        if proj is None:
            return np.zeros(others.shape, dtype=np.float64)
        _cpa_nm, intr_eta, own_eta, _vsep = proj
        return intr_eta - own_eta

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class IntruderFixVerticalSepFt(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices)
        proj = _fix_projection(
            own_idx, others, self.route_offset, self.own_eta_mode
        )
        if proj is None:
            return np.zeros(others.shape, dtype=np.float64)
        return proj[3]

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
        del own_idx
        ids = bs.traf.id
        values = np.array(
            [get_comm_message(ids[int(j)], self.channel) for j in other_indices],
            dtype=np.float64,
        )
        if self.noise_std > 0.0 and values.size:
            values += _COMM_NOISE_RNG.normal(0.0, self.noise_std, size=values.shape)
            np.clip(values, self.low, self.high, out=values)
        return values

    def bounds(self, own_idx: int) -> tuple[float, float]:
        del own_idx
        return float(self.low), float(self.high)


@dataclass(frozen=True)
class BrgFromOwnDeg(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            bs.traf.lat[other_idx],
            bs.traf.lon[other_idx],
        )[0]

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        qdr, _dist = _pair_qdr_dist(own_idx, other_indices)
        return qdr

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class BrgFromOwnRelTrkDeg(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        qdr = kwikqdrdist(
            bs.traf.lat[own_idx],
            bs.traf.lon[own_idx],
            bs.traf.lat[other_idx],
            bs.traf.lon[other_idx],
        )[0]
        return _signed_angle_delta_deg(float(qdr), float(bs.traf.trk[own_idx]))

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        qdr, _dist = _pair_qdr_dist(own_idx, other_indices)
        return (
            np.asarray(qdr, dtype=np.float32) - float(bs.traf.trk[own_idx]) + 540.0
        ) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# --------------------------------------------------------------------------- #
# BlueSky CD parameters (lookahead, protected zone) and per-intruder risk      #
# --------------------------------------------------------------------------- #
def _cd_lookahead_s() -> float:
    """Conflict-detection lookahead horizon, in seconds.

    Prefers the CD's applied default ``bs.traf.cd.dtlookahead_def`` - a scalar
    set at init from ``asas_dtlookahead`` and updated by the ``DTLOOK`` command
    (how the sandbox applies ``config.lookahead_s``), so the horizon tracks the
    scenario's actual setting rather than the config-file default. Falls back to
    ``bs.settings.asas_dtlookahead`` before CD/traffic exist. Uses the scalar
    default, not the per-aircraft ``cd.dtlookahead`` array (empty pre-traffic).
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    look = getattr(cd, "dtlookahead_def", None)
    if look is None:
        look = getattr(bs.settings, "asas_dtlookahead", 300.0)
    look = float(look)
    return look if look > 0.0 else 300.0


def _cd_rpz_m() -> float:
    """Horizontal protected-zone radius, in metres (BlueSky CD ``rpz``).

    Prefers the CD's applied default ``bs.traf.cd.rpz_def`` (metres) - set from
    ``asas_pzr`` and updated by ``ZONER`` (how the sandbox applies
    ``config.pz_radius_nm``) - falling back to ``bs.settings.asas_pzr`` before CD
    exists. A detected conflict has ``dcpa < rpz``, so this is the natural cap on
    horizontal distance at CPA.
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    rpz = getattr(cd, "rpz_def", None)
    if rpz is None:
        rpz = float(getattr(bs.settings, "asas_pzr", 5.0)) * nm
    rpz = float(rpz)
    return rpz if rpz > 0.0 else 5.0 * nm


def _cd_hpz_m() -> float:
    """Vertical protected-zone height, in metres (BlueSky CD ``hpz``).

    Prefers the CD's applied default ``bs.traf.cd.hpz_def`` (metres) - set from
    ``asas_pzh`` and updated by ``ZONEDH`` (how the sandbox applies
    ``config.pz_height_ft``) - falling back to ``bs.settings.asas_pzh`` before CD
    exists. It is the minimum vertical separation (a vertical loss of separation
    is ``|dalt| < hpz``).
    """
    cd = getattr(bs.traf, "cd", None) if bs.traf is not None else None
    hpz = getattr(cd, "hpz_def", None)
    if hpz is None:
        hpz = float(getattr(bs.settings, "asas_pzh", 1000.0)) * ft
    hpz = float(hpz)
    return hpz if hpz > 0.0 else 1000.0 * ft


@dataclass(frozen=True)
class ConflictRisk(PairObsField):
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

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        # BlueSky ConflictDetection caches ``tcpa`` per ``confpairs`` row each sim
        # step. Non-conflict rows default to the lookahead horizon -> risk 0;
        # conflict rows use their CD tcpa, graded toward 1 as the conflict nears.
        lookahead = _cd_lookahead_s()
        other_indices = _indices_array(other_indices)
        tcpa = np.full(other_indices.shape, lookahead, dtype=np.float64)
        cd = getattr(bs.traf, "cd", None)
        if cd is not None and len(cd.confpairs) > 0:
            own_id = bs.traf.id[own_idx]
            rows = {bs.traf.id[int(j)]: r for r, j in enumerate(other_indices)}
            cd_tcpa = np.asarray(cd.tcpa, dtype=np.float64)
            for k, (left, right) in enumerate(cd.confpairs):
                if left == own_id and k < cd_tcpa.size and right in rows:
                    tcpa[rows[right]] = cd_tcpa[k]
        risk = 1.0 - np.maximum(tcpa, 0.0) / lookahead
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._configured_bounds()
