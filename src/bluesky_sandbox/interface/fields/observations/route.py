"""The aircraft's own route: its active fix (or a later one), read from
BlueSky's route, with no queryable needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from .._common import (
    _M_TO_FT,
    _MIN_DYNAMIC_SPAN,
    _MIN_GS_MS,
    _MS_TO_KTS,
    _BroadcastObs,
    _signed_angle_delta_deg,
    _traf_array,
)
from .._route import _active_route_waypoint, _route_along_distance_nm, _route_index
from .._state import _ArrivalTimeBacked, arrival_time
from ..base import ObsField, ObsMeta, ObsQuantity, Unit


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
        return (
            found,
            column(0, lat),
            column(1, lon),
            column(2, missing),
            column(3, missing),
        )

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

    meta = ObsMeta(
        "active_route_waypoint_bearing_deg",
        Unit.DEG,
        ObsQuantity.BEARING,
        circular=True,
    )
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

    meta = ObsMeta(
        "active_route_waypoint_track_error_deg",
        Unit.DEG,
        ObsQuantity.TRACK,
        circular=True,
    )
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

    meta = ObsMeta("active_route_waypoint_valid", Unit.UNITLESS, ObsQuantity.INDICATOR)
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
    """1.0 when the active route waypoint carries a speed constraint - a gate,
    or a target arrival time implying one - else 0.0.

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
            span = max(abs(nominal_ft), abs(ceiling_ft - nominal_ft), _MIN_DYNAMIC_SPAN)
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
    """

    meta = ObsMeta("active_route_waypoint_ete_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "ETE lower bound, s"] = 0.0
    high: Annotated[
        float,
        "ETE upper bound, s; match the task time budget to share TimeInEnvS's scale",
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
    """

    meta = ObsMeta("active_route_waypoint_vertical_ete_s", Unit.S, ObsQuantity.TIME)
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
                f"vs_mode must be 'current' or 'capability', got {self.vs_mode!r}."
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


@dataclass(frozen=True)
class _ArrivalTimeField(_ArrivalTimeBacked, _ActiveRouteWaypointField):
    """Reads the target arrival time over the fix at ``route_offset``, assigned
    at spawn when the route step asks for one (``arrival_slack_s``)."""

    def _time_to_go(self, idx: int) -> float | None:
        """Seconds until the aircraft is due over the fix, or ``None``."""
        k = _route_index(idx, self.route_offset)
        due = None if k is None else arrival_time(idx, k)
        return None if due is None else due - float(bs.sim.simt)

    def _times_to_go(self, indices: np.ndarray) -> np.ndarray:
        """:meth:`_time_to_go` for each of ``indices``, NaN for none."""
        return np.array(
            [
                np.nan if (t := self._time_to_go(int(i))) is None else t
                for i in indices
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ActiveRouteWaypointHasArrivalTime(_ArrivalTimeField):
    """1 when the fix at ``route_offset`` has a target arrival time, else 0."""

    meta = ObsMeta(
        "active_route_waypoint_has_arrival_time", Unit.SWITCH, ObsQuantity.INDICATOR
    )
    low: Annotated[float, "no arrival time"] = 0.0
    high: Annotated[float, "an arrival time"] = 1.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        return (~np.isnan(self._times_to_go(indices))).astype(np.float64)

    def _expected(self, idx: int) -> Any:
        return 0.0 if self._time_to_go(idx) is None else 1.0


@dataclass(frozen=True)
class ActiveRouteWaypointTimeToGoS(_ArrivalTimeField):
    """Seconds until the aircraft is due over the fix at ``route_offset``:
    its target arrival time minus now - negative once overdue, 0 if the fix
    has none (pair with :class:`ActiveRouteWaypointHasArrivalTime`)."""

    meta = ObsMeta("active_route_waypoint_time_to_go_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "time to go lower bound, s (overdue)"] = -600.0
    high: Annotated[
        float, "time to go upper bound, s; match the task time budget"
    ] = 3600.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        return np.nan_to_num(self._times_to_go(indices), nan=0.0)

    def _expected(self, idx: int) -> Any:
        t = self._time_to_go(idx)
        return 0.0 if t is None else t


@dataclass(frozen=True)
class ActiveRouteWaypointArrivalErrorS(_ArrivalTimeField):
    """How late the aircraft would be over the fix at ``route_offset`` flying on
    as it is: its estimated time enroute (:class:`ActiveRouteWaypointEteS`)
    minus its time to go - positive late, negative early, 0 if the fix has no
    arrival time.

    The quantity a time-based arrival is steered on: slow down, or stretch the
    path with a vector, while it is negative; it reads the same whether the
    aircraft is on its route or on a heading, since the ETE is range over
    groundspeed.
    """

    meta = ObsMeta("active_route_waypoint_arrival_error_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "arrival error lower bound, s (early)"] = -900.0
    high: Annotated[float, "arrival error upper bound, s (late)"] = 900.0

    def _values(self, indices: np.ndarray) -> np.ndarray:
        ete = ActiveRouteWaypointEteS(route_offset=self.route_offset)._values(indices)
        return np.nan_to_num(ete - self._times_to_go(indices), nan=0.0)

    def _expected(self, idx: int) -> Any:
        t = self._time_to_go(idx)
        if t is None:
            return 0.0
        ete = ActiveRouteWaypointEteS(route_offset=self.route_offset)._expected(idx)
        return float(ete) - t
