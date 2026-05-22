"""Waypoint queryables: an aircraft's progress against a named fix."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky.tools.aero import ft as _M_PER_FT
from bluesky.tools.aero import nm as _M_PER_NM
from bluesky.tools.geo import qdrdist

from bluesky_sandbox.interface.task import (
    QueryableTemporalStateUnavailable,
    StepTime,
    UnavailableStepTime,
)
from bluesky_sandbox.sim.performance.speeds import (
    crossover_speed_state,
    within_speed_tolerance,
)

from .base import _MIN_DYNAMIC_SCALE, _ensure_navdb_loaded, _require_bound_query_result


@dataclass(frozen=True)
class WaypointTarget:
    """Configured waypoint target metadata."""

    lat: float
    lon: float
    waypoint: str | None = None
    alt_ft: float | None = None
    speed_kts: float | None = None
    reach_radius_nm: float | None = 1.0
    alt_tolerance_ft: float | None = None
    speed_tolerance_kts: float | None = None
    # Optional Mach tolerance used *above* the CAS/Mach crossover altitude. When
    # set, the speed constraint is evaluated in Mach there (matching how the
    # aircraft is controlled/assigned at altitude) and in CAS below. When None,
    # the constraint stays CAS everywhere (backwards-compatible).
    speed_tolerance_mach: float | None = None


@dataclass(frozen=True)
class WaypointCurrent:
    """Current aircraft-relative waypoint state."""

    distance_nm: float
    bearing_deg: float
    track_error_deg: float
    alt_diff_ft: float
    speed_diff_kts: float
    within_lateral: bool
    within_altitude: bool
    within_speed: bool
    satisfied: bool
    # CAS/Mach crossover speed extras (default keeps back-compat construction):
    # ``speed_diff_mach`` is current minus target Mach, and ``speed_in_mach`` is
    # True when the aircraft is above the crossover altitude (constraint in Mach).
    speed_diff_mach: float = math.nan
    speed_in_mach: bool = False

    @property
    def distance_m(self) -> float:
        return self.distance_nm * _M_PER_NM

    @property
    def alt_diff_m(self) -> float:
        return self.alt_diff_ft * _M_PER_FT


@dataclass(frozen=True)
class WaypointRoute:
    """Relationship between a waypoint queryable and the aircraft route."""

    index: int | None = None
    active: bool = False
    reached: bool = False
    future: bool = False


@dataclass(frozen=True)
class WaypointStep:
    """Waypoint facts accumulated during the current env step."""

    satisfied: bool = False
    reached: bool = False
    min_distance_nm: float = math.inf
    min_abs_alt_diff_ft: float = math.inf


@dataclass(frozen=True)
class UnavailableWaypointStep:
    """Waypoint step placeholder for query results without temporal tracking."""

    satisfied: bool = False
    reached: bool = False

    @property
    def min_distance_nm(self) -> float:
        raise QueryableTemporalStateUnavailable(
            "Waypoint step minimum distance requires "
            "track_temporal_state=True on the waypoint."
        )

    @property
    def min_abs_alt_diff_ft(self) -> float:
        raise QueryableTemporalStateUnavailable(
            "Waypoint step minimum altitude difference requires "
            "track_temporal_state=True on the waypoint."
        )


@dataclass
class WaypointResult:
    """Structured waypoint query result for one aircraft."""

    _queryable: Any | None = field(default=None, repr=False, compare=False)
    _acidx: int | None = field(default=None, repr=False, compare=False)
    _target_cache: WaypointTarget | None = field(
        default=None, repr=False, compare=False
    )
    _current_cache: WaypointCurrent | None = field(
        default=None, repr=False, compare=False
    )
    _route_cache: WaypointRoute | None = field(
        default=None, repr=False, compare=False
    )
    step: WaypointStep | UnavailableWaypointStep = field(
        default_factory=UnavailableWaypointStep
    )
    time: StepTime | UnavailableStepTime = field(default_factory=UnavailableStepTime)

    @classmethod
    def for_aircraft(
        cls,
        queryable: Any,
        acidx: int,
        *,
        target: WaypointTarget | None = None,
        current: WaypointCurrent | None = None,
        route: WaypointRoute | None = None,
        step: WaypointStep | UnavailableWaypointStep | None = None,
        time: StepTime | UnavailableStepTime | None = None,
    ) -> WaypointResult:
        """Build a waypoint result for one aircraft."""
        return cls(
            _queryable=queryable,
            _acidx=acidx,
            _target_cache=target,
            _current_cache=current,
            _route_cache=route,
            step=step if step is not None else UnavailableWaypointStep(),
            time=time if time is not None else UnavailableStepTime(),
        )

    @property
    def target(self) -> WaypointTarget:
        """Resolved waypoint target for this aircraft."""
        if self._target_cache is None:
            queryable, _acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._target_cache = queryable.target
        return self._target_cache

    @property
    def current(self) -> WaypointCurrent:
        """Current aircraft-relative waypoint state."""
        if self._current_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._current_cache = queryable.current_state(acidx)
        return self._current_cache

    @property
    def route(self) -> WaypointRoute:
        """Current route relationship for this waypoint."""
        if self._route_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._route_cache = queryable.route_state(acidx)
        return self._route_cache

    @property
    def aircraft_altitude_ceiling_ft(self) -> float:
        """Aircraft performance ceiling in feet for this query's aircraft."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        return float(bs.traf.perf.hmax[acidx] / ft)

    @property
    def altitude_error_scale_ft(self) -> float:
        """Symmetric altitude-error scale matching active waypoint obs bounds."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        nominal_ft = (
            self.target.alt_ft
            if self.target.alt_ft is not None
            else float(bs.traf.alt[acidx] / ft)
        )
        ceiling_ft = self.aircraft_altitude_ceiling_ft
        return max(
            abs(float(nominal_ft)),
            abs(ceiling_ft - float(nominal_ft)),
            _MIN_DYNAMIC_SCALE,
        )

    @property
    def speed_error_scale_kts(self) -> float:
        """Symmetric CAS-error scale matching active waypoint obs bounds."""
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        nominal_kts = (
            self.target.speed_kts
            if self.target.speed_kts is not None
            else float(bs.traf.cas[acidx] / kts)
        )
        lo_kts = float(bs.traf.perf.vmin[acidx] / kts)
        hi_kts = float(bs.traf.perf.vmax[acidx] / kts)
        return max(
            abs(float(nominal_kts) - lo_kts),
            abs(hi_kts - float(nominal_kts)),
            _MIN_DYNAMIC_SCALE,
        )

    @property
    def speed_error_normalized(self) -> float:
        """Regime-aware speed-error magnitude in ``[0, 1]`` (0 = on target).

        Uses the CAS/Mach crossover state: CAS error / CAS scale below the
        crossover altitude, Mach error / Mach scale above it - so the shaping
        magnitude is comparable across regimes and on the same axis the aircraft
        is controlled. This is the crossover-aware analogue of feeding
        ``speed_diff_kts`` / :attr:`speed_error_scale_kts` into ``err``. Returns 0
        when the target carries no speed constraint.
        """
        _queryable, acidx = _require_bound_query_result(
            self._queryable, self._acidx, type(self).__name__
        )
        if self.target.speed_kts is None:
            return 0.0
        state = crossover_speed_state(acidx, self.target.speed_kts * kts)
        return abs(state.normalized_error)


@dataclass
class Waypoint:
    """A BlueSky waypoint or ad-hoc navigation point used as a query target.

    Lifecycle is per-airspace, not per-flight: configure once on
    episode queryables and every aircraft is evaluated against
    it each step.  ``query()`` returns a :class:`WaypointResult` carrying
    current geometry, route state, tolerance satisfaction, and step timing.
    A merge point is just one use of this primitive.

    Parameters
    ----------
    lat, lon:
        Waypoint position in degrees.  Mutually exclusive with
        :attr:`waypoint`.
    waypoint:
        Name of a real BlueSky navdb waypoint (e.g. ``"EKROS"``,
        ``"VBG14"``).  Resolved to ``(lat, lon)`` at construction time
        via the BlueSky navigation database.  Mutually exclusive with
        :attr:`lat` / :attr:`lon`.
    alt_ft:
        Optional reference altitude in feet.  Drivers that have a profile
        panel draw a marker at this altitude.  When ``None``, the
        delta's ``alt_diff_ft`` / ``alt_diff_m`` come back as ``nan``.
    speed_kts:
        Optional route speed constraint in knots.  When this waypoint is
        used in ``SpawnConfig.route`` / ``SpawnRegion.route``,
        :class:`BlueskyBaseEnvironment` passes it to BlueSky's ``ADDWPT``
        command as the waypoint speed field.
    reach_radius_nm:
        Optional lateral reach radius in nautical miles. Tasks can use this
        as metadata for rendering or task-specific goal logic; it does not
        affect :meth:`query` directly.
    alt_tolerance_ft:
        Optional altitude tolerance around :attr:`alt_ft`, in feet. Like
        :attr:`reach_radius_nm`, this is metadata for task constraints and
        rendering; :meth:`query` still returns the raw altitude difference.
    speed_tolerance_kts:
        Optional speed tolerance around :attr:`speed_kts`, in knots. This is
        metadata for task constraints and route/strip presentation.
    speed_tolerance_mach:
        Optional Mach tolerance used *above* the CAS/Mach crossover altitude. When
        set, the speed constraint (``within_speed``) is evaluated in Mach there -
        matching how aircraft are controlled and ATC-assigned at altitude - and in
        CAS below the crossover. When ``None`` the constraint stays CAS at every
        altitude (backwards-compatible).
    color:
        Display color name passed through to drivers.  Defaults to ``"cyan"``.
        tsas_region:
        Optional name of another episode queryable
        (must be a :class:`QueryRegion`) used as a TSAS-strip filter:
        only aircraft inside that region appear on this waypoint's
        sequencing strip.  Lookup is by-name so the cone is configured
        once and reused by task code and the TSAS view. ``None`` means no
        filter - every aircraft is listed.
    render_shape:
        Whether the waypoint marker is drawn on map/world views.  ``True``
        by default.  Set ``False`` to keep the waypoint out of the map while
        still allowing it to be used by query logic, routes, and TSAS.
    render_tsas:
        Whether the waypoint gets a TSAS sequencing strip.  ``None`` means
        follow ``render_shape`` for backwards compatibility.
    render_label:
        Whether the waypoint's name is drawn next to its map marker.
        ``True`` by default; set ``False`` to draw the marker without
        text.  Does not affect the TSAS strip header.
    track_temporal_state:
        Whether this waypoint is sampled every physics substep to provide
        ``during_step``, accumulated ``time``, and step-minimum values.
    """

    lat: float | None = None
    lon: float | None = None
    result_type: ClassVar[type[WaypointResult]] = WaypointResult
    waypoint: str | None = None
    alt_ft: float | None = None
    speed_kts: float | None = None
    reach_radius_nm: float | None = 1.0
    alt_tolerance_ft: float | None = None
    speed_tolerance_kts: float | None = None
    speed_tolerance_mach: float | None = None
    color: str = "cyan"
    tsas_region: str | None = None
    render_shape: bool = True
    render_tsas: bool | None = None
    render_label: bool = True
    track_temporal_state: bool = False
    @staticmethod
    def _acid(acidx: int) -> str | None:
        try:
            return str(bs.traf.id[acidx])
        except (IndexError, TypeError, ValueError):
            return None

    def __post_init__(self) -> None:
        latlon_set = self.lat is not None and self.lon is not None
        if self.waypoint is not None:
            if latlon_set:
                raise ValueError(
                    "Waypoint: pass either `waypoint=...` or `lat=`/`lon=`, not both."
                )
            _ensure_navdb_loaded()
            idx = bs.navdb.getwpidx(self.waypoint)
            if idx < 0:
                raise ValueError(
                    f"Waypoint: {self.waypoint!r} not found in BlueSky navdb."
                )
            self.lat = float(bs.navdb.wplat[idx])
            self.lon = float(bs.navdb.wplon[idx])
        elif not latlon_set:
            raise ValueError(
                "Waypoint: must specify either `waypoint=...` or both "
                "`lat=` and `lon=`."
            )
    def target_from_route(self, acidx: int, route_idx: int) -> WaypointTarget:
        """Return the concrete target stored in this aircraft's BlueSky route."""
        routes = bs.traf.ap.route
        if acidx < 0 or acidx >= len(routes):
            raise IndexError(f"aircraft index {acidx} has no BlueSky route")
        route = routes[acidx]
        try:
            lat = float(route.wplat[route_idx])
            lon = float(route.wplon[route_idx])
        except (IndexError, TypeError, ValueError) as exc:
            raise IndexError(
                f"route index {route_idx} is unavailable for aircraft index {acidx}"
            ) from exc

        try:
            raw_name = route.wpname[route_idx]
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        except (IndexError, TypeError, ValueError):
            name = None
        try:
            raw_alt = float(route.wpalt[route_idx])
            alt_ft = raw_alt / ft if raw_alt >= 0.0 else None
        except (IndexError, TypeError, ValueError):
            alt_ft = None
        try:
            raw_spd = float(route.wpspd[route_idx])
            speed_kts = raw_spd / kts if raw_spd >= 0.0 else None
        except (IndexError, TypeError, ValueError):
            speed_kts = None
        return WaypointTarget(
            lat=lat,
            lon=lon,
            waypoint=name,
            alt_ft=alt_ft,
            speed_kts=speed_kts,
            reach_radius_nm=self.reach_radius_nm,
            alt_tolerance_ft=self.alt_tolerance_ft,
            speed_tolerance_kts=self.speed_tolerance_kts,
            speed_tolerance_mach=self.speed_tolerance_mach,
        )

    def _route_index(self, acidx: int, route_idx: int | None = None) -> int | None:
        if route_idx is not None:
            return route_idx
        routes = bs.traf.ap.route
        if acidx < 0 or acidx >= len(routes):
            return None
        route = routes[acidx]
        try:
            nwp = int(route.nwp or 0)
        except (TypeError, ValueError):
            return None
        if nwp <= 0:
            return None

        target_names = {str(self.waypoint).upper()} if self.waypoint else set()
        target = self.target
        tgt_lat, tgt_lon = target.lat, target.lon
        names = route.wpname
        lats = route.wplat
        lons = route.wplon
        for wp_idx in range(nwp):
            try:
                raw_name = names[wp_idx]
                wp_name = (
                    raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                ).upper()
            except (IndexError, TypeError, ValueError):
                wp_name = ""
            if wp_name and wp_name in target_names:
                return wp_idx

            try:
                wp_lat = float(lats[wp_idx])
                wp_lon = float(lons[wp_idx])
            except (IndexError, TypeError, ValueError):
                continue
            if not (math.isfinite(wp_lat) and math.isfinite(wp_lon)):
                continue
            _qdr, dist = qdrdist(wp_lat, wp_lon, tgt_lat, tgt_lon)
            if dist < 0.05:
                return wp_idx
        return None

    def route_state(
        self,
        acidx: int,
        route_idx: int | None = None,
    ) -> WaypointRoute:
        route_idx = self._route_index(acidx, route_idx)
        if route_idx is None:
            return WaypointRoute()

        routes = bs.traf.ap.route
        route = routes[acidx]
        try:
            active_idx = -1 if route.iactwp is None else int(route.iactwp)
        except (TypeError, ValueError):
            active_idx = -1

        reached = route_idx < active_idx
        try:
            if acidx in set(bs.traf.ap.idxreached):
                just_reached = active_idx - 1 if bool(bs.traf.swlnav[acidx]) else active_idx
                reached = reached or route_idx == just_reached
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        active = route_idx == active_idx
        future = route_idx >= active_idx and not reached if active_idx >= 0 else False
        return WaypointRoute(
            index=route_idx,
            active=active,
            reached=reached,
            future=future,
        )

    def reached_during_substep(self, acidx: int, route_idx: int | None = None) -> bool:
        """Return true only when this waypoint was passed in the current substep."""
        route_idx = self._route_index(acidx, route_idx)
        if route_idx is None:
            return False
        routes = bs.traf.ap.route
        route = routes[acidx]
        try:
            active_idx = -1 if route.iactwp is None else int(route.iactwp)
        except (TypeError, ValueError):
            active_idx = -1
        try:
            if acidx not in set(bs.traf.ap.idxreached):
                return False
            just_reached = active_idx - 1 if bool(bs.traf.swlnav[acidx]) else active_idx
            return route_idx == just_reached
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @property
    def target(self) -> WaypointTarget:
        return WaypointTarget(
            lat=float(self.lat),
            lon=float(self.lon),
            waypoint=self.waypoint,
            alt_ft=self.alt_ft,
            speed_kts=self.speed_kts,
            reach_radius_nm=self.reach_radius_nm,
            alt_tolerance_ft=self.alt_tolerance_ft,
            speed_tolerance_kts=self.speed_tolerance_kts,
            speed_tolerance_mach=self.speed_tolerance_mach,
        )

    def current_state(
        self,
        acidx: int,
        target: WaypointTarget | None = None,
    ) -> WaypointCurrent:
        """Return current aircraft-relative waypoint state.

        Uses an explicit route target when provided, else the static queryable
        target.
        """
        lat_deg = float(bs.traf.lat[acidx])
        lon_deg = float(bs.traf.lon[acidx])
        alt_ft = float(bs.traf.alt[acidx] / ft)
        target = target or self.target
        qdr, dist = qdrdist(lat_deg, lon_deg, target.lat, target.lon)
        alt_diff_ft = math.nan if target.alt_ft is None else alt_ft - target.alt_ft
        track_error = (
            float(qdr) - float(bs.traf.trk[acidx]) + 540.0
        ) % 360.0 - 180.0
        speed_diff = (
            math.nan
            if target.speed_kts is None
            else float(bs.traf.cas[acidx] / _M_PER_NM * 3600.0) - target.speed_kts
        )
        # Regime-aware speed state (CAS below the crossover altitude, Mach above),
        # computed whenever the target carries a speed constraint so the readout
        # and the tolerance switch regimes for *any* sampled speed/altitude.
        speed_diff_mach = math.nan
        speed_in_mach = False
        if target.speed_kts is not None:
            crossover = crossover_speed_state(acidx, target.speed_kts * kts)
            speed_diff_mach = crossover.mach_diff
            speed_in_mach = crossover.in_mach
        within_lateral = (
            True
            if target.reach_radius_nm is None
            else float(dist) <= target.reach_radius_nm
        )
        within_altitude = (
            True
            if target.alt_ft is None or target.alt_tolerance_ft is None
            else abs(alt_diff_ft) <= target.alt_tolerance_ft
        )
        # Speed constraint, evaluated in the regime the aircraft is controlled in
        # (Mach above the crossover, CAS below). The Mach tolerance defaults to the
        # CAS tolerance converted at altitude, so a single sampled CAS tolerance
        # stays well-defined no matter which regime the sample lands in.
        within_speed = target.speed_kts is None or within_speed_tolerance(
            acidx,
            target.speed_kts * kts,
            target.speed_tolerance_kts,
            target.speed_tolerance_mach,
        )
        return WaypointCurrent(
            distance_nm=float(dist),
            bearing_deg=float(qdr),
            track_error_deg=track_error,
            alt_diff_ft=float(alt_diff_ft),
            speed_diff_kts=float(speed_diff),
            within_lateral=within_lateral,
            within_altitude=within_altitude,
            within_speed=within_speed,
            satisfied=within_lateral and within_altitude and within_speed,
            speed_diff_mach=float(speed_diff_mach),
            speed_in_mach=bool(speed_in_mach),
        )
