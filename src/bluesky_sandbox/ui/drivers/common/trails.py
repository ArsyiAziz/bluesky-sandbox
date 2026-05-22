"""Shared aircraft-trail bookkeeping for human-facing drivers.

The store here is deliberately dumb - world-coordinate points and enough
bookkeeping for a renderer to cache projected geometry INCREMENTALLY.  Both
renderers used to rebuild from the full point list every frame (pygame) or
every sim step (panda3d), which made trail cost grow with elapsed sim time;
see :class:`Trail` for the index contract that lets them stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import bluesky as bs

from bluesky_sandbox.sim.queryables import QueryRegion

# ``(lat_deg, lon_deg, alt_ft, color_key)``.
TrailPoint = tuple[float, float, float, str]

_FT_PER_M = 1.0 / 0.3048
_NM_PER_DEG = 60.0

# Corridor half-width for append-time decimation: a run of points is replaced
# by the straight chord across it only while EVERY dropped point stays within
# this distance of that chord.  A straight leg therefore collapses to its two
# endpoints, and a turn keeps however many vertices it takes to stay inside
# the corridor.
#
# Deliberately SMALL, and set from the PLAN VIEW'S MAXIMUM ZOOM rather than
# from its fit projection.  A loose tolerance buys less than it looks: on a
# straight leg the true deviation is ~0, so that is where the compression comes
# from either way, and the tolerance only sets how far a gentle turn may be
# straightened.  Measured on mixed traffic (mostly cruise, some maneuvering):
#
#     tolerance   compression   error at fit zoom   error at 32x zoom
#      0.005 nm       2.9x            0.03 px             2.5 px
#      0.010 nm       5.3x            0.07 px             5.0 px
#      0.020 nm       7.7x            0.13 px            10.0 px
#      0.050 nm      12.5x            0.33 px            25.0 px
#
# 0.005 nm is the fidelity-first pick: invisible at any zoom the plan view
# actually reaches, and the compression it gives up is not where the speedup
# lives - the renderer caches carry that, and they are exact.
_DECIMATE_H_NM = 0.005
_DECIMATE_V_FT = 10.0

# Longest run of points a single chord may stand in for.  Bounds the per-append
# cost - the whole pending run is re-tested against each new chord, because the
# deepest deviation is in the MIDDLE of a run and a test that only looked at
# the most recent point would never see it (that bug collapsed a gently curving
# 400-point arc to 4 points and a 2.3 nm path error).
_DECIMATE_MAX_RUN = 48

# Slack allowed past ``trail_max_points`` before the front is evicted.
# Trimming one point per step would make eviction O(n) per step - exactly the
# cost profile the cap exists to remove - so evict in batches instead and pay
# it once per ``_TRIM_BATCH`` points.
_TRIM_BATCH = 64


def _chord_deviation(
    prev: TrailPoint,
    mid: TrailPoint,
    new: TrailPoint,
) -> tuple[float, float]:
    """``(horizontal_nm, vertical_ft)`` offset of ``mid`` from ``prev``->``new``.

    Horizontal distance is the perpendicular offset from the chord in a local
    equirectangular frame anchored at ``prev``; vertical is the offset from
    the altitude linearly interpolated along that chord, so a steady climb
    reads ~0 and a level-off reads its full deviation.
    """
    lat0, lon0, alt0, _ = prev
    lat1, lon1, alt1, _ = mid
    lat2, lon2, alt2, _ = new
    cos_lat = math.cos(math.radians(lat0))
    ax = (lon1 - lon0) * cos_lat * _NM_PER_DEG
    ay = (lat1 - lat0) * _NM_PER_DEG
    bx = (lon2 - lon0) * cos_lat * _NM_PER_DEG
    by = (lat2 - lat0) * _NM_PER_DEG
    chord_sq = bx * bx + by * by
    if chord_sq < 1e-18:
        # Degenerate chord (an aircraft holding position): fall back to the
        # raw offset from ``prev`` so a stationary point is never decimated on
        # the strength of a zero-length baseline.
        return math.hypot(ax, ay), abs(alt1 - alt0)
    # Distance to the SEGMENT, not the infinite line: a track that reverses
    # can put a point beyond an endpoint, where the line distance understates
    # how far the chord actually strays from the flown path.
    t = min(1.0, max(0.0, (ax * bx + ay * by) / chord_sq))
    h_nm = math.hypot(ax - t * bx, ay - t * by)
    return h_nm, abs(alt1 - (alt0 + (alt2 - alt0) * t))


def _run_decimatable(
    anchor: TrailPoint,
    dropped: list[TrailPoint],
    outgoing: TrailPoint,
    new: TrailPoint,
) -> bool:
    """Whether one chord ``anchor``->``new`` can stand in for the whole run.

    Tests EVERY point the chord would replace, not just the most recent one.
    That distinction is the whole correctness of this function: the deepest
    deviation sits in the middle of a run, while the most recent point is by
    construction adjacent to ``new`` and so always hugs the chord's end.

    Never across a colour-key change: that boundary is the state transition
    the trail exists to show, so its vertex survives even a perfectly straight
    run.  ``dropped`` is uniform in key by induction - a point only ever
    entered it through this test.
    """
    if not (anchor[3] == outgoing[3] == new[3]):
        return False
    for mid in dropped:
        h_nm, v_ft = _chord_deviation(anchor, mid, new)
        if h_nm > _DECIMATE_H_NM or v_ft > _DECIMATE_V_FT:
            return False
    h_nm, v_ft = _chord_deviation(anchor, outgoing, new)
    return h_nm <= _DECIMATE_H_NM and v_ft <= _DECIMATE_V_FT


@dataclass
class Trail:
    """One aircraft's trail, plus the bookkeeping renderers cache against.

    Renderers key cached geometry on ABSOLUTE point indices rather than list
    positions, because both ends of :attr:`points` move:

    * the FRONT is evicted by ``trail_max_points``, so ``points[0]`` is at
      absolute index :attr:`start`, not 0;
    * the LAST point is PROVISIONAL - decimation rewrites it in place while
      an aircraft holds a straight line, so it may change without the trail
      getting any longer.

    Only ``[start, stable_end)`` is immutable and therefore safe to bake into
    a sealed cache.  :attr:`revision` bumps on every mutation, including an
    in-place rewrite of the provisional point that leaves the length alone.
    """

    points: list[TrailPoint] = field(default_factory=list)
    dropped: int = 0
    revision: int = 0
    last_simt: float | None = None
    # Points decimated away since the last retained vertex, kept so each new
    # chord can be tested against all of them rather than only the newest.
    pending: list[TrailPoint] = field(default_factory=list)

    @property
    def start(self) -> int:
        """Absolute index of ``points[0]``."""
        return self.dropped

    @property
    def end(self) -> int:
        """Absolute index one past the last point."""
        return self.dropped + len(self.points)

    @property
    def stable_end(self) -> int:
        """Absolute index one past the last IMMUTABLE point."""
        return max(self.dropped, self.end - 1)

    def at(self, index: int) -> TrailPoint:
        """The point at an absolute index."""
        return self.points[index - self.dropped]

    def span(self, lo: int, hi: int) -> list[TrailPoint]:
        """``points`` over the absolute half-open range ``[lo, hi)``."""
        return self.points[lo - self.dropped : hi - self.dropped]

    def append(self, point: TrailPoint, max_points: int | None = None) -> None:
        """Add a point, decimating and trimming as configured."""
        self.revision += 1
        points = self.points
        pending = self.pending
        if (
            len(points) >= 2
            and len(pending) < _DECIMATE_MAX_RUN
            and _run_decimatable(points[-2], pending, points[-1], point)
        ):
            # Replace the live position rather than skipping the new point, so
            # the trail stays attached to the chevron instead of lagging behind
            # by the length of the run.  The position it replaces joins the run
            # this chord has to keep standing in for.
            pending.append(points[-1])
            points[-1] = point
            return
        pending.clear()
        points.append(point)
        if max_points is not None and len(points) > max_points + _TRIM_BATCH:
            drop = len(points) - max_points
            del points[:drop]
            self.dropped += drop


class TrailMixin:
    """Mixin that tracks per-aircraft trails in world coordinates."""

    def _init_trails(self) -> None:
        self.show_trails: bool = False
        # Per-aircraft point cap.  ``None`` keeps the full episode history,
        # which append-time decimation already keeps cheap for the
        # straight-line flight that dominates these scenarios; set an int to
        # bound the heavy-maneuvering worst case as well.  Renderers handle
        # front eviction, so this is safe to change at runtime.
        self.trail_max_points: int | None = None
        self._trails: dict[str, Trail] = {}
        # QueryRegions of the current episode, resolved lazily.  Rebuilt
        # whenever trails are cleared, which every driver does on reset - the
        # same event that swaps the episode's queryable dict.
        self._trail_regions: list[QueryRegion] | None = None

    def toggle_trails(self) -> None:
        """Flip trail rendering and discard cached points when turning it off."""
        self.show_trails = not self.show_trails
        if not self.show_trails:
            self._clear_trails()

    def _advance_trails(self) -> None:
        """Append current aircraft positions once per sim-time tick."""
        if not self.show_trails:
            return
        simt = float(bs.sim.simt)
        trails = self._trails
        max_points = self.trail_max_points
        active: set[str] = set()
        for i in range(bs.traf.ntraf):
            acid = bs.traf.id[i]
            active.add(acid)
            trail = trails.get(acid)
            if trail is None:
                trail = trails[acid] = Trail()
            elif trail.last_simt is not None and simt <= trail.last_simt:
                continue
            lat = bs.traf.lat[i]
            lon = bs.traf.lon[i]
            alt_ft = bs.traf.alt[i] * _FT_PER_M
            key = self._resolve_trail_color(acid, lat, lon, alt_ft)
            trail.append((lat, lon, alt_ft, key), max_points)
            trail.last_simt = simt

        for acid in trails.keys() - active:
            trails.pop(acid, None)

    def _resolve_trail_color(
        self,
        acid: str,
        lat_deg: float,
        lon_deg: float,
        alt_ft: float,
    ) -> str:
        """Return a colour key describing this trail point's state."""
        state = self._aircraft_state(acid)
        if state in ("los", "conflict", "violation"):
            return state

        regions = self._trail_regions
        if regions is None:
            env = getattr(self, "_env", None)
            regions = self._trail_regions = (
                [
                    qable
                    for qable in env.episode_queryables.values()
                    if isinstance(qable, QueryRegion)
                ]
                if env is not None
                else []
            )
        for qable in regions:
            if qable.bounds.contains(lat_deg, lon_deg, alt_ft):
                return qable.color
        return "normal"

    def _clear_trails(self) -> None:
        """Wipe all stored trails."""
        self._trails.clear()
        self._trail_regions = None
