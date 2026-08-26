"""HorizontalView - top-down lat/lon plan view.

Lays out the airspace, query regions, waypoints, sequencing legs, and
live aircraft on a north-up scaled lat/lon canvas.  Owns its own
projection (computed in ``on_reset`` from the airspace / spawn bounding
box) so it can run independently of any other view.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bluesky as bs
import numpy as np
import pygame
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.interface.task import WaypointReadoutKey
from bluesky_sandbox.sim.geometry.conflict import ConflictView, predicted_tlos_s
from bluesky_sandbox.ui.drivers.common import (
    CursorHint,
    CursorHintName,
    ZoomPanViewport,
)
from bluesky_sandbox.ui.drivers.pygame import colors as C
from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView

if TYPE_CHECKING:
    from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline
    from bluesky_sandbox.ui.drivers.pygame.driver import PygameSimDriver


class HorizontalView(PygameView):
    """Top-down north-up plan view: lat/lon projection of the airspace."""

    default_height_fraction = 0.7
    supports_viewport_pan_zoom = True

    _AC_LENGTH_PX = 14  # tip-to-tail length of the chevron marker
    _AC_WING_FRAC = 0.8  # tail half-width as a fraction of half-length
    _AC_NOTCH_FRAC = 0.45  # rear-notch depth as a fraction of half-length
    # (1.0 = flat back, 0 = notch reaches the centre)
    _VECTOR_WIDTH = 2
    _POLY_WIDTH = 3
    _POINT_RADIUS_PX = 6  # half-diagonal of the diamond marker
    _ROUTE_POINT_RADIUS_PX = 5
    _PROTECTION_WIDTH = 1
    _HOVER_RADIUS_PX = 18  # cursor proximity radius for aircraft hover
    _CONFLICT_PAIR_WIDTH = 1
    _LOS_PAIR_WIDTH = 2

    # Velocity vector projects the aircraft this many simulated seconds
    # ahead - line direction encodes heading + vertical rate, length
    # encodes ground speed.
    _LOOKAHEAD_S = 60.0
    _M_PER_DEG_LAT = 111_000.0
    _M_PER_NM = 1852.0

    # Slice indicator (line + arrow + centre dot) drawn by _draw_axis_indicator.
    _SLICE_LINE_WIDTH = 2
    _SLICE_TICK_HALF = 6  # tick length on either side of each end
    _SLICE_ARROW_LEN = 26  # shaft length from centre to tip
    _SLICE_ARROW_HEAD = 9  # arrow head length (tip back along shaft)
    _SLICE_ARROW_WING = 5  # arrow head half-width (perpendicular to shaft)
    _SLICE_CENTRE_DOT_PX = 6
    _SLICE_IDLE_ALPHA = 90  # 255 while dragging, this when idle

    def __init__(self, margin: float = 1.1) -> None:
        super().__init__()
        self.margin = margin

        # Projection state - set by `on_reset` from the env's bounds.
        self._center_lat: float = 0.0
        self._center_lon: float = 0.0
        self._lat_per_px: float = 1.0
        self._lon_per_px: float = 1.0

        self._viewport = ZoomPanViewport(min_zoom=0.7, max_zoom=32.0)

        # Overlays are kept in world coordinates and projected every
        # frame so viewport zoom/pan can move static render primitives
        # without replaying env renderables.
        self._polygon_overlays: list[Polygon] = []
        self._point_overlays: list[Point] = []
        self._polyline_overlays: list[Polyline] = []

        # Per-frame collision-avoidance for region/point labels.
        self._label_hits: list[tuple[pygame.Rect, dict]] = []

        # Where the slice indicator is anchored on the plan (None ->
        # airspace bbox center).  User can drag the centre dot to
        # override; cleared each on_reset.
        self._slice_center_override: tuple[float, float] | None = None

        # Cached trail geometry per aircraft, in BASE pixel space - the
        # fit projection with the pan/zoom transform NOT applied.  Both
        # transforms are affine, so the viewport is a vectorized affine
        # map over this cache at draw time; the lat/lon projection never
        # re-runs for a camera move.  Invalidated only when the fit
        # projection itself changes (``_trail_projection_signature``).
        self._trail_pixels: dict[str, _TrailPixelCache] = {}
        self._trail_projection: tuple | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_reset(self, driver: PygameSimDriver, env) -> None:
        """Recompute the projection and clear overlay caches."""
        airspace = env.episode_airspace_bounds
        if airspace is not None:
            (lat_min, lat_max), (lon_min, lon_max) = airspace.bounding_box
        else:
            spawn_bb = env.episode_spawn.resolved_bounds
            lat_min, lat_max = spawn_bb["lat_deg"]
            lon_min, lon_max = spawn_bb["lon_deg"]

        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        cos_lat = math.cos(math.radians(center_lat))

        lat_range = (lat_max - lat_min) * self.margin
        lon_range = (lon_max - lon_min) * self.margin

        plan_w = max(self.rect.width, 1)
        plan_h = max(self.rect.height, 1)
        # Pick the screen-degrees-per-pixel that makes both axes fit.
        s = max(lat_range / plan_h, lon_range * cos_lat / plan_w)
        self._lat_per_px = s
        self._lon_per_px = s / cos_lat
        self._center_lat = center_lat
        self._center_lon = center_lon
        # Pan/zoom is deliberately NOT reset here - episode boundaries fire
        # continuously while a model is running (every few hundred steps),
        # and resetting the viewport each time silently yanks the user's
        # zoom/pan out from under them, moving the tiny slice-line/dot hit
        # targets off-cursor mid-interaction. Explicit reset stays on the
        # Home/0 key (``reset_viewport``, driver._reset_viewports).

        self._polygon_overlays = []
        self._point_overlays = []
        self._polyline_overlays = []
        # Projection params just changed, every cached pixel is stale.
        # ``_draw_trails`` would catch this via its projection signature
        # anyway; dropping it here frees the arrays at the reset instead
        # of holding them until the next trail frame.
        self._trail_pixels.clear()
        self._trail_projection = None
        # Note: ``_slice_center_override`` is *not* cleared on env reset -
        # the user's drag-translated slice anchor persists across episodes.
        # Re-clamp it though, so a point that's outside a (possibly new)
        # airspace bbox gets pulled in and the dot stays visible.
        if self._slice_center_override is not None and airspace is not None:
            (la_min, la_max), (lo_min, lo_max) = airspace.bounding_box
            la, lo = self._slice_center_override
            la = max(la_min, min(la_max, la))
            lo = max(lo_min, min(lo_max, lo))
            self._slice_center_override = (la, lo)

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def project(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Project (lat, lon) -> pixel coordinates inside this view's rect."""
        cx, cy = self._viewport_center()
        x = cx + (lon_deg - self._center_lon) / self._lon_per_px
        y = cy - (lat_deg - self._center_lat) / self._lat_per_px
        return self._viewport.apply(x, y, cx, cy)

    def _viewport_center(self) -> tuple[float, float]:
        return (
            self.rect.x + self.rect.width / 2,
            self.rect.y + self.rect.height / 2,
        )

    def _effective_lat_per_px(self) -> float:
        return self._lat_per_px / max(self._viewport.zoom, 1e-9)

    def _project_many(
        self, verts: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        return [self.project(lat, lon) for lat, lon in verts]

    def _future_position(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_ft: float,
        hdg_deg: float,
        gs_ms: float,
        vs_ms: float,
    ) -> tuple[float, float, float]:
        """Project ``LOOKAHEAD_S`` seconds ahead given current state."""
        rad = math.radians(hdg_deg)
        cos_lat = math.cos(math.radians(lat_deg))
        dlat = gs_ms * math.cos(rad) * self._LOOKAHEAD_S / self._M_PER_DEG_LAT
        dlon = (
            gs_ms * math.sin(rad) * self._LOOKAHEAD_S / (self._M_PER_DEG_LAT * cos_lat)
        )
        dalt_ft = vs_ms * self._LOOKAHEAD_S / ft
        return lon_deg + dlon, lat_deg + dlat, alt_ft + dalt_ft

    # ------------------------------------------------------------------
    # Render-primitive ingestion
    # ------------------------------------------------------------------

    def add_polygon(self, driver: PygameSimDriver, polygon: Polygon) -> None:
        self._polygon_overlays.append(polygon)

    def add_point(self, driver: PygameSimDriver, point: Point) -> None:
        self._point_overlays.append(point)

    def add_polyline(self, driver: PygameSimDriver, polyline: Polyline) -> None:
        self._polyline_overlays.append(polyline)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        # Boundary labels share a placed-rects list so overlapping zones
        # (e.g. a spawn box sitting inside a query region) stack vertically
        # rather than scribbling text on top of text.  The placed rect
        # also lands in self._label_hits paired with its info dict, so a
        # later hover lookup can pop full info for any visible label.
        placed_labels: list[pygame.Rect] = []
        self._label_hits = []

        # Polylines first so polygon outlines sit cleanly on top.
        for prim in self._polyline_overlays:
            if len(prim.points) < 2:
                continue
            pts_px = self._project_many(prim.points)
            color = C.named(prim.color)
            pygame.draw.lines(canvas, color, False, pts_px, width=self._POLY_WIDTH)

        # Aircraft trails - read straight off the driver's per-aircraft
        # history dict (shared with panda3d) and project to pixels
        # here.  Painted before polygons / chevrons so the trail sits
        # *under* the live marker, never occluding it.
        if driver.show_trails:
            self._draw_trails(canvas, driver)

        for prim in self._polygon_overlays:
            verts_px = self._project_many(prim.vertices)
            color = C.named(prim.color)
            width = self._polygon_width(prim)
            pygame.draw.polygon(canvas, color, verts_px, width=width)
            rect = self._draw_polygon_label(
                canvas, driver, verts_px, prim.label, color, placed_labels
            )
            if rect is not None:
                self._label_hits.append((rect, prim.meta))

        # Point markers - drawn above polygons so a point inside a region
        # is still visible, below aircraft so an aircraft on top of a
        # point still reads as the aircraft.
        for prim in self._point_overlays:
            pos_px = self.project(prim.lat, prim.lon)
            rect = self._draw_point_marker(canvas, driver, pos_px, prim, placed_labels)
            if rect is not None:
                self._label_hits.append((rect, prim.meta))

        self._draw_route_readout(canvas, driver, placed_labels)
        self._draw_safety_pair_lines(canvas, driver)
        if getattr(driver, "show_velocity_obstacles", False):
            self._draw_velocity_obstacles(canvas, driver)
            self._draw_vo_slider(canvas, driver)

        # Live aircraft.  Conflict / LoS state comes from the env's info
        # dict (via driver._aircraft_state) so the view doesn't depend
        # on bs.traf.cd directly.
        rpz = bs.traf.cd.rpz
        for i in range(bs.traf.ntraf):
            acid = bs.traf.id[i]
            alt_ft = bs.traf.alt[i] / ft
            self._draw_aircraft_plan(
                canvas,
                driver,
                i,
                acid,
                bs.traf.lat[i],
                bs.traf.lon[i],
                alt_ft,
                bs.traf.hdg[i],
                bs.traf.gs[i],
                float(rpz[i]),
                driver._aircraft_state(acid),
                self._query_color_for_aircraft(
                    driver,
                    bs.traf.lat[i],
                    bs.traf.lon[i],
                    alt_ft,
                ),
            )

        # Compass indicator: if a VerticalView is configured, show its
        # profile axis as an arrow so the user can match what they see
        # in the profile to a direction on the plan.
        self._draw_axis_indicator(canvas, driver)
        # Wind arrow + readout (only when a wind field is active).
        self._draw_wind_indicator(canvas, driver)

    def _draw_safety_pair_lines(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        """Draw current conflict/LoS links from cached driver separation info."""
        pairs = driver._aircraft_safety_pairs()
        live_index = driver._live_index()
        for key, color, width in (
            ("conflict", C.CONF, self._CONFLICT_PAIR_WIDTH),
            ("los", C.LOS, self._LOS_PAIR_WIDTH),
        ):
            ntraf = len(bs.traf.lat)
            for acid, partner in pairs.get(key, ()):
                idx_a = live_index.get(acid)
                idx_b = live_index.get(partner)
                if idx_a is None or idx_b is None:
                    continue
                # Stale pairs can outlive their aircraft (e.g. the final frame
                # after all agents despawn) - skip out-of-range indices.
                if idx_a >= ntraf or idx_b >= ntraf:
                    continue
                start = self.project(bs.traf.lat[idx_a], bs.traf.lon[idx_a])
                end = self.project(bs.traf.lat[idx_b], bs.traf.lon[idx_b])
                pygame.draw.line(canvas, color, start, end, width=width)

    def _draw_velocity_obstacles(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        """Truncated, vertically-gated velocity obstacles matching BlueSky.

        For the tracked aircraft A, each intruder B whose vertical band overlaps
        the CD lookahead is drawn as a horizontal collision cone anchored at A
        (apex = A shifted by ``v_B``, half-angle ``asin(rpz/range)``), with the
        lookahead-truncation circle (velocities inside it reach the PZ only after
        ``dtlookahead``). A cone - and A's velocity vector - is RED iff the pair is
        an actual conflict now: ``predicted_tlos_s < T`` (the SAME 3-D ``tinconf``
        from ``StateBased.detect`` that the conflict cost mirrors); else amber.
        Vertically-separated pairs are skipped. The horizon ``T`` is the detector
        lookahead scaled by ``driver.vo_horizon_frac`` (the plan-view slider); at
        the default frac 1.0 the overlay coincides 1:1 with BlueSky's detector,
        and shorter horizons drop conflicts out as their ``tinconf`` exceeds ``T``.
        """
        acid = driver.tracked_acid() if hasattr(driver, "tracked_acid") else None
        if not acid or acid not in bs.traf.id:
            return
        i = list(bs.traf.id).index(acid)
        m_per_px = self._effective_lat_per_px() * self._M_PER_DEG_LAT
        if m_per_px <= 0:
            return

        def _scalar(arr, k, default):
            try:
                return float(arr[k])
            except (TypeError, IndexError):
                try:
                    return float(arr)
                except (TypeError, ValueError):
                    return default

        r_m = _scalar(bs.traf.cd.rpz, i, 9260.0)                     # rpz, m
        h_m = _scalar(getattr(bs.traf.cd, "hpz", 304.8), i, 304.8)   # hpz, m
        tlook = _scalar(getattr(bs.traf.cd, "dtlookahead", 300.0), i, 300.0)  # s
        # VO horizon: the plan-view slider scales the detector lookahead so the
        # cones/conflicts can be swept across shorter horizons. Velocity vectors +
        # cone span this horizon T; the truncation circle scales with it
        # (``ratio`` below). Defaults to the full detector lookahead (frac 1.0),
        # where the overlay's conflict flags match BlueSky's detector 1:1.
        frac = float(getattr(driver, "vo_horizon_frac", 1.0))
        T = tlook * frac
        lat_i, lon_i = float(bs.traf.lat[i]), float(bs.traf.lon[i])
        cos_lat = max(math.cos(math.radians(lat_i)), 1e-6)
        ax, ay = self.project(lat_i, lon_i)

        others = np.array([j for j in range(bs.traf.ntraf) if j != i], dtype=int)
        if others.size == 0:
            return
        view = ConflictView(i, others=others)
        # Per-pair status: 3-D tinconf < horizon. At frac 1.0 (T == tlook) this is
        # exactly BlueSky's detector; dragging the slider down evaluates the same
        # 3-D tinconf against the shorter horizon, so conflicts drop out as the
        # lookahead shrinks (and reappear as it grows).
        tlos = np.asarray(predicted_tlos_s(view, r_m / 1852.0, h_m / 0.3048), float)
        is_conflict = np.isfinite(tlos) & (tlos < T)
        # Vertical band overlap with [0, lookahead] (independent of horiz. velocity):
        # a horizontal cone can only ever conflict if the pair is vertically close
        # within the horizon, so skip cones for vertically-separated traffic.
        hpz_ft = h_m / 0.3048
        d0 = np.asarray(view.rel_alt_now_ft, float)                  # ft (intr - own)
        dvs = np.asarray(view.rel_vs_ft_min, float) / 60.0          # ft/s
        dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)
        t_hi = (hpz_ft - d0) / dvs
        t_lo = (-hpz_ft - d0) / dvs
        tinver = np.minimum(t_hi, t_lo)
        toutver = np.maximum(t_hi, t_lo)
        vert_relevant = (tinver < T) & (toutver > 0.0)

        def vel_offset(lat, lon, hdg_deg, gs_ms):
            rad = math.radians(hdg_deg)
            dlat = gs_ms * math.cos(rad) * T / self._M_PER_DEG_LAT
            dlon = gs_ms * math.sin(rad) * T / (self._M_PER_DEG_LAT * cos_lat)
            return lat + dlat, lon + dlon

        a_tip = self.project(
            *vel_offset(lat_i, lon_i, float(bs.traf.hdg[i]), float(bs.traf.gs[i]))
        )
        overlay = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        r_px = r_m / m_per_px
        ratio = T / max(tlook, 1e-6)
        for k in range(others.size):
            if not vert_relevant[k]:
                continue  # vertical gate - BlueSky can't flag this pair
            j = int(others[k])
            bx, by = self.project(float(bs.traf.lat[j]), float(bs.traf.lon[j]))
            dxp, dyp = bx - ax, by - ay
            dist_px = math.hypot(dxp, dyp)
            if dist_px <= r_px:  # already overlapping the PZ - no cone
                continue
            theta = math.asin(min(r_px / dist_px, 1.0))
            phi = math.atan2(dyp, dxp)  # screen A->B bearing
            apex = self.project(
                *vel_offset(lat_i, lon_i, float(bs.traf.hdg[j]), float(bs.traf.gs[j]))
            )
            length = 3.0 * dist_px
            e1 = (apex[0] + length * math.cos(phi - theta),
                  apex[1] + length * math.sin(phi - theta))
            e2 = (apex[0] + length * math.cos(phi + theta),
                  apex[1] + length * math.sin(phi + theta))
            col = C.LOS if is_conflict[k] else C.CONF
            pygame.draw.polygon(overlay, (*col, 45), [apex, e1, e2])
            pygame.draw.line(overlay, (*col, 170), apex, e1, 1)
            pygame.draw.line(overlay, (*col, 170), apex, e2, 1)
            # Lookahead truncation circle (center v_B + p/tlook, radius rpz/tlook).
            tc = (apex[0] + dxp * ratio, apex[1] + dyp * ratio)
            pygame.draw.circle(
                overlay, (*col, 170), (int(tc[0]), int(tc[1])), max(int(r_px * ratio), 1), 1
            )
        canvas.blit(overlay, (0, 0))
        vec_col = C.LOS if bool(is_conflict.any()) else C.GREEN
        pygame.draw.line(canvas, vec_col, (ax, ay), a_tip, 2)
        pygame.draw.circle(canvas, vec_col, (int(a_tip[0]), int(a_tip[1])), 3)

    # ------------------------------------------------------------------
    # VO lookahead slider - a plan-view widget (drawn only while the VO
    # overlay is on) that scales the detector horizon T via
    # ``driver.vo_horizon_frac`` so the cones/conflicts can be swept from
    # a sliver of the lookahead up to the full detector horizon.
    # ------------------------------------------------------------------

    _VO_SLIDER_W = 150       # track length, px
    _VO_SLIDER_MARGIN = 14   # inset from the panel's left / bottom edges
    _VO_SLIDER_HIT_PX = 11   # click tolerance perpendicular to the track
    _VO_HANDLE_R = 6         # handle radius, px

    def _vo_slider_track(self) -> tuple[int, int, int]:
        """``(x0, x1, y)`` pixel geometry of the VO lookahead slider track."""
        x0 = self.rect.left + self._VO_SLIDER_MARGIN
        x1 = x0 + self._VO_SLIDER_W
        y = self.rect.bottom - self._VO_SLIDER_MARGIN - 4
        return x0, x1, y

    def _vo_frac_to_x(self, frac: float, x0: int, x1: int, driver) -> float:
        lo = getattr(driver, "VO_HORIZON_FRAC_MIN", 0.05)
        t = (frac - lo) / max(1.0 - lo, 1e-6)
        return x0 + max(0.0, min(1.0, t)) * (x1 - x0)

    def _vo_x_to_frac(self, x: float, x0: int, x1: int, driver) -> float:
        lo = getattr(driver, "VO_HORIZON_FRAC_MIN", 0.05)
        t = max(0.0, min(1.0, (x - x0) / max(x1 - x0, 1e-6)))
        return lo + t * (1.0 - lo)

    def _on_vo_slider(self, pos, driver) -> bool:
        """Whether *pos* is on the (visible) VO slider track / handle."""
        if not getattr(driver, "show_velocity_obstacles", False):
            return False
        x0, x1, y = self._vo_slider_track()
        px, py = pos
        return (
            x0 - self._VO_HANDLE_R <= px <= x1 + self._VO_HANDLE_R
            and abs(py - y) <= self._VO_SLIDER_HIT_PX
        )

    def _draw_vo_slider(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        if driver.font is None:
            return
        x0, x1, y = self._vo_slider_track()
        frac = float(getattr(driver, "vo_horizon_frac", 1.0))
        # Representative detector horizon (per-aircraft, but uniform in practice)
        # for the seconds readout; falls back to 300 s like the overlay itself.
        raw = getattr(getattr(bs.traf, "cd", None), "dtlookahead", 300.0)
        try:
            tlook = float(raw[0]) if hasattr(raw, "__len__") and len(raw) else float(raw)
        except (TypeError, ValueError, IndexError):
            tlook = 300.0
        # Track: dark casing under a thin light rule.
        pygame.draw.line(canvas, C.DIVIDER, (x0, y), (x1, y), 3)
        pygame.draw.line(canvas, C.GRAY, (x0, y), (x1, y), 1)
        # Handle at the current fraction.
        hx = int(self._vo_frac_to_x(frac, x0, x1, driver))
        pygame.draw.circle(canvas, C.HIGHLIGHT, (hx, int(y)), self._VO_HANDLE_R)
        pygame.draw.circle(canvas, C.BLACK, (hx, int(y)), self._VO_HANDLE_R, 1)
        # Readout above the track.
        label = driver.font.render(
            f"VO lookahead {tlook * frac:.0f} s ({frac * 100:.0f}%)", True, C.BLACK
        )
        canvas.blit(label, (x0, y - self._VO_HANDLE_R - 3 - label.get_height()))

    def _draw_trails(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        """Multi-colour polylines of past positions per aircraft.

        Each trail point carries a colour key baked at append time on
        the driver, so a normal -> conflict -> normal sequence renders
        as three coloured runs.  Adjacent segments with the same key
        share a single ``pygame.draw.lines`` call.

        Incremental on BOTH axes that used to make this grow with
        elapsed sim time:

        * points are projected once, into base pixel space, and the
          pan/zoom transform is applied as a vectorized affine map -
          so dragging the map no longer reprojects the whole history
          from lat/lon every frame;
        * the transformed result is itself cached against the camera,
          so a frame that neither steps nor moves the camera does no
          per-point work at all;
        * the colour-run segmentation is cached alongside, so the run
          boundaries are never re-derived by walking the points.

        A frame therefore costs O(new points) when the sim stepped, one
        numpy pass when the camera moved, and one drawcall per colour
        run either way - never O(all points) in Python.
        """
        # Signature of the FIT projection only - pan/zoom is deliberately
        # absent, because the cache is stored before that transform.
        signature = (
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.rect.height,
            self._center_lat,
            self._center_lon,
            self._lat_per_px,
            self._lon_per_px,
        )
        if self._trail_projection != signature:
            self._trail_pixels.clear()
            self._trail_projection = signature

        live = driver._trails
        cache_map = self._trail_pixels
        for acid in cache_map.keys() - live.keys():
            del cache_map[acid]

        cx, cy = self._viewport_center()
        zoom = self._viewport.zoom
        off_x = cx * (1.0 - zoom) + self._viewport.pan_x
        off_y = cy * (1.0 - zoom) + self._viewport.pan_y
        clip = self.rect

        for acid, trail in live.items():
            if trail.end - trail.start < 2:
                continue
            cache = cache_map.get(acid)
            if cache is None:
                cache = cache_map[acid] = _TrailPixelCache(trail.start)
            cache.sync(
                trail,
                cx,
                cy,
                self._center_lat,
                self._center_lon,
                self._lat_per_px,
                self._lon_per_px,
            )
            total = cache.total
            if total < 2:
                continue

            points, (min_x, min_y, max_x, max_y) = cache.screen(zoom, off_x, off_y)
            # Whole-trail cull: an aircraft that has flown out of the
            # current view costs one bbox test instead of its drawcalls.
            if (
                max_x < clip.left
                or min_x > clip.right
                or max_y < clip.top
                or min_y > clip.bottom
            ):
                continue

            runs = cache.runs
            for index, (offset, key) in enumerate(runs):
                # A run owns the segments starting at its own vertices, so
                # it needs one vertex past its last segment start - which is
                # exactly the next run's first vertex.
                stop = runs[index + 1][0] + 1 if index + 1 < len(runs) else total
                if stop - offset < 2:
                    continue
                pygame.draw.lines(
                    canvas,
                    self._trail_color(key),
                    False,
                    points[offset:stop],
                    width=1,
                )

    @staticmethod
    def _trail_color(key: str) -> tuple[int, int, int]:
        """Resolve a trail-point colour key into a pygame RGB tuple.

        Mirrors the chevron's body-colour priority: ``"los"``,
        ``"conflict"``, and ``"violation"`` map to the state palette,
        ``"normal"`` falls through to :attr:`C.BLACK` (the chevron's
        default), anything else is treated as a :class:`QueryRegion`
        colour name and looked up in the shared named palette so the
        trail matches the region overlay.
        """
        if key == "los":
            return C.LOS
        if key == "conflict":
            return C.CONF
        if key == "violation":
            return C.VIOLATION
        if key == "normal":
            return C.BLACK
        return C.named(key)

    # ------------------------------------------------------------------
    # Slice-indicator helpers (anchor + endpoints)
    # ------------------------------------------------------------------

    def _slice_center(self) -> tuple[float, float]:
        """Return the (lat, lon) anchor of the slice line."""
        if self._slice_center_override is not None:
            return self._slice_center_override
        return self._center_lat, self._center_lon

    def _draw_wind_indicator(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        """Draw a wind arrow + readout when a wind field is active.

        The arrow points the way the wind blows *to* (the drift direction) and is
        oriented through the view's own projection, so it stays correct under map
        rotation. The label shows the meteorological direction (blows *from*) and
        speed. Nothing is drawn when there's no wind (``winddim == 0``).
        """
        wind = getattr(bs.traf, "wind", None)
        if driver.font is None or int(getattr(wind, "winddim", 0)) < 1:
            return
        clat, clon = self._center_lat, self._center_lon
        vn, ve = wind.getdata(
            np.array([clat]), np.array([clon]), np.array([1.0e4])
        )
        vn, ve = float(vn[0]), float(ve[0])
        spd = float(np.hypot(vn, ve))
        if spd < 0.5:
            return
        dir_from = float(np.degrees(np.arctan2(-ve, -vn)) % 360.0)
        # Screen direction of "blows to": project the centre and a small offset
        # along (vn, ve), so map rotation is respected.
        off_lat = clat + 0.01 * vn
        off_lon = clon + 0.01 * ve / max(math.cos(math.radians(clat)), 1e-6)
        p0 = self.project(clat, clon)
        p1 = self.project(off_lat, off_lon)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm

        length = 32
        ax = self.rect.right - 64
        ay = self.rect.top + 46
        tip = (ax + ux * length, ay + uy * length)
        tail = (ax - ux * length, ay - uy * length)
        color = C.named("cyan")
        pygame.draw.line(canvas, color, tail, tip, 2)
        head = math.atan2(uy, ux)
        for da in (2.618, -2.618):  # +/-150 deg for the arrowhead
            pygame.draw.line(
                canvas, color, tip,
                (tip[0] + 9 * math.cos(head + da), tip[1] + 9 * math.sin(head + da)),
                2,
            )
        label = driver.font.render(
            f"wind {dir_from:.0f}° / {spd / kts:.0f} kt", True, color
        )
        canvas.blit(label, (ax - label.get_width() // 2, ay + length + 4))

    def _draw_axis_indicator(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
    ) -> None:
        """Draw the profile's slice line directly on the plan view.

        The line is anchored at :meth:`_slice_center` and oriented at
        the vertical view's bearing.  Idle, it's translucent so it
        doesn't crowd traffic; while the user is dragging it (translate
        or rotate) it switches to fully opaque so the affordance is
        unmistakable.

        Direction cue: a filled arrow head at the "panel-right" end of
        the line, plus a short perpendicular tick at the opposite "tail"
        end so you can read which way the bearing points at a glance.
        """
        v = self._vertical_view(driver)
        if v is None or driver.font is None:
            return
        bearing_deg = float(v._axis_bearing_deg)

        # Endpoints sit at exactly the same v-extent the profile shows,
        # so the line on the plan equals the visible slice.
        near_lat, near_lon = v.v_to_latlon(v._axis_min)
        far_lat, far_lon = v.v_to_latlon(v._axis_max)
        near_xy = self.project(near_lat, near_lon)
        far_xy = self.project(far_lat, far_lon)
        # Centre dot at the live projection origin (where v = 0); this
        # is where the user grabs to translate the slice.
        ctr_lat, ctr_lon = v._axis_origin
        ctr_xy = self.project(ctr_lat, ctr_lon)
        # Re-sync the plan view's "slice anchor" so hit-testing matches.
        self._slice_center_override = (ctr_lat, ctr_lon)

        # Idle = translucent, dragging = fully opaque.
        is_dragging = (
            getattr(driver, "_view_drag", None) is not None
            and driver._view_drag[0] is self
        )
        alpha = 255 if is_dragging else self._SLICE_IDLE_ALPHA

        # Render to a per-pixel-alpha layer so the line doesn't cover
        # underlying traffic when idle.
        layer = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        ox, oy = self.rect.left, self.rect.top

        def L(p):
            return (p[0] - ox, p[1] - oy)

        line_color = (*C.HIGHLIGHT, alpha)
        line_w = self._SLICE_LINE_WIDTH

        # Main slice line - *this is the slice plane* on the plan view.
        pygame.draw.line(layer, line_color, L(near_xy), L(far_xy), width=line_w)

        # Unit vector along the slice line (from centre toward axis_max).
        dir_x = far_xy[0] - ctr_xy[0]
        dir_y = far_xy[1] - ctr_xy[1]
        mag = math.hypot(dir_x, dir_y) or 1.0
        ux, uy = dir_x / mag, dir_y / mag
        # Perpendicular: this is the *viewing* direction - the arrow
        # below points in this direction.  Sign chosen so the arrow
        # falls on the CW-perpendicular side of the slice axis (in
        # screen space, since pygame y grows downward).
        px, py = uy, -ux

        # Tick at each end of the slice line so direction vs reverse is
        # still distinguishable even without an inline arrow head.
        tick = self._SLICE_TICK_HALF
        for end in (near_xy, far_xy):
            a = (end[0] + px * tick, end[1] + py * tick)
            b = (end[0] - px * tick, end[1] - py * tick)
            pygame.draw.line(layer, line_color, L(a), L(b), width=line_w)

        # Perpendicular viewing-direction arrow at the centre - shows
        # which side of the slice plane the profile is "viewed from".
        arrow_len = self._SLICE_ARROW_LEN
        head_back = arrow_len - self._SLICE_ARROW_HEAD
        wing = self._SLICE_ARROW_WING
        arrow_tip = (ctr_xy[0] + px * arrow_len, ctr_xy[1] + py * arrow_len)
        arrow_back = (ctr_xy[0] + px * head_back, ctr_xy[1] + py * head_back)
        arrow_wing1 = (arrow_back[0] + ux * wing, arrow_back[1] + uy * wing)
        arrow_wing2 = (arrow_back[0] - ux * wing, arrow_back[1] - uy * wing)
        # Shaft from centre outward.
        pygame.draw.line(layer, line_color, L(ctr_xy), L(arrow_tip), width=line_w)
        # Filled head.
        pygame.draw.polygon(
            layer,
            line_color,
            [L(arrow_tip), L(arrow_wing1), L(arrow_wing2)],
        )

        # Centre dot (translate handle).  Drawn last so it sits on top
        # of the shaft.  Black ring for contrast.
        ring_color = (*C.BLACK, alpha)
        dot_r = self._SLICE_CENTRE_DOT_PX
        pygame.draw.circle(layer, line_color, L(ctr_xy), dot_r)
        pygame.draw.circle(layer, ring_color, L(ctr_xy), dot_r, width=1)

        canvas.blit(layer, self.rect.topleft)

        # Bearing label next to the perpendicular arrow tip - only
        # while dragging, to keep the idle plan view clean.
        if is_dragging:
            txt = driver.font.render(
                f"{int(round(bearing_deg)) % 360:03d} DEG",
                True,
                C.BLACK,
            )
            bg = pygame.Surface(
                (txt.get_width() + 6, txt.get_height() + 4),
                pygame.SRCALPHA,
            )
            bg.fill((255, 255, 255, 230))
            bg.blit(txt, (3, 2))
            canvas.blit(
                bg,
                (
                    int(arrow_tip[0]) + 6,
                    int(arrow_tip[1]) - bg.get_height() // 2,
                ),
            )

    # ------------------------------------------------------------------
    # Aircraft + label drawing (private)
    # ------------------------------------------------------------------

    def _draw_aircraft_plan(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        idx: int,
        callsign: str,
        lat_deg: float,
        lon_deg: float,
        alt_ft: float,
        hdg_deg: float,
        gs_ms: float,
        rpz_m: float,
        state,
        query_color,
    ) -> None:
        x, y = self.project(lat_deg, lon_deg)
        rad = math.radians(hdg_deg)
        dx, dy = math.sin(rad), -math.cos(rad)
        if state == "los":
            body_color = C.LOS
            pz_color = C.LOS
            pz_width = None  # sentinel: draw a translucent fill instead
        elif state == "conflict":
            body_color = C.CONF
            pz_color = C.CONF
            pz_width = self._PROTECTION_WIDTH + 1
        elif state == "violation":
            body_color = C.VIOLATION
            pz_color = C.VIOLATION
            pz_width = self._PROTECTION_WIDTH + 1
        else:
            body_color = query_color or C.BLACK
            pz_color = C.PROT_ZONE
            pz_width = self._PROTECTION_WIDTH
        if driver._aircraft_snapshot(callsign).get("background", False):
            body_color = C.dim(body_color)
            pz_color = C.dim(pz_color)

        # 1 px (in y) = lat_per_px deg lat = lat_per_px * M_PER_DEG_LAT m,
        # so radius_m / (lat_per_px * M_PER_DEG_LAT) = radius in pixels.
        radius_px = rpz_m / (self._effective_lat_per_px() * self._M_PER_DEG_LAT)
        if pz_width is None:
            # LoS: translucent fill + solid outline (same width as conflict).
            C.fill_alpha_circle(canvas, pz_color, (x, y), radius_px)
            pygame.draw.circle(
                canvas,
                pz_color,
                (x, y),
                radius_px,
                width=self._PROTECTION_WIDTH + 1,
            )
        else:
            pygame.draw.circle(canvas, pz_color, (x, y), radius_px, width=pz_width)

        future_lon, future_lat, _ = self._future_position(
            lat_deg, lon_deg, 0.0, hdg_deg, gs_ms, 0.0
        )
        xv, yv = self.project(future_lat, future_lon)
        pygame.draw.line(canvas, body_color, (x, y), (xv, yv), width=self._VECTOR_WIDTH)

        ac_half = self._AC_LENGTH_PX / 2
        # Filled plane-chevron: nose -> right wing tip -> rear notch -> left
        # wing tip.  The rear notch (sitting forward of the wing tips by
        # _AC_NOTCH_FRAC * ac_half) is what makes it read as a plane symbol
        # rather than a flat-backed arrowhead or an open V.
        # perp is 90 deg right of heading in screen space (pygame y grows
        # downward, so right-of-heading is (-dy, dx)).
        perp_x, perp_y = -dy, dx
        wing_side = ac_half * self._AC_WING_FRAC
        notch_back = ac_half * self._AC_NOTCH_FRAC
        tip = (x + dx * ac_half, y + dy * ac_half)
        wing_r = (
            x - dx * ac_half - perp_x * wing_side,
            y - dy * ac_half - perp_y * wing_side,
        )
        notch = (x - dx * notch_back, y - dy * notch_back)
        wing_l = (
            x - dx * ac_half + perp_x * wing_side,
            y - dy * ac_half + perp_y * wing_side,
        )
        pygame.draw.polygon(canvas, body_color, [tip, wing_r, notch, wing_l])

        if driver.show_callsigns and driver.font is not None:
            lines = driver.format_aircraft_marker_label_lines(idx)
            driver.blit_data_block(canvas, lines, body_color, x, y)

    def _draw_polygon_label(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        verts: list[tuple[float, float]],
        name: str,
        color: tuple[int, int, int],
        placed: list[pygame.Rect],
    ) -> pygame.Rect | None:
        if not driver.show_labels or driver.font is None or not verts or not name:
            return None
        cx = sum(v[0] for v in verts) / len(verts)
        top_y = min(v[1] for v in verts)
        bg = driver.render_text_bg(name, color)
        rect = bg.get_rect()
        rect.midbottom = (int(cx), int(top_y) - 2)
        while any(rect.colliderect(r) for r in placed):
            rect.y -= rect.height + 1
        canvas.blit(bg, rect.topleft)
        placed.append(rect)
        return rect

    def _draw_point_marker(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        pos_px: tuple[float, float],
        point: Point,
        placed: list[pygame.Rect],
    ) -> pygame.Rect | None:
        x, y = pos_px
        color = C.named(point.color)
        r = self._POINT_RADIUS_PX
        diamond = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        pygame.draw.polygon(canvas, color, diamond)
        pygame.draw.polygon(canvas, C.BLACK, diamond, width=1)

        if not driver.show_labels or driver.font is None or not point.label:
            return None
        bg = driver.render_text_bg(point.label, color)
        rect = bg.get_rect()
        rect.midbottom = (int(x), int(y) - r - 2)
        while any(rect.colliderect(other) for other in placed):
            rect.y -= rect.height + 1
        canvas.blit(bg, rect.topleft)
        placed.append(rect)
        return rect

    def _draw_route_readout(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        placed: list[pygame.Rect],
    ) -> None:
        # "Show all routes" overlays the design's *defined* routes plus every
        # live aircraft's actual route. The live routes are what reveal
        # per-aircraft sampled targets (e.g. the conflict-test geometries),
        # which the defined-route overlay collapses to a shared template point.
        # The tracked aircraft still gets labels + full readout on top.
        tracked = driver.tracked_acid()
        if getattr(driver, "show_all_routes", False):
            self._draw_defined_routes(canvas, driver)
            for acid in list(bs.traf.id):
                if acid != tracked:
                    self._draw_one_route(
                        canvas, driver, acid, placed, with_labels=False
                    )
        if tracked is not None:
            self._draw_one_route(canvas, driver, tracked, placed, with_labels=True)

    def _draw_defined_routes(
        self, canvas: pygame.Surface, driver: PygameSimDriver
    ) -> None:
        """Faint polylines through the design's defined route waypoints."""
        color = C.named("cyan")
        for pts in driver.defined_route_polylines():
            px = [self.project(lat, lon) for lat, lon, _ in pts]
            if len(px) >= 2:
                pygame.draw.lines(canvas, color, False, px, width=1)
            for x, y in px:
                pygame.draw.circle(canvas, color, (int(round(x)), int(round(y))), 2, width=1)

    def _draw_one_route(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        acid: str,
        placed: list[pygame.Rect],
        with_labels: bool,
    ) -> None:
        waypoints = driver.aircraft_route_waypoints(acid)
        if not waypoints:
            return

        idx = bs.traf.id.index(acid)
        future = [wp for wp in waypoints if wp["future"]]
        if future:
            route_pts = [self.project(bs.traf.lat[idx], bs.traf.lon[idx])]
            route_pts.extend(self.project(wp["lat"], wp["lon"]) for wp in future)
            if len(route_pts) >= 2:
                pygame.draw.lines(canvas, C.named("magenta"), False, route_pts, width=2)

        for wp in waypoints:
            x, y = self.project(wp["lat"], wp["lon"])
            color = C.HIGHLIGHT if wp["active"] else C.named("magenta")
            constraints = wp.get("constraints") or {}
            radius_nm = constraints.get(WaypointReadoutKey.RADIUS_NM)
            if radius_nm is not None and radius_nm > 0:
                radius_px = max(
                    2,
                    int(
                        round(
                            radius_nm
                            * self._M_PER_NM
                            / (self._effective_lat_per_px() * self._M_PER_DEG_LAT)
                        )
                    ),
                )
                pygame.draw.circle(
                    canvas,
                    C.named("magenta"),
                    (int(round(x)), int(round(y))),
                    radius_px,
                    width=1,
                )

            r = self._ROUTE_POINT_RADIUS_PX + (2 if wp["active"] else 0)
            pygame.draw.circle(canvas, color, (x, y), r, width=2)
            pygame.draw.line(canvas, color, (x - r, y), (x + r, y), width=1)
            pygame.draw.line(canvas, color, (x, y - r), (x, y + r), width=1)
            metadata = wp.get("metadata") or {}
            label_lines = [
                metadata.get(WaypointReadoutKey.NAME)
                or wp.get("name")
                or f"WP{wp.get('display_index', wp['index']) + 1}"
            ]
            target_alt_ft = metadata.get(
                WaypointReadoutKey.TARGET_ALT_FT, wp.get("alt_ft")
            )
            if target_alt_ft is not None:
                label_lines.append(
                    f"ALT  FL{int(round(float(target_alt_ft) / 100)):03d}"
                )
            speed_min_kts = constraints.get(WaypointReadoutKey.SPEED_MIN_KTS)
            speed_max_kts = constraints.get(WaypointReadoutKey.SPEED_MAX_KTS)
            if speed_min_kts is not None or speed_max_kts is not None:
                label_lines.extend(
                    driver.format_waypoint_speed_lines(
                        idx,
                        min_kts=speed_min_kts,
                        max_kts=speed_max_kts,
                        alt_ft=target_alt_ft,
                    )
                )
            else:
                speed_kts = metadata.get(
                    WaypointReadoutKey.TARGET_SPEED_KTS,
                    constraints.get(
                        WaypointReadoutKey.SPEED_KTS, wp.get("speed_kts")
                    ),
                )
                if speed_kts is not None:
                    label_lines.extend(
                        driver.format_waypoint_speed_lines(
                            idx,
                            target_kts=speed_kts,
                            tolerance_kts=constraints.get(
                                WaypointReadoutKey.SPEED_TOLERANCE_KTS
                            ),
                            alt_ft=target_alt_ft,
                        )
                    )
            alt_tol_ft = constraints.get(WaypointReadoutKey.ALT_TOLERANCE_FT)
            if radius_nm is not None:
                label_lines.append(f"R    {float(radius_nm):.1f} NM")
            if alt_tol_ft is not None:
                label_lines.append(f"TOL  +/-{int(round(float(alt_tol_ft)))} FT")
            label = "\n".join(label_lines)
            if not with_labels or not driver.show_labels or driver.font is None or not label:
                continue
            bg = driver.render_text_bg(str(label), color)
            rect = bg.get_rect()
            rect.midbottom = (int(x), int(y) - r - 3)
            while any(rect.colliderect(other) for other in placed):
                rect.y -= rect.height + 1
            canvas.blit(bg, rect.topleft)
            placed.append(rect)

    def _polygon_width(self, polygon: Polygon) -> int:
        name = str(polygon.meta.get("name", polygon.label))
        if name.startswith("RWY_") and name.endswith("_RUNWAY"):
            return max(self._POLY_WIDTH, 3)
        return self._POLY_WIDTH

    # ------------------------------------------------------------------
    # Hover / cross-view links
    # ------------------------------------------------------------------

    def hover(self, mouse_pos, driver: PygameSimDriver) -> dict | None:
        if not self.rect.collidepoint(mouse_pos):
            return None
        mx, my = mouse_pos
        # Label hits first - unambiguous rectangles.
        for rect, info in self._label_hits:
            if rect.collidepoint(mx, my):
                return {"kind": "label", "info": info}
        # Aircraft proximity.
        n = bs.traf.ntraf
        if n == 0:
            return None
        best_idx, best_dist_sq = None, self._HOVER_RADIUS_PX**2
        for i in range(n):
            x, y = self.project(bs.traf.lat[i], bs.traf.lon[i])
            d = (mx - x) ** 2 + (my - y) ** 2
            if d < best_dist_sq:
                best_idx, best_dist_sq = i, d
        if best_idx is None:
            return None
        return {"kind": "aircraft", "idx": best_idx}

    def aircraft_position(
        self,
        driver: PygameSimDriver,
        idx: int,
    ) -> tuple[float, float] | None:
        if not (0 <= idx < bs.traf.ntraf):
            return None
        return self.project(bs.traf.lat[idx], bs.traf.lon[idx])

    def highlight_aircraft(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        idx: int,
    ) -> None:
        pos = self.aircraft_position(driver, idx)
        if pos is None:
            return
        pygame.draw.circle(canvas, C.HIGHLIGHT, pos, 18, width=3)

    # ------------------------------------------------------------------
    # Drag the slice indicator to rotate the linked VerticalView's axis
    # directly on the plan - the line spins to follow the cursor and the
    # profile re-projects in real time.
    # ------------------------------------------------------------------

    _SLICE_HIT_PX = 8  # proximity-to-line threshold for rotate-drag
    _SLICE_DOT_PX = 9  # centre-dot click radius for translate-drag

    def _vertical_view(self, driver: PygameSimDriver):
        """Find a sibling view exposing ``_axis_spec`` / ``_bbox_cache``."""
        for v in getattr(driver, "views", []):
            if hasattr(v, "_axis_spec") and getattr(v, "_bbox_cache", None) is not None:
                return v
        return None

    def _slice_endpoints_px(
        self,
        driver: PygameSimDriver,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Pixel endpoints of the slice line drawn on this panel, or None.

        Length must match the half-length used in :meth:`_draw_axis_indicator`
        so hit-testing aligns visually with what the user sees.
        """
        v = self._vertical_view(driver)
        if v is None:
            return None
        slice_lat, _slice_lon = self._slice_center()
        math.radians(v._axis_bearing_deg)
        math.cos(math.radians(slice_lat))
        near_lat, near_lon = v.v_to_latlon(v._axis_min)
        far_lat, far_lon = v.v_to_latlon(v._axis_max)
        far = self.project(far_lat, far_lon)
        near = self.project(near_lat, near_lon)
        return near, far

    def _on_slice_line(self, pos, driver: PygameSimDriver) -> bool:
        ends = self._slice_endpoints_px(driver)
        if ends is None:
            return False
        (x1, y1), (x2, y2) = ends
        px, py = pos
        ax, ay = px - x1, py - y1
        bx, by = x2 - x1, y2 - y1
        seg_len_sq = bx * bx + by * by
        if seg_len_sq <= 0:
            return False
        t = max(0.0, min(1.0, (ax * bx + ay * by) / seg_len_sq))
        cx, cy = x1 + t * bx, y1 + t * by
        return (px - cx) ** 2 + (py - cy) ** 2 <= self._SLICE_HIT_PX**2

    def _on_slice_dot(self, pos, driver: PygameSimDriver) -> bool:
        if self._vertical_view(driver) is None:
            return False
        ctr = self.project(*self._slice_center())
        dx, dy = pos[0] - ctr[0], pos[1] - ctr[1]
        return dx * dx + dy * dy <= self._SLICE_DOT_PX**2

    def _unproject(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`project` - pixel back to (lat, lon)."""
        cx, cy = self._viewport_center()
        x, y = self._viewport.invert(x, y, cx, cy)
        lon = (
            self._center_lon
            + (x - self.rect.x - self.rect.width / 2) * self._lon_per_px
        )
        lat = (
            self._center_lat
            - (y - self.rect.y - self.rect.height / 2) * self._lat_per_px
        )
        return lat, lon

    def _clamp_to_airspace(
        self,
        driver: PygameSimDriver,
        lat: float,
        lon: float,
    ) -> tuple[float, float]:
        """Clamp (lat, lon) to the airspace bbox so it stays inside the panel."""
        if driver._env is None or driver._env.episode_airspace_bounds is None:
            return lat, lon
        (la_min, la_max), (lo_min, lo_max) = (
            driver._env.episode_airspace_bounds.bounding_box
        )
        return (max(la_min, min(la_max, lat)), max(lo_min, min(lo_max, lon)))

    def hit_test_drag(self, pos, driver: PygameSimDriver):
        # VO slider first (its own widget); a click anywhere on the track jumps
        # the handle there, then the drag tracks the cursor.
        if self._on_vo_slider(pos, driver):
            x0, x1, _ = self._vo_slider_track()
            driver.set_vo_horizon_frac(self._vo_x_to_frac(pos[0], x0, x1, driver))
            return "vo_horizon"
        # Centre dot has priority over the line - they overlap.
        if self._on_slice_dot(pos, driver):
            return "slice_translate"
        if self._on_slice_line(pos, driver):
            return "slice_rotate"
        return None

    def on_drag_motion(self, handle, pos, driver: PygameSimDriver) -> None:
        # VO slider drag is independent of the vertical view, so handle it before
        # the ``_vertical_view`` early-return below.
        if handle == "vo_horizon":
            x0, x1, _ = self._vo_slider_track()
            driver.set_vo_horizon_frac(self._vo_x_to_frac(pos[0], x0, x1, driver))
            return
        v = self._vertical_view(driver)
        if v is None:
            return
        if handle == "slice_translate":
            # Move the slice anchor visually + pan the profile by
            # shifting the live projection origin (``axis_min``/``max``
            # stay anchored to the airspace, so the airspace polygon
            # band slides on the panel - that's what makes translation
            # actually pan instead of being a no-op).
            lat, lon = self._unproject(pos[0], pos[1])
            # Clamp to the airspace bbox so the centre dot can never be
            # dragged off the panel; resize behaves the same way since
            # the airspace projection always fits inside the panel.
            lat, lon = self._clamp_to_airspace(driver, lat, lon)
            self._slice_center_override = (lat, lon)
            v.update_origin(lat, lon)
        elif handle == "slice_rotate":
            ctr = self.project(*self._slice_center())
            dx = pos[0] - ctr[0]
            dy = pos[1] - ctr[1]
            if dx * dx + dy * dy < 4:
                return
            # Pygame y grows downward, so the world's north is `-dy`.
            bearing_rad = math.atan2(dx, -dy)
            bearing_deg = math.degrees(bearing_rad) % 360.0
            # ``update_bearing`` preserves the user's translation; a
            # plain ``_resolve_axis`` would reset the origin to the
            # airspace centre, undoing the prior translate.
            v.update_bearing(bearing_deg)

    def cursor_hint(self, pos, driver: PygameSimDriver) -> CursorHintName | None:
        if self._on_vo_slider(pos, driver):
            return CursorHint.RESIZE_X
        if self._on_slice_dot(pos, driver):
            return CursorHint.MOVE
        if self._on_slice_line(pos, driver):
            return CursorHint.POINT
        return None

    def zoom_view_at(
        self,
        pos: tuple[int, int],
        factor: float,
        driver: PygameSimDriver,
    ) -> bool:
        cx, cy = self._viewport_center()
        changed = self._viewport.zoom_at(pos[0], pos[1], cx, cy, factor)
        if changed:
            self._trail_pixels.clear()
        return changed

    def pan_view_by(
        self,
        delta: tuple[int, int],
        driver: PygameSimDriver,
    ) -> bool:
        changed = self._viewport.pan_by(delta[0], delta[1])
        if changed:
            self._trail_pixels.clear()
        return changed

    def reset_viewport(self, driver: PygameSimDriver) -> bool:
        changed = self._viewport.reset()
        if changed:
            self._trail_pixels.clear()
        return changed


class _TrailPixelCache:
    """One aircraft's trail geometry in base (pre-viewport) pixel space.

    Grows alongside the driver-side :class:`~bluesky_sandbox.ui.drivers.common.trails.Trail`
    and is addressed in that trail's ABSOLUTE indices, so neither front
    eviction by the point cap nor in-place rewriting of the provisional
    last point can silently misalign it:

    * ``base[:committed]`` holds the trail's immutable prefix
      ``[start, stable_end)``, projected exactly once each;
    * row ``committed`` holds the PROVISIONAL last point, reprojected on
      every sync because decimation may have moved it.

    ``runs`` records where each colour run begins, as an offset into
    ``base``, so the renderer never re-walks the point list to find the
    run boundaries.
    """

    __slots__ = (
        "_revision",
        "_screen",
        "_screen_key",
        "base",
        "committed",
        "runs",
        "start",
        "total",
    )

    def __init__(self, start: int, capacity: int = 256) -> None:
        self.base = np.empty((capacity, 2), dtype=np.float64)
        self.committed = 0
        self.total = 0
        self.start = start
        self.runs: list[tuple[int, str]] = []
        self._revision = -1
        # Screen-space result of the last ``screen()`` call, with the camera
        # and trail state it was computed for.
        self._screen: tuple[list, tuple[float, float, float, float]] | None = None
        self._screen_key: tuple | None = None

    def screen(
        self,
        zoom: float,
        off_x: float,
        off_y: float,
    ) -> tuple[list, tuple[float, float, float, float]]:
        """``(points, bbox)`` in screen space, recomputed only when needed.

        The viewport is affine over base pixel space, so this is one numpy
        pass - but a frame where neither the sim nor the camera moved does
        not even need that, which is the common case while a model runs at
        a few steps per second and nobody is touching the mouse.
        """
        key = (zoom, off_x, off_y, self.total, self._revision)
        if self._screen_key == key and self._screen is not None:
            return self._screen
        screen = self.base[: self.total] * zoom
        screen[:, 0] += off_x
        screen[:, 1] += off_y
        low = screen.min(axis=0)
        high = screen.max(axis=0)
        result = (
            screen.tolist(),
            (float(low[0]), float(low[1]), float(high[0]), float(high[1])),
        )
        self._screen = result
        self._screen_key = key
        return result

    def _reserve(self, rows: int) -> None:
        capacity = self.base.shape[0]
        if rows <= capacity:
            return
        while capacity < rows:
            capacity *= 2
        grown = np.empty((capacity, 2), dtype=np.float64)
        grown[: self.total] = self.base[: self.total]
        self.base = grown

    def _reset(self, start: int) -> None:
        self.start = start
        self.committed = 0
        self.total = 0
        self.runs.clear()

    def sync(
        self,
        trail,
        cx: float,
        cy: float,
        center_lat: float,
        center_lon: float,
        lat_per_px: float,
        lon_per_px: float,
    ) -> None:
        """Project whatever is new, and refresh the provisional tail."""
        if self._revision == trail.revision:
            return

        # Three ways the cache can be talking about different points than the
        # trail is.  All are cheap to test and all rebuild rather than repair:
        #
        # * the point cap evicted the front, so absolute index ``start``
        #   moved - eviction is batched, so this costs one reprojection per
        #   batch instead of a memmove per point;
        # * the trail is now SHORTER than the cached prefix, or its revision
        #   ran backwards - both mean this is a different ``Trail`` object
        #   under the same callsign (trails toggled off and back on, or an
        #   episode reset), and the cached rows belong to the old one.
        if (
            trail.start > self.start
            or trail.end < self.start + self.committed
            or trail.revision < self._revision
        ):
            self._reset(trail.start)
        self._revision = trail.revision

        stable_end = trail.stable_end
        have = self.start + self.committed
        if stable_end > have:
            fresh = trail.span(have, stable_end)
            self._reserve(self.committed + len(fresh) + 1)
            base = self.base
            runs = self.runs
            row = self.committed
            for lat, lon, _alt_ft, key in fresh:
                base[row, 0] = cx + (lon - center_lon) / lon_per_px
                base[row, 1] = cy - (lat - center_lat) / lat_per_px
                if not runs or runs[-1][1] != key:
                    runs.append((row, key))
                row += 1
            self.committed = row

        # The provisional point occupies one row past the committed prefix and
        # is REWRITTEN, not appended - decimation moves it in place.  When it
        # is eventually committed it keeps this same row and key (decimation
        # only ever replaces a point with one carrying the same colour key),
        # so the run marker written here stays correct across that promotion.
        end = trail.end
        if end > self.start + self.committed:
            lat, lon, _alt_ft, key = trail.at(end - 1)
            row = self.committed
            self._reserve(row + 1)
            self.base[row, 0] = cx + (lon - center_lon) / lon_per_px
            self.base[row, 1] = cy - (lat - center_lat) / lat_per_px
            runs = self.runs
            if not runs or runs[-1][1] != key:
                runs.append((row, key))
            self.total = row + 1
        else:
            self.total = self.committed
