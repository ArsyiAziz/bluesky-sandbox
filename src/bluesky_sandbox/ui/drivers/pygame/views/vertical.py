"""VerticalView - true side-view profile (alt vs spatial axis).

A real side view, not a slot-based altitude tape: aircraft and overlays
sit at their actual position along a chosen spatial axis (longitude by
default; latitude when the airspace is more N-S than E-W).  This makes:

* **Sloped corridors visible.**  Bounds with ``alt_band_at`` produce
  tilted/curved bands automatically - each polygon vertex gets its
  altitude band evaluated at that point, so the upper and lower edges
  follow the slope without any special case here.
* **Aircraft motion legible.**  An aircraft descending toward the
  merge fix appears moving down-and-toward the apex, instead of just
  changing y in a fixed slot.
* **Spatial intuition restored.**  Constant-altitude polygons render
  as horizontal bands; the airspace ceiling/floor look like a bounding
  box; waypoints are dots placed at their lat/lon and altitude.

The "moves left/right as it flies east/west" complaint of side views
is real but at typical airspace scales the lateral travel reads
clearly as motion along the panel - no worse than any FMS-style
vertical situation display.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Literal

import bluesky as bs
import pygame
from bluesky.tools.aero import Rearth, ft
from shapely.geometry import LineString
from shapely.geometry import Polygon as _ShPolygon

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


_DEFAULT_ALT_RANGE_FT = (0.0, 45_000.0)
_BAND_FILL_ALPHA = 60  # translucent fill on alt bands
_PANEL_BG = (240, 245, 250)


class VerticalView(PygameView):
    """Side view: altitude on y, longitude (or latitude) on x."""

    default_height_fraction = 0.3
    supports_viewport_pan_zoom = True

    _MARGIN_LEFT = 60  # FL labels live here
    _MARGIN_RIGHT = 12
    _MARGIN_TOP = 8
    _MARGIN_BOTTOM = 22  # axis ticks/label live here
    _AC_RADIUS = 5
    _AC_VECTOR_WIDTH = 2
    _PROTECTION_WIDTH = 1
    _LOOKAHEAD_S = 60.0
    _FL_INTERVAL_FT = 5_000
    _POINT_RADIUS_PX = 5
    _BAND_OUTLINE = 2
    # Cross-sections used to draw the half-plane projection envelope.
    _BAND_BUCKETS = 256
    _AXIS_PAD_FRAC = 0.05  # extra room on each side of the axis range
    _ALT_PAD_FRAC = 0.10  # extra room on each side of the alt range

    def __init__(
        self,
        axis: Literal["auto", "lon", "lat"]
        | float
        | tuple[tuple[float, float], tuple[float, float]] = "auto",
    ) -> None:
        """Construct the view with a configurable spatial x-axis.

        Parameters
        ----------
        axis:
            How to slice the airspace:

            * ``"auto"`` (default) - pick ``"lon"`` or ``"lat"`` based on
              which has greater extent in nm.
            * ``"lon"`` / ``"lat"`` - force a cardinal axis.
            * ``float`` - bearing in degrees (0 = N, CW positive); the
              profile shows altitude vs distance along that bearing
              through the airspace centre.  E.g. ``axis=240`` aligns the
              profile with EHAM RWY 06's approach axis so a sloped
              corridor reads as a clean tilted band.
            * ``((lat0, lon0), (lat1, lon1))`` - explicit two-point line.
              The axis is its bearing; range covers the airspace bbox
              projected onto it.
        """
        super().__init__()
        self._axis_spec = axis
        # Resolved in `on_reset`.
        self._axis_kind: Literal["lon", "lat", "vec"] = "lon"
        self._axis_min: float = 0.0
        self._axis_max: float = 1.0
        self._axis_origin: tuple[float, float] = (0.0, 0.0)
        # Anchor for axis-range computation - fixed at airspace centre
        # by `on_reset`, never moved by translate-drag.  Origin can drift
        # away from anchor as the user drags.
        self._axis_anchor: tuple[float, float] = (0.0, 0.0)
        # (cos(bearing), sin(bearing), cos(center_lat)) - projection
        # coefficients used when ``_axis_kind == "vec"``.
        self._axis_proj: tuple[float, float, float] = (1.0, 0.0, 1.0)
        # Axis bearing in degrees (0 = N, CW positive).  Always set so the
        # horizontal view can render a matching compass arrow regardless
        # of the underlying axis kind.
        self._axis_bearing_deg: float = 90.0
        self._alt_min_view: float = 0.0
        self._alt_max_view: float = 1.0

        # Static overlays - kept in lat/lon space and clipped/projected
        # against the live slice half-plane at render time.
        # Each entry is (polygon, bounds_or_None). ``bounds_or_None`` is
        # kept when it can provide altitude interpolation.
        self._polygon_overlays: list[tuple[Polygon, object]] = []
        self._point_overlays: list[Point] = []

        # Projection cache: keyed by ``id(polygon)``, value is a list of
        # ``(axis_values, lower_alt_ft, upper_alt_ft)`` segments after
        # clipping the polygon to the current slice half-plane. Invalidated in
        # ``on_reset`` / ``update_bearing`` / ``update_origin`` - the only
        # events that change the slice geometry.
        self._envelope_cache: dict[
            int,
            list[tuple[list[float], list[float], list[float]]] | None,
        ] = {}

        # Cached airspace bbox + projection params, refreshed in
        # ``on_reset`` and reused by the live rotation drag so we don't
        # have to walk the env config on every mouse move.
        self._bbox_cache: dict | None = None

        # User's last translated origin, persisted across resets so a
        # drag-translate "sticks" through env.reset().  ``None`` means
        # "follow the airspace centre".
        self._user_origin: tuple[float, float] | None = None

        # Drag-rotate state (set by ``hit_test_drag``).
        self._drag_start_x: int = 0
        self._drag_start_bearing: float = 0.0

        self._viewport = ZoomPanViewport(min_zoom=0.7, max_zoom=32.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_reset(self, driver: PygameSimDriver, env) -> None:
        airspace = env.episode_airspace_bounds
        if airspace is not None:
            (lat_min, lat_max), (lon_min, lon_max) = airspace.bounding_box
        else:
            spawn_bb = env.episode_spawn.resolved_bounds
            lat_min, lat_max = spawn_bb["lat_deg"]
            lon_min, lon_max = spawn_bb["lon_deg"]

        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)
        cos_lat = math.cos(math.radians(center_lat))

        self._bbox_cache = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "cos_lat": cos_lat,
        }
        self._resolve_axis(spec=self._axis_spec, **self._bbox_cache)

        # Altitude axis: airspace if finite, else spawn alt, else default.
        # Polygon overlays added later via add_polygon may widen this so
        # sloped corridors reaching well above the spawn band stay visible.
        if (
            airspace is not None
            and math.isfinite(airspace.alt_min_ft)
            and math.isfinite(airspace.alt_max_ft)
        ):
            alt_min, alt_max = airspace.alt_min_ft, airspace.alt_max_ft
        else:
            spawn_alt = env.episode_spawn.resolved_bounds.get("alt_ft")
            if spawn_alt is not None and all(map(math.isfinite, spawn_alt)):
                alt_min, alt_max = spawn_alt
            else:
                alt_min, alt_max = _DEFAULT_ALT_RANGE_FT
        if alt_max <= alt_min:
            alt_max = alt_min + 1.0
        alt_pad = (alt_max - alt_min) * self._ALT_PAD_FRAC
        self._alt_min_view = alt_min - alt_pad
        self._alt_max_view = alt_max + alt_pad

        self._polygon_overlays = []
        self._point_overlays = []
        self._envelope_cache.clear()
        # Pan/zoom is deliberately NOT reset here - see the matching note in
        # HorizontalView.on_reset. Explicit reset stays on the Home/0 key.

    # ------------------------------------------------------------------
    # Axis resolution (auto / cardinal / bearing / two-point line)
    # ------------------------------------------------------------------

    def _resolve_axis(
        self,
        spec,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        center_lat: float,
        center_lon: float,
        cos_lat: float,
    ) -> None:
        """Set ``_axis_kind`` / range / projection coefficients from *spec*.

        Establishes the *anchor* at ``(center_lat, center_lon)``.  Anchor
        is the reference origin used to compute ``axis_min`` /
        ``axis_max`` from the bbox corners - *never* updated by live
        translation.  The live :attr:`_axis_origin` is initialised to
        the same point; subsequent ``update_origin`` / ``update_bearing``
        calls keep the anchor fixed so a translate genuinely pans the
        profile (and a rotate doesn't reset the user's translation).
        """
        # Resolve "auto" -> "lon" or "lat" by greater nm extent.
        if spec == "auto":
            lon_nm = (lon_max - lon_min) * cos_lat * 60.0
            lat_nm = (lat_max - lat_min) * 60.0
            spec = "lon" if lon_nm >= lat_nm else "lat"

        # Bearing + projection coefficients first.
        if spec == "lon":
            self._axis_kind = "lon"
            self._axis_bearing_deg = 90.0
            # Cardinal modes don't actually use the proj coefs, but
            # store them in a sane state in case of mode switching.
            self._axis_proj = (0.0, 1.0, cos_lat)
        elif spec == "lat":
            self._axis_kind = "lat"
            self._axis_bearing_deg = 0.0
            self._axis_proj = (1.0, 0.0, cos_lat)
        elif isinstance(spec, (int, float)):
            self._apply_vec_axis(math.radians(float(spec)), cos_lat)
        elif (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in spec)
        ):
            (la0, lo0), (la1, lo1) = spec
            dlat = la1 - la0
            dlon = (lo1 - lo0) * cos_lat
            self._apply_vec_axis(math.atan2(dlon, dlat), cos_lat)
        else:
            raise ValueError(f"Unsupported axis spec: {spec!r}")

        # Anchor at the airspace centre.  Live origin restores the
        # user's last drag-translated value if any, else snaps to anchor.
        # Clamp the persisted user origin to the (possibly different)
        # airspace bbox so it doesn't fall off-panel.
        if self._user_origin is not None:
            ula, ulo = self._user_origin
            ula = max(lat_min, min(lat_max, ula))
            ulo = max(lon_min, min(lon_max, ulo))
            self._user_origin = (ula, ulo)
        self._axis_anchor = (center_lat, center_lon)
        self._axis_origin = self._user_origin or (center_lat, center_lon)

        # Compute axis range.
        if self._axis_kind == "lon":
            self._axis_min, self._axis_max = lon_min, lon_max
        elif self._axis_kind == "lat":
            self._axis_min, self._axis_max = lat_min, lat_max
        else:
            self._axis_min, self._axis_max = self._project_corners_via_anchor(
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )

        if self._axis_kind == "vec":
            self._apply_axis_pad()

    def _apply_vec_axis(self, bearing_rad: float, cos_lat: float) -> None:
        """Set kind=vec and store projection coefficients + bearing readout."""
        self._axis_kind = "vec"
        self._axis_proj = (math.cos(bearing_rad), math.sin(bearing_rad), cos_lat)
        self._axis_bearing_deg = math.degrees(bearing_rad) % 360.0

    def _apply_axis_pad(self) -> None:
        """Widen the axis range symmetrically by ``_AXIS_PAD_FRAC``."""
        pad = max(self._axis_max - self._axis_min, 1e-9) * self._AXIS_PAD_FRAC
        self._axis_min -= pad
        self._axis_max += pad

    def update_bearing(self, bearing_deg: float) -> None:
        """Switch to a new bearing live, preserving any translated origin.

        Recomputes ``axis_min`` / ``axis_max`` from the bbox corners
        using the **anchor** as origin (so the airspace projection
        range stays anchored), but leaves :attr:`_axis_origin` alone.
        """
        if self._bbox_cache is None:
            return
        self._apply_vec_axis(math.radians(bearing_deg), self._bbox_cache["cos_lat"])
        self._axis_spec = bearing_deg
        bbox = self._bbox_cache
        self._axis_min, self._axis_max = self._project_corners_via_anchor(
            bbox["lat_min"],
            bbox["lat_max"],
            bbox["lon_min"],
            bbox["lon_max"],
        )
        self._apply_axis_pad()
        self._envelope_cache.clear()

    def update_origin(self, lat: float, lon: float) -> None:
        """Translate the live projection origin live.

        Anchor + axis range are untouched, so the bbox corners now
        project to v-values offset from their initial positions -
        producing the visible pan effect on the panel.  The translated
        position is also persisted (``_user_origin``) so it survives
        the next ``env.reset()``.
        """
        self._axis_origin = (lat, lon)
        self._user_origin = (lat, lon)
        self._envelope_cache.clear()

    def _project_corners_via_anchor(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> tuple[float, float]:
        """Project the four bbox corners using the anchor; return (min, max)."""
        la0, lo0 = self._axis_anchor
        cos_b, sin_b, cos_lat = self._axis_proj
        proj = [
            (la - la0) * cos_b + (lo - lo0) * cos_lat * sin_b
            for la in (lat_min, lat_max)
            for lo in (lon_min, lon_max)
        ]
        return min(proj), max(proj)

    def _axis_v(self, lat: float, lon: float) -> float:
        """Return the axis coordinate of (lat, lon) - the panel's x-value."""
        if self._axis_kind == "lon":
            return lon
        if self._axis_kind == "lat":
            return lat
        # "vec" - uses the *live* origin so translation pans the panel.
        la0, lo0 = self._axis_origin
        cos_b, sin_b, cos_lat = self._axis_proj
        return (lat - la0) * cos_b + (lon - lo0) * cos_lat * sin_b

    def v_to_latlon(self, v: float) -> tuple[float, float]:
        """Inverse of :meth:`_axis_v`: return the on-axis (lat, lon) at *v*.

        Lets the plan view draw a slice line whose endpoints sit at the
        same v-extent the profile shows (``axis_min`` <-> ``axis_max``),
        so the line on the plan is always exactly as long as what's
        visible on the vertical view.
        """
        la0, lo0 = self._axis_origin
        if self._axis_kind == "lon":
            return (la0, v)
        if self._axis_kind == "lat":
            return (v, lo0)
        cos_b, sin_b, cos_lat = self._axis_proj
        return (la0 + v * cos_b, lo0 + v * sin_b / max(cos_lat, 1e-9))

    def _horizontal_view(self, driver: PygameSimDriver):
        """Find a sibling plan view exposing ``project`` / ``_lat_per_px``."""
        for v in getattr(driver, "views", []):
            if v is self:
                continue
            if hasattr(v, "_lat_per_px") and hasattr(v, "project"):
                return v
        return None

    def _is_in_front(self, lat: float, lon: float) -> bool:
        """``True`` iff (lat, lon) sits on the *viewing* side of the slice plane.

        The slice line on the plan view divides the airspace into two
        half-planes; the perpendicular arrow we draw on the plan
        indicates which half is "in front" of the viewer.  Entities
        behind that line don't get rendered on the profile, so an
        observer sees only what the slice opens onto - like looking
        at a cross-section from one specific side.

        Sign matches the CW-perpendicular arrow convention used by
        :meth:`HorizontalView._draw_axis_indicator`.
        """
        la0, lo0 = self._axis_origin
        cos_b, sin_b, cos_lat = self._axis_proj
        v_perp = (lat - la0) * sin_b - (lon - lo0) * cos_lat * cos_b
        return v_perp >= 0

    # ------------------------------------------------------------------
    # Render-primitive ingestion
    # ------------------------------------------------------------------

    def add_polygon(self, driver: PygameSimDriver, polygon: Polygon) -> None:
        bounds = polygon.meta.get("bounds")
        # If the bounds object exposes per-point alt bands, keep it so
        # the slope is interpolated at render time. Otherwise fall back
        # to the polygon's static alt_range.
        has_slope = bounds is not None and hasattr(bounds, "alt_band_at")
        if polygon.alt_range is None and not has_slope:
            return
        slope_bounds = bounds if has_slope else None
        self._polygon_overlays.append((polygon, slope_bounds))
        # Widen the visible alt window so the polygon stays in frame -
        # spawn-derived alt extents are typically narrow, but a sloped
        # corridor / query region can span the full descent.
        if polygon.alt_range is not None:
            self._widen_alt(polygon.alt_range[0], polygon.alt_range[1])
        if has_slope:
            for lat, lon in polygon.vertices:
                lo, hi = bounds.alt_band_at(lat, lon)
                self._widen_alt(lo, hi)

    def add_point(self, driver: PygameSimDriver, point: Point) -> None:
        if point.alt_ft is None:
            return
        self._point_overlays.append(point)
        self._widen_alt(point.alt_ft, point.alt_ft)

    def _widen_alt(self, lo: float, hi: float) -> None:
        """Expand the visible alt window to include ``[lo, hi]`` (skip non-finite)."""
        if math.isfinite(lo):
            self._alt_min_view = min(self._alt_min_view, lo)
        if math.isfinite(hi):
            self._alt_max_view = max(self._alt_max_view, hi)

    def add_polyline(self, driver: PygameSimDriver, polyline: Polyline) -> None:
        pass  # polylines don't carry altitude

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _inner_rect(self) -> tuple[int, int, int, int]:
        """Plot area bounds: ``(left, top, right, bottom)`` excluding margins."""
        return (
            self.rect.left + self._MARGIN_LEFT,
            self.rect.top + self._MARGIN_TOP,
            self.rect.right - self._MARGIN_RIGHT,
            self.rect.bottom - self._MARGIN_BOTTOM,
        )

    def _axis_to_px(self, axis_v: float) -> float:
        """Convert an axis coordinate (output of :meth:`_axis_v`) to panel-x."""
        l, _, r, _ = self._inner_rect()
        f = (axis_v - self._axis_min) / max(self._axis_max - self._axis_min, 1e-9)
        x = l + f * max(r - l, 1)
        return self._viewport.apply_x(x, (l + r) / 2)

    def _project_x(self, lat_deg: float, lon_deg: float) -> float:
        return self._axis_to_px(self._axis_v(lat_deg, lon_deg))

    def _project_y(self, alt_ft: float) -> float:
        _, t, _, b = self._inner_rect()
        f = (alt_ft - self._alt_min_view) / max(
            self._alt_max_view - self._alt_min_view, 1e-9
        )
        y = b - f * max(b - t, 1)
        return self._viewport.apply_y(y, (t + b) / 2)

    def _alt_band_at_vertex(
        self, polygon, bounds, lat: float, lon: float
    ) -> tuple[float, float] | None:
        """Return ``(alt_lo, alt_hi)`` at vertex (lat, lon).  None if unknown."""
        if bounds is not None and hasattr(bounds, "alt_band_at"):
            return bounds.alt_band_at(lat, lon)
        return polygon.alt_range

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        pygame.draw.rect(canvas, _PANEL_BG, self.rect)

        self._draw_fl_grid(canvas, driver)
        for polygon, bounds in self._polygon_overlays:
            self._draw_polygon_band(canvas, driver, polygon, bounds)
        for point in self._point_overlays:
            self._draw_point_marker(canvas, driver, point)
        self._draw_aircraft(canvas, driver)
        self._draw_axis_label(canvas, driver)

        # Faint border around the plot area for definition.
        l, t, r, b = self._inner_rect()
        pygame.draw.rect(
            canvas,
            C.GRAY,
            pygame.Rect(l, t, max(r - l, 1), max(b - t, 1)),
            width=1,
        )

    # --- gridlines -----------------------------------------------------

    def _draw_fl_grid(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        l, t, r, b = self._inner_rect()
        interval = self._FL_INTERVAL_FT
        alt = math.ceil(self._alt_min_view / interval) * interval
        while alt <= self._alt_max_view:
            if alt > 0:
                y = self._project_y(alt)
                if t <= y <= b:
                    pygame.draw.line(canvas, C.GRID, (l, y), (r, y), width=1)
                    if driver.font is not None:
                        bg = driver.render_text_bg(
                            f"FL{int(round(alt / 100)):03d}",
                            C.GRID,
                        )
                        canvas.blit(
                            bg,
                            (
                                self.rect.left + 4,
                                y - bg.get_height() // 2,
                            ),
                        )
            alt += interval

    # --- polygon alt bands --------------------------------------------

    def _compute_envelope(
        self,
        polygon,
        bounds,
    ) -> list[tuple[list[float], list[float], list[float]]] | None:
        """Clip *polygon* to the visible slice half-plane, then project it.

        The old profile view sampled the polygon's bounding box and then
        discarded samples behind the slice. This clips the lateral polygon
        itself against the slice half-plane first, then evaluates vertical
        cross-sections of each clipped geometry component along the profile
        axis. Separate components stay separate so discontinuities do not
        get filled by the renderer.
        """
        clipped = self._clip_polygon_to_front_halfplane(polygon.vertices)
        if clipped is None or clipped.is_empty:
            return None

        segments: list[tuple[list[float], list[float], list[float]]] = []
        for part in self._iter_polygons(clipped):
            segments.extend(self._compute_envelope_part(part, polygon, bounds))
        return segments or None

    def _compute_envelope_part(
        self,
        geom,
        polygon,
        bounds,
    ) -> list[tuple[list[float], list[float], list[float]]]:
        axis_values = self._axis_values_for_geometry(geom)
        if len(axis_values) < 2:
            return []
        band = polygon.alt_range
        if bounds is None and band is not None:
            lo, hi = band
            return [
                (
                    axis_values,
                    [float(lo)] * len(axis_values),
                    [float(hi)] * len(axis_values),
                )
            ]

        lower: list[float] = []
        upper: list[float] = []
        for axis_x in axis_values:
            samples = self._cross_section_points(geom, axis_x)
            los: list[float] = []
            his: list[float] = []
            for lat, lon in samples:
                alt_band = self._alt_band_at_vertex(polygon, bounds, lat, lon)
                if alt_band is None:
                    continue
                lo, hi = alt_band
                if math.isfinite(lo) and math.isfinite(hi):
                    los.append(float(lo))
                    his.append(float(hi))
            if los and his:
                lower.append(min(los))
                upper.append(max(his))
            else:
                lower.append(math.nan)
                upper.append(math.nan)
        return self._valid_envelope_runs(axis_values, lower, upper)

    @staticmethod
    def _valid_envelope_runs(
        axis_values: list[float],
        lower: list[float],
        upper: list[float],
    ) -> list[tuple[list[float], list[float], list[float]]]:
        """Split an envelope at invalid samples instead of bridging gaps."""
        runs: list[tuple[list[float], list[float], list[float]]] = []
        cur_x: list[float] = []
        cur_lo: list[float] = []
        cur_hi: list[float] = []
        for axis_x, lo, hi in zip(axis_values, lower, upper):
            if math.isfinite(lo) and math.isfinite(hi):
                cur_x.append(axis_x)
                cur_lo.append(lo)
                cur_hi.append(hi)
                continue
            if len(cur_x) >= 2:
                runs.append((cur_x, cur_lo, cur_hi))
            cur_x = []
            cur_lo = []
            cur_hi = []
        if len(cur_x) >= 2:
            runs.append((cur_x, cur_lo, cur_hi))
        return runs

    def _clip_polygon_to_front_halfplane(self, vertices: list[tuple[float, float]]):
        """Return the lateral polygon clipped to the current visible half-plane."""
        if len(vertices) < 3:
            return None
        try:
            shape = _ShPolygon([(lon, lat) for lat, lon in vertices])
        except Exception:
            return None
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty:
            return None
        return shape.intersection(self._front_halfplane_shape(shape.bounds))

    def _front_halfplane_shape(self, bounds):
        """Large polygon covering the front side of the active slice line."""
        min_lon, min_lat, max_lon, max_lat = bounds
        la0, lo0 = self._axis_origin
        cos_b, sin_b, cos_lat = self._axis_proj
        cos_lat = max(abs(cos_lat), 1e-9)
        span = (
            max(
                abs(min_lon - lo0),
                abs(max_lon - lo0),
                abs(min_lat - la0),
                abs(max_lat - la0),
                1.0,
            )
            * 4.0
            + 1.0
        )
        axis_lat = cos_b
        axis_lon = sin_b / cos_lat
        front_lat = sin_b
        front_lon = -cos_b / cos_lat
        p0 = (la0 - axis_lat * span, lo0 - axis_lon * span)
        p1 = (la0 + axis_lat * span, lo0 + axis_lon * span)
        p2 = (p1[0] + front_lat * span, p1[1] + front_lon * span)
        p3 = (p0[0] + front_lat * span, p0[1] + front_lon * span)
        return _ShPolygon([(lon, lat) for lat, lon in (p0, p1, p2, p3)])

    def _axis_values_for_geometry(self, geom) -> list[float]:
        values: list[float] = []
        for poly in self._iter_polygons(geom):
            values.extend(self._axis_v(lat, lon) for lon, lat in poly.exterior.coords)
            for interior in poly.interiors:
                values.extend(self._axis_v(lat, lon) for lon, lat in interior.coords)
        if len(values) < 2:
            return []
        x_min = min(values)
        x_max = max(values)
        if x_max <= x_min:
            return []
        denom = max(self._BAND_BUCKETS - 1, 1)
        sampled = [
            x_min + (x_max - x_min) * i / denom for i in range(self._BAND_BUCKETS)
        ]
        return sorted(set(values + sampled))

    def _cross_section_points(self, geom, axis_x: float) -> list[tuple[float, float]]:
        """Return lat/lon samples from the clipped polygon at one profile x."""
        line = self._cross_section_line(geom.bounds, axis_x)
        return self._latlon_from_intersection(geom.intersection(line))

    def _cross_section_line(self, bounds, axis_x: float):
        min_lon, min_lat, max_lon, max_lat = bounds
        lat, lon = self.v_to_latlon(axis_x)
        cos_b, sin_b, cos_lat = self._axis_proj
        cos_lat = max(abs(cos_lat), 1e-9)
        front_lat = sin_b
        front_lon = -cos_b / cos_lat
        span = (
            max(
                abs(min_lon - lon),
                abs(max_lon - lon),
                abs(min_lat - lat),
                abs(max_lat - lat),
                1.0,
            )
            * 3.0
            + 1.0
        )
        a = (lat - front_lat * span, lon - front_lon * span)
        b = (lat + front_lat * span, lon + front_lon * span)
        return LineString([(a[1], a[0]), (b[1], b[0])])

    def _latlon_from_intersection(self, geom) -> list[tuple[float, float]]:
        if geom.is_empty:
            return []
        kind = geom.geom_type
        if kind == "Point":
            return [(geom.y, geom.x)]
        if kind == "MultiPoint":
            return [(p.y, p.x) for p in geom.geoms]
        if kind in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
            points: list[tuple[float, float]] = []
            for a, b in itertools.pairwise(coords):
                points.append((a[1], a[0]))
                points.append(((a[1] + b[1]) * 0.5, (a[0] + b[0]) * 0.5))
                points.append((b[1], b[0]))
            if len(coords) == 1:
                points.append((coords[0][1], coords[0][0]))
            return points
        if kind in {"MultiLineString", "GeometryCollection"}:
            out: list[tuple[float, float]] = []
            for part in geom.geoms:
                out.extend(self._latlon_from_intersection(part))
            return out
        if kind == "Polygon":
            out = []
            for lon, lat in geom.exterior.coords:
                out.append((lat, lon))
            return out
        if kind == "MultiPolygon":
            out = []
            for poly in geom.geoms:
                out.extend(self._latlon_from_intersection(poly))
            return out
        return []

    @staticmethod
    def _iter_polygons(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            yield geom
        elif geom.geom_type == "MultiPolygon":
            yield from geom.geoms
        elif geom.geom_type == "GeometryCollection":
            for part in geom.geoms:
                yield from VerticalView._iter_polygons(part)

    def _draw_polygon_band(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        polygon,
        bounds,
    ) -> None:
        """Draw a half-plane-clipped orthographic projection of *polygon*."""
        cache_key = id(polygon)
        if cache_key not in self._envelope_cache:
            self._envelope_cache[cache_key] = self._compute_envelope(polygon, bounds)
        cached = self._envelope_cache[cache_key]
        if cached is None:
            return
        segments = cached

        color = C.named(polygon.color)
        label_anchor: tuple[float, float] | None = None
        for axis_values, bucket_lo, bucket_hi in segments:
            upper_pts: list[tuple[float, float]] = []
            lower_pts: list[tuple[float, float]] = []
            for axis_x, lo, hi in zip(axis_values, bucket_lo, bucket_hi):
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    continue
                x_px = self._axis_to_px(axis_x)
                upper_pts.append((x_px, self._project_y(hi)))
                lower_pts.append((x_px, self._project_y(lo)))
            if len(upper_pts) < 2:
                continue

            outline = upper_pts + list(reversed(lower_pts))
            surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            local = [(p[0] - self.rect.left, p[1] - self.rect.top) for p in outline]
            pygame.draw.polygon(surf, (*color, _BAND_FILL_ALPHA), local)
            canvas.blit(surf, self.rect.topleft)
            pygame.draw.polygon(canvas, color, outline, width=self._BAND_OUTLINE)

            top_idx = min(range(len(upper_pts)), key=lambda i: upper_pts[i][1])
            candidate = upper_pts[top_idx]
            if label_anchor is None or candidate[1] < label_anchor[1]:
                label_anchor = candidate

        if driver.font is not None and polygon.label and label_anchor is not None:
            tx, ty = label_anchor
            bg = driver.render_text_bg(polygon.label, color)
            canvas.blit(
                bg,
                (
                    int(tx) - bg.get_width() // 2,
                    int(ty) - bg.get_height() - 2,
                ),
            )

    # --- point alt markers --------------------------------------------

    def _draw_point_marker(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        point,
    ) -> None:
        # Hide points behind the slice plane.
        if not self._is_in_front(point.lat, point.lon):
            return
        x = self._project_x(point.lat, point.lon)
        y = self._project_y(point.alt_ft)
        _l, t, _, b = self._inner_rect()
        if not (t <= y <= b):
            return
        color = C.named(point.color)
        r = self._POINT_RADIUS_PX
        diamond = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        pygame.draw.polygon(canvas, color, diamond)
        pygame.draw.polygon(canvas, C.BLACK, diamond, width=1)
        if driver.font is not None and point.label:
            bg = driver.render_text_bg(point.label, color)
            canvas.blit(bg, (int(x) + r + 3, int(y) - bg.get_height() // 2))

    # --- aircraft -----------------------------------------------------

    def _draw_aircraft(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        n = bs.traf.ntraf
        if n == 0:
            return
        for i in range(n):
            # Hide aircraft behind the slice plane - only the
            # front-side traffic ends up in the profile.
            if not self._is_in_front(bs.traf.lat[i], bs.traf.lon[i]):
                continue
            alt_ft = bs.traf.alt[i] / ft
            x = self._project_x(bs.traf.lat[i], bs.traf.lon[i])
            y = self._project_y(alt_ft)
            acid = bs.traf.id[i]
            state = driver._aircraft_state(acid)
            # Query-region tint mirrors the plan view - an aircraft
            # actually inside a configured :class:`QueryRegion` (e.g.
            # the merge cone) picks up that region's colour here, so
            # the side view reflects real containment rather than just
            # the axis projection (which can put a plane in a band's
            # x-range without it being inside the polygon laterally).
            query_color = self._query_color_for_aircraft(
                driver,
                bs.traf.lat[i],
                bs.traf.lon[i],
                alt_ft,
            )
            if state == "los":
                color = C.LOS
                pz_color = C.LOS
                pz_width = None  # sentinel: translucent fill via fill_alpha_rect
            elif state == "conflict":
                color = C.CONF
                pz_color = C.CONF
                pz_width = self._PROTECTION_WIDTH + 1
            elif state == "violation":
                color = C.VIOLATION
                pz_color = C.VIOLATION
                pz_width = self._PROTECTION_WIDTH + 1
            else:
                color = query_color or C.BLACK
                pz_color = C.PROT_ZONE
                pz_width = self._PROTECTION_WIDTH
            if driver._aircraft_snapshot(acid).get("background", False):
                color = C.dim(color)
                pz_color = C.dim(pz_color)
            # Protection zone cross-section: BlueSky's PZ is a vertical
            # cylinder of horizontal radius rpz[i] and vertical half-height
            # hpz[i] (both in m).  Side-view projection is a rectangle
            # centred on the aircraft - drawn as if the aircraft sits on
            # the slice plane (off-axis chord shrinkage is ignored, same
            # simplification the plan view doesn't have to make).
            rpz_m = float(bs.traf.cd.rpz[i])
            hpz_ft = float(bs.traf.cd.hpz[i]) / ft
            cos_lat = math.cos(math.radians(bs.traf.lat[i]))
            # Arc-length / radius gives angle in radians; convert to deg.
            # Longitude degrees shrink with latitude, so divide by cos_lat
            # when the slice axis is degrees-of-longitude.
            if self._axis_kind == "lon":
                rpz_axis = math.degrees(rpz_m / (Rearth * cos_lat))
            else:
                rpz_axis = math.degrees(rpz_m / Rearth)
            axis_v_ac = self._axis_v(bs.traf.lat[i], bs.traf.lon[i])
            half_w_px = self._axis_to_px(axis_v_ac + rpz_axis) - self._axis_to_px(
                axis_v_ac
            )
            y_hi_px = self._project_y(alt_ft + hpz_ft)
            y_lo_px = self._project_y(alt_ft - hpz_ft)
            pz_rect = pygame.Rect(
                int(round(x - half_w_px)),
                int(round(y_hi_px)),
                max(int(round(2 * half_w_px)), 1),
                max(int(round(y_lo_px - y_hi_px)), 1),
            )
            if pz_width is None:
                # LoS: translucent fill + solid outline (same width as conflict).
                C.fill_alpha_rect(canvas, pz_color, pz_rect)
                pygame.draw.rect(
                    canvas,
                    pz_color,
                    pz_rect,
                    width=self._PROTECTION_WIDTH + 1,
                )
            else:
                pygame.draw.rect(canvas, pz_color, pz_rect, width=pz_width)
            # Vertical climb/descent vector - encodes vs only.
            future_alt = alt_ft + bs.traf.vs[i] * self._LOOKAHEAD_S / ft
            yv = self._project_y(future_alt)
            _, t, _, b = self._inner_rect()
            yv_clip = max(t, min(b, yv))
            pygame.draw.line(
                canvas, color, (x, y), (x, yv_clip), width=self._AC_VECTOR_WIDTH
            )
            pygame.draw.circle(canvas, color, (x, y), self._AC_RADIUS)
            if driver.show_callsigns and driver.font is not None:
                driver.blit_data_block(
                    canvas,
                    driver.format_aircraft_marker_label_lines(i),
                    color,
                    x,
                    y,
                )

    # --- axis label ---------------------------------------------------

    def _draw_axis_label(
        self, canvas: pygame.Surface, driver: PygameSimDriver
    ) -> None:
        if driver.font is None:
            return
        l, _, r, b = self._inner_rect()
        # Tick marks at left / centre / right of the rotate strip, no
        # numeric labels - the compass on the plan view + the centre
        # hint below already convey orientation; per-tick lat/lon
        # numbers were just visual noise.
        for f in (0.0, 0.5, 1.0):
            x = l + f * max(r - l, 1)
            pygame.draw.line(canvas, C.GRAY, (x, b), (x, b + 3), width=1)
        # Live bearing readout + drag affordance, centred in the strip.
        # Plain ASCII so the monospace font always has the glyphs.
        center_x = (l + r) // 2
        hint = f"{int(round(self._axis_bearing_deg)) % 360:03d} DEG"
        hint_surf = driver.font.render(hint, True, C.GRAY)
        canvas.blit(
            hint_surf,
            (
                center_x - hint_surf.get_width() // 2,
                b + 4,
            ),
        )

    # ------------------------------------------------------------------
    # Hover & cross-view
    # ------------------------------------------------------------------

    def aircraft_position(
        self,
        driver: PygameSimDriver,
        idx: int,
    ) -> tuple[float, float] | None:
        n = bs.traf.ntraf
        if not (0 <= idx < n):
            return None
        # Hidden behind the slice plane -> no position to highlight or
        # link.  Returning None lets the cross-view link skip drawing
        # to a non-visible marker.
        if not self._is_in_front(bs.traf.lat[idx], bs.traf.lon[idx]):
            return None
        return (
            self._project_x(bs.traf.lat[idx], bs.traf.lon[idx]),
            self._project_y(bs.traf.alt[idx] / ft),
        )

    def highlight_aircraft(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        idx: int,
    ) -> None:
        pos = self.aircraft_position(driver, idx)
        if pos is None:
            return
        pygame.draw.circle(canvas, C.HIGHLIGHT, pos, 12, width=3)

    def hover(self, mouse_pos, driver: PygameSimDriver) -> dict | None:
        if not self.rect.collidepoint(mouse_pos):
            return None
        mx, my = mouse_pos
        n = bs.traf.ntraf
        if n == 0:
            return None
        radius_sq = 16**2
        best_idx, best_d = None, radius_sq
        for i in range(n):
            # Skip hidden aircraft so a hover doesn't grab one that
            # isn't visually present on the panel.
            if not self._is_in_front(bs.traf.lat[i], bs.traf.lon[i]):
                continue
            x = self._project_x(bs.traf.lat[i], bs.traf.lon[i])
            y = self._project_y(bs.traf.alt[i] / ft)
            d = (mx - x) ** 2 + (my - y) ** 2
            if d < best_d:
                best_idx, best_d = i, d
        if best_idx is None:
            return None
        return {"kind": "aircraft", "idx": best_idx}

    # ------------------------------------------------------------------
    # Drag-rotate: drag the bottom rotate strip horizontally to spin the
    # profile axis live.  A full panel-width drag = 360 deg rotation, so
    # ~3 pixels per degree at typical panel widths.
    # ------------------------------------------------------------------

    def _rotate_zone(self) -> pygame.Rect:
        """The hit zone for rotate-drag - the bottom margin (axis strip)."""
        l = self.rect.left + self._MARGIN_LEFT
        w = max(self.rect.width - self._MARGIN_LEFT - self._MARGIN_RIGHT, 1)
        # Reserve the bottom 4 px so the splitter divider underneath
        # remains grabbable for resize.
        h = max(self._MARGIN_BOTTOM - 4, 1)
        return pygame.Rect(l, self.rect.bottom - self._MARGIN_BOTTOM, w, h)

    def hit_test_drag(self, pos, driver) -> str | None:
        if not self._rotate_zone().collidepoint(pos):
            return None
        self._drag_start_x = pos[0]
        self._drag_start_bearing = self._axis_bearing_deg
        return "rotate"

    def on_drag_motion(self, handle, pos, driver) -> None:
        if handle != "rotate" or self._bbox_cache is None:
            return
        zone = self._rotate_zone()
        dx = pos[0] - self._drag_start_x
        # A full panel-width sweep = 360 deg rotation.  Sign chosen so
        # dragging the bottom of the panel rightward spins the axis CCW
        # (compass arrow on the plan view turns the same way the cursor
        # moves around the disk centre).
        delta_bearing = -(dx / max(zone.width, 1)) * 360.0
        new_bearing = (self._drag_start_bearing + delta_bearing) % 360.0
        # ``update_bearing`` preserves any translated origin from a
        # prior plan-view drag; ``_resolve_axis`` would reset it.
        self.update_bearing(new_bearing)

    def cursor_hint(self, pos, driver) -> CursorHintName | None:
        if self._rotate_zone().collidepoint(pos):
            return CursorHint.RESIZE_X
        return None

    def zoom_view_at(
        self,
        pos: tuple[int, int],
        factor: float,
        driver: PygameSimDriver,
    ) -> bool:
        l, t, r, b = self._inner_rect()
        return self._viewport.zoom_at(pos[0], pos[1], (l + r) / 2, (t + b) / 2, factor)

    def pan_view_by(
        self,
        delta: tuple[int, int],
        driver: PygameSimDriver,
    ) -> bool:
        return self._viewport.pan_by(delta[0], delta[1])

    def reset_viewport(self, driver: PygameSimDriver) -> bool:
        return self._viewport.reset()
