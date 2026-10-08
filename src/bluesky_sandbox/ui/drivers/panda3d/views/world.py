"""WorldView - owns the 3D scene graph for the panda3d driver.

This is the panda3d analogue of pygame's ``HorizontalView`` +
``VerticalView`` rolled into one: the 3D viewport already captures
both top-down and side-on perspectives, so there's a single view
holding the lat/lon -> local-ENU projection, every overlay primitive
(polygons / points / polylines), the live aircraft markers (chevron
+ protection-zone cylinder + label + depth pole), and the ground
reference (grid + airspace bbox + north arrow).

Anything that isn't drawn into the 3D viewport - window / events /
camera math / HUD text overlays - lives in the driver.  WorldView's
contract is "what gets painted into the 3D scene".
"""

# ruff: noqa: PLC0415 - panda3d is an optional extra ([panda3d]); every import
# below is a panda3d/direct symbol pulled in at draw time so that importing
# this module (e.g. for type hints or tests) never requires the engine.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import bluesky as bs
from bluesky.tools.aero import ft

from bluesky_sandbox.interface.task import WaypointReadoutKey
from bluesky_sandbox.ui.drivers.common import CursorHint, CursorHintName
from bluesky_sandbox.ui.drivers.panda3d.colors import (
    CHEVRON_NOTCH_FRAC,
    CHEVRON_WING_FRAC,
    HIGHLIGHT,
    M_PER_DEG,
    NAMED_COLORS,
    STATE_COLORS,
    dim_rgb,
)
from bluesky_sandbox.ui.drivers.panda3d.colors import (
    color as _color,
)
from bluesky_sandbox.ui.drivers.panda3d.views.base import Panda3DView
from bluesky_sandbox.ui.drivers.panda3d.views.tags import AircraftTag, AircraftTags

M_PER_NM = 1852.0

if TYPE_CHECKING:
    from panda3d.core import NodePath

    from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline
    from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver


class WorldView(Panda3DView):
    """The 3D scene: overlays, aircraft, ground reference, projection."""

    def __init__(self) -> None:
        # NodePath roots - populated in :meth:`on_start`.
        self._static: NodePath | None = None
        self._aircraft_root: NodePath | None = None
        self._ground: NodePath | None = None
        # Trails live under their own persistent root so they survive
        # the per-frame teardown of ``_aircraft_root``.  Each entry in
        # ``_trail_geometry`` is one aircraft's CHUNKED geometry: older
        # points are sealed into immutable NodePaths that never touch the
        # GPU again, and only a short tail is rebuilt as points arrive.
        # See :class:`_TrailGeometry`.
        self._trails_root: NodePath | None = None
        self._trail_geometry: dict[str, _TrailGeometry] = {}
        # Each aircraft's marker lives from its first frame to its last and is
        # moved, not rebuilt; what is drawn for one frame only (the tracked
        # aircraft's ring and route) goes under ``_frame_root``, under
        # ``_aircraft_root``, and is dropped with the next frame.
        self._markers: dict[str, _AircraftMarker] = {}
        self._marker_templates: _MarkerTemplates | None = None
        self._frame_root: NodePath | None = None
        self._static_label_nodes: list[NodePath] = []
        # Aircraft data blocks: flat on the screen, beside their aircraft.
        self._tags: AircraftTags | None = None

        # Local-ENU projection origin, recomputed each :meth:`on_reset`.
        self._lat0: float = 0.0
        self._lon0: float = 0.0
        self._cos_lat0: float = 1.0

        # Cache of (screen_x, screen_y, acid) for picking - refilled
        # every aircraft refresh.
        self._screen_aircraft: list[tuple[float, float, str]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self, driver: Panda3DSimDriver) -> None:
        """Attach the scene-graph roots under ``render``."""
        self._static = driver._render.attachNewNode("static")
        self._aircraft_root = driver._render.attachNewNode("aircraft")
        self._ground = driver._render.attachNewNode("ground")
        self._trails_root = driver._render.attachNewNode("trails")
        self._static_label_nodes.clear()
        self._tags = AircraftTags(driver._show.aspect2d, getattr(driver, "_ui_font", None))
        self._install_lighting(driver)

    def on_reset(self, driver: Panda3DSimDriver, env) -> None:
        """Recompute projection, rebuild ground + overlay nodes."""
        # Pick the projection origin from airspace bounds when set,
        # otherwise fall back to the union of all spawn regions.
        airspace = env.episode_airspace_bounds
        if airspace is not None:
            (lat_min, lat_max), (lon_min, lon_max) = airspace.bounding_box
        else:
            spawn_bb = env.episode_spawn.resolved_bounds
            lat_min, lat_max = spawn_bb["lat_deg"]
            lon_min, lon_max = spawn_bb["lon_deg"]

        self._lat0 = 0.5 * (lat_min + lat_max)
        self._lon0 = 0.5 * (lon_min + lon_max)
        self._cos_lat0 = math.cos(math.radians(self._lat0))

        # Drop previous nodes - env.reset() may have a different shape.
        if self._static is not None:
            self._static.removeNode()
            self._aircraft_root.removeNode()
            self._ground.removeNode()
        if self._trails_root is not None:
            self._trails_root.removeNode()
        self._static = driver._render.attachNewNode("static")
        self._aircraft_root = driver._render.attachNewNode("aircraft")
        self._ground = driver._render.attachNewNode("ground")
        self._trails_root = driver._render.attachNewNode("trails")
        self._static_label_nodes.clear()
        # The markers went with the old ``_aircraft_root``.
        self._markers.clear()
        if self._tags is not None:
            self._tags.clear()
        self._frame_root = None
        # The driver's trail dict is cleared on reset, and the old
        # ``_trails_root`` has just been removed along with every node
        # parented under it - so the cache holds nothing but dangling
        # references.
        self._trail_geometry.clear()

        self._build_ground_reference(driver, lat_min, lat_max, lon_min, lon_max)

    def clear_static(self, driver: Panda3DSimDriver) -> None:
        """Drop the region prisms and labels, to be drawn again - a region
        moved."""
        if self._static is not None:
            self._static.removeNode()
        self._static = driver._render.attachNewNode("static")
        self._static_label_nodes.clear()

    def on_step(self, driver: Panda3DSimDriver) -> None:
        """Bring every aircraft marker up to date for this frame."""
        self._refresh_aircraft(driver)

    def close(self) -> None:
        """Drop scene graph references owned by this view."""
        for node in (
            self._static,
            self._aircraft_root,
            self._ground,
            self._trails_root,
        ):
            if node is not None:
                try:
                    node.removeNode()
                except Exception:
                    pass
        self._static = None
        self._aircraft_root = None
        self._ground = None
        self._trails_root = None
        self._trail_geometry.clear()
        self._markers.clear()
        if self._tags is not None:
            self._tags.destroy()
            self._tags = None
        self._frame_root = None
        self._screen_aircraft = []

    # ------------------------------------------------------------------
    # Render-primitive vocabulary
    # ------------------------------------------------------------------

    def draw_polygon(self, driver: Panda3DSimDriver, polygon: Polygon) -> None:
        """Draw a polygon's altitude envelope as a 3D wire prism.

        Three cases:

        * ``per_vertex_alt`` set - slanted envelope (each vertex has its
          own ``(lo, hi)``).  Sloped ``RegionBounds`` objects use this
          path to produce a tilted volume.
        * ``alt_range`` set - flat band; rings at one min altitude and
          one max altitude.
        * Neither set - single ring at ground.
        """
        if self._static is None:
            return
        if not polygon.vertices:
            return
        rgba = _color(polygon.color, alpha=1.0)

        if polygon.per_vertex_alt is not None and len(polygon.per_vertex_alt) == len(
            polygon.vertices
        ):
            lo_m = [lo * ft for lo, _ in polygon.per_vertex_alt]
            hi_m = [hi * ft for _, hi in polygon.per_vertex_alt]
            self._draw_polygon_ring_varying(polygon.vertices, lo_m, rgba)
            self._draw_polygon_ring_varying(polygon.vertices, hi_m, rgba)
            self._draw_polygon_verticals_varying(polygon.vertices, lo_m, hi_m, rgba)
            label_alt_m = 0.5 * (sum(lo_m) / len(lo_m) + sum(hi_m) / len(hi_m))
        elif polygon.alt_range is not None:
            lo_ft, hi_ft = polygon.alt_range
            self._draw_polygon_ring(polygon.vertices, lo_ft * ft, rgba)
            if hi_ft != lo_ft:
                self._draw_polygon_ring(polygon.vertices, hi_ft * ft, rgba)
                self._draw_polygon_verticals(
                    polygon.vertices, lo_ft * ft, hi_ft * ft, rgba
                )
            label_alt_m = 0.5 * (lo_ft + hi_ft) * ft
        else:
            self._draw_polygon_ring(polygon.vertices, 0.0, rgba)
            label_alt_m = 0.0

        if polygon.label:
            cx = sum(v[0] for v in polygon.vertices) / len(polygon.vertices)
            cy = sum(v[1] for v in polygon.vertices) / len(polygon.vertices)
            ex, ny, _ = self._project(cx, cy, 0.0)
            label = self._attach_label(
                driver, self._static, polygon.label, ex, ny, label_alt_m, rgba
            )
            self._static_label_nodes.append(label)
            if not driver.show_labels:
                label.hide()

    def draw_point(self, driver: Panda3DSimDriver, point: Point) -> None:
        """Draw a Point as a translucent sphere with a depth pole + label.

        Waypoints (and any other Point primitive) render at ~55 %
        alpha so they read as airspace landmarks rather than solid
        objects competing with aircraft and protection zones.
        """
        if self._static is None:
            return
        from panda3d.core import LineSegs, TransparencyAttrib, Vec4

        alt_m = (point.alt_ft or 0.0) * ft
        ex, ny, up = self._project(point.lat, point.lon, alt_m)
        # Marker fill is translucent so traffic crossing the merge
        # point stays visible through the sphere.
        rgba_fill = _color(point.color, alpha=0.55)
        rgba_solid = _color(point.color, alpha=1.0)

        # Sphere radius scales with focal distance so the marker stays
        # visible at any zoom - clamped so it never grows obnoxious.
        radius = max(min(driver._distance * 0.005, 4_000.0), 200.0)
        sphere = self._make_sphere(driver, radius, rgba_fill)
        sphere.reparentTo(self._static)
        sphere.setPos(ex, ny, up)
        sphere.setTransparency(TransparencyAttrib.MAlpha)

        # Plus a vertical pole down to the ground for depth cueing.
        ls = LineSegs()
        ls.setColor(Vec4(*rgba_solid[:3], 0.4))
        ls.setThickness(1.0)
        ls.moveTo(ex, ny, 0.0)
        ls.drawTo(ex, ny, up)
        pole_np = self._static.attachNewNode(ls.create())
        pole_np.setTransparency(TransparencyAttrib.MAlpha)

        if point.label:
            label = self._attach_label(
                driver, self._static, point.label, ex, ny, up + radius * 2, rgba_solid
            )
            self._static_label_nodes.append(label)
            if not driver.show_labels:
                label.hide()

    def draw_polyline(self, driver: Panda3DSimDriver, polyline: Polyline) -> None:
        """Open chain at ground level in the polyline's color."""
        if self._static is None:
            return
        if len(polyline.points) < 2:
            return
        from panda3d.core import LineSegs, Vec4

        rgba = _color(polyline.color, alpha=1.0)
        ls = LineSegs()
        ls.setColor(Vec4(*rgba))
        ls.setThickness(2.0)
        first = True
        for lat, lon in polyline.points:
            ex, ny, _ = self._project(lat, lon, 0.0)
            if first:
                ls.moveTo(ex, ny, 0.0)
                first = False
            else:
                ls.drawTo(ex, ny, 0.0)
        self._static.attachNewNode(ls.create())

    # ------------------------------------------------------------------
    # Picking - called from the driver on a quick click.
    # ------------------------------------------------------------------

    def pick(
        self, driver: Panda3DSimDriver, mx_px: float, my_px: float
    ) -> str | None:
        """Return the nearest cached aircraft id within the pick radius."""
        best_acid: str | None = None
        best_d2 = driver._PICK_RADIUS_PX**2
        for sx, sy, acid in self._screen_aircraft:
            d2 = (sx - mx_px) ** 2 + (sy - my_px) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_acid = acid
        return best_acid

    def cursor_hint(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> CursorHintName | None:
        """Use the pointer cursor over aircraft that can be selected."""
        if driver._show is None:
            return None
        win = driver._show.win
        aspect = driver._show.getAspectRatio()
        w, h = win.getXSize(), win.getYSize()
        mx_ndc = pos[0] / aspect if aspect else 0.0
        my_ndc = pos[1]
        mx_px = (mx_ndc + 1.0) * 0.5 * w
        my_px = (1.0 - (my_ndc + 1.0) * 0.5) * h
        if self.pick(driver, mx_px, my_px) is None:
            return None
        return CursorHint.POINT

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project(
        self, lat_deg: float, lon_deg: float, alt_m: float
    ) -> tuple[float, float, float]:
        """Project geodetic ``(lat, lon, alt_m)`` into local-ENU meters."""
        east_m = (lon_deg - self._lon0) * self._cos_lat0 * M_PER_DEG
        north_m = (lat_deg - self._lat0) * M_PER_DEG
        return east_m, north_m, alt_m

    # ------------------------------------------------------------------
    # Scene graph: lighting + ground reference
    # ------------------------------------------------------------------

    def _install_lighting(self, driver: Panda3DSimDriver) -> None:
        from panda3d.core import AmbientLight, DirectionalLight, Vec4

        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.35, 0.38, 0.45, 1.0))
        driver._render.setLight(driver._render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.75, 0.75, 0.70, 1.0))
        sun_np = driver._render.attachNewNode(sun)
        sun_np.setHpr(-30, -45, 0)
        driver._render.setLight(sun_np)

    def _build_ground_reference(
        self,
        driver: Panda3DSimDriver,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> None:
        """Faint grid + bbox outline + north arrow at the origin."""
        from panda3d.core import LineSegs, Vec4

        outline = LineSegs()
        outline.setColor(Vec4(0.55, 0.70, 0.85, 1.0))
        outline.setThickness(2.0)
        corners = [
            (lat_min, lon_min),
            (lat_max, lon_min),
            (lat_max, lon_max),
            (lat_min, lon_max),
            (lat_min, lon_min),
        ]
        first = True
        for la, lo in corners:
            ex, ny, _ = self._project(la, lo, 0.0)
            if first:
                outline.moveTo(ex, ny, 0.0)
                first = False
            else:
                outline.drawTo(ex, ny, 0.0)
        self._ground.attachNewNode(outline.create())

        grid = LineSegs()
        grid.setColor(Vec4(0.30, 0.40, 0.50, 1.0))
        grid.setThickness(1.0)
        for i in range(1, 8):
            t = i / 8
            la = lat_min + t * (lat_max - lat_min)
            lo = lon_min + t * (lon_max - lon_min)
            ax, ay, _ = self._project(la, lon_min, 0.0)
            bx, by, _ = self._project(la, lon_max, 0.0)
            grid.moveTo(ax, ay, 0.0)
            grid.drawTo(bx, by, 0.0)
            cx, cy, _ = self._project(lat_min, lo, 0.0)
            dx, dy, _ = self._project(lat_max, lo, 0.0)
            grid.moveTo(cx, cy, 0.0)
            grid.drawTo(dx, dy, 0.0)
        self._ground.attachNewNode(grid.create())

        arrow = LineSegs()
        arrow.setColor(Vec4(0.85, 0.85, 0.95, 1.0))
        arrow.setThickness(2.0)
        size = max(driver._distance * 0.04, 5_000.0)
        arrow.moveTo(0, 0, 0)
        arrow.drawTo(0, size, 0)
        arrow.moveTo(0, size, 0)
        arrow.drawTo(-size * 0.2, size * 0.8, 0)
        arrow.moveTo(0, size, 0)
        arrow.drawTo(size * 0.2, size * 0.8, 0)
        self._ground.attachNewNode(arrow.create())
        self._attach_label(
            driver, self._ground, "N", 0, size + size * 0.15, 0, (0.9, 0.9, 1.0, 1.0)
        )

    # ------------------------------------------------------------------
    # Polygon rendering helpers
    # ------------------------------------------------------------------

    def _draw_polygon_ring(
        self,
        verts: list[tuple[float, float]],
        alt_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        from panda3d.core import LineSegs, Vec4

        ls = LineSegs()
        ls.setColor(Vec4(*rgba))
        ls.setThickness(2.0)
        first = True
        for lat, lon in verts:
            ex, ny, _ = self._project(lat, lon, alt_m)
            if first:
                ls.moveTo(ex, ny, alt_m)
                first = False
            else:
                ls.drawTo(ex, ny, alt_m)
        lat0, lon0 = verts[0]
        ex0, ny0, _ = self._project(lat0, lon0, alt_m)
        ls.drawTo(ex0, ny0, alt_m)
        self._static.attachNewNode(ls.create())

    def _draw_polygon_verticals(
        self,
        verts: list[tuple[float, float]],
        lo_m: float,
        hi_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        from panda3d.core import LineSegs, Vec4

        ls = LineSegs()
        ls.setColor(Vec4(rgba[0], rgba[1], rgba[2], 0.55))
        ls.setThickness(1.0)
        for lat, lon in verts:
            ex, ny, _ = self._project(lat, lon, 0.0)
            ls.moveTo(ex, ny, lo_m)
            ls.drawTo(ex, ny, hi_m)
        self._static.attachNewNode(ls.create())

    def _draw_polygon_ring_varying(
        self,
        verts: list[tuple[float, float]],
        alt_m_per_vertex: list[float],
        rgba: tuple[float, float, float, float],
    ) -> None:
        """A closed wire-loop whose altitude varies per vertex.

        For a triangle this renders as an exact tilted plane; for
        polygons with more vertices the slope follows the
        :meth:`Bounds.per_vertex_alt_range` data.
        """
        from panda3d.core import LineSegs, Vec4

        ls = LineSegs()
        ls.setColor(Vec4(*rgba))
        ls.setThickness(2.0)
        n = len(verts)
        for i in range(n + 1):
            lat, lon = verts[i % n]
            z = alt_m_per_vertex[i % n]
            ex, ny, _ = self._project(lat, lon, z)
            if i == 0:
                ls.moveTo(ex, ny, z)
            else:
                ls.drawTo(ex, ny, z)
        self._static.attachNewNode(ls.create())

    def _draw_polygon_verticals_varying(
        self,
        verts: list[tuple[float, float]],
        lo_m_per_vertex: list[float],
        hi_m_per_vertex: list[float],
        rgba: tuple[float, float, float, float],
    ) -> None:
        from panda3d.core import LineSegs, Vec4

        ls = LineSegs()
        ls.setColor(Vec4(rgba[0], rgba[1], rgba[2], 0.55))
        ls.setThickness(1.0)
        for (lat, lon), lo, hi in zip(verts, lo_m_per_vertex, hi_m_per_vertex):
            ex, ny, _ = self._project(lat, lon, 0.0)
            ls.moveTo(ex, ny, lo)
            ls.drawTo(ex, ny, hi)
        self._static.attachNewNode(ls.create())

    # ------------------------------------------------------------------
    # Sphere + label
    # ------------------------------------------------------------------

    def _make_sphere(
        self,
        driver: Panda3DSimDriver,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> NodePath:
        """Low-poly UV sphere - cached topology, instanced position."""
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
            Vec4,
        )

        fmt = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData("sphere", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        tris = GeomTriangles(Geom.UHStatic)

        rings, segs = 8, 12
        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segs + 1):
                theta = 2 * math.pi * j / segs
                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)
                vw.addData3(x * radius, y * radius, z * radius)
                nw.addData3(x, y, z)
        for i in range(rings):
            for j in range(segs):
                a = i * (segs + 1) + j
                b = a + segs + 1
                tris.addVertices(a, b, a + 1)
                tris.addVertices(b, b + 1, a + 1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("sphere_geom")
        node.addGeom(geom)
        np = driver._render.attachNewNode(node)
        np.setColor(Vec4(*rgba))
        np.detachNode()
        return np

    def _attach_label(
        self,
        driver: Panda3DSimDriver,
        parent: NodePath,
        text: str,
        x: float,
        y: float,
        z: float,
        rgba: tuple[float, float, float, float],
        *,
        screen_offset: tuple[float, float] = (0.0, 0.0),
        draw_order: int = 0,
    ) -> NodePath:
        from panda3d.core import TextNode, Vec4

        tn = TextNode("label")
        tn.setText(text)
        tn.setAlign(TextNode.ACenter)
        font = getattr(driver, "_ui_font", None)
        if font is not None:
            tn.setFont(font)
        tn.setTextColor(Vec4(0.96, 0.98, 1.0, rgba[3]))
        tn.setCardColor(0.02, 0.025, 0.03, 0.82)
        tn.setCardAsMargin(0.32, 0.32, 0.18, 0.18)
        tn.setCardDecal(True)
        np = parent.attachNewNode(tn)
        scale = self._screen_scale(driver, x, y, z, factor=0.012, minimum=1_350.0)
        np.setScale(scale)
        right, _fwd = driver._camera_ground_basis()
        offset_x = right[0] * screen_offset[0] * scale
        offset_y = right[1] * screen_offset[0] * scale
        offset_z = screen_offset[1] * scale
        np.setPos(x + offset_x, y + offset_y, z + offset_z)
        np.setBillboardPointEye()
        np.setDepthTest(False)
        np.setDepthWrite(False)
        np.setBin("fixed", 80 + draw_order)
        return np

    # ------------------------------------------------------------------
    # Aircraft refresh
    # ------------------------------------------------------------------

    def _refresh_aircraft(self, driver: Panda3DSimDriver) -> None:
        """Move, turn, size and color every aircraft's marker for this frame.

        A marker is made once, for the aircraft's first frame - from shared
        unit geometry (see :class:`_MarkerTemplates`), placed and scaled by
        its transform - and dropped once the aircraft is gone, so a frame
        builds no geometry for aircraft, only the tracked one's ring and
        route.
        """
        if self._aircraft_root is None:
            return
        if self._frame_root is not None:
            self._frame_root.removeNode()
        self._frame_root = self._aircraft_root.attachNewNode("frame")
        self._screen_aircraft = []
        if self._marker_templates is None:
            self._marker_templates = _MarkerTemplates()
        templates = self._marker_templates

        cd = bs.traf.cd
        rpz_arr = cd.rpz
        hpz_arr = cd.hpz
        default_rpz_m = 5.0 * 1_852.0  # 5 NM
        default_hpz_m = 1000.0 * 0.3048  # 1000 ft

        aircraft = driver.aircraft_frame()
        for acid in self._markers.keys() - set(aircraft.ids):
            self._markers.pop(acid).destroy()

        tracked = driver.tracked_acid()
        east, north = self._project_many(aircraft.lat, aircraft.lon)
        tags: list[AircraftTag] = []
        for i, acid in enumerate(aircraft.ids):
            ex, ny, up = east[i], north[i], aircraft.alt_m[i]

            # Color priority: display alerts > query region > default.
            state = aircraft.state(i)
            if state in ("los", "conflict", "violation"):
                base_rgb = STATE_COLORS[state]
            else:
                query = aircraft.query_color(i)
                base_rgb = _color(query)[:3] if query is not None else STATE_COLORS["normal"]
            background = aircraft.background(i)
            if background:
                base_rgb = dim_rgb(base_rgb)
            rgba = (*base_rgb, 0.55 if background else 1.0)

            marker = self._markers.get(acid)
            if marker is None:
                marker = self._markers[acid] = _AircraftMarker(self._aircraft_root, templates)

            visual_size = self._chevron_visual_size(driver, ex, ny, up)
            marker.place_chevron(ex, ny, up, aircraft.hdg[i], visual_size, rgba)

            rpz_m = float(rpz_arr[i]) if rpz_arr is not None else default_rpz_m
            hpz_m = float(hpz_arr[i]) if hpz_arr is not None else default_hpz_m
            marker.place_protection_zone(
                templates, ex, ny, up, rpz_m, hpz_m, *self._protection_zone_style(base_rgb, state)
            )
            marker.place_pole(ex, ny, up, (*base_rgb, 0.65), visual_size)

            sx, sy = self._world_to_screen(driver, ex, ny, up)
            self._screen_aircraft.append((sx, sy, acid))
            lines = driver.format_aircraft_marker_label_lines(i)
            if lines and sx > -1e8:
                tags.append(self._tag(acid, sx, sy, lines, base_rgb, state, background, acid == tracked))

            if acid == tracked:
                self._draw_highlight_ring(ex, ny, up, rpz_m * 1.10, HIGHLIGHT)

        if self._tags is not None:
            self._tags.update(driver, tags, driver.hud_rects_px())

        self._draw_selected_route(driver)

        # Sync the cached trail NodePaths exactly once per frame.
        # Cache key is just length - each point's color is baked at
        # append time on the driver, so old segments never re-color
        # and we only rebuild when new points arrive.
        self._sync_trails(driver)

    # A block's text, by separation state: alerts in their color, the rest
    # a soft white that does not compete with them.
    _TAG_TEXT = (0.86, 0.93, 1.0)

    def _tag(
        self,
        acid: str,
        sx: float,
        sy: float,
        lines: list[str],
        base_rgb: tuple[float, float, float],
        state: str,
        background: bool,
        tracked: bool,
    ) -> AircraftTag:
        """An aircraft's data block: colored by its separation state, framed
        when tracked or alerting, dimmed for background traffic."""
        alert = state in ("los", "conflict", "violation")
        rgb = STATE_COLORS[state] if alert else self._TAG_TEXT
        alpha = 0.6 if background else 1.0
        if tracked:
            frame = (*HIGHLIGHT, 0.95)
        elif alert:
            frame = (*STATE_COLORS[state], 0.85)
        else:
            frame = None
        priority = 3 if tracked else {"los": 2, "conflict": 1, "violation": 1}.get(state, 0)
        return AircraftTag(
            acid,
            float(sx),
            float(sy),
            tuple(lines),
            (*rgb, alpha),
            frame,
            (*base_rgb, 0.9 * alpha),
            priority,
        )

    def _project_many(self, lat_deg, lon_deg):
        """:meth:`_project` over arrays of latitudes and longitudes: the
        same arithmetic, all aircraft at once. Returns east and north as
        lists."""
        east_m = (lon_deg - self._lon0) * self._cos_lat0 * M_PER_DEG
        north_m = (lat_deg - self._lat0) * M_PER_DEG
        return east_m.tolist(), north_m.tolist()

    def set_static_labels_visible(self, visible: bool) -> None:
        for node in list(self._static_label_nodes):
            if node.isEmpty():
                self._static_label_nodes.remove(node)
            elif visible:
                node.show()
            else:
                node.hide()

    def _sync_trails(self, driver: Panda3DSimDriver) -> None:
        """Append new trail geometry; never rebuild what is already drawn.

        Trail cost used to grow with elapsed sim time: the cache was keyed
        on trail LENGTH, so one new point threw away the whole NodePath and
        re-walked every vertex through Python ``_project`` / ``LineSegs``
        calls plus a fresh GPU upload.  Summed over an episode that is
        O(T^2) per aircraft, and the work was pure waste - ``_project``
        yields camera-independent local-ENU meters, so a vertex's position
        never changes once appended.

        Now each aircraft's trail is chunked (:class:`_TrailGeometry`):
        completed chunks are sealed and left alone, and only a tail of at
        most ``_TRAIL_CHUNK`` vertices is rebuilt per sim step.  Camera
        orbit between steps still costs nothing at all.
        """
        if not driver.show_trails:
            if self._trail_geometry:
                for geometry in self._trail_geometry.values():
                    geometry.destroy()
                self._trail_geometry.clear()
            return

        # Prune geometry for aircraft no longer in the driver's dict
        # (deleted by env, or trails toggled off and back on).
        live = driver._trails
        cache = self._trail_geometry
        for acid in cache.keys() - live.keys():
            cache.pop(acid).destroy()

        for acid, trail in live.items():
            if trail.end - trail.start < 2:
                continue
            geometry = cache.get(acid)
            if geometry is None:
                geometry = cache[acid] = _TrailGeometry(self._trails_root)
            geometry.sync(trail, self._build_trail_node)

    def _build_trail_node(
        self,
        points: list[tuple[float, float, float, str]],
    ) -> NodePath:
        """Build a trail's multi-color polyline NodePath.

        Each ``(lat, lon, alt_ft, color_key)`` carries the state the
        aircraft was *in* when it flew that segment, so the trail
        renders normal/conflict/LoS regions in their respective
        palette colors.  Adjacent vertices with different keys share
        a single LineSegs primitive - Panda3D interpolates the
        per-vertex color along the segment, giving a soft transition
        across exactly the one segment that spans the state change.
        """
        from panda3d.core import LineSegs, TransparencyAttrib, Vec4

        ls = LineSegs()
        ls.setThickness(1.5)
        first = True
        for lat, lon, alt_ft, key in points:
            rgb = self._trail_palette(key)
            ls.setColor(Vec4(*rgb, 0.55))
            ex, ny, up = self._project(lat, lon, alt_ft * ft)
            if first:
                ls.moveTo(ex, ny, up)
                first = False
            else:
                ls.drawTo(ex, ny, up)
        np = self._trails_root.attachNewNode(ls.create())
        np.setTransparency(TransparencyAttrib.MAlpha)
        return np

    @staticmethod
    def _trail_palette(key: str) -> tuple[float, float, float]:
        """Resolve a trail point's color key against the world palette.

        Same vocabulary as :meth:`HumanSimDriver._resolve_trail_color`:
        ``"los"``, ``"conflict"``, ``"violation"``, and ``"normal"``
        map to the state palette. Anything else is treated as a named
        QueryRegion color and looked up in :data:`NAMED_COLORS` (with
        a grey fallback for unknown names so a typo is visible rather
        than crashing the renderer).
        """
        if key in STATE_COLORS:
            return STATE_COLORS[key]
        return _color(key)[:3]

    def _chevron_visual_size(
        self,
        driver: Panda3DSimDriver,
        x: float,
        y: float,
        z: float,
    ) -> float:
        """Half-length of the chevron in meters, scaled to camera distance.

        Sized to match :attr:`HorizontalView._AC_LENGTH_PX` (14 px
        tip-to-tail) so the panda3d chevron reads at the same on-screen
        size as the pygame one regardless of zoom.
        """
        return self._screen_scale(driver, x, y, z, factor=0.0051, minimum=220.0)

    @staticmethod
    def _screen_scale(
        driver: Panda3DSimDriver,
        x: float,
        y: float,
        z: float,
        *,
        factor: float,
        minimum: float,
    ) -> float:
        """Approximate stable screen sizing from camera-to-object distance."""
        distance = driver._distance
        try:
            cam_pos = driver._show.camera.getPos(driver._render)
            dx = x - cam_pos.getX()
            dy = y - cam_pos.getY()
            dz = z - cam_pos.getZ()
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        except Exception:
            pass
        return max(distance * factor, minimum)

    @staticmethod
    def _protection_zone_style(
        base_rgb: tuple[float, float, float],
        state: str,
    ) -> tuple[tuple[float, float, float, float], float]:
        """Color and line thickness of the RPZ x HPZ cylinder: they escalate
        with separation state - a LoS cylinder is unmistakable."""
        if state == "los":
            return (*STATE_COLORS["los"], 0.95), 2.5
        if state in ("conflict", "violation"):
            return (*STATE_COLORS[state], 0.85), 2.0
        return (*base_rgb, 0.45), 1.0

    def _draw_selected_route(self, driver: Panda3DSimDriver) -> None:
        if self._frame_root is None:
            return
        # "Show all routes" overlays the design's *defined* routes (resolved
        # once per episode, not per aircraft per frame); the tracked aircraft
        # still gets its live route + full readout on top.
        if getattr(driver, "show_all_routes", False):
            self._draw_defined_routes(driver)
        tracked = driver.tracked_acid()
        if tracked is not None:
            self._draw_aircraft_route(driver, tracked, with_labels=True)

    def _draw_defined_routes(self, driver: Panda3DSimDriver) -> None:
        """Faint polylines through the design's defined route waypoints."""
        from panda3d.core import LineSegs, Vec4

        for pts in driver.defined_route_polylines():
            ls = LineSegs()
            ls.setColor(Vec4(*NAMED_COLORS["cyan"], 0.7))
            ls.setThickness(1.5)
            first = True
            for lat, lon, alt_ft in pts:
                ex, ny, up = self._project(lat, lon, (alt_ft or 0.0) * ft)
                if first:
                    ls.moveTo(ex, ny, up)
                    first = False
                else:
                    ls.drawTo(ex, ny, up)
            self._frame_root.attachNewNode(ls.create())

    def _draw_aircraft_route(
        self, driver: Panda3DSimDriver, acid: str, with_labels: bool
    ) -> None:
        waypoints = driver.aircraft_route_waypoints(acid)
        if not waypoints:
            return

        from panda3d.core import LineSegs, Vec4

        idx = bs.traf.id.index(acid)
        current_alt_ft = bs.traf.alt[idx] / ft
        future = [wp for wp in waypoints if wp["future"]]
        if future:
            ls = LineSegs()
            ls.setColor(Vec4(*NAMED_COLORS["magenta"], 0.95))
            ls.setThickness(2.0)
            ex, ny, up = self._project(
                bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx]
            )
            ls.moveTo(ex, ny, up)
            for wp in future:
                alt_ft = wp["alt_ft"] if wp["alt_ft"] is not None else current_alt_ft
                ex, ny, up = self._project(wp["lat"], wp["lon"], alt_ft * ft)
                ls.drawTo(ex, ny, up)
            self._frame_root.attachNewNode(ls.create())

        radius = max(driver._distance * 0.003, 80.0)
        for wp in waypoints:
            alt_ft = wp["alt_ft"] if wp["alt_ft"] is not None else current_alt_ft
            ex, ny, up = self._project(wp["lat"], wp["lon"], alt_ft * ft)
            rgb = HIGHLIGHT if wp["active"] else NAMED_COLORS["magenta"]
            constraints = wp.get("constraints") or {}
            radius_nm = self._finite_constraint(
                constraints.get(WaypointReadoutKey.RADIUS_NM)
            )
            alt_tol_ft = self._finite_constraint(
                constraints.get(WaypointReadoutKey.ALT_TOLERANCE_FT)
            )
            if radius_nm is not None or alt_tol_ft is not None:
                self._draw_waypoint_constraints(
                    ex,
                    ny,
                    up,
                    radius_m=float(radius_nm or 0.0) * M_PER_NM,
                    alt_tolerance_m=float(alt_tol_ft or 0.0) * ft,
                )
            ls = LineSegs()
            ls.setColor(Vec4(*rgb, 1.0))
            ls.setThickness(2.0 if wp["active"] else 1.5)
            for i in range(33):
                t = 2 * math.pi * i / 32
                px = ex + radius * math.cos(t)
                py = ny + radius * math.sin(t)
                if i == 0:
                    ls.moveTo(px, py, up)
                else:
                    ls.drawTo(px, py, up)
            ls.moveTo(ex - radius, ny, up)
            ls.drawTo(ex + radius, ny, up)
            ls.moveTo(ex, ny - radius, up)
            ls.drawTo(ex, ny + radius, up)
            self._frame_root.attachNewNode(ls.create())

            metadata = wp.get("metadata") or {}
            display_index = int(wp.get("display_index", wp["index"]))
            label_lines = [
                metadata.get(WaypointReadoutKey.NAME)
                or wp.get("name")
                or f"WP{display_index + 1}"
            ]
            target_alt_ft = metadata.get(
                WaypointReadoutKey.TARGET_ALT_FT,
                wp.get("alt_ft"),
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
                            scheduled=bool(wp.get("scheduled"))
                            and WaypointReadoutKey.SPEED_KTS not in constraints
                            and WaypointReadoutKey.TARGET_SPEED_KTS not in metadata,
                        )
                    )
            label_lines.extend(driver.format_waypoint_time_lines(wp))
            if radius_nm is not None:
                label_lines.append(f"R    {float(radius_nm):.1f} NM")
            if alt_tol_ft is not None:
                label_lines.append(f"TOL  +/-{int(round(float(alt_tol_ft)))} FT")
            label = "\n".join(label_lines)
            if with_labels and label:
                self._attach_label(
                    driver,
                    self._frame_root,
                    str(label),
                    ex,
                    ny,
                    up + radius * 1.5,
                    (*rgb, 1.0),
                    screen_offset=(
                        ((display_index % 3) - 1) * 2.2,
                        1.4 + display_index * 1.0,
                    ),
                    draw_order=30 + display_index,
                )

    def _draw_waypoint_constraints(
        self,
        x: float,
        y: float,
        z: float,
        *,
        radius_m: float,
        alt_tolerance_m: float,
    ) -> None:
        if self._frame_root is None:
            return
        if radius_m <= 0.0 and alt_tolerance_m <= 0.0:
            return
        from panda3d.core import LineSegs, Vec4

        rgb = NAMED_COLORS["magenta"]
        ls = LineSegs()
        ls.setColor(Vec4(*rgb, 0.7))
        ls.setThickness(1.4)
        segs = 64
        if radius_m > 0.0:
            for level_z in {z, z - alt_tolerance_m, z + alt_tolerance_m}:
                for i in range(segs + 1):
                    t = 2 * math.pi * i / segs
                    px = x + radius_m * math.cos(t)
                    py = y + radius_m * math.sin(t)
                    if i == 0:
                        ls.moveTo(px, py, level_z)
                    else:
                        ls.drawTo(px, py, level_z)
        if alt_tolerance_m > 0.0:
            ls.moveTo(x, y, z - alt_tolerance_m)
            ls.drawTo(x, y, z + alt_tolerance_m)
            if radius_m > 0.0:
                for t in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
                    px = x + radius_m * math.cos(t)
                    py = y + radius_m * math.sin(t)
                    ls.moveTo(px, py, z - alt_tolerance_m)
                    ls.drawTo(px, py, z + alt_tolerance_m)
        self._frame_root.attachNewNode(ls.create())

    @staticmethod
    def _finite_constraint(value: object) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def _draw_highlight_ring(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        rgba: tuple[float, float, float],
    ) -> None:
        from panda3d.core import LineSegs, Vec4

        ls = LineSegs()
        ls.setColor(Vec4(*rgba, 1.0))
        ls.setThickness(2.0)
        segs = 32
        for i in range(segs + 1):
            t = 2 * math.pi * i / segs
            px = x + radius * math.cos(t)
            py = y + radius * math.sin(t)
            if i == 0:
                ls.moveTo(px, py, z)
            else:
                ls.drawTo(px, py, z)
        self._frame_root.attachNewNode(ls.create())

    def _world_to_screen(
        self,
        driver: Panda3DSimDriver,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float]:
        """Project a world point to (px, py) in window pixels.

        Returns ``(-1e9, -1e9)`` when behind the camera so the pick
        loop's distance check is guaranteed to miss it.
        """
        from panda3d.core import LPoint3, Point2

        cam = driver._show.camera
        lens = driver._show.camLens
        world_pt = LPoint3(x, y, z)
        cam_rel = cam.getRelativePoint(driver._render, world_pt)
        ndc = Point2()
        if not lens.project(cam_rel, ndc):
            return (-1e9, -1e9)
        win = driver._show.win
        w, h = win.getXSize(), win.getYSize()
        sx = (ndc.getX() + 1.0) * 0.5 * w
        sy = (1.0 - (ndc.getY() + 1.0) * 0.5) * h
        return sx, sy


# Vertices sealed into one immutable chunk NodePath.  Sized between the two
# costs it trades off: large enough that the per-step tail rebuild is short
# (the tail is at most this many vertices), small enough that sealing does not
# leave a long stretch of geometry being re-uploaded before it settles.  At 32
# a 720-step episode rebuilds ~32 vertices per step instead of ~720.
_TRAIL_CHUNK = 32


class _TrailGeometry:
    """One aircraft's trail as sealed chunks plus a live tail.

    Chunks are addressed in the trail's ABSOLUTE point indices and overlap by
    a single vertex, so consecutive chunks join without a visible gap and the
    tail starts on the last sealed vertex rather than beside it.

    Only ``[start, stable_end)`` is ever sealed: the trail's final point is
    provisional (decimation rewrites it in place while an aircraft holds a
    straight line), so it belongs to the tail, which is rebuilt anyway.
    """

    __slots__ = ("_revision", "_root", "_sealed", "_sealed_end", "_start", "_tail")

    def __init__(self, root: NodePath) -> None:
        self._root = root
        self._sealed: list[NodePath] = []
        self._sealed_end = 0
        self._start = 0
        self._tail: NodePath | None = None
        self._revision = -1

    def destroy(self) -> None:
        """Drop every NodePath this trail owns."""
        for node in self._sealed:
            node.removeNode()
        self._sealed.clear()
        if self._tail is not None:
            self._tail.removeNode()
            self._tail = None
        self._sealed_end = self._start
        self._revision = -1

    def sync(self, trail, build) -> None:
        """Seal whole chunks, then rebuild the short tail."""
        if self._revision == trail.revision:
            return  # cache hit - no sim step since the last frame

        # The cache is describing different points than the trail holds when
        # the point cap has evicted the front, or when this is a different
        # ``Trail`` object under the same callsign (toggled off and back on),
        # which shows up as a shorter trail or a revision running backwards.
        # Eviction is batched, so rebuilding costs one pass per batch rather
        # than per point, and it keeps the chunk index arithmetic honest.
        if (
            trail.start > self._start
            or trail.end < self._sealed_end
            or trail.revision < self._revision
        ):
            self.destroy()
            self._start = self._sealed_end = trail.start
        self._revision = trail.revision

        stable_end = trail.stable_end
        while self._sealed_end + _TRAIL_CHUNK < stable_end:
            lo = self._sealed_end
            hi = lo + _TRAIL_CHUNK
            # ``hi + 1`` so this chunk ends on the vertex the next one starts
            # from; without the shared vertex the trail breaks at every seam.
            self._sealed.append(build(trail.span(lo, hi + 1)))
            self._sealed_end = hi

        if self._tail is not None:
            self._tail.removeNode()
            self._tail = None
        tail = trail.span(self._sealed_end, trail.end)
        if len(tail) >= 2:
            self._tail = build(tail)


class _MarkerTemplates:
    """The geometry every aircraft marker is made of, built once at unit size:
    a marker copies it (sharing the vertices) and its transform places,
    turns and sizes it. Each part is linear in its size - the chevron in its
    half-length, the cylinder in its radius and half-height - so a scaled
    unit part is the part at that size."""

    # Line thicknesses of the protection-zone cylinder, by separation state.
    _PZ_THICKNESSES = (1.0, 2.0, 2.5)

    def __init__(self) -> None:
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
            LineSegs,
            NodePath,
        )

        # Chevron prism - eight vertices form a thin extrusion of pygame's
        # 4-point chevron silhouette - at a half-length of one meter.
        wing = CHEVRON_WING_FRAC
        notch = CHEVRON_NOTCH_FRAC
        th = 0.15
        vdata = GeomVertexData("chevron", GeomVertexFormat.getV3(), Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        vw.addData3(0, 1, th)
        vw.addData3(wing, -1, th)
        vw.addData3(0, -notch, th)
        vw.addData3(-wing, -1, th)
        vw.addData3(0, 1, -th)
        vw.addData3(wing, -1, -th)
        vw.addData3(0, -notch, -th)
        vw.addData3(-wing, -1, -th)
        tris = GeomTriangles(Geom.UHStatic)
        tris.addVertices(0, 1, 2)
        tris.addVertices(0, 2, 3)
        tris.addVertices(4, 6, 5)
        tris.addVertices(4, 7, 6)
        for top_a, top_b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            bot_a, bot_b = top_a + 4, top_b + 4
            tris.addVertices(top_a, top_b, bot_b)
            tris.addVertices(top_a, bot_b, bot_a)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("chevron")
        node.addGeom(geom)
        self.chevron = NodePath(node)
        self.chevron.setTwoSided(True)

        # A bright wireframe traces every edge of the prism so the
        # silhouette stays unambiguous against any background; the lines
        # are pulled out of the depth test so they always paint over the
        # coincident face triangles.
        outline = LineSegs()
        outline.setThickness(0.1)
        top = [(0, 1, th), (wing, -1, th), (0, -notch, th), (-wing, -1, th)]
        bot = [(lx, ly, -th) for lx, ly, _ in top]
        for ring in (top, bot):
            outline.moveTo(*ring[0])
            for p in ring[1:]:
                outline.drawTo(*p)
            outline.drawTo(*ring[0])
        for t, b in zip(top, bot):
            outline.moveTo(*t)
            outline.drawTo(*b)
        self.outline = NodePath(outline.create())
        self.outline.setDepthTest(False)
        self.outline.setDepthWrite(False)
        self.outline.setBin("fixed", 50)

        # Protection zone: top + bottom rings at z = +/-1 plus 8 vertical
        # struts so the cylinder reads as a volume - one per line thickness.
        self.cylinders = {}
        segs = 48
        ring_pts = [
            (math.cos(2 * math.pi * i / segs), math.sin(2 * math.pi * i / segs))
            for i in range(segs + 1)
        ]
        for thickness in self._PZ_THICKNESSES:
            ls = LineSegs()
            ls.setThickness(thickness)
            for ring_z in (-1.0, 1.0):
                for i, (px, py) in enumerate(ring_pts):
                    if i == 0:
                        ls.moveTo(px, py, ring_z)
                    else:
                        ls.drawTo(px, py, ring_z)
            for k in range(8):
                t = 2 * math.pi * k / 8
                ls.moveTo(math.cos(t), math.sin(t), -1.0)
                ls.drawTo(math.cos(t), math.sin(t), 1.0)
            self.cylinders[thickness] = NodePath(ls.create())

        # Depth pole from the ground up to the aircraft.
        pole = LineSegs()
        pole.setThickness(1.5)
        pole.moveTo(0, 0, 0)
        pole.drawTo(0, 0, 1)
        self.pole = NodePath(pole.create())

        # Ground mark where the pole stands: a unit ring and a cross.
        foot = LineSegs()
        foot.setThickness(1.5)
        for i in range(17):
            t = 2 * math.pi * i / 16
            (foot.moveTo if i == 0 else foot.drawTo)(math.cos(t), math.sin(t), 0.0)
        foot.moveTo(-0.5, 0, 0)
        foot.drawTo(0.5, 0, 0)
        foot.moveTo(0, -0.5, 0)
        foot.drawTo(0, 0.5, 0)
        self.foot = NodePath(foot.create())


class _AircraftMarker:
    """One aircraft's chevron, protection zone, depth pole and ground mark:
    copied from :class:`_MarkerTemplates` once, then only moved and recolored.
    Its data block is drawn flat on the screen (see :mod:`.tags`)."""

    __slots__ = (
        "_cylinder",
        "_thickness",
        "body",
        "foot",
        "outline",
        "pole",
        "root",
    )

    # The pole's height is a scale: kept off zero, which Panda3D cannot invert.
    _MIN_POLE_M = 1e-3
    # Lines carry their (template) color per vertex, which a node's color
    # replaces only at a higher priority.
    _OVER_VERTEX_COLOR = 1
    # The ground mark's radius, in chevron half-lengths.
    _FOOT_SCALE = 0.55

    def __init__(self, parent: NodePath, templates: _MarkerTemplates) -> None:
        self.root = parent.attachNewNode("marker")
        self.body = templates.chevron.copyTo(self.root)
        self.outline = templates.outline.copyTo(self.root)
        self.pole = templates.pole.copyTo(self.root)
        self.foot = templates.foot.copyTo(self.root)
        self._cylinder = None
        self._thickness = None

    def destroy(self) -> None:
        self.root.removeNode()

    def place_chevron(
        self,
        x: float,
        y: float,
        z: float,
        hdg_deg: float,
        half_length_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        """The chevron prism centered on the aircraft, along its heading;
        its outline the same hue lifted 40 % toward white, so the edges read
        as a highlight rather than a competing accent."""
        size = half_length_m
        self.body.setPosHprScale(x, y, z, -hdg_deg, 0.0, 0.0, size, size, size)
        self.body.setColor(*rgba)
        lift = 0.4
        self.outline.setPosHprScale(x, y, z, -hdg_deg, 0.0, 0.0, size, size, size)
        self.outline.setColor(
            rgba[0] + (1.0 - rgba[0]) * lift,
            rgba[1] + (1.0 - rgba[1]) * lift,
            rgba[2] + (1.0 - rgba[2]) * lift,
            1.0,
            self._OVER_VERTEX_COLOR,
        )

    def place_protection_zone(
        self,
        templates: _MarkerTemplates,
        x: float,
        y: float,
        z: float,
        radius_m: float,
        hpz_m: float,
        rgba: tuple[float, float, float, float],
        thickness: float,
    ) -> None:
        """The RPZ x HPZ cylinder around the aircraft."""
        if thickness != self._thickness:
            if self._cylinder is not None:
                self._cylinder.removeNode()
            self._cylinder = templates.cylinders[thickness].copyTo(self.root)
            self._thickness = thickness
        self._cylinder.setPosHprScale(x, y, z, 0.0, 0.0, 0.0, radius_m, radius_m, hpz_m)
        self._cylinder.setColor(*rgba, self._OVER_VERTEX_COLOR)

    def place_pole(
        self,
        x: float,
        y: float,
        z: float,
        rgba: tuple[float, float, float, float],
        half_length_m: float,
    ) -> None:
        """The pole from the ground up to the aircraft, and the mark where it
        stands - where the aircraft is over the ground, at the chevron's size."""
        self.pole.setPosHprScale(x, y, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, max(z, self._MIN_POLE_M))
        self.pole.setColor(*rgba, self._OVER_VERTEX_COLOR)
        size = half_length_m * self._FOOT_SCALE
        self.foot.setPosHprScale(x, y, 0.0, 0.0, 0.0, 0.0, size, size, 1.0)
        self.foot.setColor(*rgba, self._OVER_VERTEX_COLOR)
