"""Panda3DSimDriver - interactive 3D viewer for the BlueSky sandbox.

The driver is a third sibling to :class:`PygameSimDriver` and
:class:`QtGLSimDriver`.  It extends :class:`SandboxGUIDriver` (same
parent as :class:`PygameSimDriver`), consumes the same :mod:`render`
primitive vocabulary, and is wired into
:data:`bluesky_sandbox.ui.drivers.DRIVERS` under ``render_mode='panda3d'``.

Architecture
------------
The driver itself owns only window + event + camera + HUD-shell state.
Every piece of the 3D scene and every text overlay is encapsulated in a
:class:`Panda3DView` subclass (see :mod:`.views`):

* :class:`WorldView` - the 3D scene graph: lat/lon -> local-ENU
  projection, overlay polygons / points / polylines, live aircraft
  (chevron + protection-zone cylinder + label + depth pole), ground
  reference (grid + airspace bbox + north arrow).
* :class:`TSASView` - per-waypoint sequencing strips, rendered as a
  single right-aligned :class:`OnscreenText`.

Render-primitive dispatch (``draw_polygon`` / ``draw_point`` /
``draw_polyline``) is fanned out to every view that opted in; the
driver doesn't paint world geometry itself.  HUD elements that read
driver state directly (status badge, selected-aircraft info, key
hints) stay on the driver since they aren't view-specific.

Coordinates
-----------
Aircraft state lives in geodetic ``(lat, lon, alt)``.  The driver's
projection (in :class:`WorldView`) flattens to a local east-north-up
(ENU) tangent plane centred on the airspace using the small-region
approximation::

    east_m  = (lon - lon0) * cos(lat0_rad) * 111_320
    north_m = (lat  - lat0)               * 111_320
    up_m    = alt_ft * 0.3048

Distances are real metres in all three axes.

Camera & interaction
--------------------
Panda3D's default mouse driver is disabled and replaced with an orbit
camera around a movable focal point.

* Left-drag                  - orbit (azimuth / elevation).
* Right-drag                 - pan focal point along the ground plane.
* Mouse wheel                - dolly in / out.
* WASD or arrow keys         - strafe focal point.
* Q / E                      - raise / lower focal point.
* Left-click on aircraft     - select; HUD shows full info block + its route.
* Click on empty space       - clear selection (nothing is tracked by default).

Time controls mirror :class:`PygameSimDriver`:

* SPACE / P  - toggle pause.
* R          - toggle realtime <-> fast-time.
* + / -      - halve / double ``dtmult`` (realtime only).
* BACKSPACE  - mark every aircraft for deletion this step.
* SHIFT+BACKSPACE - abort the episode through the normal reset path.

View toggles:

* T - trails.
* L - static labels (region / spawn names).
* O - overlay the design's defined routes (in addition to the selected
      aircraft's live route).

Render loop
-----------
Panda3D drives its own render loop, but we never call ``base.run()``
because BlueSky owns the step cadence.  Each call to :meth:`step`
issues one ``bs.sim.step()`` (or ``update()``), then dispatches
``on_step`` to every view, then ``taskMgr.step()`` pumps events and
renders exactly one frame.
"""

from __future__ import annotations

import math
import time

from bluesky_sandbox.ui.drivers.common import (
    ViewPrimitiveFanoutMixin,
    preferred_panda3d_font_path,
)
from bluesky_sandbox.ui.drivers.panda3d.views import (
    Panda3DView,
    TSASView,
    WorldView,
)
from bluesky_sandbox.ui.drivers.sandbox_gui_driver import SandboxGUIDriver


class Panda3DSimDriver(ViewPrimitiveFanoutMixin, SandboxGUIDriver):
    """Interactive 3D viewer for the BlueSky multi-agent environment.

    Parameters
    ----------
    realtime:
        ``True`` (default) paces ``bs.sim`` against wall-clock time with
        ``dtmult``; ``False`` advances as fast as the CPU allows.
    window_size:
        ``(width, height)`` in pixels.
    views:
        Optional list of :class:`Panda3DView` instances composed into
        the viewer.  Defaults to ``[WorldView(), TSASView()]`` - the
        standard 3D-scene + waypoint-sequencing combo.  Pass a custom
        list to add / drop / reorder panels.
    """

    _PICK_RADIUS_PX = 22
    _HUD_MARGIN_X = 0.04
    _HUD_MARGIN_Y = 0.06

    def __init__(
        self,
        realtime: bool = True,
        window_size: tuple[int, int] = (1280, 800),
        views: list[Panda3DView] | None = None,
    ) -> None:
        # Import lazily so users who never request render_mode='panda3d'
        # don't need panda3d installed.  The import is heavyweight and
        # shells out to native graphics drivers.
        try:
            import panda3d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "render_mode='panda3d' requires panda3d. Install it with "
                '`pip install "bluesky-sandbox[panda3d]"`.'
            ) from e

        super().__init__(realtime=realtime)
        self.window_size = window_size

        # Default composition: 3D scene + waypoint sequencing strips.
        self._views: list[Panda3DView] = (
            list(views) if views is not None else [WorldView(), TSASView()]
        )
        self._world_view: WorldView | None = next(
            (v for v in self._views if isinstance(v, WorldView)), None,
        )

        # Panda objects - created in start().
        self._show = None
        self._render = None
        self._hud_root = None
        self._status_text = None
        self._info_text = None
        self._ui_font = None

        # Orbit camera state - focal point in ENU metres.
        self._focal = [0.0, 0.0, 0.0]
        self._azimuth = 45.0
        self._elevation = 30.0
        self._distance = 100_000

        # Mouse drag state.
        self._drag_kind: str | None = None
        self._drag_mouse: tuple[float, float] = (0.0, 0.0)
        self._drag_started_at_ms: int = 0
        self._drag_view: Panda3DView | None = None

        # Held-key state for WASD/QE focal-point motion.
        self._held: set[str] = set()

        # Selection - bs.traf.id of the currently-selected aircraft.
        self._selected: str | None = None

    @property
    def primitive_targets(self) -> list[Panda3DView]:
        """Views that should receive static render primitives."""
        return self._views

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the Panda3D window and let every view set itself up."""
        super().start()
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import (
            ConfigVariableBool,
            ConfigVariableString,
            WindowProperties,
        )

        ConfigVariableString("audio-library-name").setValue("null")
        ConfigVariableBool("show-frame-rate-meter").setValue(False)
        ConfigVariableBool("sync-video").setValue(False)

        self._show = ShowBase()
        self._show.disableMouse()

        # Panda3D's default lens has near=1, far=1000.  Our world is in
        # metres at airspace scale (~100s of km) so the default would
        # clip everything beyond 1 km - i.e. the whole scene.
        self._show.camLens.setNearFar(10.0, 10_000_000.0)
        self._show.camLens.setFov(50.0)

        # Background colour set the moment the window opens so the
        # first frame doesn't flash Panda's default grey.
        self._show.setBackgroundColor(0.07, 0.10, 0.14, 1.0)

        props = WindowProperties()
        props.setTitle("BlueSky Sandbox - 3D")
        props.setSize(*self.window_size)
        self._show.win.requestProperties(props)

        self._render = self._show.render
        self._ui_font = self._load_ui_font()
        self._setup_hud()
        self._bind_events()

        for view in self._views:
            view.on_start(self)

        self._update_camera()

    def on_reset(self, env=None) -> None:
        """Cache env, recentre camera on the airspace, dispatch to views."""
        if env is not None:
            self.bind_env(env)
        if self._env is None:
            raise RuntimeError("Panda3DSimDriver env has not been bound.")
        # The Panda3D window and scene graph are owned by ``start()``;
        # view.on_reset reads ``self._render`` to build node hierarchies,
        # so the window must exist before this method walks the views.
        # Lazy-open it here so ``env.reset()`` works regardless of
        # whether ``env.render()`` has been called yet.
        if not self._started:
            self.start()
        self._clear_trails()
        # New episode = new traffic, so drop the per-snapshot caches (esp. the
        # acid->index map) or stale episode-1 callsigns mask episode-2 routes.
        self._clear_aircraft_snapshot_cache()
        self._defined_routes_cache = None  # episode-static; re-resolve next access

        airspace = self._env.episode_airspace_bounds
        if airspace is not None:
            (lat_min, lat_max), (lon_min, lon_max) = airspace.bounding_box
        else:
            spawn_bb = self._env.episode_spawn.resolved_bounds
            lat_min, lat_max = spawn_bb["lat_deg"]
            lon_min, lon_max = spawn_bb["lon_deg"]

        # Camera distance + focal - fits the airspace diagonal with margin.
        cos_lat0 = math.cos(math.radians(0.5 * (lat_min + lat_max)))
        diag_lat_m = abs(lat_max - lat_min) * 111_320.0
        diag_lon_m = abs(lon_max - lon_min) * 111_320.0 * cos_lat0
        diag = math.hypot(diag_lat_m, diag_lon_m)
        self._distance = max(diag * 1.5, 20_000.0)
        self._focal = [0.0, 0.0, 0.0]
        self._selected = None

        for view in self._views:
            view.on_reset(self, self._env)

        # Now that views have projected, fan render primitives out to
        # every interested view.
        self.draw_renderables(self._env._renderable_builder.iter_renderables())

        self._update_camera()

        # Reset can happen before the first env.step(). Populate live
        # aircraft markers and HUD views immediately so the first render
        # after reset is not a static-only scene. Camera must already
        # be current because WorldView also refreshes its pick cache.
        self._dispatch_step()
        self._refresh_hud()

    def close(self) -> None:
        """Tear down each view, then destroy the Panda3D window."""
        for view in self._views:
            try:
                view.close()
            except Exception:
                pass
        if self._show is not None:
            try:
                self._show.destroy()
            except Exception:
                pass
            self._show = None
            self._render = None
            self._hud_root = None
            self._status_text = None
            self._info_text = None
            self._ui_font = None

    # ------------------------------------------------------------------
    # Sim integration
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Process events + render one frame without advancing the sim."""
        if self._show is None:
            return
        self._apply_held_motion()
        self._dispatch_step()
        self._refresh_hud()
        self._show.taskMgr.step()

    def step(self) -> None:
        """Advance the sim + render one frame, blocking while paused."""
        if self._show is None:
            super().step()
            return

        while self._paused:
            self._apply_held_motion()
            self._dispatch_step()
            self._refresh_hud()
            self._show.taskMgr.step()
            if self._show is None:
                return

        self._advance_sim()
        self._advance_trails()
        # Shared wall-clock cadence gate (see HumanSimDriver._render_due): during
        # fast-forward the sim advances every substep but the expensive scene
        # refresh + frame draw fire at most render_fps, so throughput isn't
        # capped by rendering. Input is processed inside taskMgr.step(), so it is
        # gated too - the sub-frame latency that adds is imperceptible.
        if self._render_due():
            self._apply_held_motion()
            self._dispatch_step()
            self._refresh_hud()
            self._show.taskMgr.step()
        # In realtime mode, idle out this substep's wall-clock budget while
        # keeping the window drawing + responsive (fixes slow-mo freezing).
        self._wait_realtime()

    def _draw_idle_frame(self) -> None:
        if self._show is None:
            return
        if self._render_due():
            self._apply_held_motion()
            self._dispatch_step()
            self._refresh_hud()
            self._show.taskMgr.step()

    def _dispatch_step(self) -> None:
        """Fan ``on_step`` out to every view."""
        with self._aircraft_snapshot_cache_scope():
            for view in self._views:
                view.on_step(self)

    # ------------------------------------------------------------------
    # HUD shell (status, info)
    # ------------------------------------------------------------------

    def _load_ui_font(self):
        path = preferred_panda3d_font_path()
        if path is None:
            return None
        try:
            return self._show.loader.loadFont(path)
        except Exception:
            return None

    def _setup_hud(self) -> None:
        """Status badge and selected-aircraft info block.

        These three pieces read driver-owned state (``self.realtime``,
        ``self._selected``, etc.) so they live on the driver rather
        than being broken out into their own views.  Per-view HUD -
        the right-side TSAS column - is created by :class:`TSASView`
        itself.
        """
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self._hud_root = self._show.aspect2d
        font_kwargs = {"font": self._ui_font} if self._ui_font is not None else {}

        self._status_text = OnscreenText(
            text="",
            parent=self._hud_root,
            pos=(1.20, -0.92),
            scale=0.045,
            fg=(0.95, 0.95, 1.0, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55),
            align=TextNode.ARight,
            mayChange=True,
            **font_kwargs,
        )

        self._info_text = OnscreenText(
            text="",
            parent=self._hud_root,
            pos=(-1.30, 0.92),
            scale=0.045,
            fg=(0.95, 0.95, 1.0, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55),
            align=TextNode.ALeft,
            mayChange=True,
            **font_kwargs,
        )
        self._layout_hud()

    def _refresh_hud(self) -> None:
        if self._status_text is None:
            return
        with self._aircraft_snapshot_cache_scope():
            self._layout_hud()
            self._status_text.setText(self.format_status_line())
            self._info_text.setText(self._info_block())

    def _layout_hud(self) -> None:
        if self._show is None or self._status_text is None or self._info_text is None:
            return
        aspect = self._show.getAspectRatio()
        self._status_text.setPos(
            aspect - self._HUD_MARGIN_X,
            -1.0 + self._HUD_MARGIN_Y,
        )
        self._info_text.setPos(
            -aspect + self._HUD_MARGIN_X,
            1.0 - self._HUD_MARGIN_Y,
        )

    def _info_block(self) -> str:
        tracked = self.tracked_acid()
        if tracked is None:
            return ""
        return "\n".join(self.format_aircraft_info_lines(tracked))

    # ------------------------------------------------------------------
    # Camera & input
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        sb = self._show
        sb.accept("mouse1",     self._on_left_down)
        sb.accept("mouse1-up",  self._on_left_up)
        sb.accept("mouse3",     self._on_right_down)
        sb.accept("mouse3-up",  self._on_right_up)
        sb.accept("wheel_up",   lambda: self._zoom(0.85))
        sb.accept("wheel_down", lambda: self._zoom(1.18))

        sb.accept("space", self.toggle_pause)
        sb.accept("p",     self.toggle_pause)
        sb.accept("r",     self.toggle_realtime)
        sb.accept("t",     self.toggle_trails)
        sb.accept("l",     self.toggle_labels)
        sb.accept("o",     self.toggle_all_routes)
        sb.accept("+",     lambda: self.scale_dtmult(2.0))
        sb.accept("=",     lambda: self.scale_dtmult(2.0))
        sb.accept("-",     lambda: self.scale_dtmult(0.5))
        sb.accept("backspace", self.delete_all_aircraft)
        sb.accept("shift-backspace", self.request_episode_reset)

        for key, marker in [
            ("w", "fwd"), ("arrow_up", "fwd"),
            ("s", "back"), ("arrow_down", "back"),
            ("a", "left"), ("arrow_left", "left"),
            ("d", "right"), ("arrow_right", "right"),
            ("q", "down"),
            ("e", "up"),
        ]:
            sb.accept(key,           self._hold,    [marker])
            sb.accept(f"{key}-up",   self._release, [marker])

        sb.taskMgr.add(self._camera_task, "panda3d_sim_driver_camera")
        sb.win.setCloseRequestEvent("panda3d_window_close")
        sb.accept("panda3d_window_close", self._on_window_close)

    def toggle_labels(self) -> None:
        super().toggle_labels()
        for view in self._views:
            setter = getattr(view, "set_static_labels_visible", None)
            if setter is not None:
                setter(self.show_labels)

    def _on_window_close(self) -> None:
        self.close()
        raise SystemExit("BlueSky 3D window closed")

    def _camera_task(self, task):
        # Continually update the camera so orbit drag feels live even
        # when the sim is paused.
        self._update_camera()
        if self._drag_kind == "orbit":
            self._apply_orbit_drag()
        elif self._drag_kind == "pan":
            self._apply_pan_drag()
        elif self._drag_kind == "view" and self._drag_view is not None:
            pos = self._mouse_aspect_pos()
            if pos is not None:
                self._drag_view.on_mouse_drag(self, pos)
        self._layout_hud()
        return task.cont

    def _on_left_down(self) -> None:
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return
        aspect_pos = self._mouse_aspect_pos()
        if aspect_pos is not None:
            for view in reversed(self._views):
                if view.on_mouse_down(self, aspect_pos):
                    self._drag_kind = "view"
                    self._drag_view = view
                    self._drag_mouse = (mw.getMouseX(), mw.getMouseY())
                    self._drag_started_at_ms = int(time.monotonic() * 1000)
                    return
        self._drag_kind = "orbit"
        self._drag_mouse = (mw.getMouseX(), mw.getMouseY())
        self._drag_started_at_ms = int(time.monotonic() * 1000)

    def _on_left_up(self) -> None:
        elapsed = int(time.monotonic() * 1000) - self._drag_started_at_ms
        mw = self._show.mouseWatcherNode
        moved = 0.0
        if mw.hasMouse():
            mx, my = mw.getMouseX(), mw.getMouseY()
            moved = math.hypot(mx - self._drag_mouse[0], my - self._drag_mouse[1])
        if self._drag_kind == "view":
            view = self._drag_view
            self._drag_kind = None
            self._drag_view = None
            aspect_pos = self._mouse_aspect_pos()
            if view is not None and aspect_pos is not None:
                view.on_mouse_up(self, aspect_pos)
            return
        self._drag_kind = None
        if elapsed < 250 and moved < 0.01:
            self._pick_aircraft()

    def _on_right_down(self) -> None:
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return
        self._drag_kind = "pan"
        self._drag_mouse = (mw.getMouseX(), mw.getMouseY())

    def _on_right_up(self) -> None:
        if self._drag_kind == "pan":
            self._drag_kind = None

    def _apply_orbit_drag(self) -> None:
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return
        mx, my = mw.getMouseX(), mw.getMouseY()
        dx = mx - self._drag_mouse[0]
        dy = my - self._drag_mouse[1]
        self._azimuth -= dx * 90.0
        self._elevation += dy * 90.0
        self._elevation = max(-89.0, min(89.0, self._elevation))
        self._drag_mouse = (mx, my)

    def _apply_pan_drag(self) -> None:
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return
        mx, my = mw.getMouseX(), mw.getMouseY()
        dx = mx - self._drag_mouse[0]
        dy = my - self._drag_mouse[1]
        scale = self._distance * 0.6
        right, fwd = self._camera_ground_basis()
        self._focal[0] -= (right[0] * dx + fwd[0] * dy) * scale
        self._focal[1] -= (right[1] * dx + fwd[1] * dy) * scale
        self._drag_mouse = (mx, my)

    def _camera_ground_basis(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``(right_xy, forward_xy)`` for the current camera azimuth.

        Both vectors are unit-length and live on the ground plane.
        Derived consistently with :meth:`_update_camera` - the camera
        is placed at focal + distance*(cos_el*sin(az),
        -cos_el*cos(az), sin(el)), so its forward direction projected
        onto the ground is ``(-sin(az), cos(az))`` and its right
        (forward x up) is ``(cos(az), sin(az))``.
        """
        az = math.radians(self._azimuth)
        right = (math.cos(az),  math.sin(az))
        fwd   = (-math.sin(az), math.cos(az))
        return right, fwd

    def _zoom(self, factor: float) -> None:
        self._distance = max(2_000.0, min(self._distance * factor, 3_000_000.0))

    def _mouse_aspect_pos(self) -> tuple[float, float] | None:
        if self._show is None:
            return None
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return None
        return (
            mw.getMouseX() * self._show.getAspectRatio(),
            mw.getMouseY(),
        )

    def _hold(self, marker: str) -> None:
        self._held.add(marker)

    def _release(self, marker: str) -> None:
        self._held.discard(marker)

    def _apply_held_motion(self) -> None:
        if not self._held:
            return
        step = self._distance * 0.015
        right, fwd = self._camera_ground_basis()
        if "fwd"   in self._held: self._focal[0] += fwd[0]   * step; self._focal[1] += fwd[1]   * step
        if "back"  in self._held: self._focal[0] -= fwd[0]   * step; self._focal[1] -= fwd[1]   * step
        if "right" in self._held: self._focal[0] += right[0] * step; self._focal[1] += right[1] * step
        if "left"  in self._held: self._focal[0] -= right[0] * step; self._focal[1] -= right[1] * step
        if "up"    in self._held: self._focal[2] += step
        if "down"  in self._held: self._focal[2] -= step

    def _update_camera(self) -> None:
        if self._show is None:
            return
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        cos_el = math.cos(el)
        cx = self._focal[0] + self._distance * cos_el * math.sin(az)
        cy = self._focal[1] - self._distance * cos_el * math.cos(az)
        cz = self._focal[2] + self._distance * math.sin(el)
        self._show.camera.setPos(cx, cy, cz)
        self._show.camera.lookAt(*self._focal)

    # ------------------------------------------------------------------
    # Selection / picking
    # ------------------------------------------------------------------

    def _pick_aircraft(self) -> None:
        """Delegate the pick to :class:`WorldView` (it owns the cache)."""
        if self._world_view is None:
            return
        mw = self._show.mouseWatcherNode
        if not mw.hasMouse():
            return
        win = self._show.win
        w, h = win.getXSize(), win.getYSize()
        mx_ndc, my_ndc = mw.getMouseX(), mw.getMouseY()
        mx_px = (mx_ndc + 1.0) * 0.5 * w
        my_px = (1.0 - (my_ndc + 1.0) * 0.5) * h
        self._selected = self._world_view.pick(self, mx_px, my_px)
