"""PygameSimDriver - composes :class:`PygameView`s into a window with drag-able layout.

The driver owns:

* the pygame window, event loop, and integration with the BlueSky sim;
* a layout *tree* (see :mod:`layout`) - leaves are :class:`PygameView`
  instances, splits divide a rect horizontally or vertically.  The
  user provides the initial layout via the ``views=`` kwarg as either
  a flat tuple (vertical stack - legacy default) or a nested tuple
  (alternates orientation per nesting level), or with explicit
  :func:`HSplit` / :func:`VSplit` tags;
* runtime layout edits - drag splitter dividers to resize panels, or
  click-and-hold a panel's header bar and drop on another panel's edge
  to rearrange the layout.  Drop on the centre to swap two views;
* common utilities (text rendering, info tooltip, status badge);
* cross-view orchestration: collecting hover hits from each view,
  picking a winner, asking every view to highlight the same aircraft,
  drawing connecting lines between markers across views.

Examples::

    # Default: vertical stack of profile + plan.
    PygameSimDriver()

    # Profile + TSAS stacked on top of plan.
    PygameSimDriver(views=(VerticalView, TSASView, HorizontalView))

    # Plan on the left, TSAS on the right; whole thing under a profile strip.
    from bluesky_sandbox.ui.drivers.pygame.layout import HSplit, VSplit
    PygameSimDriver(views=VSplit(VerticalView, HSplit(HorizontalView, TSASView)))

Mouse:

* Drag a splitter divider - resize.
* Drag a panel header - rearrange.  While dragging, the cursor shows a
  drop indicator on the targeted panel; release on top/bottom/left/
  right to split the target, on centre to swap.
* Mouse wheel - zoom the plan/profile under the cursor.
* Drag a plan/profile - pan that view.
* Right-drag or middle-drag also pans without click-select.
* Home / 0 - reset plan/profile zoom and pan.
"""

from __future__ import annotations

import itertools
import math
import os

import bluesky as bs
import numpy as np
import pygame

from bluesky_sandbox.ui.drivers.common import (
    UI_FONT_NAMES,
    CursorHint,
    CursorHintName,
    ViewPrimitiveFanoutMixin,
    preferred_ui_font_path,
)
from bluesky_sandbox.ui.drivers.pygame import colors as C
from bluesky_sandbox.ui.drivers.pygame import layout as L
from bluesky_sandbox.ui.drivers.pygame.layout import Leaf
from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView
from bluesky_sandbox.ui.drivers.pygame.views.horizontal import HorizontalView
from bluesky_sandbox.ui.drivers.pygame.views.vertical import VerticalView
from bluesky_sandbox.ui.drivers.sandbox_gui_driver import SandboxGUIDriver

# Hit-zone around each splitter divider for drag-resize, in pixels.
_DIVIDER_HIT_PX = 5
# Minimum view height/width after a drag, so a panel can never disappear.
_MIN_VIEW_PX = 40
# Mouse movement before a left press becomes viewport panning instead of click-select.
_PAN_START_PX = 4


_PYGAME_CURSORS = {
    CursorHint.POINT: pygame.SYSTEM_CURSOR_HAND,
    CursorHint.MOVE: pygame.SYSTEM_CURSOR_SIZEALL,
    CursorHint.RESIZE_X: pygame.SYSTEM_CURSOR_SIZEWE,
    CursorHint.RESIZE_Y: pygame.SYSTEM_CURSOR_SIZENS,
}


class _CachedFont:
    """A ``pygame.font.Font`` wrapper that memoizes ``render``.

    Glyph rasterization is the dominant per-frame cost in the software
    renderer, and the same strings/colours recur every frame (callsigns,
    headers, HUD, region labels). Caching the rendered ``Surface`` keyed by
    ``(text, antialias, colour, background)`` turns a re-raster into a dict
    lookup. Every other attribute (``size``, ``set_bold``, ``get_height`` ...)
    proxies straight through to the wrapped font.

    Returned surfaces are shared, so callers must blit *from* them and never
    draw onto them - which is how every view already uses the result.
    """

    # Cap so long-running sessions with churning text (clock, coords) don't
    # grow the cache without bound; clearing wholesale is cheap and rare.
    _MAX_ENTRIES = 4096

    def __init__(self, font: pygame.font.Font) -> None:
        self._font = font
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(self, text, antialias=True, color=(0, 0, 0), background=None):
        bg_key = tuple(background) if background is not None else None
        key = (text, bool(antialias), tuple(color), bg_key)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) >= self._MAX_ENTRIES:
                self._cache.clear()
            surf = (
                self._font.render(text, antialias, color, background)
                if background is not None
                else self._font.render(text, antialias, color)
            )
            self._cache[key] = surf
        return surf

    def set_bold(self, value: bool) -> None:
        self._font.set_bold(value)
        self._cache.clear()

    def __getattr__(self, name):
        # Proxy size(), get_height(), metrics(), etc. to the real font.
        return getattr(self._font, name)


class PygameSimDriver(ViewPrimitiveFanoutMixin, SandboxGUIDriver):
    """Pygame visualizer composed of one or more :class:`PygameView` panels."""

    _primitive_view_methods = {
        "polygon": "add_polygon",
        "point": "add_point",
        "polyline": "add_polyline",
    }

    LABEL_OFFSET = (12, -20)
    LABEL_FONT_SIZE = 16
    LABEL_BG = (255, 255, 255, 210)
    LABEL_PAD = 3

    HEADER_HEIGHT = 20  # px reserved at top of each view for the drag handle
    HEADER_BG = (50, 60, 80)
    HEADER_FG = (245, 245, 250)
    HEADER_BUTTON_BG = (74, 83, 105)
    HEADER_BUTTON_OFF_BG = (42, 48, 62)
    HEADER_BUTTON_W = 38
    HEADER_BUTTON_PAD = 4
    DROP_PREVIEW_BG = (255, 215, 30, 90)  # semi-transparent yellow

    def __init__(
        self,
        realtime: bool = True,
        views=None,
        window_size: tuple[int, int] = (1100, 1280),
        fps: int = 60,
        show_callsigns: bool = True,
    ) -> None:
        super().__init__(realtime=realtime)
        self.window_size = window_size
        self.fps = fps
        self.show_callsigns = show_callsigns

        if views is None:
            views = (VerticalView, HorizontalView)
        self._layout: L.Node = L.parse(views)

        self._window: pygame.Surface | None = None
        # Persistent off-screen canvas, reused across frames (reallocated only
        # on resize) so we don't allocate a full-window Surface every frame.
        self._canvas: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self._header_font: pygame.font.Font | None = None

        # Drag-resize state.  When non-None: (split, child_index,
        # mouse_pos_at_start, fractions_at_start).
        self._drag_resize: tuple[L.Split, int, tuple[int, int], list[float]] | None = (
            None
        )

        # Drag-rearrange state.  When non-None: the leaf being dragged.
        self._drag_view: L.Leaf | None = None
        # Drop target during drag-rearrange: (target_leaf, zone) or None.
        self._drop_target: tuple[L.Leaf, str] | None = None

        # View-private drag state.  When non-None: (view, opaque_handle).
        # Set by ``hit_test_drag`` on a view; consumed in motion/up.
        self._view_drag: tuple[PygameView, object] | None = None
        self._view_drag_cursor: CursorHintName | None = None

        # Viewport pan state.  When non-None: (view, previous_mouse_pos).
        self._viewport_drag: tuple[PygameView, tuple[int, int]] | None = None
        # Left-button viewport candidate.  Becomes `_viewport_drag` only
        # after a small movement threshold, so simple clicks still select.
        self._pending_viewport_pan: (
            tuple[PygameView, tuple[int, int], tuple[int, int]] | None
        ) = None

        # bs.traf.id of the aircraft selected by a click.
        self._selected: str | None = None

        # Video recording - driven by env vars so callers don't have to
        # thread a record_to= param through the make_env / vec_env stack.
        # Set BLUESKY_RECORD_VIDEO=/path/to.mp4 (and optionally
        # BLUESKY_RECORD_FPS) before the driver is started; the driver
        # then forces SDL into the dummy display, opens an imageio writer
        # in start(), and appends one frame per sim step (throttle off).
        self._video_writer = None
        self._record_path: str | None = os.environ.get("BLUESKY_RECORD_VIDEO") or None
        self._record_fps: int = int(os.environ.get("BLUESKY_RECORD_FPS", "30"))
        self._record_quality: int = int(os.environ.get("BLUESKY_RECORD_QUALITY", "8"))
        if self._record_path:
            self.auto_track = True
        record_size = os.environ.get("BLUESKY_RECORD_SIZE")
        if self._record_path and record_size:
            try:
                width, height = (int(part) for part in record_size.lower().split("x"))
            except ValueError:
                width, height = self.window_size
            if width > 0 and height > 0:
                self.window_size = (width, height)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def views(self) -> list[PygameView]:
        """All views in the layout, in tree order (top-to-bottom, left-to-right)."""
        return [leaf.view for leaf in L.iter_leaves(self._layout)]

    @property
    def primitive_targets(self) -> list[PygameView]:
        """Views that should receive static render primitives."""
        return self.views

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        # Headless SDL when recording - no on-screen window, but the
        # canvas Surface is still rendered and readable via surfarray.
        # Must happen before pygame.display.init().
        if self._record_path:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self._window = pygame.display.set_mode(self.window_size, self._display_flags())
        pygame.display.set_caption("BlueSky Sandbox")
        self._clock = pygame.time.Clock()
        self.font = self._make_font(self.LABEL_FONT_SIZE, bold=True)
        self._header_font = self._make_font(12, bold=True)

        if self._record_path:
            # Imported lazily so non-recording runs don't import imageio.
            # optional extra: [recording]
            import imageio  # noqa: PLC0415

            self._video_writer = imageio.get_writer(
                self._record_path,
                fps=self._record_fps,
                codec="libx264",
                quality=max(0, min(10, self._record_quality)),
                macro_block_size=1,
                output_params=["-r", str(self._record_fps)],
            )

    def on_reset(self, env=None) -> None:
        """Lay out view rects, run each view's on_reset, then dispatch renderables."""
        if env is not None:
            self.bind_env(env)
        if self._env is None:
            raise RuntimeError("PygameSimDriver env has not been bound.")
        self._compute_layout()
        self._clear_trails()
        # New episode = new traffic, so drop the per-snapshot caches (esp. the
        # acid->index map) or stale episode-1 callsigns mask episode-2 routes.
        self._clear_aircraft_snapshot_cache()
        self._defined_routes_cache = None  # episode-static; re-resolve next access
        self._selected = None
        for view in self.views:
            view.on_reset(self, self._env)
        self.draw_renderables(self._env._renderable_builder.iter_renderables())

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
            self.font = None
            self._header_font = None

    def _display_flags(self) -> int:
        # DOUBLEBUF gives a cheaper full-screen present, but recording reads the
        # framebuffer back via surfarray, which would see the swapped-out buffer
        # after a flip - so keep a single buffer when recording.
        flags = pygame.RESIZABLE
        if self._video_writer is None and not self._record_path:
            flags |= pygame.DOUBLEBUF
        return flags

    def _make_font(self, size: int, *, bold: bool = False) -> pygame.font.Font:
        return _CachedFont(self._make_raw_font(size, bold=bold))

    def _make_raw_font(self, size: int, *, bold: bool = False) -> pygame.font.Font:
        path = preferred_ui_font_path()
        if path is not None:
            try:
                font = pygame.font.Font(path, size)
                font.set_bold(bold)
                return font
            except Exception:
                pass
        for name in UI_FONT_NAMES:
            path = pygame.font.match_font(name, bold=bold)
            if path is None:
                continue
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
        return pygame.font.SysFont(None, size, bold=bold)

    # ------------------------------------------------------------------
    # Sim integration
    # ------------------------------------------------------------------

    def update(self) -> None:
        if self._window is None:
            return
        self._pump_events()
        self._render_frame()
        self._clock.tick(self.fps)

    def step(self) -> None:
        self._pump_events()
        while self._paused:
            self._render_frame()
            if self._clock is not None:
                self._clock.tick(self.fps)
            self._pump_events()
        self._advance_sim()
        self._advance_trails()
        self._render_throttled()
        # In realtime mode, idle out this substep's wall-clock budget while
        # keeping the window drawing + responsive (fixes slow-mo freezing).
        self._wait_realtime()

    def _draw_idle_frame(self) -> None:
        if self._window is None:
            return
        self._pump_events()
        # Don't capture idle frames into a recording (mirrors _render_throttled).
        if self._video_writer is not None:
            return
        if self._render_due():
            self._render_frame()

    def _render_throttled(self) -> None:
        if self._window is None:
            return
        # When recording, skip per-substep rendering. With dt=10 s and
        # simdt=0.05 s, ``env.step()`` invokes ``driver.step()`` 200
        # times - capturing a frame each time would yield 200x the
        # intended frame count. The explicit ``vec_env.render()`` call
        # per env-step (from the eval loop) is the single capture point.
        if self._video_writer is not None:
            return
        # Shared wall-clock cadence gate (see HumanSimDriver._render_due) so the
        # frame draw is decoupled from the substep loop during fast-forward.
        if self._render_due():
            self._render_frame()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _pump_events(self) -> None:
        if self._window is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("BlueSky pygame window closed")
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.size)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_p, pygame.K_SPACE):
                    self.toggle_pause()
                elif event.key == pygame.K_r:
                    self.toggle_realtime()
                elif event.key == pygame.K_t:
                    # Common API on HumanSimDriver - also bound to T
                    # on the panda3d driver, and forwarded to BlueSky's
                    # TRAIL command on qtgl.
                    self.toggle_trails()
                elif event.key == pygame.K_l:
                    self.toggle_labels()
                elif event.key == pygame.K_o:
                    # Show every aircraft's route, not just the selected one.
                    self.toggle_all_routes()
                elif event.key == pygame.K_v:
                    # Velocity-obstacle overlay for the tracked aircraft.
                    self.toggle_velocity_obstacles()
                elif event.key == pygame.K_BACKSPACE:
                    mods = pygame.key.get_mods()
                    if mods & (pygame.KMOD_SHIFT | pygame.KMOD_CTRL):
                        self.request_episode_reset()
                    else:
                        self.delete_all_aircraft()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.scale_dtmult(2.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.scale_dtmult(0.5)
                elif event.key in (pygame.K_HOME, pygame.K_0, pygame.K_KP0):
                    self._reset_viewports()
            elif event.type == pygame.MOUSEWHEEL:
                self._on_mouse_wheel(event.y, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._on_mouse_up(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (2, 3):
                self._on_viewport_pan_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
                self._on_viewport_pan_up()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                self._on_mouse_wheel(1 if event.button == 4 else -1, event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self._on_mouse_motion(event.pos)

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self._viewport_drag is not None:
            return
        self._pending_viewport_pan = None
        if self._label_button_at(pos) is not None:
            self.toggle_labels()
            return
        if self._vo_button_at(pos) is not None:
            self.toggle_velocity_obstacles()
            return
        # 1. Per-view widgets (e.g. VerticalView's bottom rotate handle)
        #    get first dibs so they can override the layout-level drags.
        for view in self.views:
            if not view.rect.collidepoint(pos):
                continue
            handle = view.hit_test_drag(pos, self)
            if handle is not None:
                self._view_drag = (view, handle)
                self._view_drag_cursor = view.cursor_hint(pos, self) or CursorHint.POINT
                self._set_cursor_hint(self._view_drag_cursor)
                return
        # 2. Header (drag-rearrange) takes priority over divider (drag-resize).
        leaf = self._leaf_with_header_at(pos)
        if leaf is not None:
            self._drag_view = leaf
            self._set_cursor_hint(CursorHint.POINT)
            return
        hit = self._divider_at(pos)
        if hit is not None:
            split, idx = hit
            self._drag_resize = (split, idx, pos, list(split.fractions))
            self._set_cursor_hint(
                CursorHint.RESIZE_Y if split.orientation == "v" else CursorHint.RESIZE_X
            )
            return
        view = self._viewport_view_at(pos)
        if view is not None:
            self._pending_viewport_pan = (view, pos, pos)
            return
        self._selected = self._pick_aircraft(pos)

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self._pending_viewport_pan is not None:
            _, _, start_pos = self._pending_viewport_pan
            self._pending_viewport_pan = None
            if self._near_pos(start_pos, pos):
                self._selected = self._pick_aircraft(pos)
            return
        if self._viewport_drag is not None:
            self._on_viewport_pan_up()
            return
        if self._view_drag is not None:
            view, handle = self._view_drag
            self._view_drag = None
            self._view_drag_cursor = None
            view.on_drag_end(handle, pos, self)
            return
        if self._drag_resize is not None:
            self._drag_resize = None
            self._refresh_after_layout_change()
            return
        if self._drag_view is not None:
            dragged = self._drag_view
            target = self._drop_target
            self._drag_view = None
            self._drop_target = None
            if target is None:
                return
            target_leaf, zone = target
            if target_leaf is dragged and zone != "center":
                return
            self._apply_drop(dragged, target_leaf, zone)
            self._refresh_after_layout_change()

    def _on_mouse_motion(self, pos: tuple[int, int]) -> None:
        if self._pending_viewport_pan is not None:
            view, previous, start_pos = self._pending_viewport_pan
            if not self._near_pos(start_pos, pos):
                dx = pos[0] - previous[0]
                dy = pos[1] - previous[1]
                view.pan_view_by((dx, dy), self)
                self._pending_viewport_pan = None
                self._viewport_drag = (view, pos)
                self._set_cursor_hint(CursorHint.MOVE)
            return
        if self._viewport_drag is not None:
            view, previous = self._viewport_drag
            dx = pos[0] - previous[0]
            dy = pos[1] - previous[1]
            view.pan_view_by((dx, dy), self)
            self._viewport_drag = (view, pos)
            self._set_cursor_hint(CursorHint.MOVE)
            return
        if self._view_drag is not None:
            view, handle = self._view_drag
            view.on_drag_motion(handle, pos, self)
            self._set_cursor_hint(self._view_drag_cursor or CursorHint.POINT)
            return
        if self._drag_resize is not None:
            self._apply_drag_resize(pos)
            split = self._drag_resize[0]
            self._set_cursor_hint(
                CursorHint.RESIZE_Y if split.orientation == "v" else CursorHint.RESIZE_X
            )
            return
        if self._drag_view is not None:
            target_leaf = L.find_leaf_at(self._layout, pos)
            if target_leaf is None:
                self._drop_target = None
            else:
                zone = L.drop_zone(target_leaf.rect, pos)
                self._drop_target = (target_leaf, zone) if zone is not None else None
            self._set_cursor_hint(CursorHint.POINT)
            return
        # Idle hover: per-view cursor hints take priority over the
        # layout-level (header / divider) ones.
        for view in self.views:
            if not view.rect.collidepoint(pos):
                continue
            cursor = view.cursor_hint(pos, self)
            if cursor is not None:
                self._set_cursor_hint(cursor)
                return
        if self._leaf_with_header_at(pos) is not None:
            self._set_cursor_hint(CursorHint.POINT)
        else:
            hit = self._divider_at(pos)
            if hit is None:
                if self._hover_cursor_hint(pos) is not None:
                    return
                self._set_cursor_hint(None)
            else:
                split, _ = hit
                self._set_cursor_hint(
                    CursorHint.RESIZE_Y
                    if split.orientation == "v"
                    else CursorHint.RESIZE_X
                )

    def _on_mouse_wheel(self, direction: int, pos: tuple[int, int]) -> None:
        if direction == 0:
            return
        view = self._viewport_view_at(pos)
        if view is None:
            return
        factor = 1.15**direction
        view.zoom_view_at(pos, factor, self)

    def _on_viewport_pan_down(self, pos: tuple[int, int]) -> None:
        view = self._viewport_view_at(pos)
        if view is None:
            return
        self._viewport_drag = (view, pos)
        self._set_cursor_hint(CursorHint.MOVE)

    def _on_viewport_pan_up(self) -> None:
        if self._viewport_drag is not None:
            self._viewport_drag = None
            self._set_cursor_hint(None)
        self._pending_viewport_pan = None

    def _viewport_view_at(self, pos: tuple[int, int]) -> PygameView | None:
        for view in self.views:
            if not view.rect.collidepoint(pos):
                continue
            if getattr(view, "supports_viewport_pan_zoom", False):
                return view
        return None

    def _reset_viewports(self) -> None:
        for view in self.views:
            view.reset_viewport(self)

    @staticmethod
    def _near_pos(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= _PAN_START_PX**2

    def _set_cursor_hint(self, hint: CursorHintName | None) -> None:
        try:
            pygame.mouse.set_cursor(
                _PYGAME_CURSORS.get(hint, pygame.SYSTEM_CURSOR_ARROW)
            )
        except pygame.error:
            # Some SDL backends used in wrapped/headless eval cannot create
            # system cursors. Interaction should continue even if the cursor
            # affordance cannot be displayed.
            pass

    def _hover_cursor_hint(self, pos: tuple[int, int]) -> CursorHintName | None:
        for view in self.views:
            if not view.rect.collidepoint(pos):
                continue
            hit = view.hover(pos, self)
            if hit is not None and hit.get("kind") == "aircraft":
                self._set_cursor_hint(CursorHint.POINT)
                return CursorHint.POINT
        return None

    def _handle_resize(self, new_size: tuple[int, int]) -> None:
        new_w, new_h = max(int(new_size[0]), 1), max(int(new_size[1]), 1)
        self.window_size = (new_w, new_h)
        self._window = pygame.display.set_mode(self.window_size, self._display_flags())
        if self._env is not None:
            self.on_reset()

    # ------------------------------------------------------------------
    # Layout / drag-resize / drag-rearrange
    # ------------------------------------------------------------------

    def _compute_layout(self) -> None:
        """Run layout, then reserve a header strip on every leaf's view rect."""
        win_rect = pygame.Rect(0, 0, self.window_size[0], self.window_size[1])
        L.compute(self._layout, win_rect)
        # Each leaf's view gets a content rect = leaf rect minus the header.
        for leaf in L.iter_leaves(self._layout):
            r = leaf.rect
            content = pygame.Rect(
                r.left,
                r.top + self.HEADER_HEIGHT,
                r.width,
                max(r.height - self.HEADER_HEIGHT, 0),
            )
            leaf.view.rect = content

    def _refresh_after_layout_change(self) -> None:
        """Recompute layout + replay env on_reset so projections + overlays update."""
        if self._env is None:
            return
        self._compute_layout()
        for view in self.views:
            view.on_reset(self, self._env)
        self.draw_renderables(self._env._renderable_builder.iter_renderables())

    def _pick_aircraft(self, pos: tuple[int, int]) -> str | None:
        """Return the aircraft under *pos* using the same hit tests as hover."""
        for view in self.views:
            if not view.rect.collidepoint(pos):
                continue
            hit = view.hover(pos, self)
            if hit is None or hit.get("kind") != "aircraft":
                continue
            idx = hit.get("idx")
            if isinstance(idx, int) and 0 <= idx < bs.traf.ntraf:
                return bs.traf.id[idx]
        return None

    def _divider_at(self, pos: tuple[int, int]) -> tuple[L.Split, int] | None:
        """Return ``(split, child_index)`` for a divider under *pos*, else ``None``.

        ``child_index`` is the index of the *upper / left* child whose far
        edge is the divider line; dragging adjusts the boundary between
        children ``i`` and ``i+1``.
        """
        for split in L.iter_splits(self._layout):
            n = len(split.children)
            for i in range(n - 1):
                child = split.children[i]
                if split.orientation == "v":
                    if (
                        abs(pos[1] - child.rect.bottom) <= _DIVIDER_HIT_PX
                        and split.rect.left <= pos[0] <= split.rect.right
                    ):
                        return split, i
                else:  # "h"
                    if (
                        abs(pos[0] - child.rect.right) <= _DIVIDER_HIT_PX
                        and split.rect.top <= pos[1] <= split.rect.bottom
                    ):
                        return split, i
        return None

    def _apply_drag_resize(self, mouse_pos: tuple[int, int]) -> None:
        if self._drag_resize is None:
            return
        split, idx, (sx, sy), start_fracs = self._drag_resize
        if split.orientation == "v":
            extent = max(split.rect.height, 1)
            delta = (mouse_pos[1] - sy) / extent
        else:
            extent = max(split.rect.width, 1)
            delta = (mouse_pos[0] - sx) / extent
        min_frac = _MIN_VIEW_PX / extent
        max_take_from_next = max(start_fracs[idx + 1] - min_frac, 0.0)
        max_give_to_next = max(start_fracs[idx] - min_frac, 0.0)
        delta = max(-max_give_to_next, min(max_take_from_next, delta))
        new = list(start_fracs)
        new[idx] = start_fracs[idx] + delta
        new[idx + 1] = start_fracs[idx + 1] - delta
        split.fractions = new
        self._compute_layout()

    def _apply_drop(self, dragged: L.Leaf, target: L.Leaf, zone: str) -> None:
        """Mutate the layout tree so *dragged* lands at *zone* of *target*."""
        if zone == "center":
            if dragged is target:
                return
            dragged.view, target.view = target.view, dragged.view
            return
        if dragged is target:
            return  # split-self drop is degenerate
        # Hold the dragged view, remove the leaf, insert a fresh leaf
        # carrying the same view at the target zone.  This keeps the
        # tree clean (no orphaned references) and lets the unwrap logic
        # merge same-orientation chains.
        view = dragged.view
        try:
            self._layout = L.remove_leaf(self._layout, dragged)
        except ValueError:
            return
        if target not in list(L.iter_leaves(self._layout)):
            # Edge case: target collapsed away.  Recover by appending the
            # dragged view at the bottom of the layout.
            self._layout = L.parse((self._layout, Leaf(view=view)))
            return
        new_leaf = L.Leaf(view=view)
        self._layout = L.insert_leaf(self._layout, target, new_leaf, zone)

    def _leaf_with_header_at(self, pos: tuple[int, int]) -> L.Leaf | None:
        """Return the leaf whose header bar contains *pos*, else ``None``."""
        for leaf in L.iter_leaves(self._layout):
            header = pygame.Rect(
                leaf.rect.left,
                leaf.rect.top,
                leaf.rect.width,
                self.HEADER_HEIGHT,
            )
            if header.collidepoint(pos):
                return leaf
        return None

    def _label_button_at(self, pos: tuple[int, int]) -> pygame.Rect | None:
        for leaf in L.iter_leaves(self._layout):
            button = self._label_button_rect(leaf)
            if button.collidepoint(pos):
                return button
        return None

    # ------------------------------------------------------------------
    # Frame composition
    # ------------------------------------------------------------------

    def _render_frame(self) -> None:
        with self._aircraft_snapshot_cache_scope():
            if self._canvas is None or self._canvas.get_size() != self.window_size:
                self._canvas = pygame.Surface(self.window_size)
            canvas = self._canvas
            canvas.fill(C.SKY_BLUE)

            for view in self.views:
                self._render_view_clipped(canvas, view)

            self._render_selection(canvas)

            # Headers on top of view content.
            for leaf in L.iter_leaves(self._layout):
                self._draw_header(canvas, leaf)

            # Splitter dividers between siblings.
            for split in L.iter_splits(self._layout):
                for i in range(len(split.children) - 1):
                    child = split.children[i]
                    if split.orientation == "v":
                        pygame.draw.line(
                            canvas,
                            C.DIVIDER,
                            (split.rect.left, child.rect.bottom),
                            (split.rect.right, child.rect.bottom),
                            width=1,
                        )
                    else:
                        pygame.draw.line(
                            canvas,
                            C.DIVIDER,
                            (child.rect.right, split.rect.top),
                            (child.rect.right, split.rect.bottom),
                            width=1,
                        )

            # Drop preview during drag-rearrange.
            if self._drag_view is not None and self._drop_target is not None:
                target_leaf, zone = self._drop_target
                preview = L.drop_preview_rect(target_leaf.rect, zone)
                overlay = pygame.Surface(preview.size, pygame.SRCALPHA)
                overlay.fill(self.DROP_PREVIEW_BG)
                canvas.blit(overlay, preview.topleft)
                pygame.draw.rect(canvas, C.HIGHLIGHT, preview, width=2)

            self._render_hover(canvas)
            self._draw_status_badge(canvas)

            self._window.blit(canvas, (0, 0))
            pygame.display.flip()

        if self._video_writer is not None:
            # surfarray.array3d returns (W, H, 3); imageio wants (H, W, 3).
            frame = pygame.surfarray.array3d(self._window).swapaxes(0, 1)
            self._video_writer.append_data(np.ascontiguousarray(frame))

    def _render_selection(self, canvas: pygame.Surface) -> None:
        tracked = self.tracked_acid()
        if tracked is None:
            return
        idx = self._live_index().get(tracked)
        if idx is None:
            return
        positions: list[tuple[float, float]] = []
        for view in self.views:
            self._highlight_view_aircraft_clipped(canvas, view, idx)
            pos = view.aircraft_position(self, idx)
            if pos is not None:
                positions.append(pos)
        for a, b in itertools.pairwise(positions):
            pygame.draw.line(canvas, C.HIGHLIGHT, a, b, width=1)

    def _render_view_clipped(
        self,
        canvas: pygame.Surface,
        view: PygameView,
    ) -> None:
        """Render one view without allowing it to paint outside its panel."""
        old_clip = canvas.get_clip()
        canvas.set_clip(view.rect)
        try:
            view.render(canvas, self)
        finally:
            canvas.set_clip(old_clip)

    def _highlight_view_aircraft_clipped(
        self,
        canvas: pygame.Surface,
        view: PygameView,
        idx: int,
    ) -> None:
        """Draw a view-local aircraft highlight clipped to that view."""
        old_clip = canvas.get_clip()
        canvas.set_clip(view.rect)
        try:
            view.highlight_aircraft(canvas, self, idx)
        finally:
            canvas.set_clip(old_clip)

    def _draw_header(self, canvas: pygame.Surface, leaf: L.Leaf) -> None:
        """Draw a small drag-handle bar with the view's class name."""
        if self._header_font is None:
            return
        header = pygame.Rect(
            leaf.rect.left,
            leaf.rect.top,
            leaf.rect.width,
            self.HEADER_HEIGHT,
        )
        bg = self.HEADER_BG
        # Tint the dragged view's header so the user sees what they're carrying.
        if self._drag_view is leaf:
            bg = C.HIGHLIGHT
        pygame.draw.rect(canvas, bg, header)
        pygame.draw.line(
            canvas,
            C.DIVIDER,
            (header.left, header.bottom - 1),
            (header.right, header.bottom - 1),
            width=1,
        )
        title = type(leaf.view).__name__
        text = self._header_font.render(title, True, self.HEADER_FG)
        canvas.blit(
            text,
            (
                header.left + 6,
                header.top + (self.HEADER_HEIGHT - text.get_height()) // 2,
            ),
        )
        button = self._label_button_rect(leaf)
        button_bg = (
            self.HEADER_BUTTON_BG if self.show_labels else self.HEADER_BUTTON_OFF_BG
        )
        pygame.draw.rect(canvas, button_bg, button, border_radius=3)
        pygame.draw.rect(canvas, C.DIVIDER, button, width=1, border_radius=3)
        label = self._header_font.render("LBL", True, self.HEADER_FG)
        canvas.blit(
            label,
            (
                button.centerx - label.get_width() // 2,
                button.centery - label.get_height() // 2,
            ),
        )
        vo_button = self._vo_button_rect(leaf)
        vo_bg = (
            self.HEADER_BUTTON_BG
            if self.show_velocity_obstacles
            else self.HEADER_BUTTON_OFF_BG
        )
        pygame.draw.rect(canvas, vo_bg, vo_button, border_radius=3)
        pygame.draw.rect(canvas, C.DIVIDER, vo_button, width=1, border_radius=3)
        vo_label = self._header_font.render("VO", True, self.HEADER_FG)
        canvas.blit(
            vo_label,
            (
                vo_button.centerx - vo_label.get_width() // 2,
                vo_button.centery - vo_label.get_height() // 2,
            ),
        )

    def _label_button_rect(self, leaf: L.Leaf) -> pygame.Rect:
        height = max(self.HEADER_HEIGHT - 2 * self.HEADER_BUTTON_PAD, 1)
        return pygame.Rect(
            leaf.rect.right - self.HEADER_BUTTON_W - self.HEADER_BUTTON_PAD,
            leaf.rect.top + self.HEADER_BUTTON_PAD,
            self.HEADER_BUTTON_W,
            height,
        )

    def _vo_button_rect(self, leaf: L.Leaf) -> pygame.Rect:
        # Sits immediately to the left of the LBL button.
        height = max(self.HEADER_HEIGHT - 2 * self.HEADER_BUTTON_PAD, 1)
        return pygame.Rect(
            leaf.rect.right - 2 * self.HEADER_BUTTON_W - 2 * self.HEADER_BUTTON_PAD,
            leaf.rect.top + self.HEADER_BUTTON_PAD,
            self.HEADER_BUTTON_W,
            height,
        )

    def _vo_button_at(self, pos: tuple[int, int]) -> pygame.Rect | None:
        for leaf in L.iter_leaves(self._layout):
            button = self._vo_button_rect(leaf)
            if button.collidepoint(pos):
                return button
        return None

    # ------------------------------------------------------------------
    # Cross-view hover
    # ------------------------------------------------------------------

    def _render_hover(self, canvas: pygame.Surface) -> None:
        if self._window is None or self.font is None:
            return
        if not pygame.mouse.get_focused():
            return
        # Suppress hover tooltip while dragging - the drop preview is
        # the user's focus right now.
        if self._drag_view is not None or self._drag_resize is not None:
            return
        mouse_pos = pygame.mouse.get_pos()

        for view in self.views:
            promoter = getattr(view, "render_hover_overlays", None)
            if promoter is not None and view.rect.collidepoint(mouse_pos):
                self._render_view_hover_overlays_clipped(
                    canvas, view, promoter, mouse_pos
                )

        label_hit = None
        ac_hit = None
        for view in self.views:
            if not view.rect.collidepoint(mouse_pos):
                continue
            hit = view.hover(mouse_pos, self)
            if hit is None:
                continue
            if hit["kind"] == "label" and label_hit is None:
                label_hit = hit
            elif hit["kind"] == "aircraft" and ac_hit is None:
                ac_hit = hit

        if label_hit is not None:
            self._blit_tooltip(
                canvas,
                self._format_label_lines(label_hit["info"]),
                mouse_pos[0],
                mouse_pos[1],
                highlight=False,
            )
            return
        if ac_hit is None:
            return

        idx = ac_hit["idx"]
        positions: list[tuple[float, float]] = []
        for view in self.views:
            self._highlight_view_aircraft_clipped(canvas, view, idx)
            pos = view.aircraft_position(self, idx)
            if pos is not None:
                positions.append(pos)
        for a, b in itertools.pairwise(positions):
            pygame.draw.line(canvas, C.HIGHLIGHT, a, b, width=1)

        if not (0 <= idx < bs.traf.ntraf):
            return
        acid = bs.traf.id[idx]
        lines = self.format_aircraft_info_lines(acid)
        if not lines:
            return
        state = self._aircraft_state(acid)
        self._blit_tooltip(canvas, lines, mouse_pos[0], mouse_pos[1], state != "normal")

    def _render_view_hover_overlays_clipped(
        self,
        canvas: pygame.Surface,
        view: PygameView,
        render_hover_overlays,
        mouse_pos: tuple[int, int],
    ) -> None:
        """Draw view-owned hover overlays without bleeding into other panels."""
        old_clip = canvas.get_clip()
        canvas.set_clip(view.rect)
        try:
            render_hover_overlays(canvas, mouse_pos)
        finally:
            canvas.set_clip(old_clip)

    # ------------------------------------------------------------------
    # Tooltip + status badge + text utilities
    # ------------------------------------------------------------------

    def _format_label_lines(self, info: dict) -> list[str]:
        bounds = info.get("bounds")
        kind = info.get("kind", "").upper()
        name = info.get("name", "")
        lines = [f"{kind}  {name}"] if kind != name else [name]
        if bounds is not None:
            (lat_min, lat_max), (lon_min, lon_max) = bounds.bounding_box
            lines.append(f"Lat   {lat_min:.3f} to {lat_max:.3f}")
            lines.append(f"Lon   {lon_min:.3f} to {lon_max:.3f}")
            alt_lo = getattr(bounds, "alt_min_ft", float("-inf"))
            alt_hi = getattr(bounds, "alt_max_ft", float("inf"))
            if math.isfinite(alt_lo) and math.isfinite(alt_hi):
                lines.append(f"Alt   {int(alt_lo):,} - {int(alt_hi):,} ft")
        spawn_alt = info.get("spawn_alt")
        if isinstance(spawn_alt, tuple) and len(spawn_alt) == 2:
            lo, hi = spawn_alt
            if math.isfinite(lo) and math.isfinite(hi):
                lines.append(f"Spawn alt  {int(lo):,} - {int(hi):,} ft")
        return lines

    def _blit_tooltip(
        self,
        canvas: pygame.Surface,
        lines: list[str],
        cursor_x: int,
        cursor_y: int,
        highlight: bool,
    ) -> None:
        if self.font is None:
            return
        text_color = C.RED if highlight else C.BLACK
        bg_color = (255, 220, 220, 240) if highlight else (255, 248, 200, 240)
        rendered = [self.font.render(line, True, text_color) for line in lines]
        w = max(s.get_width() for s in rendered)
        h = sum(s.get_height() for s in rendered)
        pad = 6
        bg = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        bg.fill(bg_color)
        pygame.draw.rect(bg, text_color, bg.get_rect(), width=1)
        cy = pad
        for s in rendered:
            bg.blit(s, (pad, cy))
            cy += s.get_height()
        win_w, win_h = self.window_size
        tx = cursor_x + 14
        ty = cursor_y - bg.get_height() - 8
        if tx + bg.get_width() > win_w:
            tx = cursor_x - bg.get_width() - 14
        if ty < 0:
            ty = cursor_y + 14
        if ty + bg.get_height() > win_h:
            ty = win_h - bg.get_height() - 2
        canvas.blit(bg, (max(0, tx), max(0, ty)))

    def _draw_status_badge(self, canvas: pygame.Surface) -> None:
        if self.font is None:
            return
        text = self.format_status_line()
        color = C.RED if self._paused else C.BLACK
        bg = self.render_text_bg(text, color)
        win_w, win_h = self.window_size
        canvas.blit(bg, (win_w - bg.get_width() - 8, win_h - bg.get_height() - 8))

    def render_text_bg(
        self,
        text: str,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        # Multi-line aware: ``\n`` splits into stacked lines under one
        # rounded background. The bg is sized to the widest line plus
        # padding; lines are left-aligned within the bg.
        lines = text.split("\n")
        rendered = [self.font.render(line, True, color) for line in lines]
        pad = self.LABEL_PAD
        w = max(s.get_width() for s in rendered)
        h = sum(s.get_height() for s in rendered)
        out = pygame.Surface(
            (w + 2 * pad, h + 2 * pad),
            pygame.SRCALPHA,
        )
        out.fill(self.LABEL_BG)
        cur_y = pad
        for s in rendered:
            out.blit(s, (pad, cur_y))
            cur_y += s.get_height()
        return out

    def blit_data_block(
        self,
        canvas: pygame.Surface,
        lines: list[str],
        color: tuple[int, int, int],
        x: float,
        y: float,
    ) -> None:
        if self.font is None:
            return
        rendered = [self.font.render(line, True, color) for line in lines]
        w = max(s.get_width() for s in rendered)
        h = sum(s.get_height() for s in rendered)
        pad = self.LABEL_PAD
        bg = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        bg.fill(self.LABEL_BG)
        cur_y = pad
        for s in rendered:
            bg.blit(s, (pad, cur_y))
            cur_y += s.get_height()
        ox, oy = self.LABEL_OFFSET
        canvas.blit(bg, (x + ox, y + oy))
