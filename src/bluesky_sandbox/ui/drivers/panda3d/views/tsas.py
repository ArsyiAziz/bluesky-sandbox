"""TSASView - draggable per-waypoint sequencing table in the Panda3D HUD."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluesky_sandbox.ui.drivers.common import (
    CursorHint,
    CursorHintName,
    TsasDataMixin,
    TsasTable,
)
from bluesky_sandbox.ui.drivers.panda3d.views.base import Panda3DView

if TYPE_CHECKING:
    from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver


class TSASView(TsasDataMixin, Panda3DView):
    """AMAN-style waypoint sequencing table rendered in aspect2d."""

    _ROWS_PER_STRIP = 8

    _POS = (1.28, 0.88)
    _WIDTH = 0.68
    _TITLE_H = 0.070
    _COL_HEADER_H = 0.052
    _ROW_H = 0.046
    _PAD_X = 0.018
    _GAP = 0.018
    _TEXT_SCALE = 0.032
    _TITLE_SCALE = 0.036
    _HEADER_SCALE = 0.027

    _BG = (0.03, 0.04, 0.06, 0.76)
    _TITLE_BG = (0.10, 0.13, 0.18, 0.92)
    _TITLE_HOVER_BG = (0.15, 0.19, 0.26, 0.94)
    _HEADER_BG = (0.17, 0.19, 0.23, 0.86)
    _ROW_BG = (0.07, 0.08, 0.10, 0.70)
    _ALT_ROW_BG = (0.10, 0.11, 0.13, 0.70)
    _ROW_HOVER_BG = (0.15, 0.17, 0.21, 0.78)
    _EMPTY_BG = (0.08, 0.08, 0.09, 0.68)
    _BORDER = (0.70, 0.74, 0.80, 0.70)
    _TEXT = (0.94, 0.95, 0.98, 1.0)
    _MUTED = (0.62, 0.67, 0.74, 1.0)
    _CONFLICT = (1.00, 0.62, 0.10, 1.0)
    _LOS = (1.00, 0.30, 0.38, 1.0)
    _VIOLATION = (150 / 255, 70 / 255, 220 / 255, 1.0)

    def __init__(self) -> None:
        self._root = None
        self._widgets: list[object] = []
        self._pos = list(self._POS)
        self._bounds: tuple[float, float, float, float] | None = None
        self._title_bounds: tuple[float, float, float, float] | None = None
        self._title_hits: list[tuple[float, float, float, float]] = []
        self._row_hits: list[tuple[tuple[float, float, float, float], str]] = []
        self._hover_pos: tuple[float, float] | None = None
        self._font = None
        self._drag_offset = (0.0, 0.0)
        self._dragging = False

    def on_start(self, driver: Panda3DSimDriver) -> None:
        self._font = getattr(driver, "_ui_font", None)
        self._root = driver._hud_root.attachNewNode("tsas_table")
        self._root.setPos(self._pos[0], 0.0, self._pos[1])

    def on_step(self, driver: Panda3DSimDriver) -> None:
        if self._root is None:
            return
        self._rebuild(driver)

    def close(self) -> None:
        self._clear_widgets()
        if self._root is not None:
            self._root.removeNode()
            self._root = None

    # ------------------------------------------------------------------
    # Mouse hooks
    # ------------------------------------------------------------------

    def on_mouse_down(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> bool:
        for bounds, acid in self._row_hits:
            if self._contains(bounds, pos):
                driver._selected = acid
                return True
        if any(self._contains(bounds, pos) for bounds in self._title_hits):
            self._dragging = True
            self._drag_offset = (self._pos[0] - pos[0], self._pos[1] - pos[1])
            return True
        return bool(self._bounds is not None and self._contains(self._bounds, pos))

    def on_mouse_drag(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> None:
        if not self._dragging or self._root is None:
            return
        x = pos[0] + self._drag_offset[0]
        y = pos[1] + self._drag_offset[1]
        self._set_pos(driver, x, y)

    def on_mouse_up(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> None:
        self._dragging = False

    def cursor_hint(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> CursorHintName | None:
        if self._dragging:
            return CursorHint.POINT
        if any(self._contains(bounds, pos) for bounds in self._title_hits):
            return CursorHint.POINT
        if any(self._contains(bounds, pos) for bounds, _acid in self._row_hits):
            return CursorHint.POINT
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rebuild(self, driver: Panda3DSimDriver) -> None:
        self._clear_widgets()
        self._row_hits = []
        self._title_hits = []
        self._hover_pos = driver._mouse_aspect_pos()
        tables = self.tsas_tables(driver, max_rows=self._ROWS_PER_STRIP)
        if not tables:
            self._bounds = None
            self._title_bounds = None
            return

        y = 0.0
        total_h = 0.0
        for table_idx, table in enumerate(tables):
            if table_idx:
                y -= self._GAP
                total_h += self._GAP
            table_h = self._table_height(table)
            self._draw_table(driver, table, y, table_h)
            y -= table_h
            total_h += table_h

        left = self._pos[0] - self._WIDTH
        right = self._pos[0]
        top = self._pos[1]
        bottom = self._pos[1] - total_h
        self._bounds = (left, right, bottom, top)
        self._title_bounds = (
            left,
            right,
            top - self._TITLE_H,
            top,
        )
        self._clamp_to_screen(driver)

    def _draw_table(
        self,
        driver: Panda3DSimDriver,
        table: TsasTable,
        y_top: float,
        height: float,
    ) -> None:
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        left = -self._WIDTH
        right = 0.0
        bottom = y_top - height

        self._add_frame(DirectFrame(
            parent=self._root,
            frameSize=(left, right, bottom, y_top),
            frameColor=self._BG,
            relief=DGG.FLAT,
        ))
        title_bounds = self._absolute_bounds(left, right, y_top - self._TITLE_H, y_top)
        title_bg = (
            self._TITLE_HOVER_BG
            if self._dragging or self._is_hovered(title_bounds)
            else self._TITLE_BG
        )
        self._add_frame(DirectFrame(
            parent=self._root,
            frameSize=(left, right, y_top - self._TITLE_H, y_top),
            frameColor=title_bg,
            relief=DGG.FLAT,
        ))
        self._title_hits.append(title_bounds)
        self._add_text(OnscreenText(
            text=self._fit_text(table.title, 26),
            parent=self._root,
            pos=(left + self._PAD_X, y_top - 0.048),
            scale=self._TITLE_SCALE,
            fg=self._waypoint_color(table.color),
            align=TextNode.ALeft,
            mayChange=False,
            **self._font_kwargs(),
        ))

        header_top = y_top - self._TITLE_H
        header_bottom = header_top - self._COL_HEADER_H
        self._add_frame(DirectFrame(
            parent=self._root,
            frameSize=(left, right, header_bottom, header_top),
            frameColor=self._HEADER_BG,
            relief=DGG.FLAT,
        ))
        self._draw_cells(
            ["CALL", "ETA", "DTM", "ST"],
            header_bottom + 0.015,
            [self._MUTED] * 4,
            self._HEADER_SCALE,
        )

        rows = table.rows
        if not rows:
            self._add_frame(DirectFrame(
                parent=self._root,
                frameSize=(left, right, bottom, header_bottom),
                frameColor=self._EMPTY_BG,
                relief=DGG.FLAT,
            ))
            self._add_text(OnscreenText(
                text="no traffic",
                parent=self._root,
                pos=(left + self._PAD_X, header_bottom - 0.033),
                scale=self._TEXT_SCALE,
                fg=self._MUTED,
                align=TextNode.ALeft,
                mayChange=False,
                **self._font_kwargs(),
            ))
        for row_idx, row in enumerate(rows):
            row_top = header_bottom - row_idx * self._ROW_H
            row_bottom = row_top - self._ROW_H
            row_bounds = self._absolute_bounds(left, right, row_bottom, row_top)
            bg = (
                self._ROW_HOVER_BG if self._is_hovered(row_bounds)
                else self._ROW_BG if row_idx % 2 == 0
                else self._ALT_ROW_BG
            )
            self._add_frame(DirectFrame(
                parent=self._root,
                frameSize=(left, right, row_bottom, row_top),
                frameColor=bg,
                relief=DGG.FLAT,
            ))
            self._row_hits.append((
                row_bounds,
                row.acid,
            ))
            color = self._row_color(row.state)
            self._draw_cells(
                [
                    self._fit_text(row.acid, 10),
                    self.tsas_eta_text(row.eta_s),
                    self.tsas_dtm_text(row.dist_nm),
                    self._state_text(row.state),
                ],
                row_bottom + 0.014,
                [color, color, self._TEXT, color],
                self._TEXT_SCALE,
            )

        self._draw_border(left, right, bottom, y_top)

    def _draw_cells(
        self,
        values: list[str],
        y: float,
        colors: list[tuple[float, float, float, float]],
        scale: float,
    ) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        x_positions = [
            -self._WIDTH + self._PAD_X,
            -0.35,
            -0.22,
            -self._PAD_X,
        ]
        aligns = [TextNode.ALeft, TextNode.ARight, TextNode.ARight, TextNode.ARight]
        for value, x, align, color in zip(values, x_positions, aligns, colors):
            self._add_text(OnscreenText(
                text=value,
                parent=self._root,
                pos=(x, y),
                scale=scale,
                fg=color,
                align=align,
                mayChange=False,
                **self._font_kwargs(),
            ))

    def _draw_border(self, left: float, right: float, bottom: float, top: float) -> None:
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectFrame

        width = 0.004
        for frame_size in (
            (left, right, top - width, top),
            (left, right, bottom, bottom + width),
            (left, left + width, bottom, top),
            (right - width, right, bottom, top),
        ):
            self._add_frame(DirectFrame(
                parent=self._root,
                frameSize=frame_size,
                frameColor=self._BORDER,
                relief=DGG.FLAT,
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_widgets(self) -> None:
        for widget in self._widgets:
            try:
                widget.destroy()
            except AttributeError:
                try:
                    widget.removeNode()
                except Exception:
                    pass
            except Exception:
                pass
        self._widgets = []

    def _add_frame(self, frame) -> None:
        self._widgets.append(frame)

    def _add_text(self, text) -> None:
        self._widgets.append(text)

    def _font_kwargs(self) -> dict:
        return {"font": self._font} if self._font is not None else {}

    def _table_height(self, table: TsasTable) -> float:
        rows = max(len(table.rows), 1)
        return self._TITLE_H + self._COL_HEADER_H + rows * self._ROW_H

    @staticmethod
    def _contains(
        bounds: tuple[float, float, float, float],
        pos: tuple[float, float],
    ) -> bool:
        left, right, bottom, top = bounds
        x, y = pos
        return left <= x <= right and bottom <= y <= top

    def _is_hovered(self, bounds: tuple[float, float, float, float]) -> bool:
        return self._hover_pos is not None and self._contains(bounds, self._hover_pos)

    def _set_pos(self, driver: Panda3DSimDriver, x: float, y: float) -> None:
        aspect = driver._show.getAspectRatio()
        half_w = self._WIDTH
        x = max(-aspect + half_w, min(aspect, x))
        if self._bounds is not None:
            height = self._bounds[3] - self._bounds[2]
            y = max(-1.0 + height, min(1.0, y))
        else:
            y = max(-1.0 + self._TITLE_H, min(1.0, y))
        self._pos = [x, y]
        self._root.setPos(x, 0.0, y)
        if self._bounds is not None:
            old_left, _, old_bottom, _ = self._bounds
            height = self._bounds[3] - self._bounds[2]
            self._bounds = (x - self._WIDTH, x, y - height, y)
            self._title_bounds = (x - self._WIDTH, x, y - self._TITLE_H, y)
            dx = (x - self._WIDTH) - old_left
            dy = (y - height) - old_bottom
            self._row_hits = [
                ((l + dx, r + dx, b + dy, t + dy), acid)
                for (l, r, b, t), acid in self._row_hits
            ]
            self._title_hits = [
                (l + dx, r + dx, b + dy, t + dy)
                for l, r, b, t in self._title_hits
            ]

    def _clamp_to_screen(self, driver: Panda3DSimDriver) -> None:
        self._set_pos(driver, self._pos[0], self._pos[1])

    def _absolute_bounds(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
    ) -> tuple[float, float, float, float]:
        return (
            self._pos[0] + left,
            self._pos[0] + right,
            self._pos[1] + bottom,
            self._pos[1] + top,
        )

    def _row_color(self, state: str) -> tuple[float, float, float, float]:
        if state == "los":
            return self._LOS
        if state == "conflict":
            return self._CONFLICT
        if state == "violation":
            return self._VIOLATION
        return self._TEXT

    @staticmethod
    def _state_text(state: str) -> str:
        if state == "conflict":
            return "CONF"
        if state == "violation":
            return "VIOL"
        if state == "normal":
            return "NORM"
        return state.upper()[:4]

    @staticmethod
    def _fit_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return text[:max_chars - 1] + "."

    @staticmethod
    def _waypoint_color(name: str) -> tuple[float, float, float, float]:
        from bluesky_sandbox.ui.drivers.panda3d.colors import NAMED_COLORS

        rgb = NAMED_COLORS.get(name.lower(), NAMED_COLORS["cyan"])
        return (*rgb, 1.0)
