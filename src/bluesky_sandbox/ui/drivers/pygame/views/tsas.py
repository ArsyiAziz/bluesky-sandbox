"""TSASView - per-waypoint sequencing strips.

For every :class:`~bluesky_sandbox.sim.queryables.Waypoint` configured
on the current episode, draws a vertical strip listing every
aircraft top-to-bottom ordered by ETA (estimated time to the waypoint).
ETA is computed from the closing component of ground speed; aircraft
heading away from the waypoint (negative closing speed) sort to the bottom
under "DTM" (distance-to-metering point) instead of an ETA.

Inspired by AMAN-style sequencing displays - the agent / observer can
read aircraft order at a glance even when several streams overlap on
the plan view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from bluesky_sandbox.ui.drivers.common import TsasDataMixin, TsasRow, TsasTable
from bluesky_sandbox.ui.drivers.pygame import colors as C
from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView

if TYPE_CHECKING:
    from bluesky_sandbox.ui.drivers.pygame.driver import PygameSimDriver


class TSASView(TsasDataMixin, PygameView):
    """Per-waypoint sequencing tables (AMAN-style ETA columns)."""

    default_height_fraction = 0.25

    _TITLE_HEIGHT      = 28
    _COL_HEADER_HEIGHT = 20
    _ROW_HEIGHT        = 22
    _STRIP_PADDING     = 6
    _DIVIDER_WIDTH     = 1
    _BG_COLOR          = (245, 245, 250)
    _TITLE_COLOR       = (50, 60, 80)
    _COL_HEADER_COLOR  = (220, 224, 232)
    _ROW_COLOR         = (250, 250, 252)
    _ALT_ROW_COLOR     = (238, 241, 246)
    _EMPTY_COLOR       = (238, 241, 246)
    _TITLE_TEXT        = (245, 245, 250)
    _MUTED             = (95, 100, 110)

    def __init__(self) -> None:
        super().__init__()
        # Per-frame: (strip_rect, table) collected during render
        # so a future hover lookup can match the cursor to a strip.
        self._strip_rects: list[tuple[pygame.Rect, TsasTable]] = []

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        self._strip_rects = []
        # Background fill.
        pygame.draw.rect(canvas, self._BG_COLOR, self.rect)

        env = driver._env
        if env is None or driver.font is None:
            return

        tables = self.tsas_tables(driver)
        if not tables:
            self._draw_centered(canvas, driver,
                                "No waypoints configured", C.GRAY)
            return

        n_strips = len(tables)
        strip_w  = self.rect.width / n_strips

        for i, table in enumerate(tables):
            strip_rect = pygame.Rect(
                int(self.rect.left + i * strip_w),
                self.rect.top,
                int(strip_w),
                self.rect.height,
            )
            self._strip_rects.append((strip_rect, table))
            self._draw_strip(canvas, driver, strip_rect, table)
            # Divider between strips (skip after the last).
            if i < n_strips - 1:
                x = strip_rect.right
                pygame.draw.line(
                    canvas, C.DIVIDER,
                    (x, strip_rect.top), (x, strip_rect.bottom),
                    width=self._DIVIDER_WIDTH,
                )

    def _draw_strip(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        strip_rect: pygame.Rect,
        table: TsasTable,
    ) -> None:
        """Header + sorted aircraft list for one waypoint."""
        title_rect = pygame.Rect(
            strip_rect.left, strip_rect.top,
            strip_rect.width, self._TITLE_HEIGHT,
        )
        pygame.draw.rect(canvas, self._TITLE_COLOR, title_rect)
        pygame.draw.line(
            canvas, C.DIVIDER,
            (title_rect.left, title_rect.bottom),
            (title_rect.right, title_rect.bottom),
            width=1,
        )
        title = self._fit_text(driver.font, table.title, title_rect.width - 2 * self._STRIP_PADDING)
        title_surf = driver.font.render(title, True, self._TITLE_TEXT)
        canvas.blit(title_surf, (
            title_rect.left + self._STRIP_PADDING,
            title_rect.top + (self._TITLE_HEIGHT - title_surf.get_height()) // 2,
        ))

        col_rect = pygame.Rect(
            strip_rect.left, title_rect.bottom,
            strip_rect.width, self._COL_HEADER_HEIGHT,
        )
        pygame.draw.rect(canvas, self._COL_HEADER_COLOR, col_rect)
        self._draw_cells(
            canvas,
            driver,
            strip_rect,
            col_rect.top,
            ["CALL", "ETA", "DTM", "STATE"],
            [self._MUTED] * 4,
            small=True,
        )

        max_rows = max(
            (strip_rect.height - self._TITLE_HEIGHT - self._COL_HEADER_HEIGHT)
            // self._ROW_HEIGHT,
            1,
        )
        if not table.rows:
            empty_rect = pygame.Rect(
                strip_rect.left, col_rect.bottom,
                strip_rect.width, self._ROW_HEIGHT,
            )
            pygame.draw.rect(canvas, self._EMPTY_COLOR, empty_rect)
            text = self._fit_text(driver.font, "no traffic", empty_rect.width - 2 * self._STRIP_PADDING)
            surf = driver.font.render(text, True, self._MUTED)
            canvas.blit(surf, (
                empty_rect.left + self._STRIP_PADDING,
                empty_rect.top + (self._ROW_HEIGHT - surf.get_height()) // 2,
            ))
            return

        y = col_rect.bottom
        for row_idx, entry in enumerate(table.rows[:max_rows]):
            bg = self._ROW_COLOR if row_idx % 2 == 0 else self._ALT_ROW_COLOR
            pygame.draw.rect(canvas, bg, (
                strip_rect.left, y, strip_rect.width, self._ROW_HEIGHT,
            ))
            self._draw_row(canvas, driver, strip_rect, y, entry)
            y += self._ROW_HEIGHT

    def _draw_row(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        strip_rect: pygame.Rect,
        y: int,
        entry: TsasRow,
    ) -> None:
        if entry.state == "los":
            color = C.LOS
        elif entry.state == "conflict":
            color = C.CONF
        elif entry.state == "violation":
            color = C.VIOLATION
        else:
            color = C.BLACK
        self._draw_cells(
            canvas,
            driver,
            strip_rect,
            y,
            [
                entry.acid,
                self.tsas_eta_text(entry.eta_s),
                self.tsas_dtm_text(entry.dist_nm),
                self._state_text(entry.state),
            ],
            [color, color, C.BLACK, color],
        )

    def _draw_cells(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        strip_rect: pygame.Rect,
        y: int,
        values: list[str],
        colors: list[tuple[int, int, int]],
        *,
        small: bool = False,
    ) -> None:
        font = driver._header_font if small and driver._header_font is not None else driver.font
        if font is None:
            return
        aligns = ["left", "right", "right", "right"]
        old_clip = canvas.get_clip()
        for text, color, rect, align in zip(values, colors, self._cell_rects(strip_rect), aligns):
            fitted = self._fit_text(font, text, rect.width)
            surf = font.render(fitted, True, color)
            tx = rect.left if align == "left" else rect.right - surf.get_width()
            canvas.set_clip(rect)
            canvas.blit(surf, (
                tx,
                y + (self._ROW_HEIGHT - surf.get_height()) // 2,
            ))
        canvas.set_clip(old_clip)

    def _cell_rects(self, strip_rect: pygame.Rect) -> list[pygame.Rect]:
        left = strip_rect.left + self._STRIP_PADDING
        right = strip_rect.right - self._STRIP_PADDING
        width = max(right - left, 4)
        gap = 4
        usable = max(width - 3 * gap, 4)
        if usable < 96:
            weights = (0.34, 0.23, 0.20, 0.23)
        else:
            weights = (0.42, 0.20, 0.18, 0.20)
        widths = [max(1, int(usable * weight)) for weight in weights]
        widths[-1] = max(1, usable - sum(widths[:-1]))

        x = left
        rects: list[pygame.Rect] = []
        for col_w in widths:
            rects.append(pygame.Rect(x, strip_rect.top, col_w, strip_rect.height))
            x += col_w + gap
        return rects

    @staticmethod
    def _fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(text)[0] <= max_width:
            return text
        marker = "."
        if font.size(marker)[0] > max_width:
            return ""
        out = text
        while out and font.size(out + marker)[0] > max_width:
            out = out[:-1]
        return out + marker if out else marker

    @staticmethod
    def _state_text(state: str) -> str:
        if state == "conflict":
            return "CONF"
        if state == "violation":
            return "VIOL"
        if state == "normal":
            return "NORM"
        return state.upper()

    def _draw_centered(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """Centre a single line of text inside the panel - used for empty states."""
        surf = driver.font.render(text, True, color)
        canvas.blit(surf, (
            self.rect.left + (self.rect.width  - surf.get_width())  // 2,
            self.rect.top  + (self.rect.height - surf.get_height()) // 2,
        ))

    # ------------------------------------------------------------------
    # Hover (aircraft only - strip rows aren't hit-tested individually)
    # ------------------------------------------------------------------

    def hover(self, mouse_pos, driver: PygameSimDriver) -> dict | None:
        if not self.rect.collidepoint(mouse_pos):
            return None
        mx, my = mouse_pos
        # Identify which strip and which row, if any.
        for strip_rect, table in self._strip_rects:
            if not strip_rect.collidepoint(mx, my):
                continue
            row_y = strip_rect.top + self._TITLE_HEIGHT + self._COL_HEADER_HEIGHT
            if my < row_y:
                return None  # cursor is on the header
            row_idx = (my - row_y) // self._ROW_HEIGHT
            max_rows = max(
                (strip_rect.height - self._TITLE_HEIGHT - self._COL_HEADER_HEIGHT)
                // self._ROW_HEIGHT,
                1,
            )
            rows = table.rows[:max_rows]
            if 0 <= row_idx < len(rows):
                return {"kind": "aircraft", "idx": rows[row_idx].idx}
        return None
