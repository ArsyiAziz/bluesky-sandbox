"""Toolkit-neutral zoom/pan primitives for human-driver viewports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoomPanViewport:
    """Pixel-space zoom/pan transform around an existing fit projection."""

    min_zoom: float = 0.5
    max_zoom: float = 24.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    version: int = 0

    def reset(self) -> bool:
        """Restore the fit projection. Return True when state changed."""
        if self.zoom == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
            return False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.version += 1
        return True

    def pan_by(self, dx: float, dy: float) -> bool:
        """Move the projected world by a pixel delta."""
        if dx == 0.0 and dy == 0.0:
            return False
        self.pan_x += dx
        self.pan_y += dy
        self.version += 1
        return True

    def zoom_at(
        self,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
        factor: float,
    ) -> bool:
        """Zoom around a screen-space point, keeping that point anchored."""
        old_zoom = self.zoom
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * factor))
        if new_zoom == old_zoom:
            return False
        ratio = new_zoom / old_zoom
        self.pan_x = x - center_x - (x - center_x - self.pan_x) * ratio
        self.pan_y = y - center_y - (y - center_y - self.pan_y) * ratio
        self.zoom = new_zoom
        self.version += 1
        return True

    def apply_x(self, x: float, center_x: float) -> float:
        return center_x + self.pan_x + (x - center_x) * self.zoom

    def apply_y(self, y: float, center_y: float) -> float:
        return center_y + self.pan_y + (y - center_y) * self.zoom

    def apply(
        self,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
    ) -> tuple[float, float]:
        return self.apply_x(x, center_x), self.apply_y(y, center_y)

    def invert_x(self, x: float, center_x: float) -> float:
        return center_x + (x - center_x - self.pan_x) / self.zoom

    def invert_y(self, y: float, center_y: float) -> float:
        return center_y + (y - center_y - self.pan_y) / self.zoom

    def invert(
        self,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
    ) -> tuple[float, float]:
        return self.invert_x(x, center_x), self.invert_y(y, center_y)
