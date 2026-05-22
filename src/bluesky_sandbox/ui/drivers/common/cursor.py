"""Toolkit-neutral cursor primitives for interactive driver surfaces."""

from __future__ import annotations

from typing import Literal

CursorHintName = Literal[
    "point",
    "move",
    "resize_x",
    "resize_y",
]


class CursorHint:
    """Semantic cursor hints shared by concrete GUI backends."""

    POINT: CursorHintName = "point"
    MOVE: CursorHintName = "move"
    RESIZE_X: CursorHintName = "resize_x"
    RESIZE_Y: CursorHintName = "resize_y"

    _INTERACTIVE = frozenset({POINT, MOVE, RESIZE_X, RESIZE_Y})

    @classmethod
    def is_interactive(cls, hint: str | None) -> bool:
        return hint in cls._INTERACTIVE
