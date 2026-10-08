"""Aircraft data blocks for the 3D view, drawn flat on the screen.

A block is text at a fixed pixel size whatever the zoom, beside its aircraft -
never on it - joined to it by a leader line, the way a radar screen shows them.
Each frame every block takes whichever of the four corners around its aircraft
overlaps least with the blocks already placed, the aircraft and the HUD; it
keeps its corner while that is as good as any, so blocks do not jump about.
The tracked aircraft's block is placed first, then those in loss of separation,
then in conflict.
"""

# ruff: noqa: PLC0415 - panda3d is an optional extra; see world.py.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panda3d.core import NodePath

    from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver

#: A screen rectangle in pixels, y down: ``(left, top, right, bottom)``.
Rect = tuple[float, float, float, float]

__all__ = ["AircraftTag", "AircraftTags", "Rect"]


@dataclass(frozen=True)
class AircraftTag:
    """One aircraft's block this frame: where its aircraft is on the screen
    (pixels), its lines, their color, its border's (None: none), its leader's,
    and how early it is placed (higher first)."""

    acid: str
    x: float
    y: float
    lines: tuple[str, ...]
    text_rgba: tuple[float, float, float, float]
    frame_rgba: tuple[float, float, float, float] | None
    leader_rgba: tuple[float, float, float, float]
    priority: int = 0


class _Block:
    """One block's text node, kept from frame to frame."""

    __slots__ = ("corner", "frame", "lines", "node", "path", "text_rgba")

    def __init__(self, parent: NodePath, font) -> None:
        from panda3d.core import TextNode

        node = TextNode("tag")
        node.setAlign(TextNode.ALeft)
        if font is not None:
            node.setFont(font)
        node.setCardColor(0.03, 0.05, 0.08, 0.88)
        node.setCardAsMargin(0.35, 0.35, 0.22, 0.12)
        node.setCardDecal(True)
        self.node = node
        self.path = parent.attachNewNode(node)
        self.lines: tuple[str, ...] | None = None
        self.text_rgba = None
        self.frame = None
        self.corner: int | None = None

    def set(self, tag: AircraftTag) -> None:
        if tag.lines != self.lines:
            self.node.setText("\n".join(tag.lines))
            self.lines = tag.lines
        if tag.text_rgba != self.text_rgba:
            self.node.setTextColor(*tag.text_rgba)
            self.text_rgba = tag.text_rgba
        if tag.frame_rgba != self.frame:
            if tag.frame_rgba is None:
                self.node.clearFrame()
            else:
                self.node.setFrameColor(*tag.frame_rgba)
                self.node.setFrameAsMargin(0.35, 0.35, 0.22, 0.12)
            self.frame = tag.frame_rgba

    def extent(self) -> tuple[float, float, float, float]:
        """The card's ``(left, right, bottom, top)``, in text units from the
        text's origin."""
        card = self.node.getCardActual()
        return card[0], card[1], card[2], card[3]


class AircraftTags:
    """Every aircraft's data block, placed each frame (see the module)."""

    #: The text's size, in pixels per text unit (about a line).
    FONT_PX = 13.0
    #: How far a block sits from its aircraft, diagonally (px).
    GAP_PX = 14.0
    #: How much of the aircraft symbol a leader leaves clear (px).
    LEADER_CLEAR_PX = 6.0
    #: Half the box an aircraft symbol takes, kept clear of blocks (px).
    AIRCRAFT_PX = 7.0
    # A corner's penalty for not being the one it had (px^2): enough that two
    # equally clear corners do not trade places, too little to keep an overlap.
    _MOVE_COST = 40.0

    def __init__(self, parent: NodePath, font) -> None:
        self._root = parent.attachNewNode("aircraft_tags", -10)
        self._font = font
        self._blocks: dict[str, _Block] = {}
        self._leaders: NodePath | None = None
        #: Where each block was drawn this frame, by callsign (px).
        self.rects: dict[str, Rect] = {}

    def destroy(self) -> None:
        self._root.removeNode()
        self._blocks.clear()
        self.rects.clear()

    def clear(self) -> None:
        """Drop every block - a new episode."""
        for block in self._blocks.values():
            block.path.removeNode()
        self._blocks.clear()
        self.rects.clear()
        if self._leaders is not None:
            self._leaders.removeNode()
            self._leaders = None

    def update(self, driver: Panda3DSimDriver, tags: list[AircraftTag], avoid: list[Rect]) -> None:
        """Place and draw ``tags``, clear of ``avoid`` (px); drop the rest."""
        from panda3d.core import LineSegs

        win = driver._show.win
        width, height = float(win.getXSize()), float(win.getYSize())
        aspect = driver._show.getAspectRatio()
        unit = 2.0 * self.FONT_PX / height  # aspect2d units per text unit

        for acid in self._blocks.keys() - {t.acid for t in tags}:
            self._blocks.pop(acid).path.removeNode()
        if self._leaders is not None:
            self._leaders.removeNode()
            self._leaders = None
        self.rects = {}

        a = self.AIRCRAFT_PX
        taken: list[Rect] = list(avoid) + [(t.x - a, t.y - a, t.x + a, t.y + a) for t in tags]
        leaders = LineSegs()
        leaders.setThickness(1.5)
        drawn = False
        ordered = sorted(tags, key=lambda t: (-t.priority, t.acid))
        for order, tag in enumerate(ordered):
            block = self._blocks.get(tag.acid)
            if block is None:
                block = self._blocks[tag.acid] = _Block(self._root, self._font)
            block.set(tag)
            left, right, bottom, top = block.extent()
            w, h = (right - left) * self.FONT_PX, (top - bottom) * self.FONT_PX
            corner, rect = self._place(tag, w, h, block.corner, taken, width, height)
            block.corner = corner
            taken.append(rect)
            self.rects[tag.acid] = rect
            # The text's origin from the card's top-left corner.
            ox = rect[0] - left * self.FONT_PX
            oy = rect[1] + top * self.FONT_PX
            block.path.setScale(unit)
            block.path.setPos((ox / width * 2.0 - 1.0) * aspect, 0.0, 1.0 - oy / height * 2.0)
            # Later blocks - the more important - on top.
            block.path.setBin("fixed", len(ordered) - order)
            block.path.show()
            # The leader: from just off the aircraft to the block's nearest corner.
            cx = rect[0] if corner in (0, 2) else rect[2]
            cy = rect[3] if corner in (0, 1) else rect[1]
            dx, dy = cx - tag.x, cy - tag.y
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            sx = tag.x + dx / length * self.LEADER_CLEAR_PX
            sy = tag.y + dy / length * self.LEADER_CLEAR_PX
            leaders.setColor(*tag.leader_rgba)
            leaders.moveTo((sx / width * 2.0 - 1.0) * aspect, 0.0, 1.0 - sy / height * 2.0)
            leaders.drawTo((cx / width * 2.0 - 1.0) * aspect, 0.0, 1.0 - cy / height * 2.0)
            drawn = True
        if drawn:
            from panda3d.core import TransparencyAttrib

            self._leaders = self._root.attachNewNode(leaders.create())
            self._leaders.setTransparency(TransparencyAttrib.MAlpha)
            self._leaders.setBin("fixed", 0)

    def _place(
        self,
        tag: AircraftTag,
        w: float,
        h: float,
        previous: int | None,
        taken: list[Rect],
        width: float,
        height: float,
    ) -> tuple[int, Rect]:
        """The corner - 0 up-right, 1 up-left, 2 down-right, 3 down-left -
        that overlaps least, and the block's rect there."""
        g = self.GAP_PX
        x, y = tag.x, tag.y
        candidates = (
            (x + g, y - g - h, x + g + w, y - g),
            (x - g - w, y - g - h, x - g, y - g),
            (x + g, y + g, x + g + w, y + g + h),
            (x - g - w, y + g, x - g, y + g + h),
        )
        own = (x - self.AIRCRAFT_PX, y - self.AIRCRAFT_PX, x + self.AIRCRAFT_PX, y + self.AIRCRAFT_PX)
        best: tuple[float, int] | None = None
        for corner, rect in enumerate(candidates):
            cost = sum(_overlap(rect, other) for other in taken if other != own)
            # Off the screen counts double: a block there cannot be read.
            inside = _overlap(rect, (0.0, 0.0, width, height))
            cost += 2.0 * (w * h - inside)
            if previous is not None and corner != previous:
                cost += self._MOVE_COST
            if best is None or cost < best[0]:
                best = (cost, corner)
        corner = best[1] if best is not None else 0
        return corner, candidates[corner]


def _overlap(a: Rect, b: Rect) -> float:
    """The area two rects share (px^2)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0
