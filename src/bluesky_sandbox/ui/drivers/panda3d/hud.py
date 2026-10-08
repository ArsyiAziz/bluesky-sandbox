"""The HUD contract (``common.hud``) drawn in Panda3D's ``aspect2d``.

:class:`PandaHud` draws a :class:`~..common.hud.HudContent` - each part in its
:data:`~..common.hud.HUD_LAYOUT` corner, pinned there by its own extent so a
block of any size stays in the window - and answers which toolbar button is at
a point, so the driver runs it and keeps the click from the scene.
"""

# ruff: noqa: PLC0415 - panda3d is an optional extra; see driver.py.

from __future__ import annotations

from typing import TYPE_CHECKING

from bluesky_sandbox.ui.drivers.common.hud import HUD_LAYOUT, Anchor, Button, HudContent
from bluesky_sandbox.ui.drivers.panda3d.colors import PAUSED

if TYPE_CHECKING:
    from panda3d.core import NodePath

#: A rectangle in aspect2d units: ``(left, right, bottom, top)``.
AspectRect = tuple[float, float, float, float]

__all__ = ["AspectRect", "PandaHud"]


class _ToolbarButton:
    """One toolbar button's text node, kept while its control is shown."""

    __slots__ = ("node", "path", "state")

    def __init__(self, parent: NodePath, font) -> None:
        from panda3d.core import TextNode

        node = TextNode("button")
        node.setAlign(TextNode.ACenter)
        if font is not None:
            node.setFont(font)
        node.setCardDecal(True)
        self.node = node
        self.path = parent.attachNewNode(node)
        self.state: tuple | None = None

    def set(self, button: Button, hovered: bool, colors) -> None:
        state = (button.text, button.on, hovered)
        if state == self.state:
            return
        on_bg, off_bg, plain_bg, on_fg, off_fg, frame = colors
        self.node.setText(button.text)
        self.node.setTextColor(*(off_fg if button.on is False else on_fg))
        self.node.setCardColor(*(plain_bg if button.on is None else on_bg if button.on else off_bg))
        self.node.setCardAsMargin(0.45, 0.45, 0.25, 0.15)
        if hovered:
            self.node.setFrameColor(*frame)
            self.node.setFrameAsMargin(0.45, 0.45, 0.25, 0.15)
        else:
            self.node.clearFrame()
        self.state = state


class PandaHud:
    """The HUD: status, aircraft information, note and toolbar (see the module)."""

    MARGIN = 0.04
    TEXT_SCALE = 0.045
    BUTTON_SCALE = 0.036
    BUTTON_GAP = 0.010
    # Between the time controls and the display toggles.
    GROUP_GAP = 0.035
    _ON_BG = (0.27, 0.32, 0.42, 0.95)
    _OFF_BG = (0.10, 0.12, 0.16, 0.85)
    _PLAIN_BG = (0.17, 0.20, 0.26, 0.92)
    _ON_FG = (0.97, 0.98, 1.0, 1.0)
    _OFF_FG = (0.60, 0.65, 0.72, 1.0)
    _HOVER = (0.85, 0.90, 1.0, 0.9)

    def __init__(self, root: NodePath, font) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self._root = root.attachNewNode("hud")
        self._font = font
        kwargs = {"font": font} if font is not None else {}

        def text(scale, align, fg=(0.95, 0.95, 1.0, 1.0), bg=(0.0, 0.0, 0.0, 0.55)):
            return OnscreenText(
                text="", parent=self._root, scale=scale, fg=fg, bg=bg, align=align, mayChange=True, **kwargs
            )

        self._parts = {
            "status": text(self.TEXT_SCALE, TextNode.ARight),
            "info": text(self.TEXT_SCALE, TextNode.ALeft),
            "note": text(0.05, TextNode.ACenter, bg=(0.05, 0.08, 0.12, 0.82)),
        }
        self._tooltip = text(0.032, TextNode.ARight, fg=(0.10, 0.10, 0.12, 1.0), bg=(1.0, 0.97, 0.80, 0.95))
        self._toolbar = self._root.attachNewNode("toolbar")
        self._buttons: dict[str, _ToolbarButton] = {}
        #: Each shown button's rect, by control name (aspect2d units).
        self.button_rects: dict[str, AspectRect] = {}
        self._content = HudContent()
        self._aspect = 1.0
        self._status_fg = None

    def destroy(self) -> None:
        self._root.removeNode()
        self._buttons.clear()
        self.button_rects.clear()

    def draw(self, content: HudContent, aspect: float, mouse: tuple[float, float] | None) -> None:
        """Show ``content`` in a window ``aspect`` wide (to 1 high), the
        button under ``mouse`` (aspect2d units) hovered."""
        self._content = content
        self._aspect = aspect
        texts = {
            "status": "\n".join(content.status),
            "info": "\n".join(content.info),
            "note": content.note or "",
        }
        for name, value in texts.items():
            part = self._parts[name]
            if part["text"] != value:
                part.setText(value)
        status_fg = (*PAUSED, 1.0) if content.paused else (0.95, 0.95, 1.0, 1.0)
        if status_fg != self._status_fg:
            self._parts["status"].setFg(status_fg)
            self._status_fg = status_fg
        self._draw_toolbar(content.toolbar, mouse)
        self.layout()

    def layout(self) -> None:
        """Pin each part to its corner (see ``HUD_LAYOUT``)."""
        aspect, m = self._aspect, self.MARGIN
        edges = {
            Anchor.TOP_LEFT: {"left": -aspect + m, "top": 1.0 - m},
            Anchor.TOP_CENTER: {"center": 0.0, "top": 1.0 - m - self._toolbar_height() - 0.02},
            Anchor.TOP_RIGHT: {"right": aspect - m, "top": 1.0 - m},
            Anchor.BOTTOM_RIGHT: {"right": aspect - m, "bottom": -1.0 + m},
        }
        for name, part in self._parts.items():
            _pin(part, **edges[HUD_LAYOUT[name]])

    def hover(self, mouse: tuple[float, float] | None) -> None:
        """Hover the button under ``mouse``, and show its tooltip."""
        self._draw_toolbar(self._content.toolbar, mouse)

    def button_at(self, pos: tuple[float, float] | None) -> str | None:
        """The control whose button is at ``pos`` (aspect2d units)."""
        if pos is None:
            return None
        x, y = pos
        for name, (left, right, bottom, top) in self.button_rects.items():
            if left <= x <= right and bottom <= y <= top:
                return name
        return None

    def rects(self) -> list[AspectRect]:
        """Where the HUD is: each part with text, and the toolbar."""
        out = []
        for part in (*self._parts.values(), self._tooltip):
            if not part["text"]:
                continue
            bounds = part.getTightBounds(part.getParent())
            if bounds is not None:
                low, high = bounds
                out.append((low.getX(), high.getX(), low.getZ(), high.getZ()))
        out.extend(self.button_rects.values())
        return out

    # ---- toolbar ------------------------------------------------------------

    def _toolbar_height(self) -> float:
        if not self.button_rects:
            return 0.0
        return max(r[3] for r in self.button_rects.values()) - min(r[2] for r in self.button_rects.values())

    def _draw_toolbar(self, buttons: tuple[Button, ...], mouse: tuple[float, float] | None) -> None:
        for name in self._buttons.keys() - {b.name for b in buttons}:
            self._buttons.pop(name).path.removeNode()
        hovered = self.button_at(mouse)
        colors = (self._ON_BG, self._OFF_BG, self._PLAIN_BG, self._ON_FG, self._OFF_FG, self._HOVER)
        scale = self.BUTTON_SCALE
        # Right to left, from the window's top-right corner.
        right = self._aspect - self.MARGIN
        top = 1.0 - self.MARGIN
        self.button_rects = {}
        tooltip = None
        group = None
        for button in reversed(buttons):
            widget = self._buttons.get(button.name)
            if widget is None:
                widget = self._buttons[button.name] = _ToolbarButton(self._toolbar, self._font)
            widget.set(button, button.name == hovered, colors)
            if group is not None and button.group != group:
                right -= self.GROUP_GAP - self.BUTTON_GAP
            group = button.group
            left_u, right_u, bottom_u, top_u = widget.node.getCardActual()
            width = (right_u - left_u) * scale
            height = (top_u - bottom_u) * scale
            x = right - width / 2.0 - (right_u + left_u) / 2.0 * scale
            y = top - top_u * scale
            widget.path.setScale(scale)
            widget.path.setPos(x, 0.0, y)
            self.button_rects[button.name] = (right - width, right, top - height, top)
            if button.name == hovered:
                tooltip = (button.tooltip, right, top - height - 0.012)
            right -= width + self.BUTTON_GAP
        if tooltip is None:
            if self._tooltip["text"]:
                self._tooltip.setText("")
        else:
            text, x, y = tooltip
            self._tooltip.setText(text)
            _pin(self._tooltip, right=x, top=y)


def _pin(text, *, left=None, right=None, center=None, top=None, bottom=None) -> None:
    """Move an :class:`OnscreenText` so its drawn extent - card included -
    meets the given edges (aspect2d units)."""
    x, y = text.getTextPos()
    bounds = text.getTightBounds(text.getParent())
    if bounds is None:  # nothing to draw
        return
    low, high = bounds
    if left is not None:
        x += left - low.getX()
    elif right is not None:
        x += right - high.getX()
    elif center is not None:
        x += center - (low.getX() + high.getX()) / 2.0
    if top is not None:
        y += top - high.getZ()
    elif bottom is not None:
        y += bottom - low.getZ()
    text.setTextPos(x, y)
