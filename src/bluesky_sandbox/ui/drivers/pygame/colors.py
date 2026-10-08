"""Pygame palette + color-name -> RGB lookup.

Shared by the driver and every view module so RGB tuples are defined in
exactly one place and the COLOR-name vocabulary stays in sync with the
QtGL driver's stack-command palette.
"""

from __future__ import annotations

import pygame

from bluesky_sandbox.ui.drivers.common.palette import BACKDROP_RGB, Backdrop, overlay_rgb, state_rgb

# Sky-blue background, black aircraft, primary overlay colors - taken from
# bluesky-gym's _render_frame for a familiar look.
#: What pygame draws its scene on: the state colors are its light-backdrop
#: shades (common.palette), the same states Panda3D shows on navy.
BACKDROP   = Backdrop.LIGHT
SKY_BLUE   = BACKDROP_RGB[BACKDROP]
BLACK      = (0, 0, 0)
GREEN      = (30, 150, 30)
CONF       = state_rgb("conflict", BACKDROP)   # predicted conflict - orange
LOS        = state_rgb("los", BACKDROP)        # loss of separation - red
VIOLATION  = state_rgb("violation", BACKDROP)  # task/path constraint violation
PAUSED     = state_rgb("paused", BACKDROP)     # the stopped clock, in the status
LOS_ALPHA  = 90               # 0-255; fill opacity for LoS protection zones
GRAY       = (80, 80, 80)
DIVIDER    = (60, 60, 60)
GRID       = (90, 110, 130)   # subtle gray-blue for FL gridlines on sky-blue bg
PROT_ZONE  = (90, 90, 110)    # protection zone - gray when not in conflict
HIGHLIGHT  = state_rgb("selected", BACKDROP)  # yellow ring on the hovered aircraft (every view)


def fill_alpha_circle(
    canvas: pygame.Surface,
    rgb: tuple[int, int, int],
    center: tuple[float, float],
    radius: float,
    alpha: int = LOS_ALPHA,
) -> None:
    """Draw a translucent filled circle at ``center`` on ``canvas``.

    pygame's ``draw.circle`` ignores per-channel alpha when drawing onto
    an opaque surface, so we render the disk on a per-pixel-alpha scratch
    surface and blit it. Used for LoS protection zones where a solid red
    fill is too visually heavy.
    """
    r = max(1, int(round(radius)))
    surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.circle(surf, (*rgb, alpha), (r, r), r)
    canvas.blit(surf, (int(round(center[0])) - r, int(round(center[1])) - r))


def fill_alpha_rect(
    canvas: pygame.Surface,
    rgb: tuple[int, int, int],
    rect: pygame.Rect,
    alpha: int = LOS_ALPHA,
) -> None:
    """Draw a translucent filled rectangle on ``canvas``.

    Same trick as :func:`fill_alpha_circle` - scratch SRCALPHA surface,
    blit. Used by the side view for LoS protection-zone cross-sections.
    """
    w = max(1, int(rect.width))
    h = max(1, int(rect.height))
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*rgb, alpha))
    canvas.blit(surf, (int(rect.left), int(rect.top)))


def dim(
    rgb: tuple[int, int, int],
    *,
    factor: float = 0.70,
) -> tuple[int, int, int]:
    """Blend a color toward the sky background for non-controllable aircraft."""
    factor = max(0.0, min(1.0, float(factor)))
    return tuple(
        int(round(channel * factor + bg * (1.0 - factor)))
        for channel, bg in zip(rgb, SKY_BLUE, strict=True)
    )


# Color names recognized by render primitives -> RGB.  Mirrors the common
# BlueSky COLOR command palette so the same color string works in either
# driver (qtgl maps the name through bs.stack, pygame through this dict).
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red":     (220,  20,  60),
    "green":   ( 30, 150,  30),
    "blue":    ( 30,  80, 200),
    "cyan":    (  0, 200, 200),
    "yellow":  (235, 235,  30),  # 60 deg: clear of the conflict alert's hue
    "orange":  (255, 140,   0),
    "purple":  (160,  70, 200),
    "magenta": (220,  60, 200),
    "violation": VIOLATION,
    "white":   (255, 255, 255),
    "black":   (  0,   0,   0),
    "gray":    ( 80,  80,  80),
}


def named(name: str) -> tuple[int, int, int]:
    """The RGB triple a design color - a name or ``#rrggbb`` - is drawn in
    (GRAY when unknown): kept clear of the alert hues (common.palette)."""
    value = name.strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            return overlay_rgb((int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)))
        except ValueError:
            pass
    return overlay_rgb(NAMED_COLORS.get(value.lower(), GRAY))
