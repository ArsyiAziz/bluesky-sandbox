"""Shared palette + scene constants for the panda3d driver.

Lives in its own module so :class:`Panda3DSimDriver` and the
:mod:`views` subpackage can both import it without a circular
dependency.  Mirrors :mod:`bluesky_sandbox.ui.drivers.pygame.colors` -
the named-colour table is intentionally aligned so the same ``COLOR``
string on a render primitive paints the same hue in either driver.
"""

from __future__ import annotations

# Earth-radius approximation for the local-ENU tangent plane.  Good to
# ~0.5 % at the typical airspace scale (~hundreds of NM) which is well
# below the visual fidelity threshold for an ATC-style view.
M_PER_DEG = 111_320.0


# Named colours mirror the pygame palette so the same COLOR strings on
# render primitives produce visually-matching output across drivers.
# Panda3D wants RGBA floats in [0, 1].
NAMED_COLORS: dict[str, tuple[float, float, float]] = {
    "red":     (220 / 255,  20 / 255,  60 / 255),
    "green":   ( 30 / 255, 150 / 255,  30 / 255),
    "blue":    ( 30 / 255,  80 / 255, 200 / 255),
    "cyan":    (  0 / 255, 200 / 255, 200 / 255),
    "yellow":  (240 / 255, 220 / 255,  30 / 255),
    "orange":  (255 / 255, 140 / 255,   0 / 255),
    "purple":  (160 / 255,  70 / 255, 200 / 255),
    "magenta": (220 / 255,  60 / 255, 200 / 255),
    "violation": (150 / 255, 70 / 255, 220 / 255),
    "white":   (1.0, 1.0, 1.0),
    "black":   (0.0, 0.0, 0.0),
    "gray":    (80 / 255, 80 / 255, 80 / 255),
}


STATE_COLORS: dict[str, tuple[float, float, float]] = {
    "normal":   (0.70, 0.90, 1.00),    # bright cyan - pops on dark-navy bg
    "violation": NAMED_COLORS["violation"],
    "conflict": (1.00, 0.55, 0.00),    # orange
    "los":      (1.00, 0.30, 0.40),    # bright crimson (lifted for visibility)
}


HIGHLIGHT = (1.00, 0.84, 0.12)         # yellow ring on selection


# Pygame chevron geometry - kept in sync with
# :class:`HorizontalView`'s ``_AC_*_FRAC`` constants so the 3D and 2D
# drivers render the same plane-symbol silhouette.
CHEVRON_WING_FRAC  = 0.8     # half-wingspan / half-length
CHEVRON_NOTCH_FRAC = 0.45    # rear-notch depth / half-length


def color(name: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Look up an RGBA tuple for a render-primitive colour name.

    Unknown names fall back to ``gray`` so a typo in a config string
    surfaces visually rather than crashing the renderer.
    """
    value = name.strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            return (
                int(value[1:3], 16) / 255,
                int(value[3:5], 16) / 255,
                int(value[5:7], 16) / 255,
                alpha,
            )
        except ValueError:
            pass
    r, g, b = NAMED_COLORS.get(value.lower(), NAMED_COLORS["gray"])
    return r, g, b, alpha


def dim_rgb(
    rgb: tuple[float, float, float],
    *,
    factor: float = 0.45,
) -> tuple[float, float, float]:
    """Blend a color toward the scene background for background aircraft."""
    factor = max(0.0, min(1.0, float(factor)))
    bg = (0.04, 0.06, 0.10)
    return tuple(
        channel * factor + bg_channel * (1.0 - factor)
        for channel, bg_channel in zip(rgb, bg, strict=True)
    )
