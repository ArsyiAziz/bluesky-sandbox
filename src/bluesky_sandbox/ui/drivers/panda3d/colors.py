"""Shared palette + scene constants for the panda3d driver.

Lives in its own module so :class:`Panda3DSimDriver` and the
:mod:`views` subpackage can both import it without a circular
dependency.  Mirrors :mod:`bluesky_sandbox.ui.drivers.pygame.colors` -
the named-color table is intentionally aligned so the same ``COLOR``
string on a render primitive paints the same hue in either driver.
"""

from __future__ import annotations

from bluesky_sandbox.ui.drivers.common.palette import BACKDROP_RGB, Backdrop, overlay_rgb, state_rgbf

# Earth-radius approximation for the local-ENU tangent plane.  Good to
# ~0.5 % at the typical airspace scale (~hundreds of NM) which is well
# below the visual fidelity threshold for an ATC-style view.
M_PER_DEG = 111_320.0


# Named colors mirror the pygame palette so the same COLOR strings on
# render primitives produce visually-matching output across drivers.
# Panda3D wants RGBA floats in [0, 1].
NAMED_COLORS: dict[str, tuple[float, float, float]] = {
    "red":     (220 / 255,  20 / 255,  60 / 255),
    "green":   ( 30 / 255, 150 / 255,  30 / 255),
    "blue":    ( 30 / 255,  80 / 255, 200 / 255),
    "cyan":    (  0 / 255, 200 / 255, 200 / 255),
    "yellow":  (235 / 255, 235 / 255,  30 / 255),
    "orange":  (255 / 255, 140 / 255,   0 / 255),
    "purple":  (160 / 255,  70 / 255, 200 / 255),
    "magenta": (220 / 255,  60 / 255, 200 / 255),
    "violation": state_rgbf("violation", Backdrop.DARK),
    "white":   (1.0, 1.0, 1.0),
    "black":   (0.0, 0.0, 0.0),
    "gray":    (80 / 255, 80 / 255, 80 / 255),
}


#: What Panda3D draws its scene on: the state colors are its dark-backdrop
#: shades (common.palette), the same states pygame shows on sky blue.
BACKDROP = Backdrop.DARK
BACKGROUND = tuple(c / 255.0 for c in BACKDROP_RGB[BACKDROP])

STATE_COLORS: dict[str, tuple[float, float, float]] = {
    state: state_rgbf(state, BACKDROP) for state in ("normal", "conflict", "los", "violation")
}


HIGHLIGHT = state_rgbf("selected", BACKDROP)  # yellow ring on selection
PAUSED = state_rgbf("paused", BACKDROP)


# Pygame chevron geometry - kept in sync with
# :class:`HorizontalView`'s ``_AC_*_FRAC`` constants so the 3D and 2D
# drivers render the same plane-symbol silhouette.
CHEVRON_WING_FRAC  = 0.8     # half-wingspan / half-length
CHEVRON_NOTCH_FRAC = 0.45    # rear-notch depth / half-length


def color(name: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """The RGBA a design color - a name or ``#rrggbb`` - is drawn in, kept
    clear of the alert hues (common.palette).

    Unknown names fall back to ``gray`` so a typo in a config string
    surfaces visually rather than crashing the renderer.
    """
    value = name.strip()
    rgb = None
    if len(value) == 7 and value.startswith("#"):
        try:
            rgb = (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
        except ValueError:
            rgb = None
    if rgb is None:
        r, g, b = NAMED_COLORS.get(value.lower(), NAMED_COLORS["gray"])
        rgb = (round(r * 255), round(g * 255), round(b * 255))
    r, g, b = overlay_rgb(rgb)
    return r / 255, g / 255, b / 255, alpha


def dim_rgb(
    rgb: tuple[float, float, float],
    *,
    factor: float = 0.45,
) -> tuple[float, float, float]:
    """Blend a color toward the scene background for background aircraft."""
    factor = max(0.0, min(1.0, float(factor)))
    bg = BACKGROUND
    return tuple(
        channel * factor + bg_channel * (1.0 - factor)
        for channel, bg_channel in zip(rgb, bg, strict=True)
    )
