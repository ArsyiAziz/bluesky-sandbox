"""The state palette (``common.palette``): each state one hue, in a shade
legible on each driver's backdrop, the states told apart - and every driver
drawing them from it, none with colors of its own."""

from __future__ import annotations

import colorsys

import pytest

from bluesky_sandbox.ui.drivers.common.palette import ALERTS, BACKDROP_RGB, STATES, Backdrop


def _luminance(rgb):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = map(lin, rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _hue(rgb):
    return colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))[0] * 360.0


@pytest.mark.parametrize("state", [*ALERTS, "paused"])
def test_a_state_is_one_hue_on_every_backdrop(state):
    light, dark = STATES[state].light, STATES[state].dark
    assert abs((_hue(light) - _hue(dark) + 180) % 360 - 180) < 8


@pytest.mark.parametrize("backdrop", list(Backdrop))
@pytest.mark.parametrize("state", ["normal", *ALERTS])
def test_a_state_reads_on_its_backdrop(state, backdrop):
    # 3:1 - the least for a mark that must be seen (WCAG, non-text).
    assert _contrast(STATES[state].on(backdrop), BACKDROP_RGB[backdrop]) >= 3.0


def test_the_alerts_are_told_apart():
    hues = [_hue(STATES[s].dark) for s in ALERTS]
    gaps = [abs((a - b + 180) % 360 - 180) for i, a in enumerate(hues) for b in hues[i + 1:]]
    assert min(gaps) >= 30


def test_every_driver_draws_the_palettes_colors():
    pytest.importorskip("pygame")
    from bluesky_sandbox.ui.drivers.panda3d import colors as panda
    from bluesky_sandbox.ui.drivers.pygame import colors as pg

    assert (pg.CONF, pg.LOS, pg.VIOLATION) == tuple(STATES[s].light for s in ("conflict", "los", "violation"))
    assert pg.SKY_BLUE == BACKDROP_RGB[Backdrop.LIGHT]
    for state in ("normal", *ALERTS):
        assert tuple(round(c * 255) for c in panda.STATE_COLORS[state]) == STATES[state].dark
    assert tuple(round(c * 255) for c in panda.BACKGROUND) == BACKDROP_RGB[Backdrop.DARK]
    from bluesky_sandbox.ui.drivers.panda3d.views.tsas import TSASView

    assert TSASView._LOS[:3] == panda.STATE_COLORS["los"]


# ---- a design's colors keep clear of the alerts --------------------------- #


def _hue_gap(rgb, state):
    return abs((_hue(rgb) - _hue(STATES[state].dark) + 180) % 360 - 180)


@pytest.mark.parametrize(
    "rgb",
    [(220, 20, 60), (255, 140, 0), (160, 70, 200), (255, 0, 0), (200, 120, 40), (150, 70, 220)],
    ids=["red", "orange", "purple", "pure red", "a custom amber", "violation itself"],
)
def test_a_design_color_near_an_alert_is_drawn_clear_of_it(rgb):
    from bluesky_sandbox.ui.drivers.common.palette import RESERVED_DEG, overlay_rgb

    drawn = overlay_rgb(rgb)
    assert all(_hue_gap(drawn, s) >= RESERVED_DEG for s in ALERTS)
    # Same saturation and brightness: the color the design chose, turned.
    before = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    after = colorsys.rgb_to_hsv(*(c / 255 for c in drawn))
    assert after[1:] == pytest.approx(before[1:], abs=0.02)


@pytest.mark.parametrize("rgb", [(0, 200, 200), (30, 150, 30), (30, 80, 200), (220, 60, 200), (235, 235, 30), (80, 80, 80)])
def test_a_design_color_clear_of_the_alerts_is_drawn_as_it_is(rgb):
    from bluesky_sandbox.ui.drivers.common.palette import overlay_rgb

    assert overlay_rgb(rgb) == rgb


def test_every_driver_draws_a_design_color_the_same_way():
    pytest.importorskip("pygame")
    from bluesky_sandbox.ui.drivers.panda3d import colors as panda
    from bluesky_sandbox.ui.drivers.pygame import colors as pg

    for name in ("red", "orange", "#ff8800", "cyan"):
        assert tuple(round(c * 255) for c in panda.color(name)[:3]) == pg.named(name)
    assert pg.named("red") != pg.NAMED_COLORS["red"]


def test_no_default_is_an_alert_color():
    from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
    from bluesky_sandbox.sim.queryables import QueryRegion
    from bluesky_sandbox.ui.drivers.common.palette import overlay_rgb
    from bluesky_sandbox.ui.drivers.pygame import colors as pg

    region = QueryRegion(RegionBounds(BoxFootprint(51, 52, 4, 5)))
    rgb = pg.NAMED_COLORS[region.color]
    assert overlay_rgb(rgb) == rgb


def test_the_designer_offers_only_colors_drawn_as_named_and_warns_of_the_rest():
    from bluesky_sandbox.ui.designer import catalog
    from bluesky_sandbox.ui.designer import spec as S
    from bluesky_sandbox.ui.designer.preview import alert_hued_colors
    from bluesky_sandbox.ui.designer.tests.test_designer import _example_design_spec

    assert set(catalog.alert_colors()) == {"red", "orange", "purple"}
    assert catalog.colors()["red"] != "#dc143c"  # reported as drawn
    spec = _example_design_spec()
    spec.queryables["goal"]["color"] = "orange"
    spec = S.DesignSpec.from_json(spec.to_json())
    ((where, color, drawn),) = [w for w in alert_hued_colors(spec) if w[1] == "orange"]
    assert where == "queryables.goal" and drawn == catalog.colors()["orange"]
