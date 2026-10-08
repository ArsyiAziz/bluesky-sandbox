"""The HUD contract (``common.hud``): controls - display toggles and actions -
declared once, every driver binding their keys and drawing their buttons from
them; one content, each part in its corner; aircraft labels showing a full
block, a callsign or nothing, with arrows for climbing and speeding up; and
Panda3D drawing them flat on the screen, clear of each other and the HUD."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.ui.drivers.common import ASCII_TRENDS, DISPLAY_TOGGLES
from bluesky_sandbox.ui.drivers.common.readouts import aircraft_label_lines

from test_driver_frame_reuse import _env, _step

pygame = pytest.importorskip("pygame")
from bluesky_sandbox.ui.drivers.pygame.driver import PygameSimDriver  # noqa: E402


@pytest.fixture
def driver():
    return PygameSimDriver(offscreen=True)


def test_a_toggle_steps_through_its_values_and_tells_the_driver(driver, monkeypatch):
    seen = []
    monkeypatch.setattr(driver, "on_toggled", lambda toggle, value: seen.append((toggle.name, value)))
    assert [driver.toggle("aircraft_labels") for _ in range(3)] == ["callsign", "off", "full"]
    driver.toggle_velocity_obstacles()  # the named methods are toggles too
    assert seen[-1] == ("show_velocity_obstacles", True)
    with pytest.raises(KeyError, match="no display toggle"):
        driver.toggle("show_everything")
    with pytest.raises(KeyError, match="not a display toggle"):
        driver.display_toggle("pause")


def test_turning_trails_off_drops_their_points(driver):
    driver.show_trails = True
    driver._trails["X"] = object()
    driver.toggle_trails()
    assert not driver.show_trails and driver._trails == {}


def test_each_driver_has_the_controls_it_can_run():
    from bluesky_sandbox.ui.drivers.common import TIME_CONTROLS
    from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver
    from bluesky_sandbox.ui.drivers.qtgl.driver import QtGLSimDriver

    shared = [c.name for c in (*TIME_CONTROLS, *DISPLAY_TOGGLES)]
    assert [c.name for c in PygameSimDriver(offscreen=True).controls] == shared
    panda = [c.name for c in Panda3DSimDriver.CONTROLS if c.name not in Panda3DSimDriver.UNSUPPORTED_CONTROLS]
    assert panda == [n for n in shared if n != "show_velocity_obstacles"]
    assert "aircraft_labels" in QtGLSimDriver.UNSUPPORTED_CONTROLS
    keys = [k for c in (*TIME_CONTROLS, *DISPLAY_TOGGLES) for k in c.keys]
    assert len(keys) == len(set(keys))  # one key, one control


def test_a_key_runs_its_control_shift_first(driver):
    assert driver.control_for_key("l") == "aircraft_labels"
    assert driver.control_for_key("l", shift=True) == "show_labels"
    assert driver.control_for_key("=", shift=True) == "faster"  # Shift+= is +
    assert driver.control_for_key("space") == "pause"
    assert driver.control_for_key("backspace", shift=True) == "reset_episode"
    assert driver.control_for_key("z") is None


def test_running_a_control_says_what_it_changed(driver):
    driver.activate("aircraft_labels")
    assert driver.aircraft_labels == "callsign" and driver.note == "aircraft labels: callsign"
    driver.activate("pause")
    assert driver._paused and driver.note == "paused"
    driver.activate("realtime")
    driver.activate("faster")
    assert driver.note == "REAL 2x"
    driver._note = ("old", 0.0)
    assert driver.note is None  # gone once its time is up


def test_the_hud_content_is_the_drivers_state():
    env = _env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        driver = env._driver
        driver._selected = bs.traf.id[0]
        content = driver.hud_content()
        assert content.toolbar == ()  # offscreen: a recorded frame has no toolbar
        assert content.status[0].startswith("T+") and content.info[0] == bs.traf.id[0]
        driver.offscreen = False
        buttons = {b.name: b for b in driver.hud_content().toolbar}
        assert set(buttons) == {c.name for c in driver.controls if c.button}
        assert buttons["slower"].on is None and buttons["pause"].on is False
        assert buttons["aircraft_labels"].text == "LBL\u00b7F" and buttons["show_trails"].group == "display"
        assert buttons["show_labels"].tooltip == "region and waypoint names (Shift+L)"
        driver.offscreen = True
    finally:
        env.close()


def test_pygame_reads_keys_and_draws_one_toolbar(monkeypatch):
    env = _env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        env.render()
        driver = env._driver
        monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_SHIFT)
        assert driver._control_for_event_key(pygame.K_l) == "show_labels"
        monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_CTRL)
        assert driver._control_for_event_key(pygame.K_BACKSPACE) == "reset_episode"
        driver.offscreen = False  # a window: the toolbar shows
        rects = driver._toolbar_rects(driver.hud_content())
        assert [b.name for b, _ in rects][0] == "show_velocity_obstacles"  # right to left
        width = driver.window_size[0]
        assert all(r.right <= width and r.top >= 0 for _, r in rects)
        button, rect = rects[-1]
        assert driver.toolbar_button_at(*rect.center) == button.name == "pause"
        driver.offscreen = True
    finally:
        env.close()


def test_a_label_shows_its_trends():
    lines = aircraft_label_lines(
        "KLM1", "B738", alt_ft=18_500, gs_kts=320, cas_kts=250, mach=0.6, alt_trend=1, speed_trend=-1
    )
    assert lines == ["KLM1  B738", "FL185↑ GS320  CAS250↓"]
    level = aircraft_label_lines("KLM1", "", alt_ft=35_000, gs_kts=450, cas_kts=260, mach=0.78, glyphs=ASCII_TRENDS)
    assert level[1] == "FL350  GS450  M.78"


def test_labels_follow_their_setting_but_the_tracked_aircraft_keeps_its_block():
    env = _env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        _step(env)
        driver = env._driver
        driver._selected = bs.traf.id[0]
        driver.aircraft_labels = "callsign"
        assert driver.format_aircraft_marker_label_lines(1) == [bs.traf.id[1]]
        assert len(driver.format_aircraft_marker_label_lines(0)) == 2
        driver.aircraft_labels = "off"
        assert driver.format_aircraft_marker_label_lines(1) == []
        bs.stack.stack(f"ALT {bs.traf.id[1]} 39000")
        driver.aircraft_labels = "full"
        _step(env)
        assert "↑" in driver.format_aircraft_marker_label_lines(1)[1]
    finally:
        env.close()


# ---- Panda3D ------------------------------------------------------------- #


def _place(tags, *rects, width=800.0, height=600.0, previous=None, w=60.0, h=30.0):
    from bluesky_sandbox.ui.drivers.panda3d.views.tags import AircraftTag

    tag = AircraftTag("A", 400.0, 300.0, ("A",), (1, 1, 1, 1), None, (1, 1, 1, 1))
    return tags._place(tag, w, h, previous, list(rects), width, height)


def test_a_block_takes_the_clearest_corner_and_keeps_it():
    from bluesky_sandbox.ui.drivers.panda3d.views.tags import AircraftTags

    tags = object.__new__(AircraftTags)
    corner, _ = _place(tags)
    assert corner == 0  # up and to the right, when all are clear
    busy = (414.0, 256.0, 474.0, 286.0)  # a block already there
    corner, rect = _place(tags, busy)
    assert corner != 0 and rect != busy
    assert _place(tags, previous=3)[0] == 3  # as good as any: it stays
    assert _place(tags, (414.0, 256.0, 474.0, 286.0), previous=0)[0] != 0  # an overlap moves it
    corner, _ = _place(tags, width=430.0)  # near the right edge
    assert corner in (1, 3)


@pytest.fixture
def panda():
    pytest.importorskip("panda3d")
    env = _env(render_mode="rgb_array", frame_driver="panda3d")
    env.reset(seed=0)
    _step(env)
    env.render()
    yield env
    env.close()


def test_panda_draws_blocks_beside_their_aircraft_and_clear_of_each_other(panda):
    from bluesky_sandbox.ui.drivers.panda3d.views.tags import _overlap

    driver = panda._driver
    world = next(v for v in driver._views if type(v).__name__ == "WorldView")
    rects = world._tags.rects
    assert rects and set(rects) <= set(bs.traf.id)
    on_screen = {acid: (x, y) for x, y, acid in world._screen_aircraft}
    for acid, rect in rects.items():
        x, y = on_screen[acid]
        assert not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])  # not on its aircraft
    overlaps = [_overlap(a, b) for i, a in enumerate(rects.values()) for b in list(rects.values())[i + 1:]]
    assert sum(overlaps) < 0.05 * sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects.values())
    driver.toggle("aircraft_labels")
    driver.toggle("aircraft_labels")  # off
    driver._selected = None
    driver.auto_track = False
    panda.render()
    assert world._tags.rects == {}


def test_panda_keeps_its_hud_inside_the_window_and_runs_its_buttons(panda):
    driver = panda._driver
    width, height = driver._show.win.getXSize(), driver._show.win.getYSize()
    driver.offscreen = False  # a window: the toolbar shows
    driver._selected = bs.traf.id[0]
    driver.activate("show_trails")
    panda.render()
    rects = driver.hud_rects_px()
    assert len(rects) >= 4  # status, info, note, buttons
    for left, top, right, bottom in rects:
        assert -1 <= left and right <= width + 1 and -1 <= top and bottom <= height + 1
    hud = driver._hud
    left, right, bottom, top = hud.button_rects["show_trails"]
    aspect = driver._show.getAspectRatio()
    x = ((left + right) / 2 / aspect + 1) / 2 * width
    y = (1 - (bottom + top) / 2) / 2 * height
    assert driver.toolbar_button_at(x, y) == "show_trails"
    assert driver.toolbar_button_at(width / 2, height / 2) is None
    driver.offscreen = True


def test_panda_falls_back_to_ascii_arrows_where_its_font_has_none():
    from bluesky_sandbox.ui.drivers.panda3d.driver import _has_glyphs

    assert not _has_glyphs(None, "↑")
