"""The pygame views draw what does not change once and copy it after: a frame
drawn from those caches is the frame drawn from scratch, whatever changed in
between - a step, a moving region, pan, zoom, a resize, labels, a reset."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import (
    BoxFootprint,
    ConstantAltitudeBand,
    DiskFootprint,
    Drift,
    LatLon,
    MovingFootprint,
    RegionBounds,
)
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

pygame = pytest.importorskip("pygame")


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        storm = MovingFootprint(DiskFootprint(LatLon(51.6, 3.6), 8.0), (Drift(90.0, 600.0),))
        queryables = {
            "core": QueryRegion(
                RegionBounds(BoxFootprint(51.6, 52.4, 4.0, 5.0), ConstantAltitudeBand(5_000, 25_000)),
                color="cyan",
            ),
            "storm": QueryRegion(RegionBounds(storm, ConstantAltitudeBand(0, 35_000)), color="red"),
            "WP1": Waypoint(lat=52.0, lon=6.0, alt_ft=20_000),
        }
        region = SpawnRegion(
            bounds=RegionBounds(BoxFootprint(51.2, 52.8, 3.5, 6.0)),
            n_aircraft=3,
            params={"alt_ft": (10_000, 30_000), "spd_kts": (250, 250)},
            route=["WP1"],
        )
        return EpisodeSpec(
            airspace_bounds=RegionBounds(
                BoxFootprint(51.0, 53.0, 3.0, 6.5), ConstantAltitudeBand(0, 40_000)
            ),
            spawn=SpawnConfig(regions=[region], aircraft_type="B744", conflict_free_spawn=False),
            queryables=queryables,
            max_aircraft=3,
        )


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = BlueskyEnv(
        scenario=_Scenario(),
        render_mode="rgb_array",
        config=EnvConfig(dt=60.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
    )
    env.reset(seed=0)
    env.render()
    yield env
    env.close()


def _step(env):
    env.step({a: np.zeros(1, dtype=np.float32) for a in env.agents})


def _views(env):
    driver = env._driver
    return {type(view).__name__: view for view in driver.views}


def _from_scratch(env) -> np.ndarray:
    """The frame with every render cache dropped first."""
    driver = env._driver
    driver._text_blocks.clear()
    for view in driver.views:
        view._static_layer.invalidate()
        if hasattr(view, "_plan"):
            view._plan = None
            view._slice_layer = None
    return env.render()


def _assert_cached_frame_is_fresh(env):
    cached = env.render()
    np.testing.assert_array_equal(cached, _from_scratch(env))


def test_an_unchanged_frame_is_copied_not_redrawn(env):
    layers = [view._static_layer for view in env._driver.views]
    before = [layer.redraws for layer in layers]
    first = env.render()
    second = env.render()
    assert [layer.redraws for layer in layers] == before
    np.testing.assert_array_equal(first, second)


def test_steps_and_a_moving_region_redraw_what_moved(env):
    storm = env.episode_queryables["storm"].shape.footprint
    layers = [view._static_layer for view in env._driver.views]
    for _ in range(3):
        version = storm.version
        before = [layer.redraws for layer in layers]
        _step(env)
        _assert_cached_frame_is_fresh(env)
        assert storm.version > version  # it moved, so it was drawn again
        assert all(layer.redraws > count for layer, count in zip(layers, before, strict=True))


def test_pan_zoom_and_viewport_reset_are_drawn_fresh(env):
    driver = env._driver
    for view in driver.views:
        assert view._viewport.zoom_at(0, 0, 0, 0, 1.5)  # a zoom alone, no pan
        _assert_cached_frame_is_fresh(env)
        assert view.zoom_view_at(view.rect.center, 2.0, driver)
        _assert_cached_frame_is_fresh(env)
        assert view.pan_view_by((37, -21), driver)
        _assert_cached_frame_is_fresh(env)
    driver._reset_viewports()
    _assert_cached_frame_is_fresh(env)


def test_slice_rotation_and_translation_are_drawn_fresh(env):
    views = _views(env)
    vertical, horizontal = views["VerticalView"], views["HorizontalView"]
    vertical.update_bearing(37.0)
    _assert_cached_frame_is_fresh(env)
    lat, lon = vertical._axis_origin
    vertical.update_origin(lat + 0.1, lon - 0.2)
    _assert_cached_frame_is_fresh(env)
    assert horizontal._slice_layer is not None


def test_resize_labels_trails_and_reset_are_drawn_fresh(env):
    driver = env._driver
    driver._handle_resize((900, 1000))
    frame = env.render()
    assert frame.shape == (1000, 900, 3)
    np.testing.assert_array_equal(frame, _from_scratch(env))

    driver.show_labels = False
    _assert_cached_frame_is_fresh(env)
    driver.show_labels = True
    _assert_cached_frame_is_fresh(env)

    driver.toggle_trails()
    for _ in range(2):
        _step(env)
        _assert_cached_frame_is_fresh(env)

    env.reset(seed=1)
    _assert_cached_frame_is_fresh(env)


def test_a_label_and_a_data_block_are_composed_once(env):
    driver = env._driver
    driver._text_blocks.clear()
    label = driver.render_text_bg("A\nB", (1, 2, 3))
    assert driver.render_text_bg("A\nB", (1, 2, 3)) is label
    assert driver.render_text_bg("A\nB", (1, 2, 4)) is not label
    canvas = pygame.Surface((80, 80))
    driver.blit_data_block(canvas, ["A", "B"], (1, 2, 3), 20, 30)
    assert len(driver._text_blocks) == 2  # the same block as the label
