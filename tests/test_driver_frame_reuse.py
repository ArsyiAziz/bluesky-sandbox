"""Views read the traffic once a frame and keep what they built: TSAS rows from
one vectorized pass, Panda3D markers moved rather than rebuilt, and TSAS
widgets updated in place."""

from __future__ import annotations

import math

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import BoxFootprint, ConstantAltitudeBand, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui.drivers.common import TsasDataMixin


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        queryables = {
            "core": QueryRegion(
                RegionBounds(BoxFootprint(51.6, 52.4, 4.0, 5.0), ConstantAltitudeBand(0, 40_000)),
                color="cyan",
            ),
            "WP1": Waypoint(lat=52.0, lon=6.0, alt_ft=20_000),
            "WP2": Waypoint(lat=51.5, lon=4.5),
        }
        region = SpawnRegion(
            bounds=RegionBounds(BoxFootprint(51.2, 52.8, 3.5, 6.0)),
            n_aircraft=5,
            params={"alt_ft": (10_000, 30_000), "spd_kts": (250, 250)},
            route=["WP1"],
        )
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[region], aircraft_type="B744", conflict_free_spawn=False),
            queryables=queryables,
            max_aircraft=5,
        )


def _env(**render):
    return BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(dt=30.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
        **render,
    )


def _step(env):
    env.step({a: np.zeros(1, dtype=np.float32) for a in env.agents})


def test_tsas_rows_are_the_waypoints_current_state():
    env = _env()
    try:
        env.reset(seed=0)
        _step(env)
        driver = env._driver
        tsas = TsasDataMixin()
        for _name, waypoint in tsas.tsas_waypoints(driver):
            rows = tsas.tsas_aircraft_rows(waypoint, driver)
            assert [row.idx for row in rows] != [] and len(rows) == bs.traf.ntraf
            for row in rows:
                current = waypoint.current_state(row.idx)
                assert row.acid == bs.traf.id[row.idx]
                assert row.dist_nm == pytest.approx(current.distance_nm, rel=1e-12)
                if math.isnan(current.alt_diff_ft):
                    assert math.isnan(row.alt_diff_ft)
                else:
                    assert row.alt_diff_ft == pytest.approx(current.alt_diff_ft)
                gs_kts = bs.traf.gs[row.idx] / (1852.0 / 3600.0)
                angle = (current.bearing_deg - bs.traf.hdg[row.idx] + 540) % 360 - 180
                closing = gs_kts * math.cos(math.radians(angle))
                eta = current.distance_nm / closing * 3600.0 if closing > 1.0 else math.inf
                assert row.eta_s == pytest.approx(eta)
            assert rows == sorted(rows, key=lambda row: (row.eta_s, row.dist_nm))
    finally:
        env.close()


def test_one_aircraft_frame_serves_a_whole_frame():
    env = _env()
    try:
        env.reset(seed=0)
        _step(env)
        driver = env._driver
        core = env.episode_queryables["core"]
        with driver._aircraft_snapshot_cache_scope():
            frame = driver.aircraft_frame()
            assert driver.aircraft_frame() is frame
            assert frame.ids == list(bs.traf.id)
            np.testing.assert_array_equal(frame.lat, bs.traf.lat)
            for i in range(frame.n):
                inside = core.shape.contains(bs.traf.lat[i], bs.traf.lon[i], bs.traf.alt[i] / 0.3048)
                assert frame.query_color(i) == ("cyan" if inside else None)
        with driver._aircraft_snapshot_cache_scope():
            assert driver.aircraft_frame() is not frame  # the next frame reads anew
    finally:
        env.close()


@pytest.fixture
def panda(monkeypatch):
    pytest.importorskip("panda3d")
    env = _env(render_mode="rgb_array", frame_driver="panda3d")
    env.reset(seed=0)
    env.render()
    yield env
    env.close()


def _view(env, name):
    return next(view for view in env._driver._views if type(view).__name__ == name)


def test_panda_moves_its_aircraft_markers_instead_of_rebuilding_them(panda):
    world = _view(panda, "WorldView")
    markers = dict(world._markers)
    assert set(markers) == set(bs.traf.id)
    positions = {acid: m.body.getPos() for acid, m in markers.items()}
    _step(panda)
    panda.render()
    assert all(world._markers[acid] is marker for acid, marker in markers.items())
    assert any(world._markers[acid].body.getPos() != pos for acid, pos in positions.items())

    gone = bs.traf.id[0]
    bs.traf.delete(0)
    panda.render()
    assert gone not in world._markers and markers[gone].root.isEmpty()
    assert set(world._markers) == set(bs.traf.id)


def test_panda_tsas_updates_its_widgets_in_place(panda):
    tsas = _view(panda, "TSASView")
    widgets = list(tsas._widgets)
    assert widgets
    for _ in range(2):
        _step(panda)
        panda.render()
        assert tsas._widgets == widgets  # the same widgets, not new ones
        tables = tsas.tsas_tables(panda._driver, max_rows=tsas._ROWS_PER_STRIP)
        for table, table_widgets in zip(tables, tsas._tables, strict=True):
            for row, (_size, _swatch, cells) in zip(table.rows, table_widgets.rows, strict=True):
                assert [cell.text.getText() for cell in cells[:3]] == [
                    row.acid,
                    tsas.tsas_eta_text(row.eta_s),
                    tsas.tsas_dtm_text(row.dist_nm),
                ]
        assert [acid for _bounds, acid in tsas._row_hits] == [
            row.acid for table in tables for row in table.rows
        ]
