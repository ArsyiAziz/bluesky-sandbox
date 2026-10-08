"""Regions that move during an episode: each motion as it says - a drift at
its speed (bouncing inside a region), a spin at its rate, growth within its
limits - its ranges drawn once an episode, its shape kept; and in the
environment, every holder of a moving region seeing it at the simulation's
time."""

from __future__ import annotations

import math

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import (
    BoxFootprint,
    DiskFootprint,
    Drift,
    Grow,
    LatLon,
    LocalFrame,
    MovingFootprint,
    RegionBounds,
    SectorFootprint,
    Spin,
    generate_regions,
    moving_in,
)
from bluesky_sandbox.sim.queryables import QueryRegion
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

C = LatLon(52.0, 4.5)
CORE = RegionBounds(BoxFootprint(51.6, 52.4, 4.0, 5.0))


def _moving(footprint, *motions) -> MovingFootprint:
    return MovingFootprint(footprint, motions)


def test_a_drift_moves_at_its_heading_and_speed_keeping_its_shape():
    f = _moving(DiskFootprint(C, 3.0), Drift(heading_deg=90.0, speed_kts=20.0))
    f.advance(1800.0)
    x, y = LocalFrame(C).to_xy_nm(f.center_point())
    assert (x, y) == pytest.approx((10.0, 0.0), abs=0.05)
    assert isinstance(f.current, DiskFootprint) and f.current.radius_nm == pytest.approx(3.0, rel=1e-3)


def test_a_drift_within_a_region_bounces_keeping_the_shape_inside():
    f = _moving(DiskFootprint(C, 3.0), Drift(heading_deg=60.0, speed_kts=80.0, within=CORE))
    outline = CORE.footprint.outline().buffer(1e-6)
    for t in range(0, 8 * 3600, 300):
        f.advance(float(t))
        assert outline.covers(f.outline())


def test_a_spin_turns_it_and_growth_stays_within_its_limits():
    sector = _moving(SectorFootprint(C, 10.0, 0.0, 20.0), Spin(rate_deg_s=0.05))
    sector.advance(1800.0)
    assert sector.current.bearing_deg == pytest.approx(90.0, abs=0.5)
    disk = _moving(DiskFootprint(C, 4.0), Grow(rate_per_hr=1.0, max_scale=1.5))
    disk.advance(1800.0)
    assert disk.current.radius_nm == pytest.approx(6.0, rel=1e-3)
    disk.advance(36000.0)
    assert disk.current.radius_nm == pytest.approx(6.0, rel=1e-3)  # at its 1.5 cap
    shrink = _moving(DiskFootprint(C, 4.0), Grow(rate_per_hr=-1.0, min_scale=0.5))
    shrink.advance(36000.0)
    assert shrink.current.radius_nm == pytest.approx(2.0, rel=1e-3)


def test_motions_compose_in_order():
    f = _moving(DiskFootprint(C, 2.0), Grow(rate_per_hr=2.0), Drift(heading_deg=0.0, speed_kts=10.0))
    f.advance(3600.0)
    assert f.current.radius_nm == pytest.approx(6.0, rel=1e-3)
    assert LocalFrame(C).to_xy_nm(f.center_point())[1] == pytest.approx(10.0, abs=0.05)


def test_ranges_are_drawn_once_an_episode_from_the_regions_stream():
    region = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=(0, 360), speed_kts=(5, 25))))
    a, b, again = (generate_regions({"cell": region}, np.random.default_rng(s))["cell"].footprint for s in (1, 2, 1))
    assert a.motions[0].heading_deg != b.motions[0].heading_deg
    assert a.motions == again.motions
    assert all(isinstance(v, float) for v in (a.motions[0].heading_deg, a.motions[0].speed_kts))


def test_its_boundary_is_of_the_shape_now():
    region = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=90.0, speed_kts=60.0)))
    before = region.boundary.nearest(C).signed_nm
    region.footprint.advance(1800.0)
    assert before > 0 and region.boundary.nearest(C).signed_nm < 0


def test_every_holder_of_a_moving_region_is_found_once():
    f = _moving(DiskFootprint(C, 3.0), Spin())
    region = RegionBounds(f)
    found = moving_in({"a": region, "b": [region, {"c": QueryRegion(region)}]})
    assert found == [f]


class _Storm:
    """An episode with one drifting region, held by a queryable."""

    def sample(self, rng):
        return self.support()

    def support(self):
        storm = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=90.0, speed_kts=60.0)))
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={"storm": QueryRegion(storm)},
            max_aircraft=0,
            bounds={"storm": storm},
        )


def test_in_the_environment_it_moves_with_simulation_time():
    env = BlueskyEnv(
        scenario=_Storm(),
        config=EnvConfig(dt=60.0, obs_fields=[obs.CasKts()], action_fields=[act.HdgDeg()]),
    )
    try:
        env.reset(seed=0)
        assert bs.traf.cre("MOV1", "B744", C.lat_deg, C.lon_deg, 90, 20_000 * ft, 250 * kts)
        env.set_aircraft_control_state("MOV1", AircraftControlState.CONTROLLED)
        storm = env.episode_spec.queryables["storm"].bounds
        assert storm is env.episode_bounds["storm"]
        start = float(bs.sim.simt)
        for _ in range(5):
            env.step({"MOV1": np.array([90.0], dtype=np.float32)})
        elapsed = float(bs.sim.simt) - start
        x, _ = LocalFrame(C).to_xy_nm(storm.center)
        assert elapsed > 0 and x == pytest.approx(60.0 * (elapsed + (start - env._motion_t0)) / 3600.0, abs=0.05)
    finally:
        env.close()


def test_a_groups_turn_turns_a_drifts_heading_and_scales_its_speed():
    from bluesky_sandbox.sim.scenario import transforms as T

    region = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=0.0, speed_kts=20.0)))
    turned = T.rotate_bounds(region, (C.lat_deg, C.lon_deg), -90.0)  # a quarter turn clockwise
    drift = turned.footprint.motions[0]
    assert drift.heading_deg == pytest.approx(90.0, abs=0.5) and drift.speed_kts == pytest.approx(20.0, rel=1e-3)
    scaled_ = T.transform_bounds(region, T.scaler((C.lat_deg, C.lon_deg), 2.0))
    assert scaled_.footprint.motions[0].speed_kts == pytest.approx(40.0, rel=1e-3)
    ranged = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=(0.0, 30.0), speed_kts=(10.0, 20.0))))
    drift = T.rotate_bounds(ranged, (C.lat_deg, C.lon_deg), -90.0).footprint.motions[0]
    assert drift.heading_deg == pytest.approx((90.0, 120.0), abs=0.5)


def test_an_update_cadence_is_substep_or_step():
    assert MovingFootprint(DiskFootprint(C, 3.0), (Spin(),)).update == "step"
    with pytest.raises(ValueError, match="'substep' or 'step'"):
        MovingFootprint(DiskFootprint(C, 3.0), (Spin(),), update="sometimes")


class _StepStorm(_Storm):
    def support(self):
        spec = super().support()
        storm = RegionBounds(
            MovingFootprint(DiskFootprint(C, 3.0), (Drift(heading_deg=90.0, speed_kts=60.0),), update="step")
        )
        spec.queryables["storm"] = QueryRegion(storm)
        spec.bounds["storm"] = storm
        return spec


def test_a_region_moved_once_a_step_holds_between_steps():
    env = BlueskyEnv(
        scenario=_StepStorm(),
        config=EnvConfig(dt=60.0, obs_fields=[obs.CasKts()], action_fields=[act.HdgDeg()]),
    )
    try:
        env.reset(seed=0)
        storm = env.episode_bounds["storm"].footprint
        moves = []
        env._hooks.on_sim_step = lambda: moves.append(storm.version)  # each substep
        assert bs.traf.cre("MOV2", "B744", C.lat_deg, C.lon_deg, 90, 20_000 * ft, 250 * kts)
        env.set_aircraft_control_state("MOV2", AircraftControlState.CONTROLLED)
        before = storm.version
        env.step({"MOV2": np.array([90.0], dtype=np.float32)})
        assert len(set(moves)) == 1  # held through the step's substeps
        assert storm.version == before + 1  # then moved, once, at its end
    finally:
        env.close()


def test_a_driver_redraws_a_moving_region_when_it_moves_and_only_then():
    from bluesky_sandbox.ui.display.overlays import BoundsResource
    from bluesky_sandbox.ui.drivers.common.render_dispatch import PrimitiveDrawMixin

    class _Driver(PrimitiveDrawMixin):
        def __init__(self):
            self.drawn, self.moved = [], []

        def draw_polygon(self, polygon):
            self.drawn.append(polygon)

        def on_polygons_moved(self, polygons):
            self.moved.append(polygons)

    storm = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=90.0, speed_kts=60.0)))
    still = RegionBounds(DiskFootprint(C, 3.0))
    driver = _Driver()
    driver.draw_renderables([BoundsResource(storm, "red", "storm"), BoundsResource(still, "blue", "still")])
    start = list(driver.drawn[0].vertices)
    assert driver.sync_moving() == []
    storm.footprint.advance(600.0)
    (moved,) = driver.sync_moving()
    assert moved is driver.drawn[0] and moved.vertices != start
    assert driver.sync_moving() == [] and len(driver.moved) == 1


def test_a_moving_regions_geometry_carries_its_next_hour():
    from bluesky_sandbox.sim.scenario.geometry_json import bounds_geometry

    storm = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=90.0, speed_kts=60.0)))
    storm.footprint.advance(120.0)
    g = bounds_geometry(storm)
    assert len(g["frames"]) == 61 and g["frames"][0]["t_s"] == 0 and g["frames"][-1]["t_s"] == 3600
    assert storm.footprint.t_s == 120.0  # left as it was
    design = RegionBounds(_moving(DiskFootprint(C, 3.0), Drift(heading_deg=(0, 360))))
    assert "frames" not in bounds_geometry(design)  # ranges not drawn: nothing to play


# ---- a driver draws a moving region where it is now ---------------------- #


def test_a_driver_brings_a_moving_regions_polygon_up_to_date():
    from bluesky_sandbox.sim.bounds import RegionBounds
    from bluesky_sandbox.sim.bounds.motion import Drift
    from bluesky_sandbox.ui.display.overlays import Polygon
    from bluesky_sandbox.ui.drivers.common.render_dispatch import PrimitiveDrawMixin

    class Driver(PrimitiveDrawMixin):
        moved: list = []

        def on_polygons_moved(self, polygons):
            self.moved.extend(polygons)

    region = RegionBounds(MovingFootprint(DiskFootprint(C, 3.0), (Drift(90.0, 360.0),)))
    polygon = Polygon(vertices=region.vertices, meta={"bounds": region})
    driver = Driver()
    driver.draw_renderables([polygon])
    assert driver.sync_moving() == [] and driver.moved == []  # it has not moved
    start = list(polygon.vertices)
    region.footprint.advance(600.0)  # ten minutes at 360 kt east: 60 nm
    assert driver.sync_moving() == [polygon] and driver.moved == [polygon]
    assert polygon.vertices == region.vertices != start
    assert driver.sync_moving() == []  # up to date until it moves again
