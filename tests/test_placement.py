"""Where a region sits each episode: a placement draws its spot - inside a
region, turned, clear of others - keeping its shape; and a region a placement
or generator refers to is this episode's draw of it."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.sim.bounds import (
    Blob,
    BoxFootprint,
    DiskFootprint,
    GeneratedFootprint,
    InRegion,
    LatLon,
    PlacedFootprint,
    PlacementError,
    PolygonFootprint,
    RegionBounds,
    SectorFootprint,
    generate_regions,
)
from bluesky_sandbox.sim.bounds.placement import moved_to

C = LatLon(52.0, 4.5)
CORE = RegionBounds(BoxFootprint(51.6, 52.4, 4.0, 5.0))


def _draw(regions, seed):
    return generate_regions(regions, np.random.default_rng(seed))


def test_a_placed_shape_keeps_its_form_and_lands_inside():
    regions = {"cell": RegionBounds(PlacedFootprint(DiskFootprint(C, 5.0), InRegion(within=CORE, keep_inside=True)))}
    outline = CORE.footprint.outline().buffer(1e-9)
    centers = set()
    for seed in range(40):
        cell = _draw(regions, seed)["cell"]
        assert isinstance(cell.footprint, DiskFootprint)
        assert cell.footprint.radius_nm == pytest.approx(5.0, rel=1e-3)
        assert outline.covers(cell.footprint.outline())
        centers.add(round(cell.center.lat_deg, 4))
    assert len(centers) > 30
    assert _draw(regions, 7)["cell"].center == _draw(regions, 7)["cell"].center


def test_a_turn_turns_it_about_its_center():
    sector = SectorFootprint(C, 10.0, 0.0, 20.0)
    turned = moved_to(sector, C, 90.0)
    assert turned.bearing_deg == pytest.approx(90.0, abs=0.5)
    assert (turned.center.lat_deg, turned.center.lon_deg) == pytest.approx((C.lat_deg, C.lon_deg))


def test_a_generated_shape_is_drawn_then_placed():
    blob = PlacedFootprint(GeneratedFootprint(Blob(center=C, radius_nm=4)), InRegion(within=CORE, keep_inside=True))
    a, b = (_draw({"wx": RegionBounds(blob)}, s)["wx"] for s in (1, 2))
    assert isinstance(a.footprint, PolygonFootprint)
    assert a.vertices != b.vertices and a.center != b.center


def test_it_keeps_clear_of_this_episodes_draw_of_a_region_it_avoids():
    wx = RegionBounds(PlacedFootprint(GeneratedFootprint(Blob(center=C, radius_nm=6)), InRegion(within=CORE, keep_inside=True)))
    avoid = RegionBounds(wx.footprint, wx.altitude, "wx")  # as a ref to "wx" stands in for it
    cell = RegionBounds(PlacedFootprint(DiskFootprint(C, 3.0), InRegion(within=CORE, keep_inside=True, avoid=(avoid,))))
    for seed in range(20):
        drawn = _draw({"cell": cell, "wx": wx}, seed)  # drawn after wx, whatever the order
        overlap = drawn["cell"].footprint.outline().intersection(drawn["wx"].footprint.outline())
        assert overlap.area == 0


def test_before_any_episode_it_is_where_it_can_be():
    placed = PlacedFootprint(DiskFootprint(C, 5.0), InRegion(within=CORE))
    (lat0, _), (lon0, _) = placed.bounding_box
    assert lat0 < 51.6 - 4.9 / 60  # the region grown by its reach
    inside = PlacedFootprint(DiskFootprint(C, 5.0), InRegion(within=CORE, keep_inside=True))
    assert inside.bounding_box == CORE.bounding_box


def test_what_cannot_be_placed_is_refused_by_region():
    huge = RegionBounds(PlacedFootprint(DiskFootprint(C, 80.0), InRegion(within=CORE, keep_inside=True, max_tries=5)))
    with pytest.raises(PlacementError, match="region 'big': InRegion found no spot inside its region"):
        _draw({"big": huge}, 0)  # noqa: B018
    with pytest.raises(ValueError, match="needs a region"):
        InRegion()


def test_a_placement_moves_with_its_region():
    from bluesky_sandbox.sim.scenario import transforms as T

    placed = PlacedFootprint(DiskFootprint(C, 5.0), InRegion(within=CORE))
    moved = placed.mapped(T.translator(20.0, 0.0))
    assert isinstance(moved, PlacedFootprint)
    assert moved.placement.within.bounding_box[1][0] > CORE.bounding_box[1][0]
