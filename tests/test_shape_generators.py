"""Shapes drawn anew each episode: every draw a valid shape inside its
generator's envelope, the same for the same seed, meeting its constraints -
and a region drawn by one redrawn each episode, its own stream untouched by
the others."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from bluesky_sandbox.sim.bounds import (
    Blob,
    BoxFootprint,
    ConvexPolygon,
    GeneratedFootprint,
    GenerationError,
    LatLon,
    RegionBounds,
    VoronoiSectors,
    generate_regions,
)
from bluesky_sandbox.sim.bounds.generators import area_nm2, valtr_polygon
from bluesky_sandbox.sim.scenario import RegionParamSampler

C = LatLon(52.0, 4.5)
AIRSPACE = RegionBounds(BoxFootprint(51.6, 52.4, 4.0, 5.0))

SINGLE = {
    "convex": ConvexPolygon(center=C, radius_nm=(8, 15), vertices=(5, 9)),
    "blob": Blob(center=C, radius_nm=10, amplitude=(0.2, 0.5)),
}


@pytest.mark.parametrize("generator", SINGLE.values(), ids=SINGLE)
def test_every_draw_is_a_valid_shape_inside_the_envelope(generator):
    envelope = generator.envelope().outline().buffer(1e-9)
    for seed in range(100):
        (shape,) = generator.generate(np.random.default_rng(seed))
        assert shape.shape.is_valid
        assert envelope.covers(shape.outline())


@pytest.mark.parametrize("generator", [*SINGLE.values(), VoronoiSectors(parent=AIRSPACE, sectors=4)], ids=[*SINGLE, "voronoi"])
def test_the_same_seed_draws_the_same_shapes_and_another_seed_others(generator):
    draw = lambda seed: [s.vertices for s in generator.generate(np.random.default_rng(seed))]  # noqa: E731
    assert draw(3) == draw(3)
    assert draw(3) != draw(4)


def test_valtr_draws_convex_polygons_of_n_vertices():
    for n in (3, 6, 20):
        xy = valtr_polygon(n, np.random.default_rng(n))
        polygon = shapely.Polygon(xy)
        assert len(xy) == n and polygon.is_valid
        assert polygon.convex_hull.area == pytest.approx(polygon.area)


def test_a_convex_polygon_has_the_vertices_and_radius_asked_for():
    generator = ConvexPolygon(center=C, radius_nm=12, vertices=7)
    for seed in range(20):
        (shape,) = generator.generate(np.random.default_rng(seed))
        assert len(shape.vertices) == 7
        assert shape.outline().convex_hull.area == pytest.approx(shape.outline().area)


def test_constraints_are_met_or_refused_by_name():
    generator = ConvexPolygon(center=C, radius_nm=10, contains=(C,), min_area_nm2=150, within=AIRSPACE)
    for seed in range(30):
        (shape,) = generator.generate(np.random.default_rng(seed))
        assert shape.contains(C.lat_deg, C.lon_deg) and area_nm2(shape) >= 150
    with pytest.raises(GenerationError, match="ConvexPolygon drew no shape of at least"):
        ConvexPolygon(center=C, radius_nm=1, min_area_nm2=1e6, max_tries=3).generate(np.random.default_rng(0))


def test_voronoi_sectors_partition_their_parent_numbered_by_bearing():
    generator = VoronoiSectors(parent=AIRSPACE, sectors=5, min_spacing_nm=5)
    sectors = generator.generate(np.random.default_rng(1))
    union = shapely.union_all([s.outline() for s in sectors])
    assert len(sectors) == 5 and generator.count == 5
    assert union.area == pytest.approx(AIRSPACE.footprint.outline().area)
    assert sum(s.outline().area for s in sectors) == pytest.approx(union.area)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: ConvexPolygon(center=C, vertices=2), "vertices"),
        (lambda: ConvexPolygon(center=C, radius_nm=(5, 1)), "range must be"),
        (lambda: Blob(center=C, amplitude=1.5), "amplitude"),
        (lambda: VoronoiSectors(parent=AIRSPACE, sectors=1), "sectors"),
        (lambda: VoronoiSectors(), "parent"),
    ],
    ids=["two vertices", "reversed range", "amplitude too large", "one sector", "no parent"],
)
def test_a_generator_that_cannot_be_is_refused(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_generated_region_is_its_envelope_until_drawn():
    region = RegionBounds(GeneratedFootprint(Blob(center=C, radius_nm=10, amplitude=0.5)))
    envelope = Blob(center=C, radius_nm=10, amplitude=0.5).envelope()
    assert region.bounding_box == envelope.bounding_box
    assert region.contains(C.lat_deg, C.lon_deg)


def test_each_region_draws_from_its_own_stream():
    wx = RegionBounds(GeneratedFootprint(Blob(center=C)))
    alone = generate_regions({"wx": wx}, np.random.default_rng(5))
    beside = generate_regions(
        {"other": RegionBounds(GeneratedFootprint(ConvexPolygon(center=C))), "wx": wx},
        np.random.default_rng(5),
    )
    assert alone["wx"].vertices == beside["wx"].vertices


def test_a_partition_adds_its_shapes_beside_the_region():
    regions = generate_regions(
        {"sectors": RegionBounds(GeneratedFootprint(VoronoiSectors(parent=AIRSPACE, sectors=3)))},
        np.random.default_rng(0),
    )
    assert sorted(regions) == ["sectors", "sectors.0", "sectors.1", "sectors.2"]
    assert isinstance(regions["sectors"].footprint, GeneratedFootprint)  # still the envelope


def test_a_scenario_redraws_its_generated_regions_each_episode():
    def regions_fn(_draw):
        return {"wx": RegionBounds(GeneratedFootprint(Blob(center=C)))}

    sampler = RegionParamSampler(regions_fn, {}, lambda regions: {"bounds": regions})
    first, second = (sampler.episode_geometry(np.random.default_rng(s))["bounds"] for s in (1, 2))
    assert first["wx"].vertices != second["wx"].vertices


def test_a_generator_moves_and_turns_with_its_region_and_stays_one():
    from bluesky_sandbox.sim.scenario import transforms as T

    moved = GeneratedFootprint(Blob(center=C, radius_nm=10)).mapped(T.translator(10.0, 0.0))
    assert isinstance(moved, GeneratedFootprint)
    assert moved.generator.center.lon_deg > C.lon_deg
    assert moved.generator.radius_nm == 10
    turned = RegionBounds(GeneratedFootprint(VoronoiSectors(parent=AIRSPACE, sectors=3)))
    turned = T.rotate_bounds(turned, (52.0, 4.5), 30.0)
    assert isinstance(turned.footprint, GeneratedFootprint)
    assert turned.footprint.generator.parent.bounding_box != AIRSPACE.bounding_box


def test_a_partitions_shapes_have_a_region_before_any_draw():
    def regions_fn(_draw):
        return {"sectors": RegionBounds(GeneratedFootprint(VoronoiSectors(parent=AIRSPACE, sectors=2)))}

    sampler = RegionParamSampler(regions_fn, {}, lambda regions: regions)
    support = sampler.support_regions()
    assert support["sectors.0"] is support["sectors"] and "sectors.1" in support
