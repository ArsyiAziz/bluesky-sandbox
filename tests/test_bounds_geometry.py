"""A region's boundary as primitives - faces (segments, arcs) and vertices - and
where a point projects onto them; its center and frames; and transforms that
keep each shape the primitive it is."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point

from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint,
    BoxFootprint,
    ConstantAltitudeBand,
    CorridorFootprint,
    DiskFootprint,
    LatLon,
    LinearAltitudeBand,
    LocalFrame,
    PolygonFootprint,
    RegionBounds,
    SectorFootprint,
)
from bluesky_sandbox.sim.bounds.geometry import Arc, Primitives, Segment, Vertex
from bluesky_sandbox.sim.scenario import transforms as _t

C = LatLon(52.0, 4.0)
F = LocalFrame(C)


def _region(fp) -> RegionBounds:
    return RegionBounds(fp)


SHAPES = {
    "disk": (DiskFootprint(C, 10.0), 0, 1, 0),
    "box": (BoxFootprint(51.8, 52.2, 3.7, 4.3), 4, 0, 4),
    "polygon": (PolygonFootprint([(51.9, 3.9), (52.1, 3.9), (52.0, 4.2)]), 3, 0, 3),
    "sector": (SectorFootprint(C, 10.0, 45.0, 30.0), 2, 1, 3),
    "annular sector": (AnnularSectorFootprint(C, 5.0, 10.0, 90.0, 40.0), 2, 2, 4),
    "ring": (AnnularSectorFootprint(C, 5.0, 10.0, 0.0, 180.0), 0, 2, 0),
    "corridor": (CorridorFootprint(C, LatLon(52.3, 4.3), 2.0), 4, 0, 4),
    "two disks": (DiskFootprint(C, 10.0) | DiskFootprint(LatLon(52.0, 4.2), 10.0), 0, 2, 2),
    "disk with a square hole": (
        DiskFootprint(C, 10.0) - BoxFootprint(51.95, 52.05, 3.9, 4.1), 4, 1, 4
    ),
    "square with a round hole": (
        BoxFootprint(51.8, 52.2, 3.7, 4.3) - DiskFootprint(C, 5.0), 4, 1, 4
    ),
    "lens": (DiskFootprint(C, 10.0) & DiskFootprint(LatLon(52.0, 4.2), 10.0), 0, 2, 2),
}


@pytest.mark.parametrize(("fp", "segments", "arcs", "vertices"), SHAPES.values(), ids=SHAPES)
def test_every_shape_is_its_true_faces_and_vertices(fp, segments, arcs, vertices):
    b = _region(fp).boundary
    assert len(b.faces.of(Segment)) == segments
    assert len(b.faces.of(Arc)) == arcs
    assert len(b.vertices) == vertices


@pytest.mark.parametrize("fp", [v[0] for v in SHAPES.values()], ids=SHAPES)
def test_the_signed_distance_always_agrees_with_inside(fp):
    region = _region(fp)
    rng = np.random.default_rng(0)
    (lat0, lat1), (lon0, lon1) = region.bounding_box
    lat = rng.uniform(lat0 - 0.1, lat1 + 0.1, 300)
    lon = rng.uniform(lon0 - 0.1, lon1 + 0.1, 300)
    signed = region.boundary.nearest(LatLon(lat, lon)).signed_nm
    inside = region.contains(LatLon(lat, lon))
    clear = np.abs(signed) > 1e-6  # off the boundary itself
    assert np.array_equal((signed > 0)[clear], inside[clear])


def test_a_circle_is_projected_onto_exactly():
    disk = _region(DiskFootprint(C, 10.0))
    for distance in (2.0, 9.0, 13.0, 40.0):
        p = F.offset(70.0, distance)
        n = disk.boundary.nearest(p)
        x, y = F.to_xy_nm(n.face.point)
        assert math.hypot(x, y) == pytest.approx(10.0, abs=1e-9)
        assert n.face.cross_nm == pytest.approx(10.0 - distance, abs=1e-9)
        # Great-circle, so a few tenths off the flat frame's bearing at 52 N.
        assert n.face.bearing_deg == pytest.approx(70.0 if distance < 10 else 250.0, abs=0.5)
        assert (n.signed_nm > 0) == (distance < 10.0)
        assert n.vertex is None


def test_a_polygon_projection_is_shapelys_nearest_point():
    region = _region(PolygonFootprint([(51.9, 3.9), (52.1, 3.85), (52.05, 4.2), (51.95, 4.1)]))
    outline = region.footprint.outline()
    frame = LocalFrame(region.center)
    rng = np.random.default_rng(1)
    for lat, lon in zip(rng.uniform(51.8, 52.2, 20), rng.uniform(3.7, 4.3, 20)):
        foot = region.boundary.nearest(LatLon(lat, lon)).face.point
        # The foot is on the outline, and as near as shapely's nearest point.
        assert outline.exterior.distance(Point(foot.lon_deg, foot.lat_deg)) < 1e-9
        ring = outline.exterior
        best = ring.interpolate(ring.project(Point(lon, lat)))
        px, py = frame.to_xy_nm(LatLon(lat, lon))
        fx, fy = frame.to_xy_nm(foot)
        bx, by = frame.to_xy_nm(LatLon(best.y, best.x))
        assert math.hypot(px - fx, py - fy) <= math.hypot(px - bx, py - by) + 1e-6


def test_two_disks_meet_at_two_vertices_on_both_circles():
    other = LatLon(52.0, 4.2)
    union = _region(DiskFootprint(C, 10.0) | DiskFootprint(other, 10.0))
    for v in union.boundary.vertices:
        for center in (C, other):
            x, y = LocalFrame(center).to_xy_nm(v.point)
            assert math.hypot(x, y) == pytest.approx(10.0, rel=1e-3)
    arcs = union.boundary.faces.of(Arc)
    assert all(a.inside and a.radius_nm == 10.0 for a in arcs)
    assert sum(a.span_deg for a in arcs) > 360.0  # more than one circle's worth


def test_a_hole_faces_out_of_the_region():
    ring = _region(AnnularSectorFootprint(C, 5.0, 10.0, 0.0, 180.0))
    inner = next(a for a in ring.boundary.faces.of(Arc) if a.radius_nm == 5.0)
    assert not inner.inside
    near_center = ring.boundary.nearest(F.offset(0.0, 1.0))
    assert near_center.face.primitive == inner
    assert near_center.face.cross_nm == pytest.approx(-4.0, abs=1e-9)
    assert near_center.signed_nm < 0


def test_primitives_are_picked_combined_and_asked_alike():
    sector = _region(SectorFootprint(C, 10.0, 0.0, 30.0))
    disk = _region(DiskFootprint(LatLon(52.0, 4.5), 3.0))
    faces = sector.boundary.faces
    assert isinstance(faces, Primitives) and isinstance(faces[:1], Primitives)
    assert isinstance(faces.of(Arc).nearest(C).primitive, Arc)
    # The sector's center is a vertex: the nearest one to it, at 0.
    both = sector.boundary.vertices + disk.boundary.vertices
    assert isinstance(both, Primitives) and len(both) == 3
    assert both.nearest(C).distance_nm == pytest.approx(0.0, abs=1e-9)
    assert disk.boundary.vertices.nearest(C) is None
    # Behind the sector's tip, the nearest point is the tip itself: a vertex.
    behind = sector.boundary.nearest(F.offset(180.0, 3.0))
    assert behind.face.at_vertex == Vertex(C)


def test_arrays_give_the_answers_one_by_one():
    region = _region(DiskFootprint(C, 10.0) - BoxFootprint(51.95, 52.05, 3.9, 4.1))
    rng = np.random.default_rng(2)
    lat, lon = rng.uniform(51.8, 52.2, 12), rng.uniform(3.7, 4.3, 12)
    many = region.boundary.nearest(LatLon(lat, lon))
    for i in range(12):
        one = region.boundary.nearest(LatLon(lat[i], lon[i]))
        assert many.face.distance_nm[i] == pytest.approx(one.face.distance_nm)
        assert many.signed_nm[i] == pytest.approx(one.signed_nm)
        assert many.face.primitive[i] == one.face.primitive
    assert np.array_equal(
        region.contains(LatLon(lat, lon)), [region.contains(a, b) for a, b in zip(lat, lon)]
    )


@pytest.mark.parametrize(
    ("fp", "center"),
    [
        (DiskFootprint(C, 10.0), C),
        (SectorFootprint(C, 10.0, 45.0, 30.0), C),
        (BoxFootprint(51.8, 52.2, 3.7, 4.3), LatLon(52.0, 4.0)),
        (CorridorFootprint(C, LatLon(52.3, 4.0), 2.0), LatLon(52.15, 4.0)),
    ],
    ids=["disk", "sector", "box", "corridor"],
)
def test_a_region_has_its_own_center(fp, center):
    got = _region(fp).center
    assert (got.lat_deg, got.lon_deg) == pytest.approx((center.lat_deg, center.lon_deg), abs=1e-9)


def test_a_frame_goes_both_ways_and_turns_with_the_region():
    corridor = _region(CorridorFootprint(C, LatLon(52.3, 4.0), 2.0))  # due north, 18 nm
    frame = corridor.frame
    end = LatLon(52.3, 4.0)
    assert frame.xy_nm(end) == pytest.approx((0.0, 9.0), abs=1e-9)
    assert frame(oriented=True).xy_nm(end) == pytest.approx((9.0, 0.0), abs=1e-9)
    p = frame.latlon(3.0, -2.0)
    assert frame.xy_nm(p) == pytest.approx((3.0, -2.0), abs=1e-9)
    # Turned 90 deg, the corridor's own coordinates are what they were.
    turned = _t.rotate_bounds(corridor, (C.lat_deg, C.lon_deg), -90.0)
    q = LatLon(*_t.rotator((C.lat_deg, C.lon_deg), -90.0)(end.lat_deg, end.lon_deg))
    assert turned.frame(oriented=True).xy_nm(q) == pytest.approx((9.0, 0.0), abs=1e-6)


def test_a_transform_keeps_the_primitive():
    pivot = (52.1, 4.1)
    turn = _t.rotator(pivot, 30.0)
    disk = _t.transform_bounds(_region(DiskFootprint(C, 10.0)), turn)
    assert isinstance(disk.footprint, DiskFootprint)
    # Turned in the pivot's flat frame, as every shape is: within its 0.1 %.
    assert disk.footprint.radius_nm == pytest.approx(10.0, rel=1e-3)
    assert len(disk.boundary.faces) == 1 and not disk.boundary.vertices
    sector = _t.transform_bounds(_region(SectorFootprint(C, 10.0, 45.0, 30.0)), turn)
    assert isinstance(sector.footprint, SectorFootprint)
    assert sector.footprint.bearing_deg == pytest.approx(15.0, abs=0.1)  # +30 counterclockwise
    scaled = _t.transform_bounds(_region(DiskFootprint(C, 10.0)), _t.scaler(pivot, 2.0))
    assert scaled.footprint.radius_nm == pytest.approx(20.0, rel=1e-9)
    # A rotated lat/lon box is no longer one: a polygon of its four corners.
    box = _t.transform_bounds(_region(BoxFootprint(51.8, 52.2, 3.7, 4.3)), turn)
    assert isinstance(box.footprint, PolygonFootprint) and len(box.boundary.vertices) == 4
    # A combined shape keeps its parts, and its arcs.
    union = _t.transform_bounds(
        _region(DiskFootprint(C, 10.0) | DiskFootprint(LatLon(52.0, 4.2), 10.0)), turn
    )
    assert union.operation == "union" and len(union.boundary.faces.of(Arc)) == 2


def test_floor_ceiling_and_parts():
    band = LinearAltitudeBand(C, LatLon(52.3, 4.0), (1000.0, 5000.0), (3000.0, 9000.0))
    region = RegionBounds(CorridorFootprint(C, LatLon(52.3, 4.0), 2.0), band)
    mid = LatLon(52.15, 4.0)
    assert region.floor_ft(mid) == pytest.approx(2000.0)
    assert region.ceiling_ft(mid) == pytest.approx(7000.0)
    assert np.allclose(region.floor_ft(LatLon(np.array([52.0, 52.3]), np.array([4.0, 4.0]))), [1000.0, 3000.0])
    assert _region(DiskFootprint(C, 10.0)).parts == []
    combined = RegionBounds(
        DiskFootprint(C, 10.0) - DiskFootprint(C, 5.0), ConstantAltitudeBand(0.0, 9000.0)
    )
    left, right = combined.parts
    assert combined.operation == "difference"
    assert isinstance(left.footprint, DiskFootprint) and left.altitude == combined.altitude
    assert right.footprint.radius_nm == 5.0
