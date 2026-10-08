"""A point bounds: a position with no area - a fix, a spawn point. Nothing is
inside it and a point drawn from it is itself; its boundary is one vertex; it
places, moves and turns like any shape, a placement keeps clear of it, and a
design refuses it wherever an area is needed."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.sim.bounds import (
    BoxFootprint,
    DiskFootprint,
    InRegion,
    LatLon,
    PointFootprint,
    RegionBounds,
)
from bluesky_sandbox.sim.bounds.geometry import Vertex
from bluesky_sandbox.sim.bounds.placement import moved_to

P = LatLon(52.0, 4.5)


def test_nothing_is_inside_a_point_and_a_draw_from_it_is_it():
    point = RegionBounds(PointFootprint(P))
    assert not point.contains(52.0, 4.5)
    assert point.sample_point(np.random.default_rng(0)) == (52.0, 4.5)
    assert point.center == P and point.bounding_box == ((52.0, 52.0), (4.5, 4.5))


def test_its_boundary_is_one_vertex_and_the_nearest_point_is_it():
    boundary = RegionBounds(PointFootprint(P)).boundary
    assert not boundary.faces and [type(v) for v in boundary.vertices] == [Vertex]
    nearest = boundary.nearest(LatLon(52.1, 4.5))
    assert nearest.face.distance_nm == pytest.approx(6.0, rel=1e-2)
    assert nearest.signed_nm == pytest.approx(-nearest.face.distance_nm)  # never inside


def test_it_moves_and_stays_a_point():
    moved = moved_to(PointFootprint(P), LatLon(52.2, 4.7), turn_deg=45.0)
    assert isinstance(moved, PointFootprint) and moved.center == pytest.approx(LatLon(52.2, 4.7))


def test_a_placed_point_lands_inside_clear_of_what_it_avoids():
    storm = DiskFootprint(P, 15.0)
    placement = InRegion(within=BoxFootprint(51.6, 52.4, 4.0, 5.0), avoid=[storm])
    rng = np.random.default_rng(0)
    for _ in range(20):
        at = placement.place(PointFootprint(P), rng).center
        assert 51.6 < at.lat_deg < 52.4 and 4.0 < at.lon_deg < 5.0
        assert not storm.contains(at.lat_deg, at.lon_deg)


def test_a_shape_keeps_clear_of_a_point_it_avoids():
    placement = InRegion(within=BoxFootprint(51.9, 52.1, 4.4, 4.6), avoid=[PointFootprint(P)])
    rng = np.random.default_rng(1)
    for _ in range(10):
        placed = placement.place(DiskFootprint(P, 2.0), rng)
        assert not placed.contains(P.lat_deg, P.lon_deg)


# ---- in a design -------------------------------------------------------- #


def _design():
    from bluesky_sandbox.ui.designer import spec as S
    from bluesky_sandbox.ui.designer.tests.test_generated_regions import _design as base

    spec = base()
    spec.regions["gate"] = {
        "type": "region",
        "footprint": {"type": "point", "center": {"lat_deg": 52.0, "lon_deg": 4.3}},
        "placement": {"type": "InRegion", "within": {"ref": "core"}},
    }
    spec.spawn["regions"][0]["bounds"] = {"ref": "gate"}
    return S.DesignSpec.from_json(spec.to_json())


def test_a_design_spawns_at_a_point_placed_each_episode():
    from bluesky_sandbox.ui.designer import spec as S
    from bluesky_sandbox.ui.designer.preview import scenario_preview

    spec = _design()
    assert S.DesignSpec.from_dict(spec.to_dict()).regions["gate"] == spec.regions["gate"]
    spots = set()
    for seed in range(3):
        preview = scenario_preview(spec, seed=seed)
        (gate,) = preview["shapes"]["gate"]["vertices"]
        assert {(a["lat"], a["lon"]) for a in preview["sampled_aircraft"]} == {tuple(gate)}
        spots.add(tuple(gate))
    assert len(spots) == 3


@pytest.mark.parametrize(
    "where",
    [
        lambda s: setattr(s, "airspace", {"ref": "gate"}),
        lambda s: s.queryables.__setitem__("q", {"type": "query_region", "bounds": {"ref": "gate"}}),
        lambda s: s.regions["wx"]["footprint"]["params"].__setitem__("within", {"ref": "gate"}),
        lambda s: s.regions.__setitem__(
            "cell",
            {"type": "region", "footprint": {"type": "disk", "center": {"lat_deg": 52, "lon_deg": 4.5}, "radius_nm": 2},
             "placement": {"type": "InRegion", "within": {"ref": "gate"}}},
        ),
    ],
    ids=["airspace", "query region", "generator", "placement"],
)
def test_a_point_is_refused_where_an_area_is_needed(where):
    from bluesky_sandbox.ui.designer.builder import BuildError, build_scenario

    spec = _design()
    where(spec)
    with pytest.raises(BuildError, match="needs an area; 'gate' is a point"):
        build_scenario(spec)


def test_the_generated_package_draws_the_point():
    from bluesky_sandbox.ui.designer import codegen

    files = codegen.generate_task(_design(), "Pointed")
    scenario = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "PointFootprint(LatLon(52.0, 4.3))" in scenario


# ---- a waypoint is at a point -------------------------------------------- #


def _waypoints():
    from bluesky_sandbox.ui.designer import spec as S

    spec = _design()
    spec.regions["fix_a"] = {"type": "region", "footprint": {"type": "point", "center": {"lat_deg": 52.15, "lon_deg": 4.7}}}
    spec.regions["ekros"] = {
        "type": "region",
        "footprint": {"type": "point", "fix": "EKROS"},  # where the navdb has it
    }
    spec.queryables["at_fix"] = {"type": "waypoint", "bounds": {"ref": "fix_a"}, "alt_ft": 3000}
    spec.queryables["at_gate"] = {"type": "waypoint", "bounds": {"ref": "gate"}}
    spec.queryables["each"] = {"type": "waypoint", "bounds": {"ref": "gate"}, "sample_per": "aircraft"}
    spec.queryables["named"] = {"type": "waypoint", "bounds": {"ref": "ekros"}}
    return S.DesignSpec.from_json(spec.to_json())


def test_a_waypoint_is_where_its_point_is_and_turns_with_its_group():
    from bluesky_sandbox.ui.designer.builder import build_scenario

    spec = _waypoints()
    at = build_scenario(spec).sample(np.random.default_rng(0)).queryables["at_fix"]
    assert (at.lat, at.lon) == pytest.approx((52.15, 4.7))
    spec.transform = {"groups": [{"id": "g", "name": "g", "members": ["fix_a"], "parent": None,
                                  "pivot": [52.0, 4.5], "angle_deg": 90.0}]}
    turned = build_scenario(spec).sample(np.random.default_rng(0)).queryables["at_fix"]
    assert (turned.lat, turned.lon) != pytest.approx((52.15, 4.7))
    from bluesky.tools.geo import kwikdist

    assert kwikdist(turned.lat, turned.lon, 52.0, 4.5) == pytest.approx(kwikdist(52.15, 4.7, 52.0, 4.5), rel=1e-3)


def test_a_waypoint_on_a_placed_point_is_where_the_point_is_this_episode():
    from bluesky_sandbox.ui.designer.builder import build_scenario

    scenario = build_scenario(_waypoints())
    for seed in range(3):
        episode = scenario.sample(np.random.default_rng(seed))
        gate = episode.bounds["gate"].center
        at = episode.queryables["at_gate"]
        assert (at.lat, at.lon) == pytest.approx((gate.lat_deg, gate.lon_deg))


def test_drawn_for_each_aircraft_a_waypoint_draws_from_its_points_region():
    from bluesky_sandbox.ui.designer.builder import lower_waypoints

    lowered = lower_waypoints(_waypoints())
    assert lowered.queryables["each"]["sample"] == {"ref": "core"}
    assert lowered.queryables["each"]["sample_per"] == "aircraft"
    assert lowered.queryables["at_gate"]["shape"] == {"ref": "gate"}  # at its point


def test_a_waypoint_at_a_fix_is_named_unless_a_group_moves_it():
    from bluesky_sandbox.ui.designer.builder import build_scenario

    spec = _waypoints()
    named = build_scenario(spec).sample(np.random.default_rng(0)).queryables["named"]
    assert named.waypoint == "EKROS"
    spec.transform = {"groups": [{"id": "g", "name": "g", "members": ["ekros"], "parent": None,
                                  "pivot": [52.0, 4.5], "angle_deg": 45.0}]}
    moved = build_scenario(spec).sample(np.random.default_rng(0)).queryables["named"]
    assert moved.waypoint is None and (moved.lat, moved.lon) != pytest.approx((named.lat, named.lon))


def test_in_code_a_waypoint_takes_its_point():
    from bluesky_sandbox.sim.queryables import Waypoint

    point = RegionBounds(PointFootprint(LatLon(52.1, 4.6)))
    assert (Waypoint(at=point).lat, Waypoint(at=point).lon) == (52.1, 4.6)
    assert Waypoint(at=LatLon(52.0, 4.5)).lat == 52.0
    fix = Waypoint(at=PointFootprint.at_fix("EKROS"))
    assert fix.waypoint == "EKROS" and fix.lat == pytest.approx(Waypoint(waypoint="EKROS").lat)


def test_a_waypoint_on_an_area_is_refused():
    from bluesky_sandbox.ui.designer.builder import BuildError, build_scenario

    spec = _waypoints()
    spec.queryables["bad"] = {"type": "waypoint", "bounds": {"ref": "core"}}
    with pytest.raises(BuildError, match="waypoint 'bad' needs a point; 'core' is an area"):
        build_scenario(spec)


def test_the_preview_and_generated_package_read_waypoints_on_points():
    from bluesky_sandbox.ui.designer import codegen
    from bluesky_sandbox.ui.designer.preview import scenario_preview

    spec = _waypoints()
    preview = scenario_preview(spec, seed=1)
    by_name = {q["name"]: q for q in preview["queryables"]}
    (gate,) = preview["shapes"]["gate"]["vertices"]
    assert (by_name["at_gate"]["lat"], by_name["at_gate"]["lon"]) == pytest.approx(tuple(gate))
    files = codegen.generate_task(spec, "Fixes")
    text = "\n".join(files.values())
    assert "'at_fix': Waypoint(at=REGIONS['fix_a']" in text
    assert "fix='EKROS')" in text


def test_the_generated_package_puts_each_waypoint_at_its_point(tmp_path):
    import importlib
    import sys

    from bluesky_sandbox.ui.designer import codegen

    files = codegen.generate_task(_waypoints(), "Fixed")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    for seed in range(3):
        episode = cls().sample(np.random.default_rng(seed))
        gate = episode.bounds["gate"].center
        at = episode.queryables["at_gate"]
        assert (at.lat, at.lon) == pytest.approx((gate.lat_deg, gate.lon_deg))
        assert episode.queryables["named"].waypoint == "EKROS"


def test_the_generated_packages_design_is_the_design_as_made():
    from bluesky_sandbox.ui.designer import codegen
    from bluesky_sandbox.ui.designer import spec as S

    spec = _waypoints()
    files = codegen.generate_task(spec, "AsMade")
    saved = next(t for p, t in files.items() if p.endswith("design.json"))
    assert S.DesignSpec.from_json(saved).to_dict() == spec.to_dict()
