"""A region drawn by a generator in a design: kept by the saved design, redrawn
each episode with everything referencing it following, shown in the preview,
generated as code that does the same, and offered in the catalog."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from bluesky_sandbox.sim.bounds import Blob, GeneratedFootprint
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import BuildError, build_scenario
from bluesky_sandbox.ui.designer.catalog import generators
from bluesky_sandbox.ui.designer.preview import scenario_preview

from .test_designer import _example_design_spec


def _design() -> S.DesignSpec:
    spec = _example_design_spec()
    spec.regions = {
        "core": {
            "type": "region",
            "footprint": {"type": "box", "lat_min_deg": 51.6, "lat_max_deg": 52.4,
                          "lon_min_deg": 4.0, "lon_max_deg": 5.0},
        },
        "wx": {
            "type": "region",
            "footprint": {
                "type": "generated",
                "generator": "Blob",
                "params": {
                    "center": {"lat_deg": 52.0, "lon_deg": 4.5},
                    "radius_nm": {"type": "range", "low": 5, "high": 9},
                    "within": {"ref": "core"},
                },
            },
        },
        "sectors": {
            "type": "region",
            "footprint": {
                "type": "generated",
                "generator": "VoronoiSectors",
                "params": {"parent": {"ref": "core"}, "sectors": 3},
            },
        },
    }
    spec.queryables["storm"] = {"type": "query_region", "bounds": {"ref": "wx"}}
    return S.DesignSpec.from_json(spec.to_json())


def test_a_generated_footprint_survives_the_saved_design():
    spec = _design()
    again = S.DesignSpec.from_dict(spec.to_dict())
    assert again.regions["wx"] == spec.regions["wx"]
    loaded = S.load({"type": "region", "footprint": {**spec.regions["wx"]["footprint"], "params": {"center": {"lat_deg": 52.0, "lon_deg": 4.5}, "radius_nm": {"type": "range", "low": 5, "high": 9}}}})
    assert isinstance(loaded.footprint, GeneratedFootprint)
    assert loaded.footprint.generator == Blob(center=S.LatLon(52.0, 4.5), radius_nm=(5.0, 9.0))
    assert S.dump(loaded)["footprint"]["params"]["radius_nm"] == {"type": "range", "low": 5.0, "high": 9.0}


def test_each_episode_redraws_it_and_its_references_follow():
    scenario = build_scenario(_design())
    a, b, again = (scenario.sample(np.random.default_rng(s)) for s in (1, 2, 1))
    assert a.bounds["wx"].vertices != b.bounds["wx"].vertices
    assert a.bounds["wx"].vertices == again.bounds["wx"].vertices
    assert a.queryables["storm"].bounds.vertices == a.bounds["wx"].vertices
    assert sorted(a.bounds) == ["core", "sectors", "sectors.0", "sectors.1", "sectors.2", "wx"]
    # Before any episode: the generator's envelope.
    assert isinstance(scenario.support().queryables["storm"].bounds.footprint, GeneratedFootprint)


def test_the_preview_shows_the_draw_its_envelope_and_a_partitions_shapes():
    regions = scenario_preview(_design(), seed=0)["shapes"]
    assert regions["wx"]["generated"] == "wx" and "envelope" in regions["wx"]
    assert regions["sectors.1"]["generated"] == "sectors" and "envelope" not in regions["sectors.1"]
    assert "generated" not in regions["core"]


def test_the_generated_package_redraws_it_too(tmp_path):
    files = codegen.generate_task(_design(), "Gen")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    assert "GeneratedFootprint(Blob(" in files[scenario_py]
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    scenario = cls()
    a, b = (scenario.sample(np.random.default_rng(s)) for s in (1, 2))
    assert a.bounds["wx"].vertices != b.bounds["wx"].vertices
    assert a.queryables["storm"].bounds.vertices == a.bounds["wx"].vertices


def test_a_generator_that_cannot_be_built_is_reported():
    spec = _design()
    spec.regions["wx"]["footprint"]["generator"] = "Nowhere"
    with pytest.raises(BuildError, match="'Nowhere' is not a shape generator"):
        build_scenario(spec)
    spec = _design()
    spec.regions["wx"]["footprint"]["params"]["within"] = {"ref": "wx"}
    with pytest.raises(BuildError, match="cannot refer to itself"):
        build_scenario(spec)


def test_the_catalog_offers_the_generators_and_their_params():
    by_name = {g["name"]: g for g in generators()}
    assert {"Blob", "ConvexPolygon", "VoronoiSectors"} <= set(by_name)
    kinds = {p["name"]: p["kind"] for p in by_name["ConvexPolygon"]["params"]}
    assert kinds["center"] == "latlon" and kinds["radius_nm"] == "value" and kinds["within"] == "region"
    assert by_name["VoronoiSectors"]["partition"] == "sectors" and by_name["Blob"]["partition"] is None


def _with_sector_queryables() -> S.DesignSpec:
    spec = _design()
    for i in range(3):
        spec.queryables[f"sector_{i}"] = {"type": "query_region", "bounds": {"ref": f"sectors.{i}"}}
    return spec


def test_each_shape_of_a_partition_can_be_its_own_queryable():
    spec = _with_sector_queryables()
    assert S.partition_names("sectors", spec.regions["sectors"]) == ["sectors.0", "sectors.1", "sectors.2"]
    scenario = build_scenario(spec)
    episode = scenario.sample(np.random.default_rng(4))
    for i in range(3):
        assert episode.queryables[f"sector_{i}"].bounds.vertices == episode.bounds[f"sectors.{i}"].vertices
    # Before any episode, each is where any of them can be.
    assert isinstance(scenario.support().queryables["sector_0"].bounds.footprint, GeneratedFootprint)
    spec.queryables["sector_9"] = {"type": "query_region", "bounds": {"ref": "sectors.9"}}
    with pytest.raises(BuildError, match="'sectors.9' is not in the design"):
        build_scenario(spec)


def test_the_generated_package_gives_each_shape_its_queryable(tmp_path):
    files = codegen.generate_task(_with_sector_queryables(), "Parts")
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
    episode = cls().sample(np.random.default_rng(4))
    assert episode.queryables["sector_2"].bounds.vertices == episode.bounds["sectors.2"].vertices


def test_a_rotated_generated_region_is_still_drawn_each_episode():
    spec = _design()
    spec.regions["wx"]["rotation_deg"] = 30
    scenario = build_scenario(spec)
    a, b = (scenario.sample(np.random.default_rng(s)) for s in (1, 2))
    assert a.bounds["wx"].vertices != b.bounds["wx"].vertices


def _placed() -> S.DesignSpec:
    spec = _design()
    spec.regions["wx"]["placement"] = {"type": "InRegion", "within": {"ref": "core"}, "keep_inside": True}
    spec.regions["cell"] = {
        "type": "region",
        "footprint": {"type": "disk", "center": {"lat_deg": 52.0, "lon_deg": 4.5}, "radius_nm": 3},
        "placement": {
            "type": "InRegion",
            "within": {"ref": "core"},
            "keep_inside": True,
            "turn_deg": {"type": "range", "low": 0, "high": 360},
            "avoid": [{"ref": "wx"}, {"ref": "sectors.0"}],
        },
    }
    spec.queryables["cell_q"] = {"type": "query_region", "bounds": {"ref": "cell"}}
    return S.DesignSpec.from_json(spec.to_json())


def test_a_placement_survives_the_saved_design():
    spec = _placed()
    assert S.DesignSpec.from_dict(spec.to_dict()).regions["cell"] == spec.regions["cell"]


def test_a_placed_region_moves_each_episode_clear_of_what_it_avoids():
    scenario = build_scenario(_placed())
    episodes = [scenario.sample(np.random.default_rng(s)) for s in range(1, 9)]
    assert len({e.bounds["cell"].center for e in episodes}) == len(episodes)
    for e in episodes:
        cell = e.bounds["cell"].footprint.outline()
        assert cell.intersection(e.bounds["wx"].footprint.outline()).area == 0
        assert cell.intersection(e.bounds["sectors.0"].footprint.outline()).area == 0
        assert e.queryables["cell_q"].bounds.vertices == e.bounds["cell"].vertices
    preview = scenario_preview(_placed(), seed=0)["shapes"]
    assert preview["cell"]["generated"] == "cell" and "envelope" in preview["cell"]


def test_the_generated_package_places_it_too(tmp_path):
    files = codegen.generate_task(_placed(), "Placed")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    assert "PlacedFootprint(DiskFootprint(" in files[scenario_py]
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    a, b = (cls().sample(np.random.default_rng(s)) for s in (1, 2))
    assert a.bounds["cell"].center != b.bounds["cell"].center
    assert a.bounds["cell"].footprint.outline().intersection(a.bounds["wx"].footprint.outline()).area == 0


def test_the_catalog_offers_the_placements():
    from bluesky_sandbox.ui.designer.catalog import placements

    (in_region,) = [p for p in placements() if p["name"] == "InRegion"]
    kinds = {p["name"]: p["kind"] for p in in_region["params"]}
    assert kinds == {"within": "region", "turn_deg": "value", "keep_inside": "bool", "avoid": "regions", "max_tries": "int"}


def _moving() -> S.DesignSpec:
    spec = _placed()
    spec.regions["cell"]["motion"] = [
        {"type": "Drift", "heading_deg": {"type": "range", "low": 0, "high": 360}, "speed_kts": 20, "within": {"ref": "core"}},
        {"type": "Grow", "rate_per_hr": 0.5},
    ]
    return S.DesignSpec.from_json(spec.to_json())


def test_a_motion_survives_the_saved_design_and_moves_every_holder():
    from bluesky_sandbox.sim.bounds import MovingFootprint, moving_in

    spec = _moving()
    assert S.DesignSpec.from_dict(spec.to_dict()).regions["cell"] == spec.regions["cell"]
    episode = build_scenario(spec).sample(np.random.default_rng(1))
    assert isinstance(episode.queryables["cell_q"].bounds.footprint, MovingFootprint)
    start = episode.queryables["cell_q"].bounds.center
    for footprint in moving_in(episode):
        footprint.advance(1800.0)
    assert episode.queryables["cell_q"].bounds.center != start
    assert episode.queryables["cell_q"].bounds.vertices == episode.bounds["cell"].vertices


def test_the_preview_plays_a_moving_regions_next_hour():
    preview = scenario_preview(_moving(), seed=0)
    frames = preview["shapes"]["cell"]["frames"]
    assert len(frames) == 61 and frames[0]["vertices"] != frames[-1]["vertices"]
    # Whatever holds it plays too: its queryable.
    (cell_q,) = [q for q in preview["queryables"] if q["name"] == "cell_q"]
    assert len(cell_q["frames"]) == 61


def test_the_generated_package_moves_it_too(tmp_path):
    from bluesky_sandbox.sim.bounds import MovingFootprint

    files = codegen.generate_task(_moving(), "Moving")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    assert "MovingFootprint(PlacedFootprint(" in files[scenario_py]
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    episode = cls().sample(np.random.default_rng(1))
    assert isinstance(episode.bounds["cell"].footprint, MovingFootprint)


def test_the_catalog_offers_the_motions():
    from bluesky_sandbox.ui.designer.catalog import motions

    assert {m["name"] for m in motions()} >= {"Drift", "Spin", "Grow"}


def test_a_motions_update_cadence_is_kept_and_generated():
    spec = _moving()
    assert build_scenario(spec).sample(np.random.default_rng(1)).bounds["cell"].footprint.update == "step"
    spec.regions["cell"]["motion_update"] = "substep"
    spec = S.DesignSpec.from_dict(spec.to_dict())
    episode = build_scenario(spec).sample(np.random.default_rng(1))
    assert episode.bounds["cell"].footprint.update == "substep"
    files = codegen.generate_task(spec, "Substepwise")
    scenario_py = next(t for p, t in files.items() if p.endswith("scenario.py"))
    assert "update='substep')" in scenario_py


def _group_moving() -> S.DesignSpec:
    """Two bounds in a group that drifts and spins; a waypoint anchored to one."""
    spec = _design()
    for name, lon in (("pad_a", 4.4), ("pad_b", 4.6)):
        spec.regions[name] = {
            "type": "region",
            "footprint": {"type": "disk", "center": {"lat_deg": 52.0, "lon_deg": lon}, "radius_nm": 2},
        }
    spec.queryables["pad_a_q"] = {"type": "query_region", "bounds": {"ref": "pad_a"}}
    spec.queryables["fix"] = {"type": "waypoint", "lat": 52.0, "lon": 4.4, "anchor": "pad_a"}
    spec.transform = {
        "groups": [
            {
                "id": "g", "name": "pads", "members": ["pad_a", "pad_b"], "parent": None, "pivot": None,
                "motion": [
                    {"type": "Drift", "heading_deg": 90, "speed_kts": 20, "within": {"ref": "core"}},
                    {"type": "Spin", "rate_deg_s": 0.05},
                ],
            }
        ]
    }
    return S.DesignSpec.from_json(spec.to_json())


def _nm(a, b) -> float:
    from bluesky.tools.geo import kwikdist

    return float(kwikdist(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg))


def test_a_group_moves_what_it_holds_as_one_and_holds_its_waypoints():
    from bluesky_sandbox.sim.bounds import MovingFootprint, moving_in

    episode = build_scenario(_group_moving()).sample(np.random.default_rng(3))
    a, b = episode.bounds["pad_a"], episode.bounds["pad_b"]
    assert isinstance(a.footprint, MovingFootprint) and a.footprint.carriers
    start_a, apart = a.center, _nm(a.center, b.center)
    fix = episode.queryables["fix"]
    fix_at = (fix.lat, fix.lon)
    for footprint in moving_in(episode):
        footprint.advance(1800.0)
    assert _nm(a.center, start_a) > 5  # 10 nm east in half an hour, and turned
    assert _nm(a.center, b.center) == pytest.approx(apart, rel=1e-6)
    # What holds a member moves with it; an anchored waypoint holds still.
    assert episode.queryables["pad_a_q"].bounds.center == a.center
    assert (fix.lat, fix.lon) == fix_at


def test_a_waypoint_anchored_to_a_bounds_turns_with_its_group():
    from bluesky_sandbox.ui.designer.builder import elements_for_region

    spec = _group_moving()
    assert "q:fix" in elements_for_region(spec, "pad_a")
    spec.transform["groups"][0].pop("motion")
    spec.transform["groups"][0]["angle_deg"] = 90.0
    spec.transform["groups"][0]["pivot"] = [52.0, 4.5]
    fix = build_scenario(spec).sample(np.random.default_rng(0)).queryables["fix"]
    assert fix.lon == pytest.approx(4.5, abs=1e-3) and fix.lat != pytest.approx(52.0, abs=1e-3)


def test_a_partitions_shapes_count_as_its_members():
    from bluesky_sandbox.ui.designer.builder import elements_for_region

    spec = _design()
    spec.queryables["s0"] = {"type": "query_region", "bounds": {"ref": "sectors.0"}}
    assert "q:s0" in elements_for_region(spec, "sectors")


def test_the_preview_plays_a_groups_moves():
    preview = scenario_preview(_group_moving(), seed=0)
    frames = preview["shapes"]["pad_a"]["frames"]
    assert len(frames) == 61 and frames[0]["vertices"] != frames[-1]["vertices"]


def test_the_generated_package_moves_the_group(tmp_path):
    files = codegen.generate_task(_group_moving(), "GroupMoving")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    scenario_py = next(p for p in files if p.endswith("scenario.py"))
    assert '"motion": (Drift(' in files[scenario_py]
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(scenario_py[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Scenario") and k != "RandomizedScenario")
    episode = cls().sample(np.random.default_rng(1))
    assert episode.bounds["pad_a"].footprint.carriers
