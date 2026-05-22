"""Verification suite for the designer backend foundation.

Runs standalone (``python -m bluesky_sandbox.ui.designer.tests.test_designer``) so
it works without pytest, but each ``test_*`` function is also pytest-collectable.

Covers the three foundation pieces:

* round-trip: every supported primitive survives ``obj -> dump -> load`` and
  ``obj -> json -> obj`` identically,
* build: a full ``DesignSpec`` compiles into scenario resources plus a static
  ``EnvConfig`` for fields and simulator settings,
* nav: the navdb query layer resolves a known fix/airport and scopes a window.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson, randint, truncnorm

from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.task import QueryableTemporalStateUnavailable
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    MinMaxNormalizer,
    SymmetricNormalizer,
)
from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint,
    BooleanFootprint,
    BoxFootprint,
    ConstantAltitudeBand,
    DiskFootprint,
    LatLon,
    LinearAltitudeBand,
    PolygonFootprint,
    RadialAltitudeBand,
    RegionBounds,
    SectorFootprint,
    VertexAltitudeBand,
)
from bluesky_sandbox.sim.performance.envelope import EnvelopeSample
from bluesky_sandbox.sim.queryables import (
    QueryRegion,
    RegionResult,
    UnavailableRegionStep,
    UnavailableWaypointStep,
    Waypoint,
    WaypointResult,
)
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui.designer import codegen, nav
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import (
    BuildError,
    build_design_config,
    build_scenario,
)
import itertools


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _roundtrip(obj):
    """obj -> dump -> load and obj -> json -> obj; return the json-reloaded obj."""
    d = S.dump(obj)
    from_dict = S.load(d)
    from_json = S.loads(S.dumps(obj))
    # dict produced from the reloaded object must equal the original dict.
    assert S.dump(from_dict) == d, f"dict round-trip mismatch for {type(obj).__name__}"
    assert S.dump(from_json) == d, f"json round-trip mismatch for {type(obj).__name__}"
    return from_json


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# footprints                                                                   #
# --------------------------------------------------------------------------- #
def test_footprint_roundtrip():
    box = BoxFootprint(51.9, 52.1, 4.4, 4.6)
    disk = DiskFootprint(LatLon(52.0, 4.5), radius_nm=10.0)
    poly = PolygonFootprint([(52.0, 4.0), (52.0, 4.5), (52.4, 4.25)])
    sector = SectorFootprint(LatLon(52.0, 4.5), 20.0, bearing_deg=90.0, half_angle_deg=30.0)
    annular = AnnularSectorFootprint(
        LatLon(52.0, 4.5), inner_radius_nm=5.0, outer_radius_nm=15.0,
        bearing_deg=45.0, half_angle_deg=20.0,
    )
    boolean = BooleanFootprint("union", box, disk)
    for fp in (box, disk, poly, sector, annular, boolean):
        _roundtrip(fp)
    try:
        S.load({
            "type": "corridor",
            "start": {"lat_deg": 52.0, "lon_deg": 4.0},
            "end": {"lat_deg": 52.0, "lon_deg": 5.0},
            "half_width_nm": 3.0,
        })
        raise AssertionError("corridor footprint should not be supported by the designer spec")
    except S.SpecError:
        pass
    print("  footprints: box/disk/polygon/sector/annular/boolean OK")


def test_altitude_roundtrip():
    const = ConstantAltitudeBand(2_000, 8_000)
    const_open = ConstantAltitudeBand()  # -inf / +inf
    linear = LinearAltitudeBand(
        LatLon(52.0, 4.0), LatLon(52.0, 5.0), (1_000, 5_000), (3_000, 9_000)
    )
    radial = RadialAltitudeBand(LatLon(52.0, 4.5), 10.0, (1_000, 4_000), (2_000, 8_000))
    vertex = VertexAltitudeBand(
        [(52.0, 4.0), (52.0, 4.5), (52.4, 4.25)], 1_000.0, [5_000.0, 6_000.0, 7_000.0]
    )
    for band in (const, const_open, linear, radial, vertex):
        _roundtrip(band)
    # explicit infinity check
    reloaded = S.load(S.dump(const_open))
    assert math.isinf(reloaded.min_ft) and reloaded.min_ft < 0
    assert math.isinf(reloaded.max_ft) and reloaded.max_ft > 0
    print("  altitude bands: constant(+inf)/linear/radial/vertex OK")


# --------------------------------------------------------------------------- #
# bounds / queryables                                                          #
# --------------------------------------------------------------------------- #
def test_bounds_and_queryables_roundtrip():
    bounds = RegionBounds(
        BoxFootprint(51.9, 52.1, 4.4, 4.6), ConstantAltitudeBand(2_000, 8_000)
    )
    _roundtrip(bounds)

    region = QueryRegion(bounds, color="cyan", render_shape=False, render_label=False)
    _roundtrip(region)

    wp_coords = Waypoint(
        lat=52.0,
        lon=4.5,
        alt_ft=3_000,
        speed_kts=220,
        reach_radius_nm=1.2,
        alt_tolerance_ft=500,
        speed_tolerance_kts=20,
        color="magenta",
    )
    reloaded_wp = _roundtrip(wp_coords)
    assert reloaded_wp.reach_radius_nm == 1.2
    assert reloaded_wp.alt_tolerance_ft == 500
    assert reloaded_wp.speed_tolerance_kts == 20

    wp_named = Waypoint(waypoint="EKROS")
    d = S.dump(wp_named)
    # reference-by-identifier: name stored, resolved coords NOT baked in
    assert d["waypoint"] == "EKROS"
    assert "lat" not in d and "lon" not in d
    reloaded = S.load(d)
    assert _approx(reloaded.lat, wp_named.lat) and _approx(reloaded.lon, wp_named.lon)
    print("  bounds + QueryRegion + Waypoint (coords & named) OK")


# --------------------------------------------------------------------------- #
# distribution values                                                         #
# --------------------------------------------------------------------------- #
def test_value_roundtrip():
    np.random.default_rng(0)

    # fixed scalar, range tuple, list pass-through
    assert S.load_value(S.dump_value(5)) == 5
    assert S.load_value(S.dump_value((1_000.0, 5_000.0))) == (1_000.0, 5_000.0)
    assert S.load_value(S.dump_value(["KL", "BA"])) == ["KL", "BA"]

    # scipy frozen distributions: compare a seeded draw
    for dist in (randint(2, 7), truncnorm(a=-2, b=2, loc=250, scale=30)):
        reloaded = S.load_value(S.dump_value(dist))
        a = dist.rvs(random_state=np.random.default_rng(42))
        b = reloaded.rvs(random_state=np.random.default_rng(42))
        assert _approx(float(a), float(b)), f"scipy {dist.dist.name} draw mismatch"

    # Categorical
    cat = Categorical({"A320": 3.0, "B738": 1.0})
    reloaded = S.load_value(S.dump_value(cat))
    assert reloaded.weights == cat.weights
    print("  values: scalar/range/list/scipy(randint,truncnorm)/categorical OK")


def test_bounded_roundtrip():
    rng = np.random.default_rng(0)

    for mode in ("truncate", "clip"):
        b = Bounded(poisson(mu=50), 1, 100, mode=mode)
        assert b.support() == (1.0, 100.0)          # finite -> designer-accepted

        reloaded = S.load_value(S.dump_value(b))
        assert isinstance(reloaded, Bounded)
        assert reloaded.support() == (1.0, 100.0)
        assert reloaded.mode == mode

        # seeded draw survives the round-trip and stays within bounds
        a = b.rvs(random_state=np.random.default_rng(42))
        c = reloaded.rvs(random_state=np.random.default_rng(42))
        assert _approx(float(a), float(c)), f"Bounded({mode}) draw mismatch"
        draws = [int(b.rvs(random_state=rng)) for _ in range(2_000)]
        assert all(1 <= x <= 100 for x in draws), f"Bounded({mode}) escaped bounds"

    # Motivating case: a bare poisson is rejected as n_aircraft (unbounded
    # support), but wrapping it makes max_n / obs-space sizing finite.
    region = SpawnRegion(
        RegionBounds(BoxFootprint(51.9, 52.7, 4.5, 5.2)),
        n_aircraft=Bounded(poisson(mu=50), 1, 100),
        params={"alt_ft": (5_000, 15_000), "spd_kts": 250.0},
        name="DENSITY",
    )
    assert region.max_n() == 100
    print("  bounded: truncate/clip round-trip + finite max_n from poisson OK")


# --------------------------------------------------------------------------- #
# spawn                                                                        #
# --------------------------------------------------------------------------- #
def test_spawn_roundtrip():
    region = SpawnRegion(
        RegionBounds(BoxFootprint(51.9, 52.7, 4.5, 5.2)),
        n_aircraft=randint(2, 7),
        params={"alt_ft": (5_000, 15_000), "spd_kts": truncnorm(a=-2, b=2, loc=250, scale=30)},
        callsign_prefixes=Categorical({"KL": 3.0, "BA": 1.0}),
        spawn_time=(0.0, 120.0),
        name="ARRIVALS",
    )
    spawn = SpawnConfig(
        regions=[region],
        aircraft_type=Categorical({"A320": 1.0}),
        route="STAR1",
        routes={"STAR1": ["EKROS", "RIVER"]},
    )
    reloaded = _roundtrip(spawn)
    assert isinstance(reloaded, SpawnConfig)
    assert reloaded.max_aircraft() == spawn.max_aircraft() == 6
    print("  spawn: SpawnRegion + SpawnConfig (dists, categorical, routes) OK")


def test_empty_spawn_config():
    spawn = SpawnConfig(regions=[])
    reloaded = _roundtrip(spawn)
    assert reloaded.max_aircraft() == 0
    assert list(reloaded.iter_spawns(np.random.default_rng(0))) == []
    assert reloaded.resolved_bounds["lat_deg"] == (0.0, 0.0)

    spec = _example_design_spec()
    spec.spawn = S.dump(spawn)
    scenario = build_scenario(spec)
    assert scenario.support().max_aircraft == 0
    files = codegen.generate_task(spec, "empty_spawn")
    assert "SpawnConfig(regions=[])" in files["empty_spawn/scenario.py"]
    print("  spawn: empty SpawnConfig round-trip/build/codegen OK")


# --------------------------------------------------------------------------- #
# full design spec -> build                                                    #
# --------------------------------------------------------------------------- #
def _example_design_spec() -> S.DesignSpec:
    airspace = RegionBounds(
        BoxFootprint(51.5, 52.5, 4.0, 5.5), ConstantAltitudeBand(0, 20_000)
    )
    spawn = SpawnConfig(
        regions=[
            SpawnRegion(
                RegionBounds(BoxFootprint(51.6, 52.4, 4.1, 5.4)),
                n_aircraft=randint(2, 6),
                params={"alt_ft": (5_000, 15_000), "spd_kts": (200, 280)},
            )
        ]
    )
    goal = QueryRegion(
        RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.6), ConstantAltitudeBand(2_000, 8_000)),
        color="cyan",
    )
    env = S.EnvSpec(
        obs_fields=[
            S.FieldRef("LatDeg"),
            S.FieldRef("LonDeg"),
            S.FieldRef("AltFt", kwargs={"normalizer": {"type": "normalizer", "name": "MinMaxNormalizer", "kwargs": {}}}),
        ],
        intruder_obs_fields=[S.FieldRef("DistToOwnNm")],
        action_fields=[
            S.FieldRef("HdgDeg"),
            S.FieldRef("SpdKts", kwargs={"normalizer": {"type": "normalizer", "name": "SymmetricNormalizer", "kwargs": {"clipped": True}}}),
        ],
        allowed_aircraft=["A320", "B738"],
    )
    return S.DesignSpec(
        env=env,
        spawn=S.dump(spawn),
        airspace=S.dump(airspace),
        queryables={"goal": S.dump(goal)},
        metadata={"name": "demo"},
    )


def test_design_spec_json_roundtrip():
    spec = _example_design_spec()
    reloaded = S.DesignSpec.from_json(spec.to_json())
    assert reloaded.to_dict() == spec.to_dict()
    print("  DesignSpec JSON round-trip OK")


def test_rotation_transform():
    spec = _example_design_spec()
    spec.transform = {"rotation": {"angle_deg": 90.0, "pivot": None}}
    scenario = build_scenario(spec)
    support = scenario.support()
    sample = scenario.sample(np.random.default_rng(0))
    # 90-deg rotation changes the airspace geometry vs the canonical support frame
    assert not np.allclose(
        support.airspace_bounds.bounding_box, sample.airspace_bounds.bounding_box, atol=1e-3
    )
    # queryables and spawn rotate as a group
    assert isinstance(sample.queryables["goal"], QueryRegion)
    # 0-deg is identity
    spec.transform = {"rotation": {"angle_deg": 0.0, "pivot": None}}
    s0 = build_scenario(spec).sample(np.random.default_rng(0))
    assert np.allclose(s0.airspace_bounds.bounding_box, support.airspace_bounds.bounding_box, atol=1e-6)
    # a range distribution is accepted for the angle
    spec.transform = {"rotation": {"angle_deg": {"type": "range", "low": -30, "high": 30}}}
    rng_scn = build_scenario(spec)
    _ = rng_scn.sample(np.random.default_rng(1))
    print("  rotation transform: group rotation sampled, 0-deg identity, range OK")


def test_group_transform():
    # A rotation group generalised to translation + scale: members are bounds
    # (named regions); the airspace references one, so the group transforms it.
    spec = _example_design_spec()
    spec.regions = {
        "core": {"type": "region",
                 "footprint": {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
                               "lon_min_deg": 4.4, "lon_max_deg": 5.0},
                 "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 9000}},
    }
    spec.airspace = {"ref": "core"}
    support = build_scenario(spec).support()
    (slat0, slat1), (slon0, slon1) = support.airspace_bounds.bounding_box

    def group(**kw):
        g = {"id": "g1", "name": "g", "members": ["core"], "parent": None, "pivot": None, **kw}
        spec.transform = {"groups": [g]}
        return build_scenario(spec).sample(np.random.default_rng(0)).airspace_bounds.bounding_box

    # Pure north translation: +60 nm = +1.0 deg latitude, longitude unchanged.
    (lat0, lat1), (lon0, lon1) = group(translation={"north_nm": 60.0})
    assert math.isclose(lat0, slat0 + 1.0, abs_tol=1e-3) and math.isclose(lat1, slat1 + 1.0, abs_tol=1e-3)
    assert math.isclose(lon0, slon0, abs_tol=1e-3) and math.isclose(lon1, slon1, abs_tol=1e-3)

    # Uniform scale about the (auto) pivot doubles the latitude extent.
    (lat0, lat1), _ = group(scale=2.0)
    assert math.isclose(lat1 - lat0, 2 * (slat1 - slat0), abs_tol=1e-2)

    # Identity: angle 0, scale 1, no translation leaves the canonical frame.
    ident = group(angle_deg=0.0, scale=1.0)
    assert np.allclose(ident, support.airspace_bounds.bounding_box, atol=1e-6)

    # Sampled ranges are accepted for every channel.
    group(angle_deg={"type": "range", "low": -20, "high": 20},
          translation={"east_nm": {"type": "range", "low": -10, "high": 10}, "north_nm": 5.0},
          scale={"type": "range", "low": 0.9, "high": 1.1})

    # A fixed lat/lon waypoint joins a group directly via "wp:<name>".
    spec.queryables["wp"] = {"type": "waypoint", "lat": 52.0, "lon": 4.7, "alt_ft": 3000}
    spec.transform = {"groups": [{"id": "g1", "name": "g", "members": ["wp:wp"],
                                  "parent": None, "pivot": None, "translation": {"north_nm": 60.0}}]}
    wp = build_scenario(spec).sample(np.random.default_rng(0)).queryables["wp"]
    assert math.isclose(wp.lat, 53.0, abs_tol=1e-3) and math.isclose(wp.lon, 4.7, abs_tol=1e-3)
    print("  group transform: translation/scale/rotation compose, identity, waypoint member, ranges OK")


def test_sampled_waypoint():
    # A waypoint with a `sample` footprint redraws its position each episode,
    # while support() stays at the region centre (stable schema).
    spec = _example_design_spec()
    box = {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
           "lon_min_deg": 4.4, "lon_max_deg": 5.0}
    spec.queryables["wp"] = {"type": "waypoint", "sample": box, "alt_ft": 3000}
    scenario = build_scenario(spec)
    assert list(scenario.sampled_waypoints) == ["wp"]
    # support = region centre
    sup = scenario.support().queryables["wp"]
    assert math.isclose(sup.lat, 52.0, abs_tol=1e-6) and math.isclose(sup.lon, 4.7, abs_tol=1e-6)
    # sampled positions stay inside the footprint and vary across seeds
    positions = set()
    for seed in range(5):
        wp = scenario.sample(np.random.default_rng(seed)).queryables["wp"]
        assert 51.8 <= wp.lat <= 52.2 and 4.4 <= wp.lon <= 5.0
        assert isinstance(wp, Waypoint) and wp.waypoint is None
        positions.add((round(wp.lat, 4), round(wp.lon, 4)))
    assert len(positions) > 1
    # codegen carries sampled waypoint regions through to the generated scenario.py
    files = codegen.generate_task(spec, "Sampled WP")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "sampled_waypoints={" in scenario_py
    assert "BoxFootprint(51.8, 52.2, 4.4, 5.0)" in scenario_py
    print("  sampled waypoint: per-episode position, centroid support, codegen OK")


def test_named_region_refs():
    # A named region in spec.regions is referenced by airspace, a query region,
    # a spawn region, and a sampled waypoint via {"ref": name}.
    spec = _example_design_spec()
    spec.regions = {
        "core": {"type": "region",
                 "footprint": {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
                               "lon_min_deg": 4.4, "lon_max_deg": 5.0},
                 "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 9000}},
    }
    spec.queryables["goal"] = {"type": "query_region", "bounds": {"ref": "core"}}
    spec.queryables["wp"] = {"type": "waypoint", "sample": {"ref": "core"}, "alt_ft": 5000}
    spec.spawn["regions"][0]["bounds"] = {"ref": "core"}
    # json round-trip preserves the regions table
    assert S.DesignSpec.from_json(spec.to_json()).regions == spec.regions
    scenario = build_scenario(spec)
    core_bbox = ((51.8, 52.2), (4.4, 5.0))
    assert scenario.support().queryables["goal"].bounds.bounding_box == core_bbox
    # the sampled waypoint draws lat/lon AND altitude from the named region's band
    alts = set()
    for seed in range(6):
        wp = scenario.sample(np.random.default_rng(seed)).queryables["wp"]
        assert 51.8 <= wp.lat <= 52.2 and 4.4 <= wp.lon <= 5.0
        assert 2000 <= wp.alt_ft <= 9000
        alts.add(round(wp.alt_ft))
    assert len(alts) > 1
    # an unknown ref is a build error
    bad = _example_design_spec()
    bad.airspace = {"ref": "nope"}
    try:
        build_scenario(bad)
        assert False, "expected BuildError for unknown region ref"
    except BuildError:
        pass
    # codegen emits local region refs inside the scenario class method.
    files = codegen.generate_task(spec, "Region Refs")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "REGIONS = {" in scenario_py and scenario_py.count("REGIONS['core']") == 3
    print("  named region refs: airspace/query/spawn/sample refs build + codegen OK")


def test_sampled_region_params():
    # A named region with a sampled footprint param (disk radius as a range):
    # spec round-trips, episodes reshape the region, support() is the union
    # (widest) shape even after sampling, and codegen emits an equivalent
    # parametric scenario.
    spec = _example_design_spec()
    spec.regions = {
        "zone": {"type": "region",
                 "footprint": {"type": "disk",
                               "center": {"lat_deg": 52.0, "lon_deg": 4.7},
                               "radius_nm": {"type": "range", "low": 10.0, "high": 30.0}},
                 "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 9000}},
    }
    spec.queryables["wp"] = {"type": "waypoint", "sample": {"ref": "zone"}, "alt_ft": 5000}
    spec.spawn["regions"][0]["bounds"] = {"ref": "zone"}
    assert S.DesignSpec.from_json(spec.to_json()).regions == spec.regions

    def _radius_nm(episode_or_support):
        (la0, la1), _ = episode_or_support.spawn.regions[0].bounds.bounding_box
        return (la1 - la0) * 60.0 / 2.0

    scenario = build_scenario(spec)
    assert _radius_nm(scenario.support()) > 29.0  # union support = widest draw
    radii = set()
    for seed in range(5):
        r = _radius_nm(scenario.sample(np.random.default_rng(seed)))
        assert 9.9 <= r <= 30.1
        radii.add(round(r, 1))
    assert len(radii) > 1  # the radius actually varies across episodes
    assert _radius_nm(scenario.support()) > 29.0  # support unchanged by sampling

    # codegen: parametric template; the generated scenario runs and matches.
    files = codegen.generate_task(spec, "Sampled Region")
    pkg = next(iter(files)).split("/", 1)[0]
    scenario_py = files[f"{pkg}/scenario.py"]
    assert "_REGION_PARAM_DISTS" in scenario_py
    assert "draw['zone.radius_nm']" in scenario_py
    ns: dict = {}
    exec(compile(scenario_py, "generated_scenario.py", "exec"), ns)
    gen = next(
        v for k, v in ns.items()
        if k.endswith("Scenario") and isinstance(v, type) and k != "RandomizedScenario"
    )()
    assert _radius_nm(gen.support()) > 29.0
    gen_radii = {round(_radius_nm(gen.sample(np.random.default_rng(s))), 1) for s in range(5)}
    assert len(gen_radii) > 1
    assert all(9.9 <= r <= 30.1 for r in gen_radii)

    # The preview exposes the sampled named-region shapes in the episode frame:
    # with a whole-geometry rotation, the region geometry rotates with the seed.
    import math

    from bluesky_sandbox.ui.designer.preview import scenario_preview

    spec.transform = {"rotation": {"angle_deg": {"type": "range", "low": 0.0, "high": 360.0},
                                   "pivot": [52.0, 4.7]}}
    zones = [scenario_preview(spec, seed=s)["regions"]["zone"] for s in (0, 1)]
    assert all(z["vertices"] for z in zones)
    # A disk is rotation-invariant, but its *sampled radius* must vary by seed.
    def _r(z):
        return max(math.hypot(v[0] - 52.0, v[1] - 4.7) for v in z["vertices"])
    assert abs(_r(zones[0]) - _r(zones[1])) > 1e-4
    print("  sampled region params: build/sample/support/codegen/preview parity OK")


def test_transform_field_preservation():
    # Regression: episode transforms must preserve ALL SpawnConfig fields.
    # rotate_spawn/_apply_groups used to reconstruct field-by-field, silently
    # dropping conflict_free_spawn from every rotated or grouped episode.
    spec = _example_design_spec()
    spec.spawn["conflict_free_spawn"] = True
    # iter_spawns samples types; the env normally normalizes this at build.
    spec.spawn["aircraft_type"] = "A320"

    for transform in (
        {"rotation": {"angle_deg": {"type": "range", "low": 0.0, "high": 360.0}}},
        {"groups": [{"id": "g", "angle_deg": {"type": "range", "low": 0.0, "high": 360.0},
                     "pivot": [52.0, 4.7], "members": []}]},
    ):
        spec.transform = transform
        scenario = build_scenario(spec)
        totals = []
        rng = np.random.default_rng(0)
        for _ in range(300):
            ep = scenario.sample(rng)
            assert ep.spawn.conflict_free_spawn is True  # survived the transform
            n = sum(1 for _ in ep.spawn.iter_spawns(np.random.default_rng(int(rng.integers(1 << 31)))))
            totals.append(n)
        assert min(totals) >= 1  # empty-episode floor
    print("  transform field preservation: OK")


def test_group_transforms_route_samples_and_preview_regions():
    # A per-aircraft sampled waypoint whose region is rotated by a *group*:
    # the episode's route-step sample bounds must carry the group transform
    # (regression: _apply_groups used to leave routes untransformed), and the
    # preview's named-region geometry must land in the same episode frame.
    import math

    from bluesky_sandbox.ui.designer.preview import scenario_preview

    spec = _example_design_spec()
    spec.regions = {
        "corridor": {"type": "region",
                     "footprint": {"type": "annular_sector",
                                   "center": {"lat_deg": 52.0, "lon_deg": 4.7},
                                   "inner_radius_nm": 10, "outer_radius_nm": 20,
                                   "bearing_deg": 90, "half_angle_deg": 10},
                     "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 9000}},
    }
    spec.queryables["goal"] = {"type": "waypoint", "sample": {"ref": "corridor"},
                               "sample_per": "aircraft", "alt_ft": 5000}
    spec.spawn["regions"][0]["route"] = ["goal"]
    spec.transform = {"groups": [{"id": "g1",
                                  "angle_deg": {"type": "range", "low": 0.0, "high": 360.0},
                                  "pivot": [52.0, 4.7],
                                  "members": ["corridor"]}]}

    def bearing(verts):
        la = sum(v[0] for v in verts) / len(verts)
        lo = sum(v[1] for v in verts) / len(verts)
        return math.degrees(math.atan2((lo - 4.7) * math.cos(math.radians(52)), la - 52)) % 360

    scenario = build_scenario(spec)
    bearings = []
    for seed in (0, 1):
        ep = scenario.sample(np.random.default_rng(seed))
        b_ep = bearing(ep.spawn.regions[0].route[0]["sample"].vertices)
        b_pv = bearing(scenario_preview(spec, seed=seed)["regions"]["corridor"]["vertices"])
        assert abs(b_ep - b_pv) < 1.0, (b_ep, b_pv)
        bearings.append(b_ep)
    assert abs(bearings[0] - bearings[1]) > 5.0  # the group angle actually varies
    print("  group transforms: route-step samples + preview regions follow the group OK")


def test_airspace_warning_ignores_shared_bounds():
    # A queryable that reuses the airspace bounds sits on its boundary; it must
    # not be flagged "outside airspace", while genuinely-outside content is.
    from bluesky_sandbox.ui.designer.preview import airspace_warnings

    spec = _example_design_spec()
    spec.regions = {
        "air": {"type": "region",
                "footprint": {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
                              "lon_min_deg": 4.4, "lon_max_deg": 5.0},
                "altitude": {"type": "constant", "min_ft": 0, "max_ft": 12000}},
        "far": {"type": "region",
                "footprint": {"type": "box", "lat_min_deg": 53.0, "lat_max_deg": 53.2,
                              "lon_min_deg": 6.0, "lon_max_deg": 6.2},
                "altitude": {"type": "constant", "min_ft": 0, "max_ft": 12000}},
    }
    spec.airspace = {"ref": "air"}
    spec.queryables = {
        "goal": {"type": "query_region", "bounds": {"ref": "air"}},
        "far": {"type": "query_region", "bounds": {"ref": "far"}},
    }
    spec.spawn["regions"] = []
    warnings = airspace_warnings(build_scenario(spec).support())
    assert "queryable 'goal'" not in warnings
    assert "queryable 'far'" in warnings
    print("  airspace warning: shared-bounds not flagged, outside flagged OK")


def test_spawn_altitude_from_bounds():
    # With no params['alt_ft'], spawn altitude is sampled from the bounds band.
    region = SpawnRegion(
        RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 5.0), ConstantAltitudeBand(4_000, 12_000)),
        n_aircraft=3,
        params={"spd_kts": (200, 250)},
    )
    rng = np.random.default_rng(0)
    alts = [region.sample_pos(rng)["alt_ft"] for _ in range(20)]
    assert all(4_000 <= a <= 12_000 for a in alts)
    assert len({round(a) for a in alts}) > 1
    assert SpawnConfig(regions=[region]).resolved_bounds["alt_ft"] == (4_000.0, 12_000.0)
    # Back-compat: an explicit params['alt_ft'] still drives the altitude.
    explicit = SpawnRegion(
        RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 5.0)),
        n_aircraft=2,
        params={"alt_ft": (5_000, 7_000), "spd_kts": (200, 250)},
    )
    assert all(5_000 <= explicit.sample_pos(rng)["alt_ft"] <= 7_000 for _ in range(20))
    envelope = SpawnRegion(
        RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 5.0)),
        n_aircraft=2,
        params={"alt_ft": EnvelopeSample(), "spd_kts": 240},
        aircraft_type="A320",
    )
    envelope_alts = [envelope.sample_pos(rng, "A320")["alt_ft"] for _ in range(20)]
    assert all(1_000 <= a <= 45_000 for a in envelope_alts)
    dumped = S.dump(envelope)
    assert dumped["params"]["alt_ft"] == {"type": "envelope"}
    assert isinstance(S.load(dumped).params["alt_ft"], EnvelopeSample)
    # Neither source -> a clear error.
    try:
        SpawnRegion(RegionBounds(BoxFootprint(51.8, 52.2, 4.4, 5.0)), n_aircraft=1, params={"spd_kts": (200, 250)})
        assert False, "expected ValueError for missing altitude source"
    except ValueError:
        pass
    print("  spawn altitude: bounds band, explicit params, envelope, missing-source error OK")


def test_per_aircraft_sampled_waypoint():
    # A waypoint with sample_per="aircraft" becomes route-step sampling metadata;
    # the Waypoint queryable itself remains a static/query definition.
    from bluesky_sandbox.sim.queryables import Waypoint

    spec = _example_design_spec()
    spec.regions = {
        "goalzone": {"type": "region",
                     "footprint": {"type": "box", "lat_min_deg": 52.5, "lat_max_deg": 52.8,
                                   "lon_min_deg": 5.5, "lon_max_deg": 5.9},
                     "altitude": {"type": "constant", "min_ft": 2000, "max_ft": 4000}},
    }
    spec.queryables = {"goal": {"type": "waypoint", "sample": {"ref": "goalzone"}, "sample_per": "aircraft", "alt_ft": 3000}}
    spec.spawn["route"] = ["goal"]
    # round-trip preserves the per-aircraft sample
    rt = S.DesignSpec.from_json(spec.to_json()).queryables["goal"]
    assert rt["sample_per"] == "aircraft" and rt["sample"] == {"ref": "goalzone"}
    # build resolves the ref onto the route step, not onto the Waypoint.
    scenario = build_scenario(spec)
    assert "goal" not in scenario.sampled_waypoints
    goal = scenario.support().queryables["goal"]
    assert isinstance(goal, Waypoint)
    route_step = scenario.support().spawn.route[0]
    assert route_step["waypoint"] == "goal"
    assert 52.5 <= route_step["sample"].bounding_box[0][0] <= 52.8
    assert goal.alt_ft == 3000
    # codegen emits sampling on the route step, not on Waypoint(...).
    files = codegen.generate_task(spec, "PerAc Demo")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "sample_region=" not in scenario_py
    assert "'sample': REGIONS['goalzone']" in scenario_py
    print("  per-aircraft sampled waypoint: route-step sampling + codegen OK")


def test_envelope_value_waypoint():
    # A waypoint with alt_ft / speed_kts == {"type": "envelope"} defers those
    # route constraints to a per-aircraft draw within the flight envelope.
    from bluesky_sandbox.sim.queryables import Waypoint

    spec = _example_design_spec()
    spec.queryables = {
        "goal": {
            "type": "waypoint",
            "lat": 56.0,
            "lon": 2.0,
            "alt_ft": {"type": "envelope"},
            "speed_kts": {"type": "envelope"},
            "alt_tolerance_ft": 500,
            "speed_tolerance_kts": 10,
        }
    }
    spec.spawn["route"] = ["goal"]
    # round-trip preserves the envelope markers
    rt = S.DesignSpec.from_json(spec.to_json()).queryables["goal"]
    assert rt["alt_ft"] == {"type": "envelope"}
    assert rt["speed_kts"] == {"type": "envelope"}
    # build resolves the markers onto the route step, leaving the queryable static.
    scenario = build_scenario(spec)
    goal = scenario.support().queryables["goal"]
    assert isinstance(goal, Waypoint)
    assert goal.alt_ft is None and goal.speed_kts is None
    route_step = scenario.support().spawn.route[0]
    assert route_step["sample_alt_from_envelope"] is True
    assert route_step["sample_speed_from_envelope"] is True
    # codegen reconstructs envelope-mode route metadata.
    files = codegen.generate_task(spec, "Envelope Demo")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "sample_alt_from_envelope=True" not in scenario_py
    assert "'sample_alt_from_envelope': True" in scenario_py
    assert "'sample_speed_from_envelope': True" in scenario_py
    print("  envelope-value waypoint: route metadata + codegen OK")


def test_active_route_waypoint_fields_catalogued():
    # The name-free active-route obs fields are offered with no queryable_spec.
    from bluesky_sandbox.ui.designer import catalog

    fields = {f["name"]: f for f in catalog.obs_fields()}
    for name in ("ActiveRouteWaypointDistanceNm", "ActiveRouteWaypointBearingDeg",
                 "ActiveRouteWaypointTrackErrorDeg", "ActiveRouteWaypointAltDiffFt",
                 "ActiveRouteWaypointSpdDiffKts"):
        assert name in fields, name
        assert fields[name].get("queryable_spec") is None
    print("  active-route waypoint fields: catalogued, name-free OK")


def test_route_composition_subroutes():
    # A route can include {"route": name} steps, expanded recursively at resolve.
    from bluesky_sandbox.sim.spawn import resolve_route

    routes = {"approach": ["FAF", "RWY"], "arrival": ["IAF", {"route": "approach"}, "MISSED"]}
    assert resolve_route(routes["arrival"], routes) == ["IAF", "FAF", "RWY", "MISSED"]
    # nested + cycle
    nested = {"a": ["w1", {"route": "b"}], "b": ["w2", {"route": "c"}], "c": ["w3"]}
    assert resolve_route(nested["a"], nested) == ["w1", "w2", "w3"]
    for bad in ({"a": [{"route": "b"}], "b": [{"route": "a"}]}, {"a": [{"route": "missing"}]}):
        try:
            resolve_route(bad["a"], bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
    # full designer round-trip + build + codegen with a subroute
    def wp(lat, lon):
        return {"type": "waypoint", "lat": lat, "lon": lon}
    spec = _example_design_spec()
    spec.queryables = {"IAF": wp(52.8, 5), "FAF": wp(52.4, 5), "RWY": wp(52.1, 5), "MISSED": wp(51.5, 5)}
    spec.spawn["routes"] = routes
    spec.spawn["regions"][0]["route"] = "arrival"
    assert S.DesignSpec.from_json(spec.to_json()).spawn["routes"]["arrival"][1] == {"route": "approach"}
    build_scenario(spec).support()  # validates (expands subroutes)
    files = codegen.generate_task(spec, "Subroute Demo")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "{'route': 'approach'}" in scenario_py
    print("  route composition: subroutes expand, validate, round-trip, codegen OK")


def test_route_composition_branches():
    # {"choice": [...]} lets a procedure diverge (SID transitions) or merge
    # (STAR entries onto a shared trunk); shared junction waypoints collapse.
    import numpy as np

    from bluesky_sandbox.sim.spawn import (
        expand_route_paths,
        resolve_route,
        sample_route_path,
    )

    routes = {
        "SID_CORE": ["DER", "ARNEM", "SUGOL"],
        "EAST": ["SUGOL", "EEL"],
        "NORTH": ["SUGOL", "EMARI"],
        "SID": [{"route": "SID_CORE"},
                {"choice": [{"route": "EAST"}, {"route": "NORTH"}], "weights": [0.75, 0.25]}],
        "ENTRY_N": ["N1", "RIVER"],
        "ENTRY_S": ["S1", "RIVER"],
        "TRUNK": ["RIVER", "RWY"],
        "STAR": [{"choice": [{"route": "ENTRY_N"}, {"route": "ENTRY_S"}]}, {"route": "TRUNK"}],
    }

    # Divergence: two paths, junction SUGOL appears once (collapsed).
    sid_paths = expand_route_paths(routes["SID"], routes)
    assert ["DER", "ARNEM", "SUGOL", "EEL"] in sid_paths
    assert ["DER", "ARNEM", "SUGOL", "EMARI"] in sid_paths
    assert all(p.count("SUGOL") == 1 for p in sid_paths)

    # Merge: both entries funnel onto the shared trunk at RIVER (once).
    star_paths = expand_route_paths(routes["STAR"], routes)
    assert ["N1", "RIVER", "RWY"] in star_paths and ["S1", "RIVER", "RWY"] in star_paths
    assert all(p.count("RIVER") == 1 for p in star_paths)

    # Weighted sampling stays on the configured split and only yields real paths.
    rng = np.random.default_rng(0)
    ends = [sample_route_path(routes["SID"], routes, rng)[-1] for _ in range(4000)]
    assert 0.70 < ends.count("EEL") / len(ends) < 0.80
    # resolve_route stays deterministic (first branch) for back-compat.
    assert resolve_route(routes["SID"], routes) == ["DER", "ARNEM", "SUGOL", "EEL"]

    # Validation: weight/branch mismatch and empty branch list are rejected.
    for bad in ([{"choice": ["A", "B"], "weights": [1.0]}], [{"choice": []}]):
        try:
            expand_route_paths(bad, {"A": ["a"], "B": ["b"]})
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("  route composition: choice diverge/merge, dedupe, weights, validate OK")


def test_route_step_crossing_restrictions():
    # A {"waypoint": name, speed_kts, alt_ft} step is a route-local crossing
    # restriction: preserved for ADDWPT, but names-only for viz/validation/hooks.
    import numpy as np

    from bluesky_sandbox.sim.spawn import (
        expand_route_paths,
        resolve_route,
        route_step_names,
        sample_route_path,
    )

    routes = {"STAR": ["RIVER", {"waypoint": "EEL", "speed_kts": 250, "alt_ft": 10000}]}
    rng = np.random.default_rng(0)
    sampled = sample_route_path(routes["STAR"], routes, rng)
    assert sampled == ["RIVER", {"waypoint": "EEL", "speed_kts": 250, "alt_ft": 10000}]
    # Names-only views drop the restriction.
    assert route_step_names(sampled) == ["RIVER", "EEL"]
    assert expand_route_paths(routes["STAR"], routes) == [["RIVER", "EEL"]]
    assert resolve_route(routes["STAR"], routes) == ["RIVER", "EEL"]

    # Validation rejects bad override keys / non-finite values.
    for bad in (
        [{"waypoint": "EEL", "speed_kts": "fast"}],
        [{"waypoint": "EEL", "speed_kts": 250}],
        [{"waypoint": "EEL", "bogus": 1}],
        [{"speed_kts": 250}],
    ):
        try:
            expand_route_paths(bad, {})
            assert False, "expected ValueError"
        except ValueError:
            pass

    # A constrained step survives the designer round-trip + build + codegen.
    def wp(lat, lon):
        return {"type": "waypoint", "lat": lat, "lon": lon}
    spec = _example_design_spec()
    spec.queryables = {"RIVER": wp(52.5, 5.0), "EEL": wp(52.3, 5.4)}
    spec.spawn["routes"] = {
        "STAR": ["RIVER", {"waypoint": "EEL", "speed_kts": 250, "alt_ft": 10000}]
    }
    spec.spawn["regions"][0]["route"] = "STAR"
    rt = S.DesignSpec.from_json(spec.to_json())
    assert rt.spawn["routes"]["STAR"][1] == {
        "waypoint": "EEL",
        "speed_kts": 250,
        "alt_ft": 10000,
    }
    build_scenario(spec).support()  # validates the constrained step
    files = codegen.generate_task(spec, "Crossing Demo")
    scenario_py = files[f"{next(iter(files)).split('/', 1)[0]}/scenario.py"]
    assert "'waypoint': 'EEL'" in scenario_py and "'speed_kts': 250" in scenario_py
    print("  route steps: crossing restrictions preserved, validate, round-trip, codegen OK")


def test_route_speed_requires_resolved_altitude():
    spec = _example_design_spec()
    spec.queryables = {
        "FAST": {"type": "waypoint", "lat": 52.0, "lon": 4.5, "speed_kts": 250}
    }
    spec.spawn["route"] = ["FAST"]
    env_cls = type("RouteSpeedEnv", (BlueskyEnv,), {})
    env = env_cls(
        scenario=build_scenario(spec),
        config=build_design_config(spec),
        render_mode=None,
    )
    try:
        env.reset(seed=0)
    except ValueError as e:
        assert "speed constraints require alt_ft" in str(e)
    else:
        raise AssertionError("expected speed-only route target to fail")
    finally:
        env.close()
    print("  route speed constraints require resolved altitude OK")


def test_env_hooks_catalog_and_codegen():
    # Hooks are discovered by introspection (no hard-coded list) and only the
    # customised ones are emitted - no super() boilerplate for the rest.
    from bluesky_sandbox.ui.designer import catalog

    hook_names = {h["name"] for h in catalog.hooks()}
    assert {"on_aircraft_spawned", "define_agent_context", "on_episode_reset"} <= hook_names
    # clean (annotation-free, underscore-stripped) signatures for codegen
    by_name = {h["name"]: h for h in catalog.hooks()}
    assert by_name["on_aircraft_spawned"]["def_signature"] == "(self, callsign, route)"

    spec = _example_design_spec()
    spec.env.hooks = {"on_aircraft_spawned": "bs.stack.stack(f'SPD {callsign} 250')"}
    assert S.DesignSpec.from_json(spec.to_json()).env.hooks == spec.env.hooks
    files = codegen.generate_task(spec, "Hooks Demo")
    env_py = files[f"{next(iter(files)).split('/', 1)[0]}/env.py"]
    assert "def on_aircraft_spawned(self, callsign, route):" in env_py
    assert "{callsign}" in env_py  # body uses the natural param name
    # uncustomised hooks are NOT emitted (inherited; no super() stub)
    assert "def on_sim_step" not in env_py
    print("  env hooks: introspected catalog + customised-only codegen OK")


def test_completion_context_uses_built_config_and_hook_protocols():
    from bluesky_sandbox.ui.designer import catalog
    from bluesky_sandbox.ui.designer.api import _spec_completion_context

    spec = _example_design_spec()
    spec.queryables["wp"] = S.dump(Waypoint(lat=52.0, lon=4.5, alt_ft=3000, color="magenta"))
    spec.env.hook_setup = "import numpy as np\nSCALE = np.ones(1)"
    spec.env.task_info_setup = "from math import sqrt as root\nLIMIT = root(4)"
    ctx = _spec_completion_context(spec)
    assert ctx["ok"] is True
    assert ctx["hook_setup"]["imports"]["np"] == "numpy"
    assert {symbol["name"] for symbol in ctx["hook_setup"]["symbols"]} >= {"np", "SCALE"}
    assert ctx["task_info_setup"]["imports"]["root"] == "math.sqrt"
    assert {symbol["name"] for symbol in ctx["task_info_setup"]["symbols"]} >= {"root", "LIMIT"}
    assert [field["name"] for field in ctx["obs_fields"]] == ["lat_deg", "lon_deg", "alt_ft"]
    assert [field["name"] for field in ctx["action_fields"]] == ["hdg_deg", "spd_kts"]
    reward_params = {param["name"]: param["detail"] for param in ctx["hooks"]["reward"]["params"]}
    assert "numpy.ndarray" in reward_params["obs"]
    assert reward_params["terminated"] == "bool"
    assert reward_params["truncated"] == "bool"
    context_members = {member["name"] for member in ctx["task_info"]["members"]["context"]}
    assert {"acid", "acidx", "airspace", "query", "queryable"} <= context_members
    airspace_members = {member["name"] for member in ctx["airspace_result_members"]}
    assert {"current", "step", "time"} <= airspace_members
    airspace_current_members = {
        member["name"]
        for member in ctx["airspace_result_nested_members"]["current"]
    }
    assert "inside" in airspace_current_members
    info_members = {member["name"]: member for member in ctx["task_info"]["members"]["info"]}
    assert info_members["acid"]["access"] == "item"
    rng_members = {member["name"] for member in ctx["task_info"]["members"]["rng"]}
    assert {"normal", "uniform"} <= rng_members
    wp_queryable = next(symbol for symbol in ctx["queryables"] if symbol["insert"] == '"wp"')
    assert wp_queryable["color"] == "magenta"
    waypoint_members = {member["name"] for member in ctx["query_result_members"]["wp"]}
    assert {
        "current",
        "route",
        "target",
        "step",
        "time",
        "aircraft_altitude_ceiling_ft",
        "altitude_error_scale_ft",
        "speed_error_scale_kts",
    } <= waypoint_members
    waypoint_current_members = {
        member["name"]
        for member in ctx["query_result_nested_members"]["wp"]["current"]
    }
    assert {"distance_nm", "bearing_deg", "satisfied"} <= waypoint_current_members
    queryable_members = {member["name"] for member in ctx["queryable_members"]["wp"]}
    assert "target_for" not in queryable_members
    assert "current" not in queryable_members
    assert "query" not in queryable_members
    assert "_current_result" not in queryable_members
    assert "_assigned_targets" not in queryable_members
    task_info_types = {item["name"] for item in catalog.task_info_types()}
    # Only the auto-cost provider is offered; the others were removed from the
    # library because a task defines its own against the protocols.
    assert {"AutoCostConstraintTaskInfoProvider"} == task_info_types
    by_type = {item["name"]: item for item in catalog.task_info_types()}
    autocost_scaffold = by_type["AutoCostConstraintTaskInfoProvider"]["scaffold"]
    assert "def auto_cost_constraint_extrinsic_cost" in autocost_scaffold["setup"]
    assert "def auto_cost_constraint_intrinsic_cost" in autocost_scaffold["setup"]
    assert "AutoCostConstraintTaskInfoProvider(" in autocost_scaffold["setup"]
def test_codegen_task_info_provider_object_scaffold():
    from bluesky_sandbox.ui.designer import catalog

    spec = _example_design_spec()
    scaffold = {
        item["name"]: item for item in catalog.task_info_types()
    }["AutoCostConstraintTaskInfoProvider"]["scaffold"]
    var = scaffold["provider_var"]
    spec.env.task_info_setup = scaffold["setup"]
    spec.env.task_info = [S.TaskInfoSpec("constraints", scaffold["body"])]

    files = codegen.generate_task(spec, "Constraint Demo")
    pkg = next(iter(files)).split("/", 1)[0]
    env_py = files[f"{pkg}/env.py"]
    setup_py = files[f"{pkg}/setup.py"]
    # The provider CLASS is written into the task, not imported from the
    # library - that is the whole point of scaffolding it.
    assert "class AutoCostConstraintTaskInfoProvider:" in setup_py
    assert "from bluesky_sandbox.interface.task import AutoCostConstraintTaskInfoProvider" not in setup_py
    # ...and the library still supplies the protocols it is written against.
    assert "from bluesky_sandbox.interface.task import (" in setup_py
    # The provider object is module-level setup, so it lands in setup.py; env.py
    # keeps the hook that references it.
    assert f"{var} = AutoCostConstraintTaskInfoProvider(" in setup_py
    assert "def constraints(obs, action, info, context, rng)" not in setup_py
    assert f"return [{var}]" in env_py
    assert var in env_py.split("class ")[0]  # imported

    cfg = build_design_config(spec)
    assert len(cfg.task_info_providers) == 1
    print("  codegen: task-info provider-object scaffold returns provider directly OK")


def test_bounds_rotation_deg():
    # A non-square box rotated 90deg about its centre changes its lon/lat extent.
    base = {"type": "region",
            "footprint": {"type": "box", "lat_min_deg": 51.8, "lat_max_deg": 52.2,
                          "lon_min_deg": 4.0, "lon_max_deg": 5.0},
            "altitude": {"type": "constant", "min_ft": 0, "max_ft": 10000}}
    plain = S.load(base)
    rotated = S.load({**base, "rotation_deg": 90.0})
    (plat0, plat1), (plon0, plon1) = plain.bounding_box
    (rlat0, rlat1), (rlon0, rlon1) = rotated.bounding_box
    # rotation swaps the (wider lon / narrower lat) extents -> taller, narrower
    assert (rlat1 - rlat0) > (plat1 - plat0)
    assert (rlon1 - rlon0) < (plon1 - plon0)
    # rotation_deg = 0 / absent is identity
    assert S.load({**base, "rotation_deg": 0}).bounding_box == plain.bounding_box
    print("  bounds rotation_deg: rotates footprint about centre OK")


def test_build_scenario_and_env_config():
    spec = _example_design_spec()
    spec.env.hook_setup = "import math\nimport math\nHOOK_SCALE = math.sqrt(4.0)"
    spec.env.hooks["reward"] = "return HOOK_SCALE"
    spec.env.task_info_setup = "import numpy as np\nLIMITS = np.zeros(1, dtype=np.float32)"
    spec.env.task_info = [
        S.TaskInfoSpec("task_metric", 'info["task"]["metric"] = float(LIMITS[0] + 1.0)')
    ]

    scenario = build_scenario(spec)
    support = scenario.support()
    sample = scenario.sample(np.random.default_rng(0))
    assert support.max_aircraft == sample.max_aircraft == 5
    assert support.airspace_bounds is not None
    assert "goal" in support.queryables and isinstance(support.queryables["goal"], QueryRegion)

    cfg = build_design_config(spec)
    assert [f.meta.name for f in cfg.obs_fields] == ["lat_deg", "lon_deg", "alt_ft"]
    assert cfg.intruder_obs_fields is not None and len(cfg.intruder_obs_fields) == 1
    assert len(cfg.action_fields) == 2
    assert isinstance(cfg.obs_fields[2].normalizer, MinMaxNormalizer)
    assert isinstance(cfg.action_fields[1].normalizer, SymmetricNormalizer)
    assert cfg.action_fields[1].normalizer.clipped is True
    assert cfg.allowed_aircraft == ["A320", "B738"]
    assert len(cfg.task_info_providers) == 1
    # Config remains static; scenario airspace is not injected into field bounds.
    lat_field = cfg.obs_fields[0]
    assert _approx(lat_field.low, -90.0) and _approx(lat_field.high, 90.0)
    print("  build_scenario + build_design_config OK (fields, code-refs)")


def test_stacked_field_transform():
    # Frame stacking is the one transform that expands ONE spec entry into
    # several observation channels, so it is the one that can silently disagree
    # with the observation space. Checked on both list kinds (an ownship
    # ObsField and an intruder PairObsField) and through codegen, where the
    # emitted `.stacked(...)` sits inside a tuple that EnvConfig must flatten.
    spec = _example_design_spec()
    norm = {"type": "normalizer", "name": "MinMaxNormalizer", "kwargs": {}}
    stacked_ref = S.FieldRef(
        "LonDeg",
        kwargs={"normalizer": norm},
        transform="stacked",
        transform_kwargs={"depth": 3},
    )
    spec.env.obs_fields = [S.FieldRef("LatDeg"), stacked_ref]
    spec.env.intruder_obs_fields = [
        S.FieldRef("DistToOwnNm", transform="stacked", transform_kwargs={"depth": 2}),
    ]

    cfg = build_design_config(spec)
    assert [f.meta.name for f in cfg.obs_fields] == [
        "lat_deg", "lon_deg", "lon_deg_lag1", "lon_deg_lag2",
    ]
    assert [f.meta.name for f in cfg.intruder_obs_fields] == [
        "dist_to_own_nm", "dist_to_own_nm_lag1",
    ]
    # Lag channels inherit the live field's bounds and normalizer, which is what
    # lets them share its calibration.
    live, lag1 = cfg.obs_fields[1], cfg.obs_fields[2]
    assert lag1.bounds(0) == live.bounds(0)
    assert isinstance(lag1.normalizer, MinMaxNormalizer)

    # depth=1 is the identity (live only), not an error.
    spec.env.obs_fields[1] = S.FieldRef(
        "LonDeg", transform="stacked", transform_kwargs={"depth": 1}
    )
    assert [f.meta.name for f in build_design_config(spec).obs_fields] == [
        "lat_deg", "lon_deg",
    ]

    # Spec round-trip keeps the transform, and codegen emits it.
    spec.env.obs_fields[1] = stacked_ref
    reloaded = S.DesignSpec.from_json(spec.to_json())
    assert reloaded.env.obs_fields[1].transform == "stacked"
    assert reloaded.env.obs_fields[1].transform_kwargs == {"depth": 3}
    files = codegen.generate_task(spec, "Stacked Demo")
    pkg = next(iter(files)).split("/", 1)[0]
    config_py = files[f"{pkg}/config.py"]
    assert ".stacked(depth=3)" in config_py
    assert ".stacked(depth=2)" in config_py
    compile(config_py, "config.py", "exec")
    print("  stacked transform: expands to lag channels, round-trips, codegen OK")


# --------------------------------------------------------------------------- #
# nav                                                                          #
# --------------------------------------------------------------------------- #
def test_nav_resolve_and_window():
    airport = nav.resolve_airport("EHAM")
    assert airport.icao == "EHAM"
    assert 52.0 < airport.lat_deg < 52.7 and 4.0 < airport.lon_deg < 5.2

    wp = nav.resolve_waypoint("EKROS")  # a real fix near Schiphol
    assert _approx(wp.lat_deg, 52.237, tol=0.1) and _approx(wp.lon_deg, 4.620, tol=0.1)

    # window scoping: a tight box around Schiphol contains EHAM and few features
    bounds = RegionBounds(BoxFootprint(52.0, 52.6, 4.3, 5.0))
    payload = nav.features_in_bounds(bounds, waypoint_limit=50, airport_limit=20)
    icaos = {a.icao for a in payload["airports"]}
    assert "EHAM" in icaos
    for w in payload["waypoints"]:
        assert bounds.bounding_box[0][0] - 1 <= w.lat_deg <= bounds.bounding_box[0][1] + 1

    try:
        nav.resolve_waypoint("ZZZZNOTAFIX")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown waypoint")
    print("  nav: resolve_airport/resolve_waypoint/window scoping/loud-miss OK")


def test_nav_search():
    res = nav.search("EHAM")
    assert "EHAM" in {a.icao for a in res["airports"]}
    # prefix search returns several Dutch airports
    res = nav.search("EH", limit=10)
    assert all(a.icao.upper().startswith("EH") or "EH" in (a.name or "").upper()
               for a in res["airports"])
    assert nav.search("")["waypoints"] == []
    res = nav.search("EKROS")
    assert any(w.ident == "EKROS" for w in res["waypoints"])
    print("  nav search: airport/waypoint prefix + empty-query OK")


# --------------------------------------------------------------------------- #
# codegen                                                                      #
# --------------------------------------------------------------------------- #
def test_codegen_generates_importable_package():
    import importlib
    import sys
    import tempfile

    spec = _example_design_spec()
    spec.env.hook_setup = "import math\nimport math\nHOOK_SCALE = math.sqrt(4.0)"
    spec.env.hooks["reward"] = "return HOOK_SCALE"
    spec.env.task_info_setup = "import numpy as np\nLIMITS = np.zeros(1, dtype=np.float32)"
    spec.env.task_info = [
        S.TaskInfoSpec("task_metric", 'info["task"]["metric"] = float(LIMITS[0] + 1.0)')
    ]
    files = codegen.generate_task(spec, "My Demo Task!")
    pkg = next(iter(files)).split("/", 1)[0]
    assert pkg == "my_demo_task"
    # reward/terminated/truncated are emitted as hooks in env.py, so no task.py.
    expected = {f"{pkg}/{n}" for n in
                ("__init__.py", "design.json", "scenario.py", "config.py", "setup.py",
                 "env.py", "__main__.py", "README.md")}
    assert set(files) == expected
    assert S.DesignSpec.from_json(files[f"{pkg}/design.json"]).to_dict() == spec.to_dict()

    # Runtime files are emitted as Python; design.json is for designer reload.
    scenario_py = files[f"{pkg}/scenario.py"]
    assert "RegionBounds(" in scenario_py and "SpawnConfig(" in scenario_py
    assert "AIRSPACE =" not in scenario_py and "SPAWN =" not in scenario_py
    assert "class MyDemoTaskScenario(RandomizedScenario):" in scenario_py
    assert "def __init__(self) -> None:" in scenario_py
    assert "def _design_scenario(self)" not in scenario_py
    assert "def sample(self" not in scenario_py and "def support(self" not in scenario_py
    assert "with_max_aircraft" not in scenario_py
    assert "def make_scenario" not in scenario_py
    config_py = files[f"{pkg}/config.py"]
    assert "obs.LatDeg(" in config_py
    assert "allowed_aircraft=list(['A320', 'B738'])" in config_py
    assert "TASK_INFO_PROVIDERS" not in config_py
    assert "INTRUDER_OBS_FIELDS" not in config_py
    env_py = files[f"{pkg}/env.py"]
    setup_py = files[f"{pkg}/setup.py"]
    assert "def define_obs_fields(self):" not in env_py
    # Module-level setup lives in setup.py; env.py keeps the hooks and the class.
    assert setup_py.count("import math") == 1
    assert "import numpy as np" in setup_py
    assert "LIMITS = np.zeros(1, dtype=np.float32)" in setup_py
    assert "def task_metric(obs, action, info, context, rng) -> None:" in setup_py
    assert 'info["task"]["metric"] = float(LIMITS[0] + 1.0)' in setup_py
    assert "HOOK_SCALE = math.sqrt(4.0)" in setup_py
    assert "def define_task_info_providers(self):" in env_py
    assert "return [task_metric]" in env_py
    assert "return HOOK_SCALE" in env_py
    # ...and imports exactly the setup names its hooks read.
    header = env_py.split("class ")[0]
    assert "from .setup import (" in header
    assert "HOOK_SCALE," in header and "task_metric," in header
    assert "LIMITS," not in header  # only task_metric's body reads it
    assert "MinMaxNormalizer()" in config_py and "SymmetricNormalizer(clipped=True)" in config_py
    assert "allowed_aircraft = tuple(ALLOWED_AIRCRAFT)" not in env_py
    assert "make_config" not in env_py and "make_env" not in env_py
    assert "config=CONFIG" in env_py

    # write it out, import it, and verify it builds a scenario + env config
    with tempfile.TemporaryDirectory() as tmp:
        codegen.write_task(spec, "My Demo Task!", tmp)
        sys.path.insert(0, tmp)
        try:
            mod = importlib.import_module(pkg)
            scenario = mod.Scenario()
            assert scenario.support().max_aircraft == 5
            cfg = importlib.import_module(f"{pkg}.config").CONFIG
            assert [f.meta.name for f in cfg.action_fields] == ["hdg_deg", "spd_kts"]
            assert len(cfg.task_info_providers) == 0
            env = mod.Env(render_mode=None)
            assert len(env.config.task_info_providers) == 1
            env.close()
        finally:
            sys.path.remove(tmp)
            for name in list(sys.modules):
                if name == pkg or name.startswith(pkg + "."):
                    del sys.modules[name]
    print("  codegen: package generates, imports, builds scenario + env config OK")


# --------------------------------------------------------------------------- #
# custom code: editable task.py + custom observation field                    #
# --------------------------------------------------------------------------- #
_CUSTOM_FIELDS_PY = '''
from dataclasses import dataclass
from typing import Any
from bluesky_sandbox.interface.fields.base import ObsField, ObsMeta, Unit, ObsQuantity


@dataclass(frozen=True)
class DoubleLat(ObsField):
    meta = ObsMeta("double_lat", Unit.DEG, ObsQuantity.LATITUDE)
    low: float = -180.0
    high: float = 180.0

    def get(self, idx: Any) -> Any:
        return 2.0 * 0.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()
'''

_TASK_PY = '''
def reward(obs, action, terminated, truncated, context, info, rng):
    return 1.5

def terminated(obs, action, context, info, rng):
    return False

def truncated(obs, action, context, info, rng):
    return False
'''


def test_reward_hooks_migration_from_task_py():
    # An old-model spec (reward via task.py + env.reward_fn) migrates on load:
    # the function bodies are lifted into reward/terminated/truncated hooks and
    # task.py is dropped.
    spec = _example_design_spec()
    d = spec.to_dict()
    d["env"]["reward_fn"] = "task:reward"
    d["env"]["termination_fn"] = "task:terminated"
    d["env"]["truncation_fn"] = "task:truncated"
    d["code"] = {"task.py": _TASK_PY}
    migrated = S.DesignSpec.from_dict(d)
    assert migrated.env.hooks.get("reward") == "return 1.5"
    assert migrated.env.hooks.get("terminated") == "return False"
    assert "task.py" not in migrated.code
    # and it builds + codegens with the reward hook
    files = codegen.generate_task(migrated, "Migrated")
    env_py = files[f"{next(iter(files)).split('/', 1)[0]}/env.py"]
    assert "return 1.5" in env_py and "task.py" not in [k.split("/")[-1] for k in files]
    print("  reward migration: task.py functions lifted into hooks OK")


def _spec_with_custom_code() -> S.DesignSpec:
    spec = _example_design_spec()
    spec.code = {"custom_fields.py": _CUSTOM_FIELDS_PY}
    # reward is now an env hook, not a task.py function
    spec.env.hooks = {"reward": "return 1.5"}
    spec.env.obs_fields = [S.FieldRef("LatDeg"), S.FieldRef("custom_fields:DoubleLat")]
    return spec


def test_custom_code_fields_resolve_in_build():
    spec = _spec_with_custom_code()
    cfg = build_design_config(spec)
    # the custom observation field resolved via "custom_fields:DoubleLat"
    assert [f.meta.name for f in cfg.obs_fields] == ["lat_deg", "double_lat"]
    print("  custom code: custom_fields.py ObsField resolves OK")


def test_queryable_obs_fields_resolve_in_build_and_codegen():
    spec = _example_design_spec()
    spec.queryables["merge"] = S.dump(Waypoint(lat=52.0, lon=4.7, alt_ft=3000.0))
    spec.env.obs_fields = [
        S.FieldRef("WaypointRouteIndex", kwargs={"query_name": "merge"}),
    ]
    cfg = build_design_config(spec)
    assert [f.meta.name for f in cfg.obs_fields] == ["merge_route_index"]

    files = codegen.generate_task(spec, "queryable_field_pkg")
    config_py = files["queryable_field_pkg/config.py"]
    assert "from bluesky_sandbox.interface.fields import queryables as qobs" in config_py
    assert "qobs.WaypointRouteIndex(query_name='merge')" in config_py
    print("  queryable fields: bare refs resolve + codegen uses qobs alias OK")


def test_temporal_queryable_fields_require_tracking_flag():
    spec = _example_design_spec()
    spec.queryables["zone"] = {
        "type": "query_region",
        "bounds": S.dump(RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.8))),
    }
    spec.queryables["merge"] = S.dump(Waypoint(lat=52.0, lon=4.7, alt_ft=3000.0))
    spec.env.obs_fields = [
        S.FieldRef("QueryRegionInsideDuringStep", kwargs={"query_name": "zone"}),
        S.FieldRef("WaypointSatisfiedDuringStep", kwargs={"query_name": "merge"}),
    ]
    cfg = build_design_config(spec)
    assert [f.meta.name for f in cfg.obs_fields] == [
        "zone_inside_during_step",
        "merge_satisfied_during_step",
    ]
    support = build_scenario(spec).support()
    assert support.queryables["zone"].track_temporal_state is True
    assert support.queryables["merge"].track_temporal_state is True

    files = codegen.generate_task(spec, "temporal_queryable_pkg")
    scenario_py = files["temporal_queryable_pkg/scenario.py"]
    assert scenario_py.count("track_temporal_state=True") >= 2
    print("  queryable temporal fields: scenario tracking inferred + codegens OK")


def test_unavailable_temporal_query_state_rejects_temporal_access():
    region_step = UnavailableRegionStep()
    region_result = RegionResult()
    waypoint_step = UnavailableWaypointStep()
    waypoint_result = WaypointResult()
    for access in (
        lambda: region_step.inside,
        lambda: region_result.time.total_s,
        lambda: region_result.time.during_step_s,
        lambda: waypoint_step.min_distance_nm,
        lambda: waypoint_step.min_abs_alt_diff_ft,
        lambda: waypoint_result.time.total_s,
        lambda: waypoint_result.time.during_step_s,
    ):
        try:
            access()
        except QueryableTemporalStateUnavailable as e:
            assert "track_temporal_state" in str(e)
        else:
            raise AssertionError("unavailable temporal access should fail")
    print("  unavailable query temporal state rejects access OK")


def test_unbound_query_results_reject_lazy_current_access():
    for access in (
        lambda: RegionResult().current,
        lambda: bool(RegionResult()),
        lambda: WaypointResult().target,
        lambda: WaypointResult().current,
        lambda: WaypointResult().route,
        lambda: WaypointResult().aircraft_altitude_ceiling_ft,
        lambda: WaypointResult().altitude_error_scale_ft,
        lambda: WaypointResult().speed_error_scale_kts,
    ):
        try:
            access()
        except RuntimeError as e:
            assert "for_aircraft" in str(e)
        else:
            raise AssertionError("unbound query result access should fail")
    print("  unbound query results reject lazy current access OK")


def test_codegen_with_custom_code_imports():
    import importlib
    import sys
    import tempfile

    spec = _spec_with_custom_code()
    files = codegen.generate_task(spec, "custom_pkg")
    assert "custom_pkg/custom_fields.py" in files and "custom_pkg/task.py" not in files
    # the custom field is package-qualified in the emitted config.py
    env_py = files["custom_pkg/env.py"]
    config_py = files["custom_pkg/config.py"]
    assert "import custom_pkg.custom_fields as custom_fields" in config_py
    assert "custom_fields.DoubleLat(" in config_py
    # reward is emitted as a hook method body (not imported from task.py)
    assert "def reward(self, obs, action, terminated, truncated, context, info, rng):" in env_py
    assert "return 1.5" in env_py and "_reward_fn" not in env_py

    with tempfile.TemporaryDirectory() as tmp:
        codegen.write_task(spec, "custom_pkg", tmp)
        sys.path.insert(0, tmp)
        try:
            cfg = importlib.import_module("custom_pkg.config").CONFIG
            assert [f.meta.name for f in cfg.obs_fields] == ["lat_deg", "double_lat"]
        finally:
            sys.path.remove(tmp)
            for name in list(sys.modules):
                if name == "custom_pkg" or name.startswith("custom_pkg."):
                    del sys.modules[name]
    print("  codegen+custom: generated package with custom field imports + builds OK")


# --------------------------------------------------------------------------- #
# scenario code: module-level setup + per-episode geometry hook                 #
# --------------------------------------------------------------------------- #
def test_scenario_hooks():
    # The scenario-side twin of env hook_setup/hooks: sampling the structured
    # spec cannot express (here, nudging a waypoint by a drawn amount) stays IN
    # the design rather than forcing a hand-edit of the generated scenario.py.
    spec = _example_design_spec()
    # The fixture's only queryable is a query_region; the hook needs a waypoint
    # with a position to move.
    spec.queryables["fix"] = {
        "type": "waypoint", "lat": 52.0, "lon": 4.5, "alt_ft": 10000,
    }
    spec.scenario_setup = (
        "from dataclasses import replace\n"
        "\n"
        "SHIFT_DEG = 0.25\n"
        "\n"
        "\n"
        "def nudge(queryables, rng):\n"
        "    out = dict(queryables)\n"
        "    for name, q in out.items():\n"
        "        if hasattr(q, 'lat'):\n"
        "            out[name] = replace(q, lat=q.lat + rng.uniform(0, SHIFT_DEG))\n"
        "    return out\n"
    )
    spec.scenario_hooks = {
        "episode_geometry": (
            'return {**geometry, "queryables": nudge(geometry["queryables"], rng)}\n'
        )
    }

    # design.json carries both verbatim
    round_tripped = S.DesignSpec.from_json(spec.to_json())
    assert round_tripped.scenario_setup == spec.scenario_setup
    assert round_tripped.scenario_hooks == spec.scenario_hooks

    # an unknown hook name is an error, not a silent drop - a body under a
    # misspelled key would look live in the designer and never be emitted
    bad = spec.to_dict()
    bad["scenario_hooks"] = {"not_a_hook": "return geometry"}
    try:
        S.DesignSpec.from_dict(bad)
        raise AssertionError("expected SpecError for unknown scenario hook")
    except S.SpecError:
        pass

    # the DESIGNER path (live preview) runs the hook
    scenario = build_scenario(spec)
    base_lat = {n: q["lat"] for n, q in spec.queryables.items() if "lat" in q}
    assert base_lat, "fixture lost its waypoint"
    seen = set()
    for seed in range(20):
        qs = scenario.sample(np.random.default_rng(seed)).queryables
        for name, lat0 in base_lat.items():
            shifted = qs[name].lat - lat0
            assert 0.0 <= shifted <= 0.25 + 1e-9, f"{name} moved {shifted}"
            seen.add(round(shifted, 6))
    assert len(seen) > 1, "hook produced a constant shift; rng not threaded through"

    # the GENERATED path emits the same two strings, and the hook is chained
    # after the structured rebuild rather than replacing it
    files = codegen.generate_task(spec, "Hooked Scenario")
    pkg = next(iter(files)).split("/", 1)[0]
    scenario_py = files[f"{pkg}/scenario.py"]
    assert "SHIFT_DEG = 0.25" in scenario_py
    assert "def _episode_geometry(geometry, rng):" in scenario_py
    assert "self._episode_geometry(dict(sampler.episode_geometry(rng)), rng)" in scenario_py
    compile(scenario_py, "scenario.py", "exec")

    # a design with no scenario code is untouched: same template, no hook wiring
    plain = codegen.generate_task(_example_design_spec(), "Plain Scenario")
    plain_py = plain[f"{next(iter(plain)).split('/', 1)[0]}/scenario.py"]
    assert "_episode_geometry" not in plain_py
    print("  scenario hooks: setup+hook round-trip, run in builder AND codegen OK")


# --------------------------------------------------------------------------- #
# runner                                                                       #
# --------------------------------------------------------------------------- #
def _all_tests():
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    failures = 0
    for fn in _all_tests():
        try:
            print(f"- {fn.__name__}")
            fn()
        except Exception as e:  # noqa: BLE001 - test runner reports all failures
            failures += 1
            import traceback

            print(f"  FAILED: {e}")
            traceback.print_exc()
    total = len(_all_tests())
    print(f"\n{total - failures}/{total} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_runner_repo_root_is_the_package_parent():
    """The designer's "run design" subprocess gets this on its PYTHONPATH.

    Regression: it was ``Path(__file__).parents[2]``, which was the repo root
    while this module lived at ``bluesky_sandbox/designer/`` and became the
    package itself once ``designer`` moved under ``ui``. The child then failed
    with ``ModuleNotFoundError: No module named 'bluesky_sandbox'`` - a break
    invisible from inside the parent process, which imports fine either way.
    """
    from pathlib import Path

    from bluesky_sandbox.ui.designer.runner import _REPO_ROOT

    root = Path(_REPO_ROOT)
    assert (root / "bluesky_sandbox" / "__init__.py").is_file(), (
        f"{root} is not the parent of the bluesky_sandbox package"
    )
    assert root.name != "bluesky_sandbox"
    print("  runner repo root: OK")
