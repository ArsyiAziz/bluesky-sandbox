"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

from scipy.stats import randint
from bluesky_sandbox.sim.spawn import Route, TowardRoute
from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint, BooleanFootprint, BoxFootprint, ConstantAltitudeBand,
    DiskFootprint, LatLon, LinearAltitudeBand, PointFootprint, PolygonFootprint,
    RadialAltitudeBand, RegionBounds, SectorFootprint, VertexAltitudeBand,
    GeneratedFootprint, Blob, ConvexPolygon, VoronoiSectors, PlacedFootprint, InRegion,
    MovingFootprint, Drift, Spin, Grow,
)
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.spawn import (
    PlanContext, PlannedSource, SpawnConfig, SpawnRegion, SpawnRequest, nearest_entry,
)
from bluesky.tools.geo import kwikqdrdist, qdrpos



class CamrnFiveScenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {
            'airspace': RegionBounds(DiskFootprint(LatLon(39.35, -74.45), radius_nm=200, n_vertices=72), None),
    'RADDS': RegionBounds(PointFootprint(LatLon(38.648556, -75.088467), fix='RADDS'), None),
    'FISSH': RegionBounds(PointFootprint(LatLon(38.891425, -74.383956), fix='FISSH'), None),
    'BOTON': RegionBounds(PointFootprint(LatLon(39.414472, -74.454744), fix='BOTON'), None),
    'SIE': RegionBounds(PointFootprint(LatLon(39.0955, -74.80033333), fix='SIE'), None),
    'HOGGS': RegionBounds(PointFootprint(LatLon(39.582847, -74.270575), fix='HOGGS'), None),
    'KARRS': RegionBounds(PointFootprint(LatLon(39.840872, -73.985989), fix='KARRS'), None),
    'CAMRN': RegionBounds(PointFootprint(LatLon(40.017303, -73.861058), fix='CAMRN'), None),
    'J121_ENTRY': RegionBounds(PointFootprint(LatLon(38.4259, -75.2313)), ConstantAltitudeBand(24000, 28000)),
    'B24_ENTRY': RegionBounds(PointFootprint(LatLon(38.7472, -74.1134)), ConstantAltitudeBand(24000, 28000)),
    'PANZE': RegionBounds(PointFootprint(LatLon(39.676, -74.168194), fix='PANZE'), None)
        }
        transform = None
        rotation = None
        groups = None
        if transform and transform.get("rotation"):
            rot = transform["rotation"]
            pivot = rot.get("pivot")
            rotation = {"angle": rot["angle_deg"], "pivot": tuple(pivot) if pivot else None}
        if transform and transform.get("groups"):
            groups = tuple(transform["groups"])
        super().__init__(
            airspace_bounds=REGIONS['airspace'],
            spawn=SpawnConfig(regions=[
    SpawnRegion(
        shape=REGIONS['J121_ENTRY'],
        n_aircraft=randint(low=3, high=6),
        params={'spd_kts': (280, 300), 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'B738': 1, 'A321': 1, 'B739': 1, 'E190': 1}),
        spawn_time=(0, 900),
        route='VIA_J121',
        name='J121',
        conflict_free_spawn=True,
        spawn_gap_s=120.0,
    ),
    SpawnRegion(
        shape=REGIONS['B24_ENTRY'],
        n_aircraft=randint(low=3, high=6),
        params={'spd_kts': (280, 300), 'hdg_deg': TowardRoute()},
        aircraft_type=Categorical({'A320': 1, 'B738': 1, 'A321': 1, 'B739': 1, 'E190': 1}),
        spawn_time=(0, 900),
        route='VIA_B24',
        name='B24',
        conflict_free_spawn=True,
        spawn_gap_s=120.0,
    ),
], routes={'J121': Route(['RADDS', 'SIE'], color='green'), 'B24': Route(['FISSH', 'SIE'], color='blue'), 'CAMRN5': Route(['SIE', 'BOTON', {'waypoint': 'HOGGS', 'alt_ft': 18000}, 'PANZE', 'KARRS', {'waypoint': 'CAMRN', 'alt_ft': 11000, 'speed_kts': 250}], color='magenta'), 'VIA_J121': [{'route': 'J121'}, {'route': 'CAMRN5'}], 'VIA_B24': [{'route': 'B24'}, {'route': 'CAMRN5'}]}),
            queryables={
            'RADDS': Waypoint(at=REGIONS['RADDS'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'FISSH': Waypoint(at=REGIONS['FISSH'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'BOTON': Waypoint(at=REGIONS['BOTON'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'SIE': Waypoint(at=REGIONS['SIE'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'HOGGS': Waypoint(at=REGIONS['HOGGS'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'KARRS': Waypoint(at=REGIONS['KARRS'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'CAMRN': Waypoint(at=REGIONS['CAMRN'], reach_radius_nm=2, color='cyan', track_temporal_state=False),
    'J121_ENTRY': Waypoint(at=REGIONS['J121_ENTRY'], reach_radius_nm=2, color='magenta', track_temporal_state=False),
    'B24_ENTRY': Waypoint(at=REGIONS['B24_ENTRY'], reach_radius_nm=2, color='yellow', track_temporal_state=False),
    'PANZE': Waypoint(at=REGIONS['PANZE'], reach_radius_nm=2, color='cyan', track_temporal_state=False)
        },
            rotation=rotation,
            groups=groups,
            sampled_waypoints={
            
        },
            waypoint_fields={
            
        },
            shapes=REGIONS,
        )

