"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

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

ALT_FT = 20_000
CAS_KTS = 280.0
AC_TYPE = "A320"
START_NM = 40.0          # each aircraft's distance from the center at spawn
MAX_OFFSET_NM = 2.0      # the second aircraft's along-track shift: sets the miss distance


def _point(center, bearing_deg, dist_nm):
    """The point dist_nm from center, on that bearing from it."""
    lat, lon = qdrpos(center.lat_deg, center.lon_deg, bearing_deg, dist_nm)
    return LatLon(float(lat), float(lon))


def _exit_route(at, hdg_deg, ctx):
    """A route to where the straight line from ``at`` along ``hdg_deg`` leaves the airspace."""
    lat, lon = at.lat_deg, at.lon_deg
    while ctx.airspace.contains(lat, lon):
        lat, lon = qdrpos(lat, lon, hdg_deg, 1.0)
    return [{"waypoint": "exit", "sample": RegionBounds(PointFootprint(LatLon(float(lat), float(lon))))}]


# Spawn sources: each plans its aircraft at reset.
def _plan_crossing(rng, ctx):
    """Spawn source 'crossing': this episode's aircraft."""
    # This episode's aircraft, each a SpawnRequest with its time: read any data
    # (scenario setup can load a file once), draw from any distribution.
    # ctx: the episode's shapes, queryables, airspace and spawn config.

    center = ctx.airspace.center
    rot = rng.uniform(0.0, 360.0)
    offset = rng.uniform(-MAX_OFFSET_NM, MAX_OFFSET_NM)
    out = []
    for k, (hdg, dist) in enumerate([(rot, START_NM), (rot + 90.0, START_NM + offset)]):
        hdg %= 360.0
        at = _point(center, hdg + 180.0, dist)          # start on the reciprocal: flying at the center
        out.append(SpawnRequest(at=at, alt_ft=ALT_FT, spd_kts=CAS_KTS, hdg_deg=hdg, actype=AC_TYPE,
                                callsign=f"CROSS{k + 1}", route=_exit_route(at, hdg, ctx)))
    return out


SPAWN_SOURCES = [
    PlannedSource(_plan_crossing, name='crossing', when_blocked='defer', max_aircraft=2),
]



class Crossing90Scenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {
            'airspace': RegionBounds(DiskFootprint(LatLon(56, 2), radius_nm=60, n_vertices=72), None),
    'exit': RegionBounds(PointFootprint(LatLon(56, 2)), None)
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
            spawn=SpawnConfig(regions=[], sources=SPAWN_SOURCES),
            queryables={
            'exit': Waypoint(at=REGIONS['exit'], reach_radius_nm=2, color='cyan', render_shape=False, render_label=False, track_temporal_state=False)
        },
            rotation=rotation,
            groups=groups,
            sampled_waypoints={
            
        },
            waypoint_fields={
            
        },
            shapes=REGIONS,
        )

