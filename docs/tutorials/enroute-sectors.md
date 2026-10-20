# En route sectors

```{figure} ../media/tutorials/enroute-sectors.gif
:alt: Four flows cross a rectangular center split into four Voronoi sectors; aircraft in conflict are drawn in orange and red
:width: 620px
:align: center

Eastbound, westbound, northbound and southbound flows cross a high-altitude
center. The yellow lines are the sector boundaries, which are drawn again in
each episode. Orange aircraft are in conflict, and red aircraft have lost
separation.
```

This task is a high-altitude center of about 120 by 140 nm, with four flows
that cross in the middle. In each episode the center is cut into four new
sectors, so a policy cannot learn one fixed sector map. Each aircraft flies
straight to its exit fix. The agent resolves conflicts with temporary vectors.

**You will learn to:**

- partition a region into sectors with a generator, and query each sector
- sample the traffic count and the flight levels
- set up conflict detection: protected zone and look-ahead time
- use the pair fields that describe a conflict geometry
- end an aircraft's episode at the exit on its own route

The finished design is
[`designs/enroute_sectors.json`](designs/enroute_sectors.json). The script that
writes it is short enough to read: see the end of this page.

## 1. The center and its sectors

The airspace is a box from 39.5°N to 41.5°N and from 82°W to 79°W, from
24,000 to 45,000 ft. Name it `center`.

Add a second shape with the `VoronoiSectors` generator, and give it `center` as
its parent:

```json
"sectors": {
  "type": "region",
  "footprint": {
    "type": "generated", "generator": "VoronoiSectors",
    "params": {"parent": {"ref": "center"}, "sectors": 4, "min_spacing_nm": 40}
  }
}
```

In each episode, the generator draws four seed points inside the parent, at
least `min_spacing_nm` apart, and cuts the parent into their Voronoi cells. The
generator is a *partition*: it makes one shape for each sector, named
`sectors.0` to `sectors.3`. The sectors are numbered clockwise from north, by
the bearing of their seed from the parent's center. Each of these shapes can be
used as any other shape. Add a query region on each one:

```json
"SECTOR_0": {"type": "query_region", "shape": {"ref": "sectors.0"}},
"SECTOR_1": {"type": "query_region", "shape": {"ref": "sectors.1"}},
"SECTOR_2": {"type": "query_region", "shape": {"ref": "sectors.2"}},
"SECTOR_3": {"type": "query_region", "shape": {"ref": "sectors.3"}}
```

Click through some episodes on the map to see different partitions. To keep
one fixed sector map, draw the sectors as polygons.

## 2. Exits and flows

Add four exit points, one inside each edge of the box, and a waypoint on each:
`EAST_EXIT`, `WEST_EXIT`, `NORTH_EXIT` and `SOUTH_EXIT`. Give each a reach radius
of 5 nm, and turn on **track temporal state**, as in the
[storm tutorial](storm-arrival.md#2-fixes-and-the-route), so that an aircraft
that passes its exit on a vector is still detected. Each route is one step long, so the aircraft fly direct:

```json
"routes": {"eastbound": ["EAST_EXIT"], "westbound": ["WEST_EXIT"],
           "northbound": ["NORTH_EXIT"], "southbound": ["SOUTH_EXIT"]}
```

:::{note}
Keep each exit's reach ring inside the airspace. If an exit point is within
5 nm of the edge, `design check` warns `outside the airspace: waypoint
'EAST_EXIT'`, because the check also tests the reach ring.
:::

Add a thin box along each edge as an entry shape, and a spawn region on it.
The west region:

```json
{
  "type": "spawn_region", "name": "west", "shape": {"ref": "west_entry"},
  "n_aircraft": {"type": "scipy", "name": "randint", "args": [4, 7], "kwds": {}},
  "spawn_time": {"type": "range", "low": 0, "high": 1500},
  "params": {"alt_ft":  {"type": "envelope", "alt_floor_ft": 33000, "alt_step_ft": 2000},
             "spd_kts": {"type": "range", "low": 270, "high": 300},
             "hdg_deg": 90},
  "aircraft_type": {"type": "categorical", "weights": {"A320": 1, "B738": 1, "A333": 1, "B77W": 1}},
  "route": "eastbound"
}
```

- `n_aircraft` uses a `scipy.stats` distribution: `randint(4, 7)` gives 4 to 6
  aircraft from each edge. `design check` reports the upper bound (`at most 24
  aircraft`).
- `alt_ft` of type `envelope` draws a level from the aircraft type's own flight
  envelope. With `alt_floor_ft` and `alt_step_ft`, the draws are multiples of
  2,000 ft at or above 33,000 ft, below the type's ceiling. Over four seeds,
  this gives FL340 and FL360, and FL380 for some A333s.
- The east, south and north regions are the same, with headings 270, 0 and
  180, and routes `westbound`, `northbound` and `southbound`.

There are few levels for each flow, and the flows cross in the middle, so
conflicts are frequent.

## 3. Conflict detection

In the **Config** tab, under **Conflict detection**:

| Setting | Design key | Value |
|---|---|---|
| method | `cd_method` | `CSTATEBASED` |
| resolution | `reso_method` | off (the agent resolves) |
| zone radius | `pz_radius_nm` | 5 nm |
| zone height | `pz_height_ft` | 1,000 ft |
| lookahead | `lookahead_s` | 300 s |

The environment applies these settings again at each reset, because BlueSky
does not keep them across `sim.reset()`. They define `context.separation.conflict`
(a loss of separation is predicted within the look-ahead time) and
`context.separation.los` (the zone is violated now). The conflict observation
fields also use them.

## 4. Observations and the action

The ownship part tells the agent where it is: its level, the distance and track
error to its exit, the time left on its clearance, and which sector it is in:

```json
"obs_fields": [
  {"field": "AltFt", "kwargs": {"low": 24000, "high": 45000}},
  {"field": "ActiveRouteWaypointDistanceNm", "kwargs": {"high": 200}},
  {"field": "ActiveRouteWaypointTrackErrorDeg"},
  {"field": "ClearanceTimeLeftS", "kwargs": {"target": "active_route_waypoint_hdg_delta_deg"}},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "SECTOR_0"}},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "SECTOR_1"}},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "SECTOR_2"}},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "SECTOR_3"}}
]
```

The intruder part describes the conflict geometry of each nearby aircraft:

```json
"intruder_obs_fields": [
  {"field": "DistToOwnNm", "kwargs": {"high": 60}},
  {"field": "BrgFromOwnRelTrkDeg"},
  {"field": "ConflictTcpaS"},
  {"field": "ConflictHorizontalDistAtCpaNm"},
  {"field": "ConflictVerticalSepAtCpaFt"}
]
```

`ConflictTcpaS`, `ConflictHorizontalDistAtCpaNm` and `ConflictVerticalSepAtCpaFt` leave `low` and `high`
blank. They then take their bounds from the conflict detection settings at run
time: `ConflictTcpaS` is limited to ± the look-ahead time, and `ConflictHorizontalDistAtCpaNm`
to the zone radius. When you change the zone in **Config**, these fields follow.

The action is the same heading clearance as in
[Arrival through moving weather](storm-arrival.md#5-what-the-agent-sees-and-does):
`ActiveRouteWaypointHdgDeltaDeg` (±45°) as a clearance of 60 to 600 s, locked
for its duration. After the clearance ends, the aircraft flies direct to its
exit again.

## 5. Reward and episode end

```python
# reward
reward = 0.0
if context.separation.los:
    reward -= 1.0
elif context.separation.conflict:
    reward -= 0.1
return reward
```

Every aircraft can see all four exits as queryables. The west exit is close to
the west entry box, so an eastbound aircraft can spawn inside the reach radius
of `WEST_EXIT`. If termination tested "any exit satisfied", that aircraft would
leave at once. Test only the exit on the aircraft's own route instead.
`route.index` is `None` for a waypoint that is not on the aircraft's route:

```python
# terminated
# Reached the exit on its own route (route.index is None for the others),
# checked on every substep.
for name in ("EAST_EXIT", "WEST_EXIT", "NORTH_EXIT", "SOUTH_EXIT"):
    exit_fix = context.query(name)
    if exit_fix.route.index is not None and (exit_fix.step.reached or exit_fix.step.satisfied):
        return True
return False
```

```python
# truncated
return not context.airspace.current.inside
```

## 6. Check it with a policy that does nothing

Fly seed 0 with no clearances (`binary` set to 1), as in the storm tutorial.
The episode draws 17 aircraft, and all of them reach their exits in 487 steps.
Six of them are in a conflict at some point, and the same six lose separation
for a total of 48 steps. The total return is -90.3. A policy that vectors early
can remove those losses and keeps the aircraft near their direct tracks.

## Things to try

- Make the sectors part of the task: add a state field or a task-info provider
  that counts aircraft per sector, and penalize a sector that holds more than
  a set number of aircraft. A policy then has to spread the traffic, and not
  only separate it.
- Give the policy a second axis: add `AltDeltaFt` as a second clearance, with
  `"kwargs": {"low": -2000, "high": 2000, "grid": {"type": "grid", "step": 1000}}`
  so that the values are whole flight levels.
- Rotate the whole problem in each episode: in **Config**, select
  **+ rotation group**, add every shape, and give the angle a range of 0 to 360°.
  The flows then cross at a different orientation in each episode.
- Turn on BlueSky's own resolution (**resolution** `MVP`, `"reso_method": "MVP"`)
  and fly the hold policy, as a baseline to compare with the learned policy.

## The script behind the design

The design file was written by a short Python script. When a design has many
similar elements, such as four flows and four sectors, a script is often easier
to read than the JSON:

```python
import json

lat0, lat1, lon0, lon1 = 39.5, 41.5, -82.0, -79.0
latc, lonc = (lat0 + lat1) / 2, (lon0 + lon1) / 2

def box(a, b, c, d):
    return {"type": "box", "lat_min_deg": a, "lat_max_deg": b, "lon_min_deg": c, "lon_max_deg": d}

def point(lat, lon):
    return {"type": "region", "footprint": {"type": "point", "center": {"lat_deg": lat, "lon_deg": lon}},
            "altitude": {"type": "constant", "min_ft": 24000, "max_ft": 45000}}

levels = {"type": "constant", "min_ft": 29000, "max_ft": 39000}
shapes = {
    "center": {"type": "region", "footprint": box(lat0, lat1, lon0, lon1),
               "altitude": {"type": "constant", "min_ft": 24000, "max_ft": 45000}},
    "sectors": {"type": "region", "footprint": {"type": "generated", "generator": "VoronoiSectors",
                "params": {"parent": {"ref": "center"}, "sectors": 4, "min_spacing_nm": 40}}},
    "west_entry":  {"type": "region", "footprint": box(40.0, 41.0, -81.95, -81.80), "altitude": levels},
    "east_entry":  {"type": "region", "footprint": box(40.0, 41.0, -79.20, -79.05), "altitude": levels},
    "south_entry": {"type": "region", "footprint": box(39.55, 39.65, -81.0, -80.0), "altitude": levels},
    "north_entry": {"type": "region", "footprint": box(41.35, 41.45, -81.0, -80.0), "altitude": levels},
    "west_exit": point(latc, lon0 + 0.2), "east_exit": point(latc, lon1 - 0.2),
    "south_exit": point(lat0 + 0.15, lonc), "north_exit": point(lat1 - 0.15, lonc),
}
# ... queryables, spawn regions and env as shown above ...
```

The complete design is in
[`designs/enroute_sectors.json`](designs/enroute_sectors.json). Run
`bluesky-sandbox design check` on any file that you write this way. It reports
every problem in the design, and it shows each problem at the place in the
design where it occurs.
