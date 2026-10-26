# Final approach spacing

```{figure} ../media/tutorials/final-approach.gif
:alt: Arrivals from the east and south merge at the intermediate fix and fly down the final approach to JFK runway 31R
:width: 620px
:align: center

Two arrival streams join at the intermediate fix (IF) and fly the final
approach course to runway 31R. Red aircraft have lost separation: with no
speed control, they bunch up at the merge.
```

Arrivals for JFK runway 31R come from two directions: from the east, and from
the south. Both streams join the final approach course at the same
intermediate fix. From there the aircraft fly down a constrained profile: 3,000 ft
and 210 kt at the IF, 1,900 ft and 170 kt at the final approach fix (FAF), and
the runway threshold. The agent controls the speed of each arrival, so that the
two streams merge with enough space between the aircraft.

**You will learn to:**

- place waypoints by latitude and longitude, where the navigation database
  has no fix
- put altitude and speed constraints on route steps
- turn on VNAV with a lifecycle hook, so that BlueSky flies the constraints
- use intruder fields that sequence aircraft at a shared fix
- set the protected zone for a terminal task

The finished design is
[`designs/final_approach.json`](designs/final_approach.json).

## 1. The approach geometry

Runway 31R's threshold is at 40.64517°N, 73.76247°W, and the runway points to
301° true. The IF and the FAF are on the extended centerline (bearing 121° from
the threshold), at 11 nm and 5.5 nm:

| Point | Latitude | Longitude | Distance from the threshold |
|---|---|---|---|
| `RW31R` | 40.64517 | -73.76247 | 0 nm |
| `FAF31R` | 40.5979 | -73.6590 | 5.5 nm |
| `IF31R` | 40.5506 | -73.5557 | 11 nm |

On the **Map**, add each point as a point shape with a lat/lon (in the shape
editor, a point can be a navigation fix or a position), and add a waypoint on
each one:

```json
"shapes": {
  "rw31r":  {"type": "region", "footprint": {"type": "point", "center": {"lat_deg": 40.64517, "lon_deg": -73.76247}}},
  "faf31r": {"type": "region", "footprint": {"type": "point", "center": {"lat_deg": 40.5979,  "lon_deg": -73.6590}}},
  "if31r":  {"type": "region", "footprint": {"type": "point", "center": {"lat_deg": 40.5506,  "lon_deg": -73.5557}}}
},
"queryables": {
  "IF31R":  {"type": "waypoint", "shape": {"ref": "if31r"},  "reach_radius_nm": 2},
  "FAF31R": {"type": "waypoint", "shape": {"ref": "faf31r"}, "reach_radius_nm": 2},
  "RW31R":  {"type": "waypoint", "shape": {"ref": "rw31r"},  "reach_radius_nm": 1.5}
}
```

The airspace is a 45 nm disk around the IF, from the surface to 14,000 ft.

:::{tip}
If you want to compute such points yourself, use BlueSky's geodesy:
`bluesky.tools.geo.qdrpos(lat, lon, bearing_deg, dist_nm)` returns the point at
that bearing and distance.
:::

## 2. A constrained route

A route step can be a plain waypoint name, or a dict with a crossing
restriction. In the **Route** tab, build the route `final`:

```json
"routes": {
  "final": [
    {"waypoint": "IF31R",  "alt_ft": 3000, "speed_kts": 210},
    {"waypoint": "FAF31R", "alt_ft": 1900, "speed_kts": 170},
    {"waypoint": "RW31R",  "alt_ft": 100}
  ]
}
```

These restrictions go into BlueSky's flight plan (`ADDWPT callsign wpt,alt,spd`).
A speed restriction needs an altitude restriction on the same step, because
BlueSky requires it.

## 3. Two feeder streams

Make two annular sectors around the IF, 28 to 34 nm out: one at bearing 80°
(east) and one at bearing 170° (south), each 24° wide. Put a spawn region on
each, with headings toward the IF and both on the route `final`:

```json
{
  "type": "spawn_region", "name": "east", "shape": {"ref": "east_feed"},
  "n_aircraft": 5,
  "spawn_time": {"type": "range", "low": 0, "high": 1500},
  "params": {"alt_ft":  {"type": "range", "low": 8000, "high": 10000},
             "spd_kts": {"type": "range", "low": 240, "high": 250},
             "hdg_deg": 260},
  "aircraft_type": {"type": "categorical", "weights": {"A320": 2, "B738": 2, "E190": 1}},
  "route": "final"
}
```

The `south` region is the same, with `"hdg_deg": 350`. The `categorical` weights
make A320s and B738s twice as likely as E190s.

## 4. Turn on VNAV

BlueSky turns on LNAV when it receives a flight plan, but it leaves VNAV off.
Without VNAV, an aircraft flies the speed restrictions but stays at its spawn
altitude, and it crosses the threshold at 8,000 to 10,000 ft.

Turn VNAV on as each aircraft spawns. In the **Code** tab, under **env hooks**,
override `on_aircraft_spawned`. Then import BlueSky in the hook setup:

```python
# hook setup
import bluesky as bs
```

```python
# on_aircraft_spawned(self, callsign, route)
# BlueSky leaves VNAV off after ADDWPT: turn it on so the route's
# altitude constraints are flown.
bs.stack.stack(f"VNAV {callsign} ON")
```

With VNAV on, a measured run crosses the IF at 3,000 ft and 210 kt, the FAF at
about 2,300 ft (still in its descent) and 170 kt, and is at about 500 ft when
it reaches the 1.5 nm ring around the threshold. This is close to a 3° glidepath.

Only hooks that you customize go into the generated `env.py`. All other hooks
keep the default from `BlueskyEnv`.

## 5. Observations and the speed action

The ownship part of the observation describes where the aircraft is on the
profile:

```json
"obs_fields": [
  {"field": "AltFt", "kwargs": {"low": 0, "high": 12000}},
  {"field": "CasKts"},
  {"field": "ActiveRouteWaypointDistanceNm", "kwargs": {"high": 40}},
  {"field": "ActiveRouteWaypointEteS", "kwargs": {"high": 900}}
]
```

`AltFt` takes both `low` and `high`, or neither (then the field reads its bounds
from the aircraft's performance at run time). With only one of them,
`design check` fails with `AltFt bounds must provide both low and high`.

The intruder part is the sequencing information:

```json
"intruder_obs_fields": [
  {"field": "DistToOwnNm", "kwargs": {"high": 40}},
  {"field": "IntruderFixApproachDistNm", "kwargs": {"high": 40}},
  {"field": "IntruderFixArrivalDeltaS", "kwargs": {"low": -300, "high": 300}}
]
```

- `IntruderFixApproachDistNm` is how near the intruder's present path passes
  *my* active fix. A small value means that the intruder flies to the same fix.
- `IntruderFixArrivalDeltaS` is the intruder's ETA at my fix minus my ETA.
  A negative value means that the intruder arrives first, so I go behind it.

Use the two together. The arrival delta alone also gives a value for an
aircraft that only crosses near the fix.

The action is `ActiveRouteWaypointSpdDeltaKts` with `low` -40 and `high` 20.
This action is relative to the speed restriction of the active waypoint. A
value of 0 flies the published speed, and -40 slows the aircraft down by 40 kt.
The aircraft stays in LNAV, so it does not leave the approach.

## 6. Reward, termination and the protected zone

```python
# reward
reward = 0.0
# Spacing: an intruder whose path passes within 3 nm of my next fix, and
# which is due there within 90 s of me, is too close in the sequence.
intruders = context.obs["intruders"]
same_fix = intruders["intruder_fix_approach_dist_nm"] < 3.0
too_close = abs(intruders["intruder_fix_arrival_delta_s"]) < 90.0
reward -= 0.1 * float((same_fix & too_close).sum())
if context.separation.los:
    reward -= 1.0
if context.query("RW31R").current.satisfied:
    reward += 10.0
return reward
```

`context.obs` contains raw values, in each field's own unit, and not the
normalized values that the policy sees. The intruder part contains one array
for each field, with one row per intruder.

```python
# terminated
return context.query("RW31R").current.satisfied
```

```python
# truncated
return not context.airspace.current.inside
```

En route separation is 5 nm. On final approach it is 3 nm. Set it in the
**Config** tab, under **Conflict detection**: zone radius 3 nm and zone height
1,000 ft. In the design these settings are `"pz_radius_nm": 3.0` and
`"pz_height_ft": 1000`. Detection and `context.separation` both use this zone.
Set **step (dt)** to 5 s.

## 7. Check it with a policy that does nothing

```python
import numpy as np
from final_approach import Env

env = Env(render_mode=None)
env.reset(seed=0)
total = 0.0
while not env.episode_done:
    _, rewards, *_ = env.step({agent: np.zeros(1, np.float32) for agent in env.agents})
    total += sum(rewards.values())
print(total)
```

When each action is 0, every aircraft flies the published speeds. With seed 0,
all ten aircraft reach the runway in 405 steps. Seven of them lose separation on
the way, where the two streams meet, and the total return is -247. The policy has to slow the aircraft early, while they are still 20 to 30
nm out. Near the fix, a speed change is too late to make a gap.

## Things to try

- Give the IF an arrival time: add `"arrival_slack_s": {"type": "range", "low":
  0, "high": 120}` to its route step. Then observe
  `ActiveRouteWaypointArrivalErrorS`. Turn on **meet arrival times (RTA)** in
  **Config → Automation** to let BlueSky fly the times itself, as a baseline.
- Use `own_eta_mode: "route"` on the intruder fields, so that my ETA comes from
  the planned route and not from a straight-line projection.
- Make one stream background traffic: set `"controlled": false` on the south
  spawn region. The policy must then fit the east stream around aircraft that it
  cannot command.
- Add a frame stack to `IntruderFixArrivalDeltaS` (the **frame stack** option in
  the field settings, or `"transform": "stacked", "transform_kwargs": {"depth": 3}`),
  so that the policy can see whether a gap opens or closes.

## Next: train a policy

The [SB3 training tutorial](train-sb3.md) generates this task with the SB3
template and trains a speed policy on it.
