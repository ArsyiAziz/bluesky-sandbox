# Arrival through moving weather

```{figure} ../media/tutorials/storm-arrival.gif
:alt: Arrivals fly from SIE to CAMRN while two yellow storm cells drift and change shape across the route
:width: 620px
:align: center

Six arrivals fly SIE - BOTON - HOGGS - PANZE - KARRS - CAMRN. STORM_1 and
STORM_2 have a new shape and drift direction in each episode, and they move
while the aircraft fly.
```

Aircraft arrive from the south over the Sea Isle VORTAC (SIE) and fly the
CAMRN arrival towards New York. Two convective cells sit near the route. They
drift, grow, shrink and turn during the episode. The agent controls each
arrival. It can issue a temporary heading clearance to keep the aircraft out of
the weather. When the clearance ends, the aircraft returns to its own
navigation and continues along the route.

**You will learn to:**

- add real fixes from the navigation database and join them into a route
- spawn traffic in an arc and spread the spawn times over the episode
- draw a shape with a generator, so that the shape is different in each episode
- make a shape move during the episode
- give the policy a heading action as a *clearance*, with a duration and a
  return to own navigation
- write reward, termination and truncation hooks

The finished design is
[`designs/storm_arrival.json`](designs/storm_arrival.json).

## 1. The airspace

Open the designer, go to the **Map** tab and pan to southern New Jersey. Draw a
box from 38.8°N to 40.3°N and from 75.3°W to 73.4°W, with altitudes 0 to
30,000 ft. Name it `terminal_area` and select it as the airspace.

```json
"terminal_area": {
  "type": "region",
  "footprint": {"type": "box", "lat_min_deg": 38.8, "lat_max_deg": 40.3,
                "lon_min_deg": -75.3, "lon_max_deg": -73.4},
  "altitude": {"type": "constant", "min_ft": 0, "max_ft": 30000}
}
```

The airspace has two functions. It is the area that the renderers draw, and a
hook can read it to see whether an aircraft is still inside. This task uses it
for truncation in step 6.

## 2. Fixes and the route

Type `SIE` in the search box (**search fixes / airports…**) and click the add
button beside the result to add it as a waypoint. Do the same for `BOTON`,
`HOGGS`, `PANZE`, `KARRS` and `CAMRN`. Each fix becomes a point shape that
refers to the navigation database by name, plus a waypoint that refers to that
shape:

```json
"shapes":     {"camrn": {"type": "region", "footprint": {"type": "point", "fix": "CAMRN"}}},
"queryables": {"CAMRN": {"type": "waypoint", "shape": {"ref": "camrn"},
                           "reach_radius_nm": 5, "track_temporal_state": true}}
```

Set two values on each waypoint:

- **reach radius** 5 nm. The reach radius is the distance at which the waypoint
  counts as *satisfied*. The default of 1 nm is too tight for jets at 290 kt.
- **track temporal state** (under **Advanced**). Without it, a waypoint is only
  checked at the end of each environment step, at the aircraft's position at
  that moment. With it, the waypoint is checked on every simulator substep,
  and `step.reached` and `step.satisfied` record what happened at any moment of
  the step.

The second setting matters as soon as the policy vectors aircraft. A vectored
aircraft can **overstep** a fix: it passes the fix between two checks, or
outside the reach ring. With only the end-of-step check, the termination hook
does not see the passage, and the aircraft flies on until it leaves the
airspace. In 12 episodes with random clearances, the end-of-step check missed 3
of the 18 aircraft that reached CAMRN. All three came back to LNAV after a
vector and passed CAMRN outside the reach ring, so no end-of-step position was
inside it. Substep tracking caught all 18.

Next, in the **Route** tab, create a route named `CAMRN5` with the six fixes in
order:

```json
"routes": {"CAMRN5": ["SIE", "BOTON", "HOGGS", "PANZE", "KARRS", "CAMRN"]}
```

At spawn, the environment gives this route to BlueSky as a flight plan. The
aircraft then flies it in LNAV.

## 3. Traffic

Add a spawn region southwest of SIE. An *annular sector* (a ring segment)
spawns the aircraft at 8 to 12 nm from the VOR, in a 60° arc:

```json
"entry": {
  "type": "region",
  "footprint": {"type": "annular_sector",
                "center": {"lat_deg": 39.0955, "lon_deg": -74.80033333},
                "inner_radius_nm": 8, "outer_radius_nm": 12,
                "bearing_deg": 225, "half_angle_deg": 30},
  "altitude": {"type": "constant", "min_ft": 19000, "max_ft": 24000}
}
```

Then set up the spawn region on this shape:

```json
{
  "type": "spawn_region", "name": "south", "shape": {"ref": "entry"},
  "n_aircraft": 6,
  "spawn_time": {"type": "range", "low": 0, "high": 3000},
  "params": {"alt_ft":  {"type": "range", "low": 19000, "high": 24000},
             "spd_kts": {"type": "range", "low": 280, "high": 300},
             "hdg_deg": 40},
  "aircraft_type": {"type": "categorical", "weights": {"A320": 1, "A321": 1, "B738": 1, "B739": 1}},
  "route": "CAMRN5"
}
```

Each sampled value (`n_aircraft`, `spawn_time`, `alt_ft`, `spd_kts`) can be a
fixed number, a uniform `range`, or any `scipy.stats` distribution. The
designer's value editor offers all three. Here, `spawn_time` spreads the six
arrivals over the first 50 minutes, so the aircraft meet the weather at
different stages.

Also turn on **conflict-free spawn**. Then an aircraft does not spawn where it is
predicted to come into conflict with traffic that is already in the airspace.

Click the episode arrows on the map, or run `bluesky-sandbox design preview`,
to see the draws for different seeds:

```text
seed 0: 6 aircraft (at most 6)
  t=   68.0 s  A321   38.9849  -74.9822    19621 ft    293 kt  SIE > BOTON > HOGGS > PANZE > KARRS > CAMRN
  t=   98.3 s  A321   38.9738  -74.9968    19083 ft    296 kt  SIE > BOTON > HOGGS > PANZE > KARRS > CAMRN
  ...
```

## 4. Weather that moves

A storm cell is a shape that a **generator** draws. The `Blob` generator draws
a smooth irregular outline about a center: a radius that changes with bearing
as a random Fourier series. Each episode draws a new outline.

```json
"storm_1": {
  "type": "region",
  "footprint": {
    "type": "generated", "generator": "Blob",
    "params": {"center": {"lat_deg": 39.62, "lon_deg": -74.12},
               "radius_nm": {"type": "range", "low": 7, "high": 11},
               "amplitude": 0.3}
  },
  "motion": [
    {"type": "Drift", "heading_deg": {"type": "range", "low": 0, "high": 360},
                      "speed_kts":   {"type": "range", "low": 10, "high": 25},
                      "within": {"ref": "weather_box"}},
    {"type": "Grow",  "rate_per_hr": {"type": "range", "low": -0.3, "high": 0.5},
                      "min_scale": 0.6, "max_scale": 1.6}
  ]
}
```

`motion` is a list of motions that apply in order. In the shape editor, you add
them under **moves**:

| Motion | Parameters | Effect |
|---|---|---|
| `Drift` | `heading_deg`, `speed_kts`, `within` | Translates the shape. With `within`, it bounces inside that region and stays fully inside it. |
| `Grow` | `rate_per_hr`, `min_scale`, `max_scale` | Scales the shape about its center. A negative rate shrinks it. |
| `Spin` | `rate_deg_s` | Turns the shape about its center. A positive rate turns it clockwise. |

The episode draws each ranged parameter once, so a storm keeps one heading and
speed for the whole episode. `weather_box` is a plain box shape that holds the
two cells near the route. `STORM_2` uses `Drift` and `Spin`.

The map's play control shows the next hour of motion. Every element that refers
to the shape moves with it: its query region, the renderers, and
`context.shape("storm_1")` in a hook.

To let fields and hooks ask whether an aircraft is inside a storm, add a
**query region** on each storm shape:

```json
"STORM_1": {"type": "query_region", "shape": {"ref": "storm_1"}, "color": "yellow"},
"STORM_2": {"type": "query_region", "shape": {"ref": "storm_2"}, "color": "yellow"}
```

## 5. What the agent sees and does

In the **Spaces** tab, add these ownship observations:

```json
"obs_fields": [
  {"field": "AltFt"},
  {"field": "CasKts"},
  {"field": "ActiveRouteWaypointDistanceNm", "kwargs": {"high": 60}},
  {"field": "ActiveRouteWaypointTrackErrorDeg"},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "STORM_1"}},
  {"field": "QueryRegionInside", "kwargs": {"query_name": "STORM_2"}},
  {"field": "ClearanceTimeLeftS", "kwargs": {"target": "active_route_waypoint_hdg_delta_deg"}}
]
```

Then add two intruder observations, which give one row per nearby aircraft:
`DistToOwnNm` (with `high` set to 40) and `BrgFromOwnRelTrkDeg`.

For the action, add `ActiveRouteWaypointHdgDeltaDeg`. This action steers
relative to the bearing to the active route waypoint, so a value of 0 flies
straight at the waypoint. In its settings, make it a **clearance**:

```json
"action_fields": [{
  "field": "ActiveRouteWaypointHdgDeltaDeg", "kwargs": {"low": -60, "high": 60},
  "clearance": {"duration": [60, 600], "lock": "duration",
                "duration_grid": {"type": "grid", "step": 60}}
}]
```

A clearance changes the action space. The action becomes a dict:

```text
Dict('binary': MultiBinary(1), 'continuous': Box([-60. 60.], [60. 600.], (2,), float32))
```

- `continuous[0]` is the heading offset, and `continuous[1]` is how long the
  clearance lasts, in seconds. The duration is rounded to whole minutes
  because of `duration_grid`.
- `binary[0]` is the mask. **1 means "no new clearance"**: the aircraft
  continues on what it flies now.
- `lock: "duration"` refuses a new heading until the current clearance has run
  out. A vector is flown, not only started.
- When the duration ends, own navigation (LNAV) takes the heading back, and
  the aircraft continues along `CAMRN5`.

:::{important}
Do not use the plain `ActiveRouteWaypointHdgDeltaDeg` action for a route task.
A heading command takes the aircraft out of LNAV. BlueSky then never moves it
on to the next waypoint, and the aircraft circles the first fix forever. The
clearance form gives LNAV back after each vector.
:::

## 6. Reward and episode end

In the **Code** tab, open **env hooks** and write the three hooks.
`context.query(name)` evaluates a queryable for the current aircraft.

```python
# reward
camrn = context.query("CAMRN").step
reward = -0.01  # time: a detour or a turn-back costs
if context.query("STORM_1").current.inside or context.query("STORM_2").current.inside:
    reward -= 1.0
if context.separation.los:
    reward -= 1.0
if camrn.reached or camrn.satisfied:
    reward += 10.0
return reward
```

```python
# terminated
# Checked on every substep: passed by LNAV, or inside the reach ring
# at any moment of the step.
camrn = context.query("CAMRN").step
return camrn.reached or camrn.satisfied
```

`step.reached` is true when BlueSky's LNAV sequenced past the fix during the
step. `step.satisfied` is true when the aircraft was inside the reach ring at
any substep. Use both: an aircraft on a vector can enter the ring without
LNAV, and an aircraft in LNAV can pass the fix outside the ring.

The time penalty is there for the oversteps that remain. When a clearance ends
after the aircraft has passed a fix outside its ring, LNAV turns the aircraft
back to that fix. The penalty makes this detour cost the policy something.

```python
# truncated: left the terminal area without reaching CAMRN
return not context.airspace.current.inside
```

When an aircraft is terminated or truncated, it leaves the episode. The episode
ends when all aircraft have spawned and left.

:::{note}
Without substep tracking, do not use `route.reached` to detect the **last**
fix of a route. BlueSky does not advance past the final waypoint, so
`route.reached` for that fix is only true during the single substep of the
passage, and an end-of-step hook almost never sees it. `step.reached` keeps it
for the whole step.
:::

:::{note}
On the step in which an aircraft spawns, its hooks run before it has received
an action, so `context.action` is empty. If your reward reads an action, read
it with a default, for example
`context.action.get("active_route_waypoint_hdg_delta_deg", 0.0)`.
:::

Set **step (dt)** to 5 s in the **Config** tab.

## 7. Check it with a policy that does nothing

Build the package and fly one episode in which no aircraft receives a
clearance (`binary` set to 1):

```python
import numpy as np
from storm_arrival import Env

env = Env(render_mode=None)            # "pygame" to watch it
env.reset(seed=0)
hold = {"binary": np.ones(1, np.int8), "continuous": np.zeros(2, np.float32)}
returns = {}
while not env.episode_done:
    _, rewards, *_ = env.step({agent: hold for agent in env.agents})
    for agent, r in rewards.items():
        returns[agent] = returns.get(agent, 0.0) + r
print(returns)
```

With seed 0, all six aircraft reach CAMRN within about an hour of simulated
time (732 steps). Three of them fly through a storm on the way, for 12 to 18
steps each. An aircraft that stays clear returns about +8.7 (+10 at CAMRN,
less 0.01 per step), and each step inside a cell costs 1 more. A random policy issues long random vectors, and
many of its aircraft leave the airspace before they reach CAMRN. A trained
policy has to do better than both.

## Things to try

- Change the storm `Drift` to fly along the route (`heading_deg` near 40°), so
  that a cell follows the traffic.
- Add `"motion_update": "substep"` to a storm so that it moves on every
  simulator substep and not only once per environment step.
- Change the reward to a cost: move the storm term into the `cost` hook, so that
  a constrained-RL learner reads it from `info["cost"]`.
- Add wind in the **Config** tab, so that the same heading flies a different
  track.
