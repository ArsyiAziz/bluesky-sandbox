# Queryables

Queryables are named scenario entities—such as waypoints, routes, or airspace regions—referenced by observation fields. Fields declare the specific types and counts of entities they require, while the scenario supplies them by name. This abstraction allows identical field definitions to function across diverse task setups.


## `bluesky_sandbox.sim.queryables.base`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.base
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.queryables.waypoints`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.waypoints
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.arrival`

A waypoint can carry `arrival_slack_s` - seconds, or a distribution of them - giving every aircraft whose route crosses it a target arrival time there, assigned at spawn. `obs.ActiveRouteWaypointTimeToGoS`, `obs.ActiveRouteWaypointArrivalErrorS` and `obs.ActiveRouteWaypointHasArrivalTime` read it back, for the active fix or, with `route_offset`, a later one.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.arrival
   :members:
```

## `bluesky_sandbox.sim.queryables.regions`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.regions
   :members:
   :show-inheritance:
```
