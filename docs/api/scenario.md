# Scenario and spawning

Scenarios define three core elements per episode: airspace geometry, aircraft spawn parameters and performance profiles, and named queryable entities referenced by observation fields.


## `bluesky_sandbox.sim.scenario.base`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.scenario.base
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.scenario.randomized`

Implements domain randomization by sampling scenario parameters dynamically per episode instead of relying on fixed configurations.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.scenario.randomized
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.scenario.transforms`

Applies spatial transformations—rotation, translation, scaling, and composition—to entire scenarios while keeping geometry, spawn configurations, and queryables in sync.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.scenario.transforms
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.spawn.regions`

Manages spatial spawn volumes, arrival distributions, and episode aircraft spawn queues.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.spawn.regions
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.spawn.routes`

Defines the route-specification for flight paths assigned to newly spawned aircraft.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.spawn.routes
   :members:
   :show-inheritance:
```
