# API reference

These pages document each package concept at its point of definition, with direct source links to the underlying implementation.

Task architecture
- [Environment](env.md): Subclasses {class}~bluesky_sandbox.env.BlueskyEnv to supply reward, termination, and task-info hooks (see Environment and configuration).
- Interface:  Declares agent observations and actions via fields on {class}~bluesky_sandbox.config.EnvConfig (see [Observation and action fields](fields.md) and [Task types and hooks](task.md).
- World: Defines airspace geometry, spawn regions, routes, and queryable entities (see [Scenario and spawning](scenario.md), [Geometry and bounds](geometry.md), and [Queryables](queryables.md).
- Plumbing: Manages wrappers, normalizers, parallel-environment utilities, and render drivers (see [Wrappers](wrappers.md), [Integrations](integrations.md) and
  [Rendering and drivers](rendering.md)).


```{toctree}
:maxdepth: 2

env
fields
task
scenario
geometry
queryables
performance
sampling
wrappers
integrations
rendering
```
