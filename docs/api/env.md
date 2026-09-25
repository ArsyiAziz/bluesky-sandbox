# Environment and configuration

## `bluesky_sandbox.env`

The class a task subclasses.

```{eval-rst}
.. automodule:: bluesky_sandbox.env
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.config`

Static configuration: which observation and action fields make up the interface, and the
simulator settings the episode runs under.

```{eval-rst}
.. automodule:: bluesky_sandbox.config
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.base_environment`

The PettingZoo `ParallelEnv` implementation underneath. Most tasks never touch this
directly, but the hook protocol and the aircraft lifecycle states are defined here.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.base_environment
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.layout`

Which columns of an observation or action belong to which field. The action space
is a `Box` while every action is continuous; with a switch it is a `Dict` of a
`continuous` Box and a `binary` MultiBinary:

```python
env.action_space(agent)     # Dict('binary': MultiBinary(2), 'continuous': Box(-1.0, 1.0, (4,)))
env.action_layout(agent)    # {"continuous": [Slot("active_route_waypoint_alt_delta_ft", 0:1), ...],
                            #  "binary":     [Slot("autopilot_lnav", 0:1), Slot("autopilot_vnav", 1:2)]}
env.step({agent: {"continuous": cont, "binary": bits}})
```

A layout depends only on the config, so `observation_layout(config)` and
`action_layout(config)` give it without an environment - in a vector env's main
process, say.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.layout
   :members:
.. automodule:: bluesky_sandbox.core.slot
   :members:
```

## `bluesky_sandbox.core.runtime`

The BlueSky process lifecycle — startup, stepping, teardown.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.runtime
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.services`

Services the environment exposes to fields and task hooks during a step.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.services
   :members:
   :show-inheritance:
```
