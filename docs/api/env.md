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
