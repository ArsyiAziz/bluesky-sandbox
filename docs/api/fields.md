# Observation and action fields

Fields define the interface between the BlueSky and BlueSky-Sandbox: an observation field contributes channels to what an agent sees, an action field contributes  to what it can command. 

## `bluesky_sandbox.interface.fields.base`

Base classes and metadata: field kinds, units, quantities, control axes, and the queryable-requirement declarations a field uses to state what the scenario must provide.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.base
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.interface.fields.observations`

The built-in observation fields, one module per kind of quantity. Every field is importable from the package itself, whichever module defines it. Imported as `obs` from the top level:

```python
from bluesky_sandbox import obs
```

### `observations.kinematics`

Position, heading and track, altitude, and speeds - of whichever aircraft is observed.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.kinematics
   :members:
   :show-inheritance:
```

### `observations.autopilot`

The autopilot's selections and the aircraft's error from each.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.autopilot
   :members:
   :show-inheritance:
```

### `observations.performance`

Capability descriptors and the flight phase.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.performance
   :members:
   :show-inheritance:
```

### `observations.route`

The aircraft's own route fixes, read from BlueSky's route.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.route
   :members:
   :show-inheritance:
```

### `observations.relative`

Another aircraft relative to the ownship: position, motion, closest point of approach.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.relative
   :members:
   :show-inheritance:
```

### `observations.conflict`

BlueSky's conflict detection and the shared conflict geometry, per intruder.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.conflict
   :members:
   :show-inheritance:
```

### `observations.comm`

What each intruder broadcast last step.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.comm
   :members:
   :show-inheritance:
```

### `observations.episode`

Time in the environment and the previous action.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.episode
   :members:
   :show-inheritance:
```

### `observations.transforms`

Fields built from other fields: differences and lagged values.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.transforms
   :members:
   :show-inheritance:
```

### `observations.queryable`

Fields read from a queryable the scenario configures - distance to a waypoint, whether an aircraft is inside a region, and so on. Also importable as `qobs`, the name generated configs use:

```python
from bluesky_sandbox import qobs
```

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations.queryable
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.interface.fields.actions`

The built-in action fields, one module per kind of command. Every action is importable from the package itself, whichever module defines it. Imported as `actions` from the top level:

```python
from bluesky_sandbox import actions
```

### `actions.kinematics`

A heading, speed or altitude target on the aircraft's own state, absolute or a delta from the current value.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions.kinematics
   :members:
   :show-inheritance:
```

### `actions.route`

A deviation from the active route waypoint's guidance, so a zero action flies the nominal.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions.route
   :members:
   :show-inheritance:
```

### `actions.autopilot`

The autopilot's selected heading, speed and altitude, relative to the current value, and its LNAV and VNAV modes.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions.autopilot
   :members:
   :show-inheritance:
```

### `actions.mask`

A 0/1 action that, set to 1, skips another action that step: the aircraft keeps the last command it was given. The policy then decides *when* to act as well as how, e.g. giving a heading vector once and letting the aircraft fly it out. The target is named by its class, an instance or its name:

```python
action_fields = [
    actions.HdgDeg(),
    actions.AutopilotLnav(),
    actions.ActionMask(target=actions.HdgDeg),
    actions.ActionMask(target="autopilot_lnav"),  # switches too
]
obs_fields = [obs.PrevActionMasked(target=actions.HdgDeg)]
```

A masked switch keeps its state, even when a switch turned on requires it, and suppresses nothing. With a mask configured, every agent's `info["action_applied"]` says which values of its action took effect, shaped as the action: 1 for each value applied, 0 for each one skipped, whether masked or on an axis a switch suppressed. A custom loss can use it to leave skipped values out of the policy gradient. `PrevActionMasked` shows the policy what it masked last.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions.mask
   :members: ActionMask
   :show-inheritance:
```

### `actions.comm`

A learned message, with no effect on the aircraft.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions.comm
   :members:
   :show-inheritance:
```
