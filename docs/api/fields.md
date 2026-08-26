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

The built-in observation fields. Imported as `obs` from the top level:

```python
from bluesky_sandbox import obs
```

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.observations
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.interface.fields.actions`

The built-in action fields. Imported as `actions` from the top level:

```python
from bluesky_sandbox import actions
```

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.actions
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.interface.fields.queryables`

Observation fields defined against scenario queryables — distance to a waypoint, whether an
aircraft is inside a region, and so on. Imported lazily as `qobs`:

```python
from bluesky_sandbox import qobs
```

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.fields.queryables
   :members:
   :show-inheritance:
```
