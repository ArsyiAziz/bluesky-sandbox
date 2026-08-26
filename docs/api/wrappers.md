# Wrappers

Wrappers adapt environment behavior without modifying task code—handling observation normalization, step limits, intruder filtering, feature padding, and agent ID persistence across dynamic episodes.


## Normalizers

Applies per-field transformations, enabling continuous linear scaling for metrics like altitude alongside circular wrapping for angular headings within the same observation.


```{eval-rst}
.. automodule:: bluesky_sandbox.interface.wrappers.observations.normalizer
   :members:
   :show-inheritance:
```

## Intruder selection and padding

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.wrappers.observations.keep
   :members:
   :show-inheritance:

.. automodule:: bluesky_sandbox.interface.wrappers.observations.pad
   :members:
   :show-inheritance:
```

## Episode lifecycle

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.wrappers.lifecycle.time_limit
   :members:
   :show-inheritance:
```

## Compatibility

Many learners assume a fixed, stable agent set. Aircraft spawn and land, so agent IDs churn;
this wrapper hides that.

```{eval-rst}
.. automodule:: bluesky_sandbox.interface.wrappers.compat.stable_ids
   :members:
   :show-inheritance:
```
