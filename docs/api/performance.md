# Performance models

Performance models dictate aircraft availability and valid kinematic envelopes, directly constraining initial spawn altitudes and reachable speed boundaries. OpenAP is bundled by default, whereas BADA is proprietary and requires separate installation. Refer to 
[Verification and diagnostics](../diagnostics.md) to observe the possible datasets

## `bluesky_sandbox.sim.performance.models`

Queries available aircraft types, spawnable configurations, and kinematic performance limits.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.performance.models
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.performance.envelope`

Calculates feasible altitude-speed envelopes and provides sampling methods for scenario generation.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.performance.envelope
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.performance.speeds`

Handles conversions between Calibrated Airspeed (CAS) and Mach number, including transition altitude crossovers.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.performance.speeds
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.performance.bada`

Manages local BADA dataset discovery, path resolution, and coefficient loading.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.performance.bada
   :members:
   :show-inheritance:
```
