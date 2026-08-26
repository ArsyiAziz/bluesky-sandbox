# Geometry and bounds

Airspace volumes pair a horizontal footprint with a vertical altitude range. Footprints compose modularly using primitive operations—such as unions, intersections, sectors, and corridors—allowing arbitrary airspace structures to be constructed

## `bluesky_sandbox.sim.bounds.base`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.bounds.base
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.bounds.coordinates`

Latitude/longitude and the local tangent frame that nautical-mile offsets are measured in.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.bounds.coordinates
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.bounds.footprints`

Primitive horizontal shapes.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.bounds.footprints
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.bounds.derived`

Footprints built from other footprints.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.bounds.derived
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.bounds.altitude`

Vertical extent: constant, linear, radial, or per-vertex.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.bounds.altitude
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.geometry.conflict`

Pairwise conflict geometry — separation, time-to-loss, and the windowed minima that
separation-based rewards and terminations are written against.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.geometry.conflict
   :members:
   :show-inheritance:
```
