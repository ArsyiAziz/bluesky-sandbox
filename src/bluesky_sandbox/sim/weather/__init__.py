"""Weather: environmental fields that act on every aircraft.

Currently wind. These are *models*, not simulator plumbing - the runtime is
what pushes a field into BlueSky, and the environment is what advances it.
"""

from __future__ import annotations

from .wind import UniformWind, WindField, wind_from_config

__all__ = ["UniformWind", "WindField", "wind_from_config"]
