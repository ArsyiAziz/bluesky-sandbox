"""Backwards-compatible re-export of the geometric transforms.

The transforms now live in :mod:`bluesky_sandbox.sim.scenario.transforms` (core API) so that
generated task packages depend only on the main API, not the designer. This
shim keeps existing ``bluesky_sandbox.ui.designer.transforms`` imports working.
"""

from __future__ import annotations

from bluesky_sandbox.sim.scenario.transforms import (  # noqa: F401
    bbox_center,
    rotate_bounds,
    rotate_queryable,
    rotate_spawn,
    sample_scalar,
)
