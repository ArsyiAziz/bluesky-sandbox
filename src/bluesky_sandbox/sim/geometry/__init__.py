"""Geometry shared across the step: predicted conflicts between aircraft.

Airspace *volumes* are :mod:`bluesky_sandbox.sim.bounds`; this is the pairwise
closest-approach machinery the task layer scores against.
"""

from __future__ import annotations

from .conflict import (
    ConflictGeometry,
    ConflictView,
    conflict_geometry,
    predicted_tlos_s,
    safety_margin,
    safety_margin_pairs,
    windowed_min_hsep_nm,
    windowed_min_vsep_ft,
    windowed_signed_vsep_at_entry_ft,
)

__all__ = [
    "ConflictGeometry",
    "ConflictView",
    "conflict_geometry",
    "predicted_tlos_s",
    "safety_margin",
    "safety_margin_pairs",
    "windowed_min_hsep_nm",
    "windowed_min_vsep_ft",
    "windowed_signed_vsep_at_entry_ft",
]
