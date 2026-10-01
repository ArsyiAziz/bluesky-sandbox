"""A design's grid: one step per quantity, applied everywhere it belongs.

``DesignSpec.grid`` - ``{"alt_ft": 1000, "spd_kts": 10, "hdg_deg": 10}`` - is the
one place a design says "aircraft fly 1,000 ft levels, are given speeds in
10 kt steps and turns in 10 deg". :func:`apply_grid` expands it into the
library's per-field pieces before anything is built or generated, so the env
and the generated task get exactly the same thing:

* Altitude - the levels:
  - a waypoint altitude drawn from the envelope, or taken from where its leg
    starts (``{"type": "start"}``), is put on them;
  - a spawn altitude drawn from the envelope is, where it is marked
    ``"levels": true`` (spawns are otherwise as drawn) - a mark that needs the
    grid's altitude, and fails without it;
  - EVERY altitude target action commands them (``command_step``), step action
    or not: every aircraft flies the same levels.
* Every step action (a ``StepNormalizer`` with no ``step`` of its own) takes its
  quantity's step: levels for altitude, the speed step, the heading step.

Actions are read as the builder reads them - built-in or from the design's own
code - and one the grid cannot read is an error, never skipped. What a field or
marker states itself is kept: the grid fills in, it does not override.
"""

from __future__ import annotations

import copy
from typing import Any

from bluesky_sandbox.interface.fields import actions as _actions
from bluesky_sandbox.interface.fields.base import ControlAxis

from . import spec as _spec

__all__ = ["apply_grid"]

_AXIS_KEY = {
    ControlAxis.ALTITUDE: "alt_ft",
    ControlAxis.SPEED: "spd_kts",
    ControlAxis.HEADING: "hdg_deg",
}


def apply_grid(spec: _spec.DesignSpec) -> _spec.DesignSpec:
    """``spec`` with its grid expanded into its fields and markers (a copy; the
    same spec when it has no grid)."""
    grid = spec.grid or {}
    marked = [
        region
        for region in spec.spawn.get("regions", []) or []
        if _levels_marked((region.get("params") or {}).get("alt_ft"))
    ]
    if marked and "alt_ft" not in grid:
        names = [region.get("name") for region in marked]
        raise _spec.SpecError(
            f"spawn regions {names} spawn on the grid's levels, but the design has "
            "no altitude grid: set grid.alt_ft (Config > Grid)"
        )
    if not grid:
        return spec
    spec = copy.deepcopy(spec)
    for ref in spec.env.action_fields:
        _grid_action(_action_class(spec, ref), ref, grid)
    level = grid.get("alt_ft")
    for q in spec.queryables.values():
        if not (isinstance(q, dict) and q.get("type") == "waypoint"):
            continue
        alt = q.get("alt_ft")
        if level is not None and (
            _spec.is_envelope_value(alt) or _spec.is_start_value(alt)
        ):
            alt.setdefault("alt_step_ft", level)
    for region in spec.spawn.get("regions", []) or []:
        alt = (region.get("params") or {}).get("alt_ft")
        if _levels_marked(alt):
            alt.pop("levels")
            alt.setdefault("alt_step_ft", level)
    return spec


def _levels_marked(alt: Any) -> bool:
    """An envelope altitude marked to be drawn on the grid's levels."""
    return (
        isinstance(alt, dict)
        and _spec.is_envelope_value(alt)
        and bool(alt.get("levels"))
    )


def _action_class(spec: _spec.DesignSpec, ref: Any) -> type:
    """The class an action ref names - a built-in, or a custom one from the
    design's own code (``"module:Class"``) - as the builder resolves it."""
    if ":" in ref.name:
        # Imported here: the builder imports this module.
        from .builder import install_code_modules, resolve_callable  # noqa: PLC0415

        install_code_modules(spec.code)
        try:
            return resolve_callable(ref.name)
        except Exception as e:
            raise _spec.SpecError(
                f"the grid cannot read action {ref.name!r}: {e}"
            ) from e
    cls = getattr(_actions, ref.name, None)
    if cls is None:
        raise _spec.SpecError(
            f"the grid cannot read action {ref.name!r}: no such action"
        )
    return cls


def _grid_action(cls: type, ref: Any, grid: dict[str, float]) -> None:
    """One action field: its step action's step, and an altitude target's
    levels, from the grid."""
    meta = getattr(cls, "meta", None)
    key = _AXIS_KEY.get(getattr(meta, "control_axis", None))
    if key is None or key not in grid:
        return
    normalizer = ref.kwargs.get("normalizer")
    if (
        isinstance(normalizer, dict)
        and normalizer.get("name") == "StepNormalizer"
        and "step" not in (normalizer.get("kwargs") or {})
    ):
        normalizer.setdefault("kwargs", {})["step"] = grid[key]
    fields = getattr(cls, "__dataclass_fields__", {})
    if (
        key == "alt_ft"
        and "command_step" in fields
        and "command_step" not in ref.kwargs
    ):
        ref.kwargs["command_step"] = grid[key]
