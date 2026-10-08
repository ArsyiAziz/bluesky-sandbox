"""A design's MDP at a glance: its spaces, field by field, and each normalizer.

For every column of the observation and the action: the field it belongs to, its
raw value (unit, range), the normalizer between that and what the policy sees,
the range it lands in, and the mapping itself sampled as a curve - raw to
normalized for an observation, the policy's value to the command for an action.

Built from the config and the layouts, like the environment's own spaces.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from typing import Any

import numpy as np

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import action_layout, observation_parts, slots
from bluesky_sandbox.core.services import _action_parts, _field_normalizer
from bluesky_sandbox.interface.fields.base import ActionKind, action_kind
from bluesky_sandbox.interface.fields.observations import LaggedObs, LaggedPair

from .builder import build_design_config
from .catalog import normalizers as normalizer_catalog
from .spec import DesignSpec

__all__ = ["mdp_summary"]

#: Samples per curve, and how far past the range each end reaches - far enough
#: to show a clip, near enough to keep the range itself the plot.
_SAMPLES = 41
_OVERSHOOT = 0.1


def mdp_summary(spec: DesignSpec) -> dict[str, Any]:
    """The design's MDP, for the designer's summary view."""
    config = build_design_config(spec)
    return {
        "ok": True,
        # The order normalizers are colored in: the catalog's, so a normalizer
        # keeps its color whatever the design uses.
        "normalizers": [n["name"] for n in normalizer_catalog()],
        "observation": [
            {
                "part": part,
                "rows": "per intruder" if part.endswith("intruders") else "per agent",
                "width": sum(s.width for s in slots(fields)),
                "fields": [
                    _field(f, slot, "observation", fields, slots(fields))
                    for f, slot in zip(fields, slots(fields))
                ],
            }
            for part, fields in observation_parts(config).items()
        ],
        "action": _actions(config),
    }


def _actions(config: EnvConfig) -> dict[str, Any]:
    parts = _action_parts(config.action_fields)
    layout = action_layout(config)
    return {
        "space": "Box" if set(layout) <= {"continuous"} else "Dict",
        "parts": [
            {
                "part": kind.value,
                "width": sum(s.width for s in layout[kind.value]),
                "fields": [
                    _field(f, slot, "action")
                    for f, slot in zip(parts[kind], layout[kind.value])
                ],
            }
            for kind in parts
        ],
    }


def _field(
    field: Any,
    slot: Any,
    role: str,
    part: list[Any] = (),
    part_slots: list[Any] = (),
) -> dict[str, Any]:
    normalizer = _field_normalizer(field)
    raw = _raw(field)
    out = {
        "name": slot.name,
        "class": type(field).__name__,
        "doc": (inspect.getdoc(type(field)) or "").split("\n\n", 1)[0],
        "columns": [slot.columns.start, slot.columns.stop],
        # A binary action takes the set {0, 1}; everything else a range.
        "binary": role == "action" and action_kind(field) is ActionKind.BINARY,
        # A choice among a few values: the action is an index (MultiDiscrete).
        "discrete": role == "action" and action_kind(field) is ActionKind.DISCRETE,
        "lag": _lag(field, part, part_slots),
        "raw": raw,
        "normalizer": None,
        "output": _output(field, normalizer, raw),
        "curve": None,
        # An action, stage by stage: the policy's value, its normalizer, and its
        # grid (the value's, or the target's) - each shown, none applied out of
        # sight.
        "pipeline": _pipeline(field, normalizer, raw) if role == "action" else None,
        # A crossover speed action acting in Mach above its crossover: its
        # mapping there, on the Mach scale (the curve above is the knots').
        "crossover": _above_crossover(field) if role == "action" else None,
    }
    if role == "action" and action_kind(field) is ActionKind.BINARY:
        # A switch is a choice too: off or on.
        out["curve"] = {
            "x": [0.0, 1.0],
            "series": [[0.0, 1.0]],
            "x_label": "choice",
            "y_label": "switch",
            "discrete": True,
            "labels": ["off", "on"],
        }
    if normalizer is None and out["curve"] is None and raw["width"] == 1:
        # Raw: the value is passed as is - drawn too, so every field shows what
        # the policy gets from it.
        out["curve"] = _identity(raw, role)
    if normalizer is not None:
        out["normalizer"] = {
            "name": type(normalizer).__name__,
            "params": _params(normalizer),
        }
        try:
            out["curve"] = _curve(field, normalizer, raw, role)
        except Exception as error:  # a mapping it cannot sample is still listed
            out["curve_note"] = str(error)
        else:
            if out["curve"].get("discrete"):
                out["curve_note"] = (
                    f"{len(out['curve']['x'])} choices; a step an aircraft cannot "
                    "take now is clipped to the largest one it can"
                )
    return out


def _pipeline(field: Any, normalizer: Any, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """An action's mapping from the policy's value to the command, in order."""
    unit = raw["unit"]
    kind = action_kind(field)
    if kind is ActionKind.BINARY:
        policy = "a switch: 0 or 1"
    elif kind is ActionKind.DISCRETE:
        policy = "a choice: an index"
    else:
        policy = "a value in a range"
    stages = [{"stage": "policy", "text": policy}]
    grid = getattr(field, "grid", None)
    if normalizer is not None:
        text = type(normalizer).__name__
        if getattr(normalizer, "discrete", False):
            every = getattr(normalizer, "step", None) or 1.0
            if grid is not None and grid.on == "target":
                values = "grid values" if every == 1 else f"× {every:g} grid values"
                text += f": k {values} above or below the present one (0: the nearest)"
            elif grid is not None or getattr(normalizer, "step", None) is not None:
                step = grid.step * every if grid is not None else normalizer.step
                text += f": k whole steps of {_number(step):g} {unit}".rstrip()
        stages.append({"stage": "normalizer", "text": text})
    mode = getattr(getattr(field.meta, "mode", None), "value", None)
    stages.append({"stage": "value", "text": "a delta" if mode == "delta" else "the value"})
    if grid is not None:
        if grid.on == "target" and mode == "delta":
            what = "nominal + delta"
        else:
            what = "the delta" if mode == "delta" else "the value"
        stages.append(
            {
                "stage": "grid",
                "text": f"{what}, on its nearest grid value (every {_number(grid.step):g} {unit})",
                "step": _number(grid.step),
            }
        )
    stages.append({"stage": "command", "text": "the target BlueSky is given"})
    return stages


def _raw(field: Any) -> dict[str, Any]:
    """The field's raw value: its width, unit, and range - or that its range
    is each aircraft's own, resolved at runtime."""
    width = field.output_size() if callable(getattr(field, "output_size", None)) else 1
    unit = str(getattr(field.meta, "unit", "") or "")
    # Decided from what the fields declare, not by asking for bounds: with
    # traffic in this process, an aircraft's own range would pass for a fixed one.
    low = high = None
    if not _needs_aircraft(field):
        try:
            low, high = (_number(v) for v in field.bounds(0))
        except Exception:
            pass
    per_aircraft = low is None or high is None
    return {
        "width": width,
        "unit": "" if unit == "unitless" else unit,
        "low": low,
        "high": high,
        "per_aircraft": per_aircraft,
    }


def _needs_aircraft(field: Any) -> bool:
    """Whether the field's range is read from each aircraft: not when it fixes
    both ends; for a field built on others (a lag, a difference), when any of
    them does; otherwise when its meta says so."""
    if (
        getattr(field, "low", None) is not None
        and getattr(field, "high", None) is not None
    ):
        return False
    wrapped = _wrapped(field)
    if wrapped:
        return any(_needs_aircraft(f) for f in wrapped)
    return bool(getattr(getattr(field, "meta", None), "dynamic_bounds", False))


def _wrapped(field: Any) -> list[Any]:
    """The fields ``field`` is built on."""
    if not dataclasses.is_dataclass(field):
        return []
    values = (getattr(field, f.name, None) for f in dataclasses.fields(field))
    return [v for v in values if hasattr(v, "bounds") and hasattr(v, "meta")]


def _lag(field: Any, part: list[Any], part_slots: list[Any]) -> dict[str, Any] | None:
    """For a lag: how many steps back, and the name of the field it lags when
    that field is in the same part."""
    if not isinstance(field, (LaggedObs, LaggedPair)):
        return None
    of = next((s.name for f, s in zip(part, part_slots) if f == field.inner), None)
    return {"steps": int(field.steps), "of": of, "inner": field.inner.meta.name}


def _output(field: Any, normalizer: Any, raw: dict[str, Any]) -> dict[str, Any]:
    """The range each output column lands in."""
    if normalizer is None:
        width = raw["width"]
        return {"low": [raw["low"]] * width, "high": [raw["high"]] * width}
    low, high = normalizer.output_bounds(field)
    return {"low": [_number(v) for v in low], "high": [_number(v) for v in high]}


def _curve(
    field: Any, normalizer: Any, raw: dict[str, Any], role: str
) -> dict[str, Any]:
    """The mapping sampled: raw -> normalized for an observation, the policy's
    value -> the field's value for an action. A field whose range is each
    aircraft's own is sampled over its position in that range."""
    if role == "action" and getattr(normalizer, "discrete", False):
        return _choices(field, normalizer)
    unit_axis = raw["per_aircraft"]
    low, high = (0.0, 1.0) if unit_axis else (raw["low"], raw["high"])
    field = _Ranged(field, low, high, unit_axis=unit_axis)
    raw_label = "position in range (low → high)" if unit_axis else "raw"
    if role == "observation" or normalizer.output_size(field) > 1:
        # Raw -> what the policy sees. For an action that takes several values
        # (an angle as sin, cos), this is what the policy gives for a command.
        xs = _span(low, high)
        ys = [normalizer.normalize(field, x, 0) for x in xs]
        series = [list(column) for column in zip(*ys)]
        x_label = (
            raw_label
            if role == "observation"
            else ("command" if not unit_axis else raw_label)
        )
        y_label = "normalized" if role == "observation" else "policy's values"
    else:
        # The policy's value -> the command it gives.
        out_low, out_high = normalizer.output_bounds(field)
        xs = _span(float(out_low[0]), float(out_high[0]))
        series = [[float(_on_grid(field, normalizer.denormalize(field, [x], 0))) for x in xs]]
        x_label = "policy's value"
        y_label = raw_label if unit_axis else "command"
    return {
        "x": [round(float(x), 6) for x in xs],
        "series": [[round(float(y), 6) for y in column] for column in series],
        "x_label": x_label,
        "y_label": y_label,
    }


def _above_crossover(field: Any) -> dict[str, Any] | None:
    """For a crossover speed action with a Mach regime: its mapping above the
    crossover, on that regime's own scale - the policy's value to a change in
    Mach - as the main curve is the change in knots below it."""
    regime = getattr(field, "above_crossover", None)
    if regime is None:
        return None
    from bluesky.tools.aero import ft  # noqa: PLC0415

    from bluesky_sandbox.interface.fields.actions.crossover import _MachView  # noqa: PLC0415

    level = f"FL{round(regime.crossover.altitude_m / ft / 100):03d}"
    view = _MachView(field)
    normalizer = view.normalizer
    raw = {
        "width": 1,
        "unit": "Mach",
        "low": None if view.low is None else float(view.low),
        "high": None if view.high is None else float(view.high),
        "per_aircraft": view.low is None,
    }
    try:
        if normalizer is None:
            curve = _identity(raw, "action")
        else:
            curve = _curve(view, normalizer, raw, "action")
    except Exception as error:  # a mapping it cannot sample is still named
        return {"title": f"above the crossover ({level})", "note": str(error)}
    curve["title"] = f"above the crossover ({level}): a change in Mach"
    curve["unit"] = "Mach"
    if not curve.get("discrete"):
        curve["y_label"] = "Mach change" if not raw["per_aircraft"] else "position in range"
    else:
        curve["y_label"] = "Mach change"
    return curve


def _on_grid(field: Any, value: float) -> float:
    """``value`` on the action's grid, as the environment puts it - drawn from
    a nominal of 0. A range that is each aircraft's own is drawn as a position
    in it, which no grid in the action's unit fits: left as is."""
    grid = getattr(field, "grid", None)
    if grid is None or getattr(field, "_unit_axis", False):
        return value
    return grid.apply(field, value, 0)


def _identity(raw: dict[str, Any], role: str) -> dict[str, Any]:
    """A raw field's mapping: the value as is - over its range, or over the
    position in each aircraft's own."""
    unit_axis = raw["per_aircraft"]
    low, high = (0.0, 1.0) if unit_axis else (raw["low"], raw["high"])
    xs = list(np.linspace(low, high, _SAMPLES))
    if role == "observation":
        labels = (
            "position in range (low → high)" if unit_axis else "raw",
            "policy sees (as is)",
        )
    else:
        labels = (
            "policy's value",
            "position in range (low → high)" if unit_axis else "command (as is)",
        )
    return {
        "x": [round(float(x), 6) for x in xs],
        "series": [[round(float(x), 6) for x in xs]],
        "x_label": labels[0],
        "y_label": labels[1],
    }


def _choices(field: Any, normalizer: Any) -> dict[str, Any]:
    """A discrete action (a choice, e.g. whole steps): each choice's number of
    steps, and the command it gives - before any aircraft's reach clips it - as
    points, not a curve."""
    unbounded = _Ranged(field, -1e12, 1e12)
    steps = normalizer.steps()
    commands = [
        float(normalizer.denormalize(unbounded, [i], 0)) for i in range(len(steps))
    ]
    return {
        "x": [float(k) for k in steps],
        "series": [[round(c, 6) for c in commands]],
        "x_label": "steps (choice)",
        "y_label": "command",
        "discrete": True,
    }


class _Ranged:
    """``field`` with a given range: what a normalizer scales against - its
    bounds and its reach - and nothing else changed."""

    def __init__(
        self, field: Any, low: float, high: float, *, unit_axis: bool = False
    ) -> None:
        self._field = field
        self._range = (low, high)
        self._unit_axis = unit_axis

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._range

    def reach(self, idx: int) -> tuple[float, float]:
        return self._range

    def nominal(self, idx: int) -> float:
        # Drawn from a nominal of 0, on the grid: no aircraft is read.
        return 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._field, name)


def _span(low: float, high: float) -> list[float]:
    reach = (high - low) * _OVERSHOOT
    return list(np.linspace(low - reach, high + reach, _SAMPLES))


def _params(normalizer: Any) -> dict[str, Any]:
    """The normalizer's constructor arguments, as it holds them."""
    out = {}
    for name in inspect.signature(type(normalizer)).parameters:
        value = getattr(normalizer, name, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[name] = value
        elif isinstance(value, tuple):
            out[name] = list(value)
    return out


def _number(value: Any) -> float | None:
    """A JSON number; None for an unbounded end."""
    value = float(value)
    return None if math.isinf(value) or math.isnan(value) else value
