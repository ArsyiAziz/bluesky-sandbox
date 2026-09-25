"""Raw field values for the current step: computed once, batched, then shared.

Assembling observations computes every observation field for every aircraft at
once, and each pair field for every ownship against every aircraft. Those arrays
are kept here for the step, so the raw values a hook reads come from the same
computation as the observation. Read before the observation, a field is
computed the same batched way, and the observation reuses it.

The values belong to the traffic they were computed from: they are dropped when
the sim time or the set of aircraft changes, and on reset. The actions applied
this step are kept until the next step starts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import bluesky as bs
import numpy as np

from bluesky_sandbox.interface.fields.base import PairObsField

from .slot import Slot, unique_names

__all__ = ["RawObservation", "StepValues", "raw_action_values"]

#: The key of each intruder's callsign in a raw intruder part.
ACID = "acid"


class StepValues:
    """This step's raw field values, one batched computation per field."""

    def __init__(self) -> None:
        self._stamp: tuple[float, tuple[str, ...]] | None = None
        self._values: dict[int, np.ndarray] = {}
        self._pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._actions: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        """Forget everything: a new episode."""
        self._stamp = None
        self._values.clear()
        self._pairs.clear()
        self._actions.clear()

    def begin_step(self) -> None:
        """Forget the previous step's actions."""
        self._actions.clear()

    def _current(self) -> None:
        """Drop the values if the traffic has moved on since they were computed."""
        stamp = (float(bs.sim.simt), tuple(bs.traf.id))
        if stamp != self._stamp:
            self._stamp = stamp
            self._values.clear()
            self._pairs.clear()

    def values(self, field: Any) -> np.ndarray:
        """``field`` for every live aircraft, in traffic order."""
        self._current()
        values = self._values.get(id(field))
        if values is None:
            values = np.asarray(field.get_many(tuple(range(bs.traf.ntraf))))
            self._values[id(field)] = values
        return values

    def pair_matrix(self, field: Any, owns: np.ndarray) -> np.ndarray:
        """Pair ``field`` for each ownship in ``owns`` against every aircraft."""
        self._current()
        cached = self._pairs.get(id(field))
        if cached is not None and np.array_equal(cached[0], owns):
            return cached[1]
        matrix = np.asarray(field.get_pair_matrix(owns))
        self._pairs[id(field)] = (np.array(owns, dtype=np.intp), matrix)
        return matrix

    def pair_row(
        self, field: Any, own: int, owns: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """Ownship ``own``'s row of pair ``field`` against every aircraft.

        Read from the matrix already computed when it has that ownship; else
        computed for ``owns()`` - the ownships the observation will need - so
        the observation reuses it.
        """
        self._current()
        cached = self._pairs.get(id(field))
        if cached is not None:
            rows = np.flatnonzero(cached[0] == own)
            if rows.size:
                return cached[1][rows[0]]
        candidates = np.asarray(owns(), dtype=np.intp)
        if own not in candidates:
            candidates = np.array([own], dtype=np.intp)
        matrix = self.pair_matrix(field, candidates)
        return matrix[int(np.flatnonzero(candidates == own)[0])]

    def record_action(self, acid: str, values: dict[str, Any]) -> None:
        """The raw values of the action ``acid`` was given this step."""
        self._actions[acid] = values

    def action(self, acid: str) -> dict[str, Any]:
        """The raw action ``acid`` was given this step; empty if none."""
        return self._actions.get(acid, {})


class RawObservation(Mapping[str, Mapping[str, Any]]):
    """One aircraft's observation in raw values, by part and field name.

    ``raw["ownship"]["alt_ft"]`` is a value in the field's unit - a circular
    field its angle, a multi-column field its array. ``raw["intruders"][name]``
    is an array with one row per intruder, in the observation's intruder order;
    ``raw["intruders"]["acid"]`` names the aircraft of each row. A part is
    built on first read, from values already computed this step.
    """

    def __init__(
        self,
        values: StepValues,
        parts: Mapping[str, Any],
        acidx: int,
        owns: Callable[[], np.ndarray],
    ) -> None:
        self._values = values
        # Keyed as the observation is: ownship always, the rest when configured.
        self._parts = {
            key: list(fields or ())
            for key, fields in parts.items()
            if fields or key == "ownship"
        }
        self._acidx = acidx
        self._owns = owns
        self._built: dict[str, dict[str, Any]] = {}

    def __getitem__(self, part: str) -> dict[str, Any]:
        built = self._built.get(part)
        if built is None:
            if part not in self._parts:
                raise KeyError(
                    f"no observation part {part!r}; this environment has "
                    f"{sorted(self._parts)}"
                )
            built = self._build(part, self._parts[part])
            self._built[part] = built
        return built

    def __iter__(self) -> Iterator[str]:
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __repr__(self) -> str:
        return f"RawObservation(parts={list(self._parts)})"

    def _build(self, part: str, fields: Any) -> dict[str, Any]:
        names = unique_names_of(fields)
        if not part.endswith("intruders"):
            return {
                name: _scalar(self._values.values(field)[self._acidx])
                for name, field in zip(names, fields)
            }
        others = np.array(
            [i for i in range(bs.traf.ntraf) if i != self._acidx], dtype=np.intp
        )
        out: dict[str, Any] = {ACID: np.array(bs.traf.id, dtype=object)[others]}
        for name, field in zip(names, fields):
            if isinstance(field, PairObsField):
                row = self._values.pair_row(field, self._acidx, self._owns)
                out[name] = row[others]
            else:
                out[name] = self._values.values(field)[others]
        return out


def unique_names_of(fields: Any) -> list[str]:
    """The fields' names, suffixed ``#2``... where repeated - as a layout names them."""
    placeholder = slice(0, 0)
    return [
        s.name for s in unique_names([Slot(f.meta.name, placeholder) for f in fields])
    ]


def raw_action_values(values: list[tuple[Any, Any]]) -> dict[str, Any]:
    """The action fields' values by name, from ``(field, value)`` in config order."""
    fields = [field for field, _value in values]
    return {
        name: _scalar(value)
        for name, (_field, value) in zip(unique_names_of(fields), values)
    }


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array
