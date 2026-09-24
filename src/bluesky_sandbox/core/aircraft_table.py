"""Per-aircraft numeric arrays that follow BlueSky's changing aircraft list."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Column:
    """One array in an :class:`AircraftTable`.

    ``fill`` is what a new aircraft (or queryable) starts with. A ``per_step``
    array is refilled by :meth:`AircraftTable.reset_step`; the others keep
    their values for the whole episode.
    """

    fill: float
    dtype: Any
    per_step: bool = False


class AircraftTable:
    """Named numeric arrays with one row per aircraft.

    With ``keyed_columns`` every array also has one column per named key (the
    episode's queryables); without, the arrays are 1-D. Rows and columns are
    matched *by key*, so an aircraft keeps its values when others spawn or
    leave, and a key keeps its values when others are added.

    The table knows nothing of BlueSky: callers choose the row keys. The
    monitors pass each aircraft's uid rather than its callsign - BlueSky reuses
    a callsign once its aircraft is deleted, and a row keyed by callsign would
    hand the new aircraft the old one's values.

    :attr:`ids` is replaced by a new tuple only when the rows actually change,
    so a caller may cache work against it by identity.
    """

    def __init__(self, arrays: dict[str, Column], *, keyed_columns: bool = False):
        self._columns_spec = dict(arrays)
        self._keyed = keyed_columns
        self.ids: tuple[Hashable, ...] = ()
        self.row: dict[Hashable, int] = {}
        self.columns: tuple[str, ...] = ()
        self.col: dict[str, int] = {}
        self._arrays = {name: self._blank(name, 0, 0) for name in self._columns_spec}

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        # Reached by ``table[name] += x``: numpy updates the array in place and
        # Python then assigns that same array back. Anything else would replace
        # an array behind the table's fills, shapes and dtypes, so refuse it.
        if value is not self._arrays[name]:
            raise TypeError(
                f"AircraftTable array {name!r} is updated in place, not replaced."
            )

    def __len__(self) -> int:
        return len(self.ids)

    def sync_rows(self, ids: Iterable[Hashable]) -> np.ndarray | None:
        """Match the rows to the row keys ``ids``.

        Survivors keep every array's values, newcomers get each array's fill,
        and departed aircraft are dropped. Returns, for each new row, the old
        row it came from (``-1`` for a newcomer), so a caller can carry its own
        per-row data the same way - or ``None`` when nothing changed.
        """
        ids = tuple(ids)
        if ids == self.ids:
            return None
        row = {key: i for i, key in enumerate(ids)}
        if len(row) != len(ids):
            raise ValueError(f"AircraftTable rows must be unique, got {ids!r}.")
        take = np.fromiter(
            (self.row.get(key, -1) for key in ids), dtype=np.intp, count=len(ids)
        )
        kept = take >= 0
        for name, old in self._arrays.items():
            new = self._blank(name, len(ids), len(self.columns))
            new[kept] = old[take[kept]]
            self._arrays[name] = new
        self.ids = ids
        self.row = row
        return take

    def set_columns(self, names: Iterable[str]) -> None:
        """Match the columns to ``names``, the way :meth:`sync_rows` does rows."""
        if not self._keyed:
            raise TypeError("This AircraftTable has no keyed columns.")
        names = tuple(names)
        if names == self.columns:
            return
        take = np.fromiter(
            (self.col.get(name, -1) for name in names), dtype=np.intp, count=len(names)
        )
        kept = take >= 0
        for name, old in self._arrays.items():
            new = self._blank(name, len(self.ids), len(names))
            new[:, kept] = old[:, take[kept]]
            self._arrays[name] = new
        self.columns = names
        self.col = {name: i for i, name in enumerate(names)}

    def reset_step(self) -> None:
        """Refill the ``per_step`` arrays."""
        for name, spec in self._columns_spec.items():
            if spec.per_step:
                self._arrays[name].fill(spec.fill)

    def clear(self) -> None:
        """Drop every row; keyed columns stay."""
        self.ids = ()
        self.row = {}
        self._arrays = {
            name: self._blank(name, 0, len(self.columns)) for name in self._columns_spec
        }

    def _blank(self, name: str, rows: int, columns: int) -> np.ndarray:
        spec = self._columns_spec[name]
        shape = (rows, columns) if self._keyed else (rows,)
        return np.full(shape, spec.fill, dtype=spec.dtype)
