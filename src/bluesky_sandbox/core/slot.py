"""A field's columns in one part of a space; see :mod:`.layout`.

Kept free of imports so the observation wrappers, which :mod:`.layout`'s own
imports load, can build slots too.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    """One field's columns in a part: ``name`` is its ``meta.name``, suffixed
    ``#2``, ``#3``... where the part holds the same name more than once."""

    name: str
    columns: slice

    @property
    def width(self) -> int:
        return self.columns.stop - self.columns.start


def unique_names(part: list[Slot]) -> list[Slot]:
    """``part`` with a repeated name suffixed by its occurrence: ``#2``, ``#3``."""
    seen: Counter[str] = Counter()
    out = []
    for slot in part:
        base = slot.name.split("#", 1)[0]
        seen[base] += 1
        name = base if seen[base] == 1 else f"{base}#{seen[base]}"
        out.append(Slot(name, slot.columns))
    return out
