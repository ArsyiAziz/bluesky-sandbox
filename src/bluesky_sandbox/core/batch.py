"""Every agent's step at once, for the batched hooks.

A per-agent hook - ``reward``, ``terminated``, ``truncated`` - is called once for
each agent. Its batched counterpart - ``reward_batch`` and so on - is called once
with a :class:`StepBatch`: the same information for every agent, stacked into
arrays, so the hook can be one NumPy expression. Both are optional; a task
defines at most one of each pair.

Every agent's intruders are all the other aircraft, so the intruder parts are
regular ``(n_agents, n_intruders)`` arrays - no padding or masks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

import bluesky as bs
import numpy as np

from bluesky_sandbox.interface.fields.base import PairObsField
from bluesky_sandbox.interface.task import AgentStepContext, DesignKeys

from .step_values import ACID, StepValues, unique_names_of

if TYPE_CHECKING:
    from .services import QueryBatch

__all__ = ["RawBatch", "StepBatch"]


class RawBatch(Mapping[str, Mapping[str, np.ndarray]]):
    """Every agent's observation in raw values, stacked: row ``k`` is agent ``k``.

    ``raw["ownship"]["alt_ft"]`` is ``(n_agents,)``; ``raw["intruders"]
    ["dist_to_own_nm"]`` is ``(n_agents, n_intruders)``, row ``k`` column ``i``
    being agent ``k``'s intruder ``i``; ``raw["intruders"]["acid"]`` names them.
    A multi-column field adds a trailing axis. Built part by part on first read.
    """

    def __init__(
        self, values: StepValues, parts: Mapping[str, Any], acidx: np.ndarray
    ) -> None:
        self._values = values
        self._parts = dict(parts)
        self._acidx = acidx
        self._traffic = values.traffic()
        self._built: dict[str, dict[str, np.ndarray]] = {}

    def __getitem__(self, part: str) -> dict[str, np.ndarray]:
        built = self._built.get(part)
        if built is None:
            if part not in self._parts:
                raise KeyError(
                    f"no observation part {part!r}; this environment has "
                    f"{sorted(self._parts)}"
                )
            self._values.check_current(self._traffic)
            built = self._build(part, self._parts[part])
            self._built[part] = built
        return built

    def __iter__(self) -> Iterator[str]:
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __repr__(self) -> str:
        return f"RawBatch(parts={list(self._parts)}, agents={self._acidx.size})"

    def _build(self, part: str, fields: Any) -> dict[str, np.ndarray]:
        names = unique_names_of(fields)
        acidx = self._acidx
        if not part.endswith("intruders"):
            return {
                name: np.asarray(self._values.values(f))[acidx]
                for name, f in zip(names, fields)
            }
        others = intruder_indices(acidx)
        rows = np.arange(acidx.size)[:, None]
        out = {ACID: np.array(bs.traf.id, dtype=object)[others]}
        for name, f in zip(names, fields):
            if isinstance(f, PairObsField):
                out[name] = self._values.pair_matrix(f, acidx)[rows, others]
            else:
                out[name] = np.asarray(self._values.values(f))[others]
        return out


def intruder_indices(acidx: np.ndarray) -> np.ndarray:
    """``(n_agents, n_intruders)``: row ``k`` holds agent ``k``'s intruders -
    every other aircraft, in traffic order, as the observation lists them."""
    n = int(bs.traf.ntraf)
    everyone = np.arange(n, dtype=np.intp)
    return np.array(
        [everyone[everyone != own] for own in acidx], dtype=np.intp
    ).reshape(acidx.size, max(n - 1, 0))


@dataclass
class StepBatch:
    """This step for every controlled agent, stacked: index ``k`` is ``acids[k]``.

    ``obs`` holds the observations as the policy sees them, stacked per part;
    ``raw_obs`` the raw values by part and field name (:class:`RawBatch`);
    ``raw_action`` each action field's value by name, NaN for an agent given no
    action (``has_action``). ``terminated`` and ``truncated`` are filled in
    once decided, for ``reward_batch``. ``query(name)`` reads a queryable for
    every agent at once; ``context(k)`` is agent ``k``'s per-agent context.
    """

    acids: tuple[str, ...]
    acidx: np.ndarray
    obs: dict[str, np.ndarray]
    raw_obs: Annotated[RawBatch, DesignKeys("observation", batched=True)]
    raw_action: Annotated[dict[str, np.ndarray], DesignKeys("action", batched=True)]
    has_action: np.ndarray
    intruder_idx: np.ndarray
    infos: list[dict[str, Any]]
    rng: np.random.Generator
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    _query: Callable[[str, np.ndarray], Any] = field(default=None, repr=False)
    _context: Callable[[int], Any] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.acids)

    def query(
        self, name: Annotated[str, DesignKeys("queryable", batched=True)]
    ) -> QueryBatch:
        """Queryable ``name`` for every agent at once, as arrays."""
        return self._query(name, self.acidx)

    def context(self, k: int) -> AgentStepContext:
        """Agent ``k``'s :class:`~bluesky_sandbox.interface.task.AgentStepContext`."""
        return self._context(int(self.acidx[k]))


def stack_observations(observations: list[Any]) -> dict[str, np.ndarray]:
    """Per-agent observations stacked per part; a plain vector becomes ``"ownship"``."""
    if not observations:
        return {}
    if not isinstance(observations[0], Mapping):
        return {"ownship": np.stack([np.asarray(o) for o in observations])}
    return {
        key: np.stack([np.asarray(o[key]) for o in observations])
        for key in observations[0]
    }


def stack_actions(
    names: list[str], actions: list[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Raw actions by name, one row per agent, NaN where an agent had none."""
    has_action = np.array([bool(a) for a in actions], dtype=bool)
    stacked = {}
    for name in names:
        column = [a.get(name, np.nan) for a in actions]
        stacked[name] = np.asarray(column, dtype=np.float64)
    return stacked, has_action
