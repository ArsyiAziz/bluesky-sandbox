"""Every field of a design checked against itself, on live traffic.

A field may compute the same value several ways - in bulk for every aircraft,
one at a time, in a pair matrix, normalized as a batch or one value at a time
- and the env reads whichever is quickest where it is. If they disagree, an
agent observes one number and its reward is computed from another, and
nothing else notices. Each field family says how its own ways compare
(``check_consistency``); this runs them, and its normalizer's, for every field
in a config - a custom one as a built-in, and against its ``expected`` where
it states one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any

import bluesky as bs
import numpy as np

from bluesky_sandbox.core import services
from bluesky_sandbox.interface.fields._consistency import differs

__all__ = ["FieldCheck", "FieldsReport", "check_fields", "normalization_findings"]


@dataclass(frozen=True)
class FieldCheck:
    field: str
    findings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class FieldsReport:
    results: tuple[FieldCheck, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> tuple[FieldCheck, ...]:
        return tuple(r for r in self.results if not r.ok)

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError(
                f"{len(self.failures)} of {len(self.results)} fields disagree with themselves:\n"
                + "\n".join(f"  {r.field}: {'; '.join(r.findings[:3])}" for r in self.failures)
            )

    def __str__(self) -> str:
        return "\n".join(f"{r.field}: {'ok' if r.ok else '; '.join(r.findings[:3])}" for r in self.results)


def check_fields(env: Any, *, steps: int = 4, seed: int = 0) -> FieldsReport:
    """Every field in ``env``'s config that can check itself, on its live
    traffic: ``env`` reset for episode ``seed``, then stepped ``steps`` times
    with actions drawn from its own action space (stateful fields get a
    history; aircraft move into each other's way)."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if not env.agents:
            break
        env.step({agent: _drawn(env.action_space(agent), rng) for agent in env.agents})
    indices = list(range(bs.traf.ntraf))
    return FieldsReport(tuple(FieldCheck(_name(f), tuple(_findings(f, indices))) for f in _checkable(env.config)))


def normalization_findings(field: Any, own: int, others: Any) -> list[str]:
    """Where ``field``'s normalizer gives a different value for ``own``'s
    intruders normalized as a batch than one at a time."""
    raw = field.values_about(own, others)
    batched = services._normalize_field_values_batch(field, raw, own)
    single = [services._normalize_field_value(field, raw[k], own) for k in range(len(others))]
    return differs(batched, single, f"own {own}: normalized as a batch vs one at a time")


def _findings(field: Any, indices: list[int]) -> Iterator[str]:
    yield from field.check_consistency(indices)
    if getattr(field, "normalizer", None) is not None and len(indices) > 1:
        for own in (indices[0], indices[len(indices) // 2]):
            yield from normalization_findings(field, own, [j for j in indices if j != own])


def _checkable(config: Any) -> Iterator[Any]:
    """Each field of ``config`` that checks itself, once - from every list it
    holds, whatever the list is called."""
    seen: set[int] = set()
    for config_field in fields(config):
        value = getattr(config, config_field.name)
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            if callable(getattr(item, "check_consistency", None)) and id(item) not in seen:
                seen.add(id(item))
                yield item


def _drawn(space: Any, rng: np.random.Generator) -> Any:
    space.seed(int(rng.integers(2**31)))
    return space.sample()


def _name(field: Any) -> str:
    query = getattr(field, "query_name", "")
    return type(field).__name__ + (f"({query!r})" if query else "")
