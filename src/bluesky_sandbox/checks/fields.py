"""Every field of a design checked against itself, on live traffic.

A field may compute the same value several ways - in bulk for every aircraft,
one at a time, in a pair matrix, normalized as a batch or one value at a time
- and the env reads whichever is quickest where it is. If they disagree, an
agent observes one number and its reward is computed from another, and
nothing else notices. Each field family says how its own ways compare
(``check_consistency``); this runs them, and its normalizer's, for every field
in a config - a custom one as a built-in, and against its ``expected`` where
it states one.

Over the same run it watches each field's values: one that is not a number
(NaN, infinite) breaks training, and one that differs when the episode is
flown again with the same seed cannot be reproduced - both fail. One outside
the field's bounds is noted, not failed: a clipped normalizer loses it, and
whether that matters is the designer's call.
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
    """One field's check: what fails it (``findings``) and what is worth
    knowing without failing it (``notes``)."""

    field: str
    findings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class FieldsReport:
    results: tuple[FieldCheck, ...]
    #: What fails the whole episode rather than one field: it is not
    #: repeatable - the same seed flies other traffic.
    episode_findings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.episode_findings and all(r.ok for r in self.results)

    @property
    def failures(self) -> tuple[FieldCheck, ...]:
        return tuple(r for r in self.results if not r.ok)

    def assert_ok(self) -> None:
        if not self.ok:
            lines = [f"  episode: {f}" for f in self.episode_findings]
            lines += [f"  {r.field}: {'; '.join(r.findings[:3])}" for r in self.failures]
            raise AssertionError(f"the field checks failed:\n" + "\n".join(lines))

    def __str__(self) -> str:
        lines = [f"episode: {f}" for f in self.episode_findings]
        for r in self.results:
            lines.append(f"{r.field}: {'ok' if r.ok else '; '.join(r.findings[:3])}")
            lines += [f"  note: {n}" for n in r.notes]
        return "\n".join(lines)


def check_fields(env: Any, *, steps: int = 20, seed: int = 0) -> FieldsReport:
    """Every field in ``env``'s config that can check itself, on its live
    traffic: ``env`` reset for episode ``seed``, then stepped ``steps`` times
    with actions drawn from its own action space (stateful fields get a
    history; aircraft move into each other's way), each field's values kept at
    every step - then flown again the same way, to compare."""
    checked = list(_checkable(env.config))
    first = _fly(env, checked, steps, seed)
    # The state the consistency checks read: the end of the first flight.
    indices = list(range(bs.traf.ntraf))
    consistency = {id(f): list(_findings(f, indices)) for f in checked}
    second = _fly(env, checked, steps, seed)
    traffic = _traffic_differs(first, second)
    results = []
    for f in checked:
        samples = [step[1][id(f)] for step in first]
        findings = consistency[id(f)] + _not_numbers(samples)
        if not traffic:
            findings += _not_repeated(samples, [step[1][id(f)] for step in second])
        results.append(FieldCheck(_name(f), tuple(findings), tuple(_outside_bounds(samples))))
    return FieldsReport(tuple(results), tuple(traffic))


#: One step's record: the traffic (callsigns), and each field's values and
#: bounds - by the field's id.
_Step = tuple[tuple[str, ...], dict[int, Any]]


def _fly(env: Any, checked: list[Any], steps: int, seed: int) -> list[_Step]:
    """``env``'s episode ``seed``, ``steps`` steps of actions drawn from its
    action space by a generator seeded ``seed``, recorded after the reset and
    each step."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    record = [_record(checked)]
    for _ in range(steps):
        if not env.agents:
            break
        env.step({agent: _drawn(env.action_space(agent), rng) for agent in env.agents})
        record.append(_record(checked))
    return record


def _record(checked: list[Any]) -> _Step:
    return tuple(bs.traf.id), {id(f): _sample(f) for f in checked}


def _sample(field: Any) -> Any:
    """``field``'s values for every aircraft now (a pair field's, each with
    every other) - ``(values, low, high, defined)`` - or the error reading them
    raised."""
    try:
        return field.values_and_bounds(list(range(bs.traf.ntraf)))
    except Exception as e:  # noqa: BLE001 - reported as the field's
        return e


def _traffic_differs(first: list[_Step], second: list[_Step]) -> list[str]:
    for k, ((a, _), (b, _)) in enumerate(zip(first, second, strict=False)):
        if a != b:
            return [f"not repeatable: flown again with the same seed, step {k} has other aircraft ({len(a)} vs {len(b)})"]
    if len(first) != len(second):
        return [f"not repeatable: flown again with the same seed, it ran {len(second) - 1} steps, not {len(first) - 1}"]
    return []


def _not_numbers(samples: list[Any]) -> list[str]:
    for k, sample in enumerate(samples):
        if isinstance(sample, Exception):
            return [f"step {k}: reading it raised {type(sample).__name__}: {sample}"]
        values, _low, _high, defined = sample
        bad = int((~np.isfinite(values) & defined).sum())
        if bad:
            return [f"step {k}: {bad} value(s) not a number (NaN or infinite)"]
    return []


def _not_repeated(first: list[Any], second: list[Any]) -> list[str]:
    for k, (a, b) in enumerate(zip(first, second, strict=False)):
        if isinstance(a, Exception) or isinstance(b, Exception):
            continue
        if not np.array_equal(a[0], b[0], equal_nan=True):
            return [f"not repeatable: flown again with the same seed, its values differ at step {k}"]
    return []


def _outside_bounds(samples: list[Any]) -> list[str]:
    values, lows, highs = [], [], []
    for sample in samples:
        if isinstance(sample, Exception):
            return []
        v, low, high, defined = sample
        defined = defined & np.isfinite(v)
        values.append(v[defined])
        lows.append(np.broadcast_to(low, v.shape)[defined])
        highs.append(np.broadcast_to(high, v.shape)[defined])
    v, low, high = (np.concatenate(x) for x in (values, lows, highs))
    outside = (v < low) | (v > high)
    if not outside.any():
        return []
    share = outside.mean()
    return [
        f"outside its bounds in {share:.1%} of samples (seen {_num(v.min())} to {_num(v.max())}, "
        f"bounds {_num(low.min())} to {_num(high.max())})"
    ]


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


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
