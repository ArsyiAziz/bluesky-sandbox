"""A design's fields reported over sampled episodes: what a step costs, phase by
phase and field by field, and where each field's values fall against its
bounds.

The episodes are the design's own - its scenario, each seed given - flown with
actions drawn from the action space, every step timed by the env as it runs
(:class:`~bluesky_sandbox.core.step_timing.StepTimer`). The steps are grouped
by how many aircraft were in the air. The sampled episodes may never fly as
many aircraft as the design may, nor as few: past the most they reached, each
episode's traffic is topped up with copies of its aircraft
(:func:`.cost.fill_traffic`) and stepped again; below the fewest, the
episode is set again with none of its traffic and copies of one of its
aircraft placed, as many as each count. So every count from one aircraft to
the design's most is measured, none projected.

The same steps give each field's values: how many fall below and above its
bounds, the range seen, raw and normalized, and a histogram against the
bounds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import bluesky as bs
import numpy as np

from bluesky_sandbox.checks.cost import Like, counts_to, fill_traffic
from bluesky_sandbox.checks.fields import _checkable, _drawn, batched_path, check_fields, spawned_all
from bluesky_sandbox.checks.placement import without_traffic
from bluesky_sandbox.checks.probe import probe
from bluesky_sandbox.core import services
from bluesky_sandbox.interface.fields.base import ActionKind, ObsField, PairObsField, action_kind

__all__ = ["ActionCheck", "BoundsTally", "CostReport", "FieldReport", "Spread", "field_checks", "field_report"]

#: The phase of what the env's wrappers add around its step.
WRAPPERS = "wrappers"


@dataclass
class Spread:
    """A series at each count: its median and the percentiles about it."""

    median: list[float | None] = field(default_factory=list)
    low: list[float | None] = field(default_factory=list)
    high: list[float | None] = field(default_factory=list)


@dataclass
class CostReport:
    """What a step costs at each count of aircraft in ``aircraft``.

    ``sampled[k]``: whether count ``k`` was flown by the sampled episodes
    (else topped up). ``total`` is the whole step, ``phases`` each phase's own
    time (they add up to the total), ``fields`` each field's raw computation -
    by its place in the config, ``(list, index)``."""

    aircraft: list[int] = field(default_factory=list)
    sampled: list[bool] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    total: Spread = field(default_factory=Spread)
    phases: dict[str, Spread] = field(default_factory=dict)
    fields: dict[tuple[str, int], list[float | None]] = field(default_factory=dict)
    percentiles: tuple[float, float] = (10.0, 90.0)


@dataclass
class BoundsTally:
    """One field's values over the sampled steps, against its bounds."""

    name: str
    where: tuple[str, int]
    samples: int = 0
    below: int = 0
    above: int = 0
    min: float | None = None
    max: float | None = None
    #: The normalized values of the least and the greatest seen.
    normalized_min: float | None = None
    normalized_max: float | None = None
    #: The bounds, when every aircraft has the same; else None (each its own).
    fixed_bounds: tuple[float, float] | None = None
    #: The normalizer (its class's name, "" for none), its output range and
    #: whether it clips to it.
    normalizer: str = ""
    output: tuple[float, float] | None = None
    clips: bool = False
    #: Samples per bin: bin ``k`` holds the values ``k / bins`` to
    #: ``(k + 1) / bins`` of the way from the low bound to the high one - bins
    #: below 0 and from ``bins`` up are outside, as far as the bounds' width
    #: again either side; bin ``-bins - 1`` holds all further below, bin
    #: ``2 * bins`` all further above (the range seen says how far).
    bins: int = 0
    histogram: dict[int, int] = field(default_factory=dict)
    _varying: bool = False

    def add(self, values: np.ndarray, low: np.ndarray, high: np.ndarray, defined: np.ndarray, normalize) -> None:
        values = np.asarray(values, dtype=np.float64)
        low = np.broadcast_to(np.asarray(low, dtype=np.float64), values.shape)
        high = np.broadcast_to(np.asarray(high, dtype=np.float64), values.shape)
        ok = np.asarray(defined, dtype=bool) & np.isfinite(values)
        if not ok.any():
            return
        v, lo, hi = values[ok], low[ok], high[ok]
        self.samples += int(v.size)
        self.below += int((v < lo).sum())
        self.above += int((v > hi).sum())
        self._track_bounds(lo, hi)
        least, most = int(np.argmin(v)), int(np.argmax(v))
        where = np.argwhere(ok)
        if self.min is None or v[least] < self.min:
            self.min = float(v[least])
            self.normalized_min = normalize(self.min, tuple(where[least]))
        if self.max is None or v[most] > self.max:
            self.max = float(v[most])
            self.normalized_max = normalize(self.max, tuple(where[most]))
        span = hi - lo
        bounded = np.isfinite(span) & (span > 0)
        if bounded.any():
            k = np.floor((v[bounded] - lo[bounded]) / span[bounded] * self.bins).astype(np.int64)
            # The high bound itself is inside: the last bin's, not the first out.
            k[v[bounded] == hi[bounded]] = self.bins - 1
            k = np.clip(k, -self.bins - 1, 2 * self.bins)
            for b, n in zip(*np.unique(k, return_counts=True), strict=True):
                self.histogram[int(b)] = self.histogram.get(int(b), 0) + int(n)

    def _track_bounds(self, lo: np.ndarray, hi: np.ndarray) -> None:
        if self._varying:
            return
        pair = (float(lo.min()), float(hi.max()))
        same = lo.min() == lo.max() and hi.min() == hi.max()
        if not same or (self.fixed_bounds is not None and self.fixed_bounds != pair):
            self._varying, self.fixed_bounds = True, None
        else:
            self.fixed_bounds = pair


@dataclass
class FieldReport:
    cost: CostReport
    bounds: list[BoundsTally]
    #: Each field's name and whether it reads every aircraft at once, by its
    #: place in the config.
    fields: dict[tuple[str, int], tuple[str, bool | None]]


def field_report(
    env: Any,
    seeds: Sequence[int],
    steps: int,
    *,
    most: int | None = None,
    percentiles: tuple[float, float] = (10.0, 90.0),
    bins: int = 20,
) -> FieldReport:
    """Fly ``env``'s episode for each of ``seeds``, ``steps`` steps each with
    actions drawn from its action space (each episode's draws seeded by its
    seed), timing every step with aircraft in the air - it ends early once
    none are and none are still to spawn. Then, past the most aircraft that
    episode flew, top it up to each count of :func:`.cost.counts_to` up to
    ``most`` (the design's most) and step it once more at each; and below
    the fewest, set it again with none of its traffic and that count of
    copies of one of its aircraft, and step it once at each. Each field's
    values are tallied against its bounds over the sampled steps, in ``bins``
    bins from its low bound to its high one."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    most = int(base.episode_max_aircraft if most is None else most)
    located = dict(_checkable(base.config))
    names = {where: (_field_name(f), batched_path(f)) for where, f in located.items()}
    where_of = {id(f): where for where, f in located.items()}
    tallies = {where: _tally(f, where, bins) for where, f in located.items()}
    rows: list[tuple[int, bool, float, dict[str, float], dict[tuple[str, int], float]]] = []
    timer = base.step_timer
    timer.enabled = True
    try:
        for seed in seeds:
            env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed))
            flown: list[int] = []
            like: Like | None = None
            for _ in range(int(steps)):
                if bs.traf.ntraf == 0 and spawned_all(base):
                    break
                if bs.traf.ntraf == 0:
                    # Waiting for the first to spawn: nothing to time.
                    env.step({agent: _drawn(env.action_space(agent), rng) for agent in env.agents})
                    continue
                rows.append(_timed_step(env, base, rng, True, where_of))
                if bs.traf.ntraf:
                    flown.append(bs.traf.ntraf)
                    like = Like.of(0)
                    for where, f in located.items():
                        _sample(f, tallies[where])
            if not flown or like is None:
                continue
            for n in counts_to(most):
                if n <= max(flown) or not env.agents:
                    continue
                if fill_traffic(base, n) < n:
                    break  # the design refuses more
                rows.append(_timed_step(env, base, rng, False, where_of))
            fewer = [n for n in counts_to(most) if n < min(flown)]
            if fewer:
                with without_traffic(base):
                    env.reset(seed=int(seed))
                for n in fewer:
                    if fill_traffic(base, n, like) < n:
                        break
                    rows.append(_timed_step(env, base, rng, False, where_of))
    finally:
        timer.enabled = False
    return FieldReport(_cost(rows, percentiles), list(tallies.values()), names)


def _timed_step(env, base, rng, sampled, where_of):
    actions = {agent: _drawn(env.action_space(agent), rng) for agent in env.agents}
    start = time.perf_counter()
    env.step(actions)
    outer = (time.perf_counter() - start) * 1000.0
    timing = base.step_timer.last
    phases = dict(timing.phases)
    if outer > timing.total_ms:
        phases[WRAPPERS] = outer - timing.total_ms
    fields = {where_of[k]: v for k, v in timing.fields.items() if k in where_of}
    return bs.traf.ntraf, sampled, outer, phases, fields


def _cost(rows, percentiles) -> CostReport:
    """The steps grouped by their count of aircraft: a count the sampled
    episodes flew from their steps only; one they did not, from the topped-up."""
    by_count: dict[int, list] = {}
    for row in rows:
        if row[0]:  # a step that ended with none in the air reads no field
            by_count.setdefault(row[0], []).append(row)
    rows = [r for group in by_count.values() for r in group]
    report = CostReport(percentiles=percentiles)
    phase_names = sorted({name for row in rows for name in row[3]})
    field_keys = sorted({key for row in rows for key in row[4]})
    report.phases = {name: Spread() for name in phase_names}
    report.fields = {key: [] for key in field_keys}
    for n in sorted(by_count):
        group = by_count[n]
        sampled = [r for r in group if r[1]]
        group = sampled or group
        report.aircraft.append(n)
        report.sampled.append(bool(sampled))
        report.steps.append(len(group))
        _append(report.total, [r[2] for r in group], percentiles)
        for name in phase_names:
            _append(report.phases[name], [r[3].get(name, 0.0) for r in group], percentiles)
        for key in field_keys:
            times = [r[4][key] for r in group if key in r[4]]
            report.fields[key].append(float(np.median(times)) if times else None)
    return report


def _append(spread: Spread, values: list[float], percentiles) -> None:
    low, median, high = np.percentile(values, [percentiles[0], 50.0, percentiles[1]])
    spread.median.append(float(median))
    spread.low.append(float(low))
    spread.high.append(float(high))


def _tally(field_obj: Any, where: tuple[str, int], bins: int) -> BoundsTally:
    normalizer = getattr(field_obj, "normalizer", None)
    output = None
    if normalizer is not None:
        try:
            lows, highs = normalizer.output_bounds(field_obj)
            output = (float(min(lows)), float(max(highs)))
        except Exception:  # noqa: BLE001 - a range it cannot state is not shown
            output = None
    return BoundsTally(
        _field_name(field_obj),
        where,
        normalizer="" if normalizer is None else type(normalizer).__name__,
        output=output,
        clips=bool(getattr(normalizer, "clipped", False)),
        bins=int(bins),
    )


def _sample(field_obj: Any, tally: BoundsTally) -> None:
    every = list(range(bs.traf.ntraf))
    try:
        values, low, high, defined = field_obj.values_and_bounds(every)
    except Exception:  # noqa: BLE001 - the field's checks report it
        return

    def normalize(value: float, at: tuple[int, ...]) -> float | None:
        if getattr(field_obj, "normalizer", None) is None:
            return None
        try:
            out = services._normalize_field_value(field_obj, value, int(at[0]))
        except Exception:  # noqa: BLE001 - a value it cannot normalize is not shown
            return None
        return float(out[0]) if len(out) == 1 else None

    tally.add(values, low, high, defined, normalize)


def _field_name(field_obj: Any) -> str:
    """A field as the designer lists it: its class, a lag's from what it lags,
    a difference of a field from the ownship's as ``<field> − own``."""
    left, right = getattr(field_obj, "left", None), getattr(field_obj, "right", None)
    if isinstance(left, ObsField) and isinstance(right, ObsField):
        if type(left) is type(right):
            return f"{_field_name(left)} − own"
        return f"{_field_name(left)} − {_field_name(right)}"
    inner = getattr(field_obj, "inner", None)
    steps = getattr(field_obj, "steps", None)
    if isinstance(inner, (ObsField, PairObsField)) and isinstance(steps, int):
        return f"{_field_name(inner)}_lag{steps}"
    query = getattr(field_obj, "query_name", "")
    name = type(field_obj).__name__
    return f"{name}({query})" if query else name


@dataclass(frozen=True)
class ActionCheck:
    """One action probed once: whether the target it commands is the one it
    states (``agrees``), with both; or why it could not be probed."""

    where: tuple[str, int]
    name: str
    agrees: bool | None = None
    actual: float | None = None
    expected: float | None = None
    command: tuple[str, ...] = ()
    error: str = ""


def field_checks(env: Any, *, seed: int, steps: int) -> tuple[Any, list[ActionCheck]]:
    """Every field of ``env`` checked against itself over episode ``seed``,
    ``steps`` steps flown twice (:func:`.fields.check_fields`), and each action
    probed once at the end: what the checks found, field by field."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    report = check_fields(base, steps=int(steps), seed=int(seed))
    actions = []
    for i, field_obj in enumerate(base.config.action_fields):
        where = ("action_fields", i)
        name = _field_name(field_obj)
        if bs.traf.ntraf == 0:
            actions.append(ActionCheck(where, name, error="no aircraft in the air to probe it on"))
            continue
        try:
            result = probe(base, field_obj, str(bs.traf.id[0]), give=_middle(field_obj, 0))
        except Exception as e:  # noqa: BLE001 - reported as the action's
            actions.append(ActionCheck(where, name, error=f"{type(e).__name__}: {e}"))
            continue
        target = next((stage for stage in result.stages if stage.name == "target"), None)
        actions.append(
            ActionCheck(
                where,
                name,
                agrees=None if target is None else target.agrees,
                actual=None if target is None or target.actual is None else float(target.actual),
                expected=None if target is None or target.expected is None else float(target.expected),
                command=tuple(result.command),
            )
        )
    return report, actions


def _middle(field_obj: Any, idx: int) -> float | list[float]:
    """What the policy gives an action to probe it: a switch, the value that
    turns it on; any other, the middle of its bounds for aircraft ``idx``, in
    its own unit, through its normalizer if it has one - whatever the
    normalizer (a choice, an angle's two values)."""
    if action_kind(field_obj) is ActionKind.BINARY:
        return float(field_obj.switch_on_value())
    acting = services._acting(field_obj, idx)
    low, high = acting.bounds(idx)
    middle = (float(low) + float(high)) / 2.0
    normalizer = getattr(acting, "normalizer", None)
    return middle if normalizer is None else [float(v) for v in normalizer.normalize(acting, middle, idx)]


def checks_plain(report: Any, actions: list[ActionCheck]) -> dict[str, Any]:
    """:func:`field_checks`' findings as JSON values, config places as
    ``"list:index"``."""
    return {
        "episode": list(report.episode_findings),
        "fields": [
            {"field": f"{r.where[0]}:{r.where[1]}", "findings": list(r.findings), "notes": list(r.notes)}
            for r in report.results
            if r.where is not None
        ],
        "actions": [
            {
                "field": f"{a.where[0]}:{a.where[1]}", "name": a.name, "agrees": a.agrees,
                "actual": a.actual, "expected": a.expected, "command": list(a.command), "error": a.error,
            }
            for a in actions
        ],
    }


def as_plain(report: FieldReport) -> dict[str, Any]:
    """``report`` as JSON values: config places as ``"list:index"``."""

    def key(where: tuple[str, int]) -> str:
        return f"{where[0]}:{where[1]}"

    def num(x: Any) -> Any:
        return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x

    cost = report.cost
    return {
        "cost": {
            "aircraft": cost.aircraft,
            "sampled": cost.sampled,
            "steps": cost.steps,
            "percentiles": list(cost.percentiles),
            "total": vars(cost.total),
            "phases": {name: vars(s) for name, s in cost.phases.items()},
            "fields": {key(k): v for k, v in cost.fields.items()},
        },
        "fields": {key(k): {"name": name, "batched": batched} for k, (name, batched) in report.fields.items()},
        "bounds": [
            {
                "field": key(t.where), "name": t.name, "samples": t.samples, "below": t.below, "above": t.above,
                "min": num(t.min), "max": num(t.max),
                "normalized_min": num(t.normalized_min), "normalized_max": num(t.normalized_max),
                "fixed_bounds": list(t.fixed_bounds) if t.fixed_bounds else None,
                "normalizer": t.normalizer, "output": list(t.output) if t.output else None, "clips": t.clips,
                "bins": t.bins, "histogram": {str(k): v for k, v in sorted(t.histogram.items())},
            }
            for t in report.bounds
        ],
    }
