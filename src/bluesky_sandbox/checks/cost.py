"""What a field costs to read, measured with as many aircraft in the air as it
is timed for - never projected from fewer.

Aircraft are added as copies of one aircraft, spread out: where they are does
not matter, how many does. Each way the field computes is timed on every
aircraft, each time at a new sim time: what a field keeps for a sim time (a
query's result, a pair geometry) is computed again, as at the first read of a
step, so a time includes getting what it reads, not only arithmetic on it.

:func:`cost_curve` times a field from one aircraft (a pair field: two) up to
the design's most, doubling: how each way grows with the traffic. The
simulation's part of an env step - its ``dt / simdt`` substeps - is timed on
the same traffic.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.checks.fields import batched_path
from bluesky_sandbox.checks.placement import without_traffic
from bluesky_sandbox.interface.fields.base import ObsField, PairObsField

__all__ = ["CostCurve", "FieldCost", "Like", "cost_curve", "counts_to", "field_cost", "fill_traffic", "sim_step_ms"]

#: Spacing of the aircraft added, in degrees: apart, so none share a point.
_SPREAD_DEG = 0.25


@dataclass(frozen=True)
class FieldCost:
    """Each way a field computes, timed on ``aircraft`` aircraft: median ms."""

    aircraft: int
    ms: dict[str, float] = field(default_factory=dict)
    #: Whether its bulk path is a batched one, or one aircraft at a time.
    batched: bool | None = None
    #: The way the env reads it, among ``ms``.
    env_path: str = ""


@dataclass(frozen=True)
class CostCurve:
    """A field timed with each count of aircraft in ``aircraft`` in the air:
    each way's median ms at each, and the simulation's part of an env step."""

    aircraft: tuple[int, ...]
    ms: dict[str, tuple[float | None, ...]]
    sim_step_ms: tuple[float, ...]
    batched: bool | None = None
    env_path: str = ""


@dataclass(frozen=True)
class Like:
    """What the aircraft added copy: where, how high, how fast, which way."""

    lat: float
    lon: float
    alt_ft: float
    cas_kts: float
    hdg_deg: float
    actype: str

    @classmethod
    def of(cls, idx: int) -> Like:
        """The aircraft at ``idx``, as it is now."""
        return cls(
            float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), float(bs.traf.alt[idx]) / ft,
            float(bs.traf.cas[idx]) / kts, float(bs.traf.hdg[idx]), str(bs.traf.type[idx]),
        )


def counts_to(most: int) -> tuple[int, ...]:
    """One aircraft, doubling, up to ``most``: ``1, 2, 4, ... most``."""
    most = max(1, int(most))
    return tuple(sorted({*(2**k for k in range(int(math.log2(most)) + 1)), most}))


def fill_traffic(env: Any, aircraft: int | None = None, like: Like | None = None) -> int:
    """Top the traffic up to ``aircraft`` (the design's most, by default) with
    copies of ``like`` (the first aircraft in the air, by default), spread on a
    grid beside it - or as many as the design takes, if it refuses one. How
    many are in the air after."""
    target = int(env.episode_max_aircraft if aircraft is None else aircraft)
    if like is None:
        if bs.traf.ntraf == 0:
            return 0
        like = Like.of(0)
    # The grid the design's most would fill: each place the same, however many.
    side = math.ceil(math.sqrt(max(target, int(env.episode_max_aircraft))))
    k = bs.traf.ntraf
    while bs.traf.ntraf < target:
        row, col = divmod(k, side)
        k += 1
        made = env.spawn(
            (like.lat + row * _SPREAD_DEG, like.lon + col * _SPREAD_DEG),
            alt_ft=like.alt_ft, spd_kts=like.cas_kts, hdg_deg=like.hdg_deg, actype=like.actype,
        )
        if made is None:
            break  # the design refuses more: time what there is
    return bs.traf.ntraf


def field_cost(field_obj: Any, *, repeats: int = 5, every_way: bool = True) -> FieldCost:
    """Each way ``field_obj`` computes - or, not ``every_way``, the way the env
    reads it - for every aircraft in the air, timed ``repeats`` times, each at
    a new sim time."""
    paths, env_path = _paths(field_obj)
    if not every_way:
        paths = {env_path: paths[env_path]} if env_path else {}
    ms = {}
    for name, path in paths.items():
        try:
            ms[name] = _cold_ms(path, repeats)
        except Exception:  # noqa: BLE001 - a path that cannot run is not timed
            continue
    return FieldCost(bs.traf.ntraf, ms, batched_path(field_obj), env_path)


def cost_curve(
    env: Any,
    field_obj: Any,
    *,
    like: Like,
    seed: int = 0,
    most: int | None = None,
    repeats: int = 5,
) -> CostCurve:
    """``field_obj`` timed every way with 1, 2, 4, ... up to ``most`` (the
    design's most) aircraft in the air - copies of ``like``, in ``env``'s
    episode ``seed`` without its own traffic; a pair field from two, the
    fewest that make a pair. It leaves ``env`` so."""
    fewest = 2 if isinstance(field_obj, PairObsField) else 1
    counts = [n for n in counts_to(env.episode_max_aircraft if most is None else most) if n >= fewest]
    ms: dict[str, list[float | None]] = {}
    steps: list[float] = []
    flown: list[int] = []
    with without_traffic(env):
        env.reset(seed=seed)
    for n in counts:
        if fill_traffic(env, n, like) < n:
            break  # the design refuses more
        cost = field_cost(field_obj, repeats=repeats)
        for name in [*cost.ms, *(k for k in ms if k not in cost.ms)]:
            ms.setdefault(name, [None] * len(flown)).append(cost.ms.get(name))
        steps.append(sim_step_ms(env, repeats))
        flown.append(n)
    return CostCurve(
        tuple(flown), {k: tuple(v) for k, v in ms.items()}, tuple(steps), batched_path(field_obj), _paths(field_obj)[1]
    )


def sim_step_ms(env: Any, repeats: int = 5) -> float:
    """The simulation's part of one of ``env``'s steps - BlueSky's ``dt /
    simdt`` substeps - on the traffic in the air (ms, median)."""
    substeps = max(1, round(float(env.config.dt) / float(bs.sim.simdt)))
    times = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        for _ in range(substeps):
            bs.sim.step()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _paths(field_obj: Any) -> tuple[dict[str, Callable[[], Any]], str]:
    """Each way ``field_obj`` computes for every aircraft in the air, and the
    one the env reads."""
    every = list(range(bs.traf.ntraf))
    if isinstance(field_obj, PairObsField):
        return {
            "pair matrix": lambda: field_obj.get_pair_matrix(np.asarray(every)),
            "per ownship": lambda: [field_obj.get_pairs(i, [j for j in every if j != i]) for i in every],
            "per pair": lambda: [field_obj.get_pair(i, j) for i in every for j in every if j != i],
        }, "pair matrix"
    if isinstance(field_obj, ObsField):
        return {
            "batched": lambda: field_obj.get_many(every),
            "one at a time": lambda: [field_obj.get(i) for i in every],
        }, "batched"
    return {}, ""


def _cold_ms(path: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(max(1, repeats)):
        bs.sim.step()  # a new sim time: nothing kept for the last is used
        start = time.perf_counter()
        path()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)
