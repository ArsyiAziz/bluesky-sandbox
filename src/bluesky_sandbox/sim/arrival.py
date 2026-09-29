"""An aircraft's route plan: the level and speed it flies each leg at, and when
it is due over each fix.

Both are assigned once, at spawn, leg by leg from where the aircraft is. A leg
is planned at the level and calibrated airspeed carried into it - the spawn
state for the first, then each fix's gates as the aircraft passes them, a fix
without a gate carrying the last one on (:func:`planned_legs`).

A leg's nominal time is the longer of flying it at its planned speed and making
its altitude change at the aircraft's own maximum climb or descent rate. A sampled slack is added, so an aircraft may have to lose time (or gain
it), and the leg is never due sooner than the aircraft could fly it at its
maximum speed. A fix with no slack given has no target time; the times after it
still run on from its nominal.

Times are in simulator seconds, as ``bs.sim.simt``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vcas2tas
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.sampling.distributions import sample_scalar

__all__ = ["arrival_times", "planned_legs"]

# Floors that keep a division finite for an aircraft that is not moving or a
# performance model that reports no rate; flying traffic never reaches them.
_MIN_SPEED_MS = 1.0
_MIN_RATE_MS = 0.1


def planned_legs(idx: int, targets: Sequence[Any]) -> tuple[tuple[float, float], ...]:
    """The ``(level_m, cas_ms)`` aircraft ``idx`` flies each leg of its route at,
    unless cleared otherwise: its spawn state into the first fix, then each
    fix's altitude and speed gates from where it passes them."""
    level_m, cas_ms = float(bs.traf.alt[idx]), float(bs.traf.cas[idx])
    plan = []
    for target in targets:
        plan.append((level_m, cas_ms))
        if target.alt_ft is not None:
            level_m = float(target.alt_ft) * ft
        if target.speed_kts is not None:
            cas_ms = float(target.speed_kts) * kts
    return tuple(plan)


def arrival_times(
    idx: int,
    targets: Sequence[Any],
    slacks: Sequence[Any],
    rng: np.random.Generator,
    now_s: float,
) -> tuple[float | None, ...]:
    """The time aircraft ``idx`` is due over each of ``targets``, or ``None``
    for a target with no slack (``slacks[k] is None``).

    ``targets`` are the route's fixes in order (``lat``, ``lon``, ``alt_ft``,
    ``speed_kts``, as :class:`~bluesky_sandbox.sim.queryables.WaypointTarget`);
    each slack is a number of seconds or a distribution of them.
    """
    lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
    alt_m = float(bs.traf.alt[idx])
    plan = planned_legs(idx, targets)
    climb_ms = max(float(bs.traf.perf.vsmax[idx]), _MIN_RATE_MS)
    descent_ms = max(abs(float(bs.traf.perf.vsmin[idx])), _MIN_RATE_MS)
    vmax_cas_ms = float(bs.traf.perf.vmax[idx])

    t = float(now_s)
    out: list[float | None] = []
    for target, slack, (level_m, cas_ms) in zip(targets, slacks, plan, strict=True):
        speed_ms = max(float(vcas2tas(cas_ms, level_m)), _MIN_SPEED_MS)
        _, dist_nm = kwikqdrdist(lat, lon, float(target.lat), float(target.lon))
        dist_m = float(dist_nm) * nm
        to_alt_m = alt_m if target.alt_ft is None else float(target.alt_ft) * ft
        climb = to_alt_m - alt_m
        vertical_s = abs(climb) / (climb_ms if climb > 0.0 else descent_ms)
        nominal_s = max(dist_m / speed_ms, vertical_s)
        fastest_ms = max(
            float(vcas2tas(vmax_cas_ms, 0.5 * (alt_m + to_alt_m))), _MIN_SPEED_MS
        )
        earliest_s = max(dist_m / fastest_ms, vertical_s)
        leg_s = nominal_s
        if slack is not None:
            leg_s = max(nominal_s + float(sample_scalar(slack, rng)), earliest_s)
        t += leg_s
        out.append(t if slack is not None else None)
        lat, lon, alt_m = float(target.lat), float(target.lon), to_alt_m
    return tuple(out)
