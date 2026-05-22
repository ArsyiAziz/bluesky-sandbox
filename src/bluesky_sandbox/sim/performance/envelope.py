"""Sample feasible (altitude, CAS) targets from an aircraft's flight envelope.

Used by per-aircraft waypoint sampling (``Waypoint`` envelope mode) so a goal is
reachable for the aircraft it is assigned to: altitude within the performance
ceiling, and calibrated airspeed within the speed envelope *at that altitude*
(the CAS upper limit shrinks with altitude as the Mach limit ``MMO`` takes over
from the ``VMO`` limit).

The aircraft must already exist in ``bs.traf`` (it is created before its
waypoint is assigned), so the live performance model supplies the ceiling and
the stall/min CAS; OpenAP's ``VMO``/``MMO`` give the CAS upper limit at an
arbitrary altitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import warnings
from functools import cache, lru_cache

import bluesky as bs
from bluesky.tools.aero import ft, kts, vmach2cas


# NOTE: the caches are keyed by MODEL as well as type. Keying by type alone
# looks harmless and is not: the performance model is chosen when an EnvConfig
# is built, which can happen after something has already asked about a type.
# A ``None`` cached during the default-OpenAP phase then survives the switch to
# BADA, and every later lookup returns it - the symptom is "openap ceiling data
# is unavailable" for a type the configured model knows perfectly well.

_WARNED_MODELS: set[str] = set()


def active_performance_model() -> str:
    """The performance model this process asked for.

    Not ``bs.settings.performance_model`` directly: ``bs.init()`` re-reads
    ``settings.cfg`` and overwrites it, so a BADA design reverts to whatever
    that file says as soon as the runtime initialises.
    """
    from bluesky_sandbox.config import requested_performance_model

    return requested_performance_model()


def _warn_type_data_mismatch(kind: str) -> None:
    """Say once that type data came from OpenAP while the sim flies something else.

    OpenAP is the only per-TYPE database this package can read without a BADA
    licence, so envelope limits and MTOW fall back to it. Under BADA that means
    the numbers used to sample targets are not the numbers the simulator flies
    with - survivable, but it must not be silent, or an envelope that quietly
    disagrees with the aircraft looks like a policy problem.
    """
    key = f"{active_performance_model()}:{kind}"
    if key in _WARNED_MODELS:
        return
    _WARNED_MODELS.add(key)
    warnings.warn(
        f"performance_model={active_performance_model()!r} but {kind} is read "
        f"from OpenAP's type database (the only one readable without a BADA "
        f"licence). Sampled envelopes may differ from what the simulator flies.",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class EnvelopeSample:
    """Marker for values sampled from an aircraft's feasible flight envelope."""

    alt_floor_ft: float = 1000.0


def _aircraft_limits(actype: str) -> dict | None:
    """Type limits under the model configured *right now*."""
    return _aircraft_limits_cached(active_performance_model(), actype)


@cache
def _aircraft_limits_cached(model: str, actype: str) -> dict | None:
    """Normalised type limits, from whichever provider ``model`` names.

    Falls back to OpenAP (with a one-time warning) when the configured model
    has no record of the type: OpenAP is the only per-type database readable
    without a licence, so it is better than nothing - but it must be said, or a
    sampled envelope silently disagrees with what the simulator flies.
    """
    from bluesky_sandbox.sim.performance.models import type_limits

    limits = type_limits(actype, model)
    if limits is not None and limits.get("ceiling_ft") is not None:
        return limits
    if model != "openap":
        _warn_type_data_mismatch("flight-envelope limits")
        return type_limits(actype, "openap")
    return limits


def _ceiling_ft_for_type(actype: str) -> float | None:
    """Certified ceiling in feet under the model configured right now."""
    return _ceiling_ft_cached(active_performance_model(), actype)


@cache
def _ceiling_ft_cached(model: str, actype: str) -> float | None:
    # Ask for THIS model's limits, not the currently-configured one: the model
    # is a cache key here, and re-deriving it from global state would make the
    # key a lie (and quietly return OpenAP numbers for a BADA lookup).
    limits = _aircraft_limits_cached(model, actype)
    if not limits:
        return None
    # Every provider normalises to feet on the way out, so there is nothing to
    # infer here.
    ceiling_ft = limits.get("ceiling_ft")
    return None if ceiling_ft is None else float(ceiling_ft)


def _vmax_cas_kt(actype: str, alt_ft: float) -> float | None:
    """CAS upper limit (kt) at ``alt_ft`` from VMO/MMO, or ``None`` if unavailable.

    Below the crossover altitude ``VMO`` binds; above it the Mach limit ``MMO``
    binds, and its equivalent CAS falls with altitude.
    """
    try:
        from openap import aero
    except Exception:
        return None
    limits = _aircraft_limits(actype)
    if not limits:
        return None
    try:
        vmo_kt = float(limits["VMO"])
        mmo = float(limits["MMO"])
    except Exception:
        return None
    mach_cas_kt = float(aero.mach2cas(mmo, alt_ft * aero.ft)) / aero.kts
    return min(vmo_kt, mach_cas_kt)


def feasible_alt_for_type(
    actype: str,
    rng,
    alt_floor_ft: float = 1000.0,
    alt_min_ft: float | None = None,
    alt_max_ft: float | None = None,
) -> float:
    """Draw a feasible spawn altitude for an aircraft type before creation."""
    ceiling_ft = _ceiling_ft_for_type(actype)
    if ceiling_ft is None:
        raise ValueError(
            f"Cannot sample envelope altitude for aircraft type {actype!r}: "
            f"{active_performance_model()} ceiling data is unavailable "
            f"(run `python -m bluesky_sandbox.doctor`)."
        )
    floor_ft = min(float(alt_floor_ft), ceiling_ft)
    lo = floor_ft if alt_min_ft is None else max(floor_ft, float(alt_min_ft))
    hi = ceiling_ft if alt_max_ft is None else min(ceiling_ft, float(alt_max_ft))
    if hi < lo:
        lo, hi = floor_ft, ceiling_ft
    return float(rng.uniform(lo, hi))


def fleet_ceiling_ft(actypes) -> float | None:
    """Highest altitude EVERY type in ``actypes`` can reach - the min ceiling.

    A *shared* target (a merge fix several streams cross at one level) has to be
    feasible for the whole fleet, not for one aircraft, so the binding limit is
    the least-capable type. ``None`` when no type has usable OpenAP data - the
    caller should then leave its own band alone rather than clamp to a guess.
    """
    ceilings = [
        c
        for c in (_ceiling_ft_for_type(str(t)) for t in actypes or ())
        if c is not None
    ]
    return min(ceilings) if ceilings else None


def fleet_max_cas_kt(actypes, alt_ft: float) -> float | None:
    """Fastest CAS EVERY type in ``actypes`` can hold at ``alt_ft`` - the min VMO/MMO.

    Mirrors :func:`fleet_ceiling_ft` for the speed axis: a crossing speed
    assigned at a shared fix must be holdable by the whole fleet at that
    altitude, and the Mach limit makes the CAS ceiling fall as altitude rises.
    Types whose ceiling is below ``alt_ft`` are skipped - they cannot be there at
    all, which is :func:`fleet_ceiling_ft`'s constraint, not this one. ``None``
    when no type has usable data.
    """
    limits = []
    for t in actypes or ():
        actype = str(t)
        ceiling = _ceiling_ft_for_type(actype)
        if ceiling is not None and ceiling < float(alt_ft):
            continue
        vmax = _vmax_cas_kt(actype, float(alt_ft))
        if vmax is not None:
            limits.append(vmax)
    return min(limits) if limits else None


@lru_cache(maxsize=1)
def _sim_limit_table() -> dict:
    """BlueSky's fixed-wing limit table - what the simulator actually enforces.

    ``perfoap.OpenAP`` loads this into ``bs.traf.perf``: ``hmax``, the enroute
    CAS band ``vminer``/``vmaxer``, and ``mmo``, all SI. It is a *different*
    data source from the certified placard in :func:`_aircraft_limits`, and the
    two disagree - OpenAP's aircraft profile puts the A321 ceiling at FL410
    while this table stops it at FL346, and ``perf.limits()`` clamps the
    selected altitude to the latter. Anything a whole fleet must be able to fly
    to has to respect this table, not the placard.

    Empty dict when the table cannot be loaded; the fleet helpers below then
    return ``None`` so callers leave their own bands alone rather than clamp to
    a guess.
    """
    try:
        from bluesky.traffic.performance.openap import coeff

        return dict(coeff.Coefficient().limits_fixwing)
    except Exception:
        return {}


def _sim_limits(actype: str) -> dict | None:
    return _sim_limit_table().get(str(actype))


def fleet_sim_ceiling_ft(actypes) -> float | None:
    """Highest altitude EVERY type in ``actypes`` can be *flown* to.

    The simulator-enforced counterpart of :func:`fleet_ceiling_ft`: the minimum
    over the fleet of ``perf.hmax`` (see :func:`_sim_limit_table`), which is
    what BlueSky clamps the selected altitude against. Use this - not the
    placard ceiling, which sits thousands of feet higher for much of a typical
    fleet - to bound a *shared* altitude gate, since a gate above ``hmax`` is
    permanently uncapturable for the binding type.
    """
    ceilings = [
        float(row["hmax"]) / ft
        for row in (_sim_limits(t) for t in actypes or ())
        if row is not None and "hmax" in row
    ]
    return min(ceilings) if ceilings else None


def fleet_speed_band_kt(actypes, alt_ft: float) -> tuple[float, float] | None:
    """CAS band ``(lo_kt, hi_kt)`` EVERY type in ``actypes`` can hold at ``alt_ft``.

    The fleet-wide counterpart of :func:`_speed_band_kt`, read from static
    tables so a *shared* crossing speed can be drawn before any aircraft
    exists. ``lo`` is the fastest enroute minimum over the fleet and ``hi`` the
    slowest maximum, intersecting every cap the simulator enforces or the
    placard declares: the certified VMO/MMO (:func:`fleet_max_cas_kt`), and
    BlueSky's own ``vmaxer`` and ``mmo`` from :func:`_sim_limit_table`. Types
    that cannot reach ``alt_ft`` are skipped, as in :func:`fleet_max_cas_kt`.

    ``None`` when no type has usable data, or when the caps cross so no single
    speed is holdable fleet-wide at that altitude - the caller should then
    assign no speed constraint rather than an unholdable one.
    """
    alt_ft = float(alt_ft)
    lo_kt = None
    hi_kt = fleet_max_cas_kt(actypes, alt_ft)
    for t in actypes or ():
        row = _sim_limits(t)
        if row is None:
            continue
        if float(row.get("hmax", math.inf)) / ft < alt_ft:
            continue
        vminer_kt = float(row["vminer"]) / kts
        lo_kt = vminer_kt if lo_kt is None else max(lo_kt, vminer_kt)
        caps = [float(row["vmaxer"]) / kts]
        mmo = float(row.get("mmo", 0.0))
        if mmo > 0.0:
            caps.append(float(vmach2cas(mmo, alt_ft * ft)) / kts)
        cap = min(caps)
        hi_kt = cap if hi_kt is None else min(hi_kt, cap)
    if lo_kt is None or hi_kt is None or hi_kt <= lo_kt:
        return None
    return lo_kt, hi_kt


def _speed_band_kt(acidx: int, alt_ft: float) -> tuple[float, float]:
    """Sampling band ``(vmin_kt, vmax_kt)`` the aircraft can actually HOLD at
    ``alt_ft``.

    The upper bound intersects every cap the simulator enforces or the placard
    data declares:

    * live ``perf.vmax`` - the CAS cap ``perf.limits()`` applies (OpenAP's
      phase-level ``vmaxer``); the static placard VMO alone over-promises by
      10-30 kt for most types (and OpenAP's B789 VMO data is broken outright);
    * the *simulator's own* Mach limit ``vmach2cas(perf.mmo, alt_ft)`` -
      ``limits()`` clamps TAS to ``mmo`` after the CAS clamp, and BlueSky's
      ``mmo`` is stricter than OpenAP's placard MMO for most of the fleet
      (7-41 kt lower at FL350), so at high altitude this is the binding cap;
    * the static VMO/MMO placard from :func:`_vmax_cas_kt` where available
      (realism: keeps types whose placard sits below the sim envelope, e.g.
      C550, from being assigned placard-illegal speeds).

    A target drawn above any enforced cap can never be flown, and a speed-gated
    waypoint becomes permanently uncapturable.
    """
    vmin_kt = float(bs.traf.perf.vmin[acidx]) / kts
    vmax_kt = float(bs.traf.perf.vmax[acidx]) / kts
    mmo_arr = getattr(bs.traf.perf, "mmo", None)
    if mmo_arr is not None:
        mmo = float(mmo_arr[acidx])
        if math.isfinite(mmo) and mmo > 0.0:
            vmax_kt = min(vmax_kt, float(vmach2cas(mmo, float(alt_ft) * ft)) / kts)
    static_kt = _vmax_cas_kt(bs.traf.type[acidx], float(alt_ft))
    if static_kt is not None:
        vmax_kt = min(vmax_kt, static_kt)
    if vmax_kt <= vmin_kt:
        vmin_kt = min(vmin_kt, vmax_kt)
    return vmin_kt, vmax_kt


def feasible_cas_at_alt(acidx: int, alt_ft: float, rng) -> float:
    """Draw a feasible CAS (kt) for the live aircraft at a *fixed* altitude.

    Uniform over the holdable speed envelope at ``alt_ft`` (see
    :func:`_speed_band_kt`). Used for envelope-sampled *spawn* speed, where the
    altitude is already fixed (e.g. by the region band) and only the speed is
    drawn.
    """
    vmin_kt, vmax_kt = _speed_band_kt(acidx, float(alt_ft))
    return float(rng.uniform(vmin_kt, vmax_kt))


def feasible_alt_cas(
    acidx: int,
    rng,
    alt_floor_ft: float = 1000.0,
    alt_min_ft: float | None = None,
    alt_max_ft: float | None = None,
) -> tuple[float, float]:
    """Draw a feasible ``(alt_ft, cas_kts)`` for the aircraft at ``acidx``.

    ``alt`` is uniform over ``[alt_floor_ft, ceiling]`` (the live performance
    ceiling), and ``cas`` is uniform over the holdable speed envelope at that
    altitude (see :func:`_speed_band_kt`).

    ``alt_min_ft``/``alt_max_ft`` further clamp the altitude window to a
    caller-supplied band (e.g. a sampling region's vertical extent), intersected
    with the envelope. If that intersection is empty (the band lies entirely
    outside the aircraft's envelope) the full envelope window is used instead.
    """
    ceiling_ft = float(bs.traf.perf.hmax[acidx]) / ft
    floor_ft = min(float(alt_floor_ft), ceiling_ft)
    lo = floor_ft if alt_min_ft is None else max(floor_ft, float(alt_min_ft))
    hi = ceiling_ft if alt_max_ft is None else min(ceiling_ft, float(alt_max_ft))
    if hi < lo:
        # Band lies outside the envelope -> fall back to the full envelope.
        lo, hi = floor_ft, ceiling_ft
    alt_ft = float(rng.uniform(lo, hi))

    vmin_kt, vmax_kt = _speed_band_kt(acidx, alt_ft)
    cas_kt = float(rng.uniform(vmin_kt, vmax_kt))
    return alt_ft, cas_kt
