"""One interface over the performance models, with a provider per model.

BlueSky can fly OpenAP or BADA, and both answer the same two questions: which
aircraft types do you carry, and what are this type's limits? Everything else
in the package asks here rather than importing a specific database, so adding a
model means adding a provider, not editing every call site.

Only *finding* the data differs enough to live elsewhere: OpenAP installs with
pip, BADA is licensed and has to be located on disk. That is what
:mod:`.bada` is for - it is not this module's BADA half.

Limits are normalised on the way out, so callers never unit-guess:

    ceiling_ft   feet          certified ceiling
    VMO          knots CAS     max operating speed
    MMO          Mach          max operating Mach
    MTOW         kilograms     max takeoff weight
"""

from __future__ import annotations

from functools import cache

from bluesky.tools.aero import ft


def _openap_types() -> frozenset[str]:
    import openap

    return frozenset(openap.prop.available_aircraft(use_synonym=True))


def _openap_limits(actype: str) -> dict | None:
    try:
        import openap

        raw = dict(openap.prop.aircraft(actype)["limits"])  # copy: openap caches its own
    except Exception:  # noqa: BLE001 - unknown type or unusable database
        return None
    ceiling_m = raw.get("ceiling")
    return {
        # OpenAP reports ceiling in metres for every type it carries (checked
        # across all 37: 11 300-16 000). Converted here so no caller has to
        # infer the unit from magnitude, which is what this used to do.
        "ceiling_ft": None if ceiling_m is None else float(ceiling_m) / ft,
        "VMO": raw.get("VMO"),
        "MMO": raw.get("MMO"),
        "MTOW": raw.get("MTOW"),
    }


def _bada_types() -> frozenset[str]:
    from .bada import bada_aircraft_types

    return bada_aircraft_types()


def _bada_limits(actype: str) -> dict | None:
    from .bada import bada_coefficients

    data = bada_coefficients(actype)
    if data is None:
        return None
    mtow_t = getattr(data, "m_max", None)
    return {
        # ``h_MO`` (max operating altitude), not ``h_max`` (max at MTOW): the
        # envelope sampler asks what the TYPE is certified to, not what one
        # aircraft can reach at today's weight. Already feet.
        "ceiling_ft": float(data.h_MO),
        "VMO": float(data.VMO),
        "MMO": float(data.MMO),
        # BADA masses are tonnes; OpenAP's MTOW is kilograms.
        "MTOW": float(mtow_t) * 1000.0 if mtow_t else None,
    }


#: model name -> (available types, per-type limits)
_PROVIDERS = {
    "openap": (_openap_types, _openap_limits),
    "bada": (_bada_types, _bada_limits),
}

MODELS = tuple(_PROVIDERS)


#: Keys the envelope sampler needs: without a ceiling there is no altitude to
#: draw, without a speed limit no CAS.
_REQUIRED_BOUNDS = ("ceiling_ft", "VMO")


@cache
def available_types(model: str) -> frozenset[str]:
    """Every ICAO type ``model`` carries, lowercased.

    Not filtered by whether bounds exist. A model can fly a type it has no
    per-type ceiling record for - 22 of OpenAP's do - and 36 existing task
    designs list exactly those. Only the envelope sampler needs the bounds, so
    only it (and :func:`spawnable_types`) may insist on them.
    """
    model = model.lower()
    if model not in _PROVIDERS:
        raise RuntimeError(
            f"Unknown BlueSky performance model {model!r}. Known: {', '.join(MODELS)}."
        )
    return frozenset(str(t).lower() for t in _PROVIDERS[model][0]())


@cache
def spawnable_types(model: str) -> frozenset[str]:
    """Types ``model`` can also supply envelope bounds for.

    What a chooser should offer: picking one of these means an envelope-sampled
    spawn can actually draw an altitude and speed. Picking outside them is
    legal but only works for designs that never envelope-sample.
    """
    limits = _PROVIDERS[model.lower()][1]
    return frozenset(
        t for t in available_types(model)
        if (lim := limits(t.upper())) and all(lim.get(k) is not None for k in _REQUIRED_BOUNDS)
    )


def type_limits(actype: str, model: str) -> dict | None:
    """Normalised limits for ``actype`` under ``model``, or ``None`` if unknown."""
    model = model.lower()
    if model not in _PROVIDERS:
        return None
    return _PROVIDERS[model][1](str(actype).upper())
