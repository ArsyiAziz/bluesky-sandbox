"""One interface over the performance models, with a provider per model.

BlueSky can fly OpenAP or BADA, and both answer the same questions: which
aircraft types do you carry, what are this type's limits, and what is it? Each
is asked of the model BlueSky flies - BlueSky's own OpenAP data, fixed-wing and
rotorcraft (a helicopter, drones); the BADA files - not a separate library, so
a type BlueSky carries is offered, and nothing is listed by hand. Everything
else in the package asks here rather than importing a specific database, so
adding a model means adding a provider, not editing every call site.

Only *finding* the data differs enough to live elsewhere: OpenAP installs with
pip, BADA is licensed and has to be located on disk. That is what
:mod:`.bada` is for - it is not this module's BADA half.

Limits are normalized on the way out, so callers never unit-guess:

    ceiling_ft   feet          certified ceiling
    VMO          knots CAS     max operating speed
    MMO          Mach          max operating Mach
    MTOW         kilograms     max takeoff weight
"""

from __future__ import annotations

from functools import cache

from bluesky.tools.aero import ft, kts

from .bada import bada_aircraft_types, bada_coefficients

#: Rotorcraft lighter than this (kg) are drones; heavier, helicopters.
DRONE_MAX_MTOW_KG = 150.0


@cache
def _openap_model():
    """BlueSky's OpenAP coefficients - what it flies OpenAP aircraft with:
    fixed-wing types (with the synonyms it flies them as) and rotorcraft."""
    # expensive: ~3 s - OpenAP's import and BlueSky's data
    from bluesky.traffic.performance.openap import coeff  # noqa: PLC0415

    return coeff.Coefficient()


def _openap_types() -> frozenset[str]:
    model = _openap_model()
    return frozenset([*model.actypes_fixwing, *model.actypes_rotor])


def _openap_limits(actype: str) -> dict | None:
    rotor = _openap_model().limits_rotor.get(str(actype).upper())
    if rotor is not None:
        acs = _openap_model().acs_rotor.get(str(actype).upper(), {})
        return {
            "ceiling_ft": float(rotor["hmax"]) / ft,
            # Its top speed, low and slow enough that TAS is CAS near enough.
            "VMO": float(rotor["vmax"]) / kts,
            "MMO": None,  # no Mach limit
            "MTOW": acs.get("mtow"),
        }
    try:
        # expensive: openap costs ~1.8s to import
        import openap  # noqa: PLC0415

        raw = dict(openap.prop.aircraft(actype)["limits"])  # copy: openap caches its own
    except Exception:  # noqa: BLE001 - unknown type or unusable database
        return None
    ceiling_m = raw.get("ceiling")
    return {
        # OpenAP reports ceiling in meters for every type it carries (checked
        # across all 37: 11 300-16 000). Converted here so no caller has to
        # infer the unit from magnitude, which is what this used to do.
        "ceiling_ft": None if ceiling_m is None else float(ceiling_m) / ft,
        "VMO": raw.get("VMO"),
        "MMO": raw.get("MMO"),
        "MTOW": raw.get("MTOW"),
    }


def _bada_types() -> frozenset[str]:
    return bada_aircraft_types()


def _bada_limits(actype: str) -> dict | None:
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


def _openap_info(actype: str) -> dict | None:
    model = _openap_model()
    key = str(actype).upper()
    if key in model.acs_rotor:
        acs = model.acs_rotor[key]
        return {"name": acs.get("name"), "lift": "rotor", "engine": "turboshaft",
                "engines": acs.get("n_engines"), "mtow_kg": acs.get("mtow"), "stand_in": None}
    acs = model.acs_fixwing.get(key)
    if acs is None:
        return None
    engine = acs.get("engine") or {}
    stand_in = None if key.lower() in _openap_native() else acs.get("aircraft")
    return {
        # A stand-in's data is another type's: its name is that type's, not this one's.
        "name": None if stand_in else acs.get("aircraft"),
        "lift": "fixed-wing",
        "engine": engine.get("type"),
        "engines": engine.get("number"),
        "mtow_kg": acs.get("mtow"),
        # Flown on another type's data (BlueSky's synonyms): that type.
        "stand_in": stand_in,
    }


@cache
def _openap_native() -> frozenset[str]:
    """The fixed-wing types OpenAP has data of its own for."""
    import openap  # noqa: PLC0415

    return frozenset(t.lower() for t in openap.prop.available_aircraft(use_synonym=False))


def _bada_info(actype: str) -> dict | None:
    data = bada_coefficients(actype)
    if data is None:
        return None
    mtow_t = getattr(data, "m_max", None)
    return {
        "name": None,
        "lift": "fixed-wing",
        "engine": str(getattr(data, "engtype", "") or "").lower() or None,
        "engines": None,
        "mtow_kg": float(mtow_t) * 1000.0 if mtow_t else None,
        "stand_in": None,
    }


#: model name -> (available types, per-type limits, per-type description)
_PROVIDERS = {
    "openap": (_openap_types, _openap_limits, _openap_info),
    "bada": (_bada_types, _bada_limits, _bada_info),
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
    """Normalized limits for ``actype`` under ``model``, or ``None`` if unknown."""
    model = model.lower()
    if model not in _PROVIDERS:
        return None
    return _PROVIDERS[model][1](str(actype).upper())


def type_info(actype: str, model: str) -> dict | None:
    """What ``actype`` is under ``model``: its name, lift (``fixed-wing`` or
    ``rotor``), engine, engine count, MTOW (kg), the type whose data it is
    flown on where not its own (``stand_in``) - and its :func:`type_tags`.
    ``None`` where the model does not carry it."""
    model = model.lower()
    if model not in _PROVIDERS:
        return None
    info = _PROVIDERS[model][2](str(actype).upper())
    if info is None:
        return None
    return {**info, "type": str(actype).upper(), "tags": type_tags(info)}


#: ICAO wake turbulence categories by MTOW (kg): up to each bound.
_WAKE = ((7_000.0, "Light"), (136_000.0, "Medium"), (500_000.0, "Heavy"), (float("inf"), "Super"))
_ENGINES = {"turbofan": "Jet", "jet": "Jet", "turboprop": "Turboprop", "piston": "Piston", "turboshaft": "Turboshaft"}


def type_tags(info: dict) -> list[str]:
    """Words for what a type is, from its model data: what it is (fixed-wing,
    helicopter, drone), its propulsion, its ICAO wake category and its
    mass; and the type it is flown on, where not its own."""
    mtow = info.get("mtow_kg")
    rotor = info.get("lift") == "rotor"
    drone = rotor and mtow is not None and float(mtow) < DRONE_MAX_MTOW_KG
    tags = ["Drone" if drone else "Helicopter" if rotor else "Fixed-wing"]
    if drone:
        if info.get("engines"):
            tags.append(f"{int(info['engines'])} rotors")
    else:
        engine = _ENGINES.get(str(info.get("engine") or "").lower())
        if engine:
            tags.append(engine)
        if mtow:
            tags.append(f"Wake {next(name for bound, name in _WAKE if float(mtow) <= bound)}")
    if mtow:
        mtow = float(mtow)
        tags.append(f"{mtow / 1000:.0f} t" if mtow >= 10_000 else f"{mtow / 1000:.1f} t" if mtow >= 1000 else f"{mtow:g} kg")
    if info.get("stand_in"):
        tags.append(f"Flown as {info['stand_in']}")
    return tags
