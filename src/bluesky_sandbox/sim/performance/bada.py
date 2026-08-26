"""Locating the user's BADA data.

BADA (EUROCONTROL's Base of Aircraft Data) is licensed and cannot be
redistributed, so it ships with neither this package nor BlueSky - BlueSky
carries only an empty folder and a README. A user with a licence installs their
own copy.

Where it goes matters once this is a pip-installed package. BlueSky searches
its resource roots in order, the user data directory (``~/bluesky``) first and
the installed package's ``resources`` second, so both locations work. Only the
first survives an upgrade: reinstalling the wheel replaces the package tree,
taking any licensed files put there with it, and writing into site-packages may
need root besides. So this points at the user directory, even though BlueSky's
bundled README suggests its own folder.

``bs.resource()`` is the wrong function to ask. It returns the first root that
*has* the files, which before installation is the read-only bundled one - it
answers "where are they" and the question here is "where do I put them".
"""

from __future__ import annotations

import importlib
from pathlib import Path

import bluesky as bs

#: Files a real BADA 3.x distribution contains. The bundled directory holds a
#: README and nothing else, so "directory is non-empty" is not the test.
_BADA_MARKERS = ("SYNONYM.NEW", "*.OPF", "*.APF")


def _user_resource_root() -> Path:
    """BlueSky's writable resource root, whether or not ``bs.init`` has run."""
    cfg = getattr(bs.settings, "_cfgfile", None)
    if cfg:
        parent = Path(str(cfg)).parent
        # Before ``bs.init`` runs, ``_cfgfile`` is a bare filename whose parent
        # is ".", which would make this return a relative path - and a hint
        # that says "put your files in performance/BADA" is worse than no hint.
        if parent.is_absolute() and parent.name != "resources":
            return parent
    return Path.home() / "bluesky"


def ensure_user_resource_root() -> Path:
    """Register ``~/bluesky`` as a BlueSky resource root, if it is not already.

    BADA is resolved through ``bs.resource()``, which searches the roots
    registered on ``bs.resource.path``. Out of the box that is the installed
    package alone; ``bs.init()`` adds the user directory, but init also loads
    the nav database and plugins, which is far too much to require before
    asking "which aircraft types exist". Inserting the root directly is the
    cheap half of what init would have done, and is idempotent.

    Returns the root, registered or not.
    """
    root = _user_resource_root()
    try:
        already = {str(b) for b in bs.resource.path.bases()}
        if root.is_dir() and str(root) not in already:
            bs.resource.path.insertpath(root, 0)
    except Exception:  # noqa: BLE001 - a missing hook must not break lookups
        pass
    return root


def bada_data_dir() -> Path:
    """Where a user's BADA files belong."""
    rel = getattr(bs.settings, "perf_path_bada", "performance/BADA")
    return ensure_user_resource_root() / rel


def bada_available() -> bool:
    """True when a real BADA database is installed in any search root."""
    roots = [bada_data_dir()]
    try:
        roots.extend(
            Path(str(base)) / getattr(bs.settings, "perf_path_bada", "performance/BADA")
            for base in bs.resource.path.bases()
        )
    except Exception:
        pass
    for d in roots:
        if d.is_dir() and any(next(d.glob(m), None) for m in _BADA_MARKERS):
            return True
    return False


def bada_install_hint() -> str:
    """One line telling the user what to do, with the real path."""
    return (
        f"BADA is licensed and ships with neither this package nor BlueSky. "
        f"Extract your licensed BADA 3.x files into {bada_data_dir()} "
        f"(create the directory if needed), or use performance_model='openap', "
        f"which is bundled."
    )


#: Where BlueSky's BADA implementation lives, across versions. The class was
#: ``PerfBADA`` historically and is ``BADA`` in current BlueSky.
_PERFBADA_MODULES = (
    "bluesky.traffic.performance.bada.perfbada",
    "bluesky.bs_traf.performance.bada.perfbada",
)
_PERFBADA_CLASSES = ("BADA", "PerfBADA")


def load_perf_bada():
    """Import and return BlueSky's BADA performance class.

    Note the ordering requirement: BADA data is found through
    ``bs.resource()``, whose search roots only include the user directory once
    ``bs.init()`` has run. Importing this beforehand looks in the installed
    package alone and reports the data as missing when it is simply not
    registered yet.
    """
    ensure_user_resource_root()
    errors = []
    for mod in _PERFBADA_MODULES:
        try:
            module = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - any failure means "not usable"
            errors.append(f"{mod}: {e}")
            continue
        for cls in _PERFBADA_CLASSES:
            if hasattr(module, cls):
                return getattr(module, cls)
        errors.append(f"{mod}: none of {_PERFBADA_CLASSES} present")
    raise ImportError("; ".join(errors))


def bada_aircraft_types() -> frozenset[str]:
    """ICAO types the installed BADA database covers, lowercased.

    Read from the coefficient tables rather than the performance class, which
    exposes no type listing. ``coeff_bada.init()`` is what populates them - the
    tables are empty until it runs.
    """
    load_perf_bada()  # surfaces a missing/broken install with the right message
    for mod in ("bluesky.traffic.performance.bada.coeff_bada",
                "bluesky.bs_traf.performance.bada.coeff_bada"):
        try:
            coeff = importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            continue
        if not getattr(coeff, "synonyms", None):
            coeff.init()
        # ``synonyms`` maps ICAO types onto coefficient sets and is the
        # user-facing list. ``accoeffs`` is deliberately NOT merged in: its keys
        # are BADA's own file stems (``GA____``, ``TP2M__``, 6 of them covering
        # these 136 types), which are not aircraft and carry no per-type bounds.
        types = set(getattr(coeff, "synonyms", {}) or {})
        if types:
            return frozenset(t.lower() for t in types)
    raise RuntimeError("BADA coefficient tables are empty after init.")


def bada_coefficients(actype: str):
    """BADA's ``ACData`` for ``actype``, or ``None`` if the tables lack it.

    The raw record; :mod:`.models` turns it into normalised limits.
    """
    try:
        load_perf_bada()  # registers the user resource root
        for mod in ("bluesky.traffic.performance.bada.coeff_bada",
                    "bluesky.bs_traf.performance.bada.coeff_bada"):
            try:
                coeff = importlib.import_module(mod)
            except Exception:  # noqa: BLE001
                continue
            if not getattr(coeff, "synonyms", None):
                coeff.init()
            found = coeff.getCoefficients(str(actype).upper())
            # On an unknown type BlueSky returns ``(False, "<message>")`` - a
            # truthy 2-tuple whose second element is a string, not ACData. Test
            # for the data, not for the tuple.
            data = found[1] if isinstance(found, tuple) and len(found) > 1 else found
            return data if hasattr(data, "h_MO") else None
    except Exception:  # noqa: BLE001 - an unusable database is "no data"
        return None
    return None
