"""A design's test cases: situations placed by hand, and the value each field
should give in them - the data the Tests tab edits, checked here and
generated into the task package as ``cases.py`` for
:mod:`bluesky_sandbox.checks` to run::

    "tests": {
      "situations": [
        {"name": "head-on", "seed": 0, "aircraft": [
          {"acid": "OWN", "lat": 0, "lon": 0, "track_deg": 90, "gs_kts": 250,
           "alt_ft": 10000, "actype": "B744"},
          {"acid": "INTR", "relative_to": "OWN", "distance_nm": 10, "bearing_deg": 90,
           "track_deg": 270, "gs_kts": 250, "alt_ft": 10000, "actype": "B744"}]}],
      "cases": [
        {"situation": "head-on", "field": {"field": "ClosingRateKts"}, "of": "INTR",
         "expected": 500, "tolerance": {"rel": 0.01}}]
    }

An aircraft's keys and a case's are those of :class:`~bluesky_sandbox.checks.Aircraft`
and :class:`~bluesky_sandbox.checks.Case` - read from them, so they cannot drift.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance

from .emit import _Emitter, field_imports
from .spec import DesignSpec, FieldRef, SpecError

__all__ = ["cases_module", "design_cases", "design_situations", "situation_of"]


def _names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls) if f.name != "_"}


def design_situations(spec: DesignSpec) -> list[Situation]:
    """The design's situations, each as :mod:`bluesky_sandbox.checks` takes it."""
    out = [situation_of(d, i) for i, d in enumerate((spec.tests or {}).get("situations") or [])]
    names = [s.name for s in out]
    if len(names) != len(set(names)):
        raise SpecError("tests: two situations share a name")
    return out


def situation_of(d: Any, index: int) -> Situation:
    """Situation ``d`` - the ``index``-th - as :mod:`bluesky_sandbox.checks`
    takes it; a :class:`SpecError` saying where, for one that is not one."""
    where = f"tests: situation {d.get('name') or index + 1!r}" if isinstance(d, dict) else f"tests: situation {index + 1}"
    if not isinstance(d, dict) or not d.get("name"):
        raise SpecError(f"{where} needs a name")
    needed = [
        f.name
        for f in dataclasses.fields(Aircraft)
        if f.name != "_" and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    aircraft = []
    for k, a in enumerate(d.get("aircraft") or []):
        what = f"{where}, aircraft {a.get('acid') or k + 1!r}" if isinstance(a, dict) else f"{where}, aircraft {k + 1}"
        if not isinstance(a, dict):
            raise SpecError(f"{what} is not a mapping")
        unknown = set(a) - _names(Aircraft)
        if unknown:
            raise SpecError(f"{what}: no {', '.join(sorted(unknown))}")
        missing = [name for name in needed if a.get(name) is None]
        if missing:
            raise SpecError(f"{what} needs {', '.join(missing)}")
        try:
            aircraft.append(Aircraft(**a))
        except (TypeError, ValueError) as e:
            raise SpecError(f"{what}: {e}") from e
    try:
        return Situation(str(d["name"]), tuple(aircraft), seed=int(d.get("seed") or 0))
    except ValueError as e:
        raise SpecError(f"{where}: {e}") from e


def design_cases(spec: DesignSpec) -> list[tuple[dict[str, Any], FieldRef]]:
    """The design's cases, each with its field reference - checked against
    its situations. Each case's dict is as stored; its field is resolved
    where the cases run."""
    situations = {s.name: s for s in design_situations(spec)}
    keys = _names(Case)
    out = []
    for i, d in enumerate((spec.tests or {}).get("cases") or []):
        where = f"tests: case {i + 1}"
        if not isinstance(d, dict):
            raise SpecError(f"{where} is not a mapping")
        unknown = set(d) - keys
        if unknown:
            raise SpecError(f"{where}: no {', '.join(sorted(unknown))}")
        missing = [k for k in ("situation", "field", "expected", "tolerance") if d.get(k) is None]
        if missing:
            raise SpecError(f"{where} needs {', '.join(missing)}")
        situation = situations.get(d["situation"])
        if situation is None:
            raise SpecError(f"{where}: no situation is named {d['situation']!r}")
        placed = {a.acid for a in situation.aircraft}
        for role in ("aircraft", "of"):
            if d.get(role) and d[role] not in placed:
                raise SpecError(f"{where}: situation {situation.name!r} places no aircraft {d[role]!r}")
        try:
            Tolerance(**d["tolerance"])
        except TypeError as e:
            raise SpecError(f"{where}: tolerance - {e}") from e
        out.append((d, FieldRef.from_dict(d["field"])))
    return out


def cases_module(spec: DesignSpec, package: str | None = None) -> str | None:
    """The design's cases as a Python module - ``SITUATIONS`` and ``CASES`` -
    or None for a design without tests."""
    if not spec.tests:
        return None
    situations = design_situations(spec)
    cases = design_cases(spec)
    em = _Emitter(package=package)
    case_lines = [_case(d, em.field(ref, "obs")) for d, ref in cases]
    situation_lines = [_situation(s) for s in situations]
    return f'''"""This design's test cases: situations placed by hand, and the value each
field should give in them. Run them with ``bluesky_sandbox.checks.run_cases``.
"""

from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance
{field_imports(em, package)}

SITUATIONS = (
{"".join(situation_lines)})

CASES = (
{"".join(case_lines)})
'''


def _situation(s: Situation) -> str:
    aircraft = "".join(f"            {_aircraft(a)},\n" for a in s.aircraft)
    return f"    Situation(\n        {s.name!r},\n        (\n{aircraft}        ),\n        seed={s.seed!r},\n    ),\n"


def _aircraft(a: Aircraft) -> str:
    default = {f.name: f.default for f in dataclasses.fields(Aircraft)}
    given = [
        f"{f.name}={getattr(a, f.name)!r}"
        for f in dataclasses.fields(Aircraft)
        if f.name not in ("_", "acid") and getattr(a, f.name) != default.get(f.name, dataclasses.MISSING)
    ]
    return f"Aircraft({a.acid!r}, {', '.join(given)})"


def _case(d: dict[str, Any], field_expr: str) -> str:
    tolerance = ", ".join(f"{k}={float(v)!r}" for k, v in sorted(d["tolerance"].items()))
    extra = "".join(f", {k}={d[k]!r}" for k in ("of", "aircraft", "note") if d.get(k))
    return f"    Case({d['situation']!r}, {field_expr}, {d['expected']!r}, Tolerance({tolerance}){extra}),\n"
