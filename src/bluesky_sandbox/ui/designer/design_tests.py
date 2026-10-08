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
import re
from typing import Any

from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance

from .emit import _Emitter, field_imports
from .spec import DesignSpec, FieldRef, SpecError

__all__ = [
    "TEST_FILE",
    "cases_module",
    "design_cases",
    "design_situations",
    "design_test_files",
    "situation_of",
    "package_test_files",
]

#: What a design's own test file may be named: pytest finds it by the prefix.
TEST_FILE = re.compile(r"^test_\w+\.py$")
#: The files the package's tests folder holds already: a design's own may not.
_GENERATED = ("conftest.py", "test_design.py")


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


def design_cases(spec: DesignSpec) -> list[tuple[dict[str, Any], dict[str, FieldRef]]]:
    """The design's cases, each with its field references (``field``, and
    ``apply`` where it applies an action) - checked against its situations.
    Each case's dict is as stored; its fields are resolved where it runs."""
    design_test_files(spec)
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
        missing = [k for k in _required(Case) if d.get(k) is None]
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
            tolerance = Tolerance(**d["tolerance"])
        except TypeError as e:
            raise SpecError(f"{where}: tolerance - {e}") from e
        # Built as the checks build it - its own rules (an action and a value,
        # or neither; flying forward) - with the references still as written.
        optional = {k: v for k, v in d.items() if k not in _required(Case)}
        try:
            Case(d["situation"], d["field"], d["expected"], tolerance, **optional)
        except (TypeError, ValueError) as e:
            raise SpecError(f"{where}: {e}") from e
        refs = {k: FieldRef.from_dict(d[k]) for k in _REFERENCES if d.get(k) is not None}
        out.append((d, refs))
    return out


#: A case's keys that name a field - written as field references.
_REFERENCES = ("field", "apply")


def _required(cls: type) -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(cls)
        if f.name != "_" and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]


def design_test_files(spec: DesignSpec) -> dict[str, str]:
    """The design's own test files - pytest modules, by name - checked."""
    tests = spec.tests or {}
    unknown = set(tests) - {"situations", "cases", "files"}
    if unknown:
        raise SpecError(f"tests: no {', '.join(sorted(unknown))}")
    files = tests.get("files") or {}
    if not isinstance(files, dict):
        raise SpecError("tests: files are a mapping, name to source")
    for name, source in files.items():
        if not TEST_FILE.match(str(name)):
            raise SpecError(f"tests: a test file is named test_<name>.py, not {name!r}")
        if name in _GENERATED:
            raise SpecError(f"tests: {name} is the package's own - name the file differently")
        if not isinstance(source, str):
            raise SpecError(f"tests: {name} is not source code")
    return dict(files)


def package_test_files(spec: DesignSpec, package: str) -> dict[str, str]:
    """The design's tests, as files of its generated package (paths relative
    to it): ``cases.py``, and a ``tests`` folder - a ``design_env`` fixture,
    a test for its field checks and each case, and its own test files - so
    ``pytest`` runs them all. None for a design without tests."""
    if not spec.tests:
        return {}
    files = design_test_files(spec)
    out = {"cases.py": cases_module(spec, package) or "", "tests/conftest.py": _conftest(package)}
    out["tests/test_design.py"] = _design_test(package)
    out.update({f"tests/{name}": source for name, source in files.items()})
    return out


def _conftest(package: str) -> str:
    return f'''"""Fixtures for this design's tests. Run them with ``pytest`` from the folder
holding this package, or ``bluesky-sandbox design test`` on the design.
"""

import pytest

from {package} import Env


@pytest.fixture(scope="session")
def design_env():
    """The design's env, built once for the session. Reset it as a test needs:
    the checks reset it themselves."""
    env = Env(render_mode=None)
    yield env.unwrapped
    env.close()
'''


def _design_test(package: str) -> str:
    return f'''"""The design's own checks: each field against itself, and each test case
(``cases.py``)."""

import warnings

import pytest

from bluesky_sandbox.checks import check_fields, run_cases
from {package}.cases import CASES, SITUATIONS


def test_each_field_agrees_with_itself(design_env):
    report = check_fields(design_env)
    # What is worth knowing without failing - a field outside its bounds.
    for r in report.results:
        for note in r.notes:
            warnings.warn(f"{{r.field}}: {{note}}", stacklevel=1)
    report.assert_ok()


@pytest.mark.parametrize(
    "case", CASES, ids=[f"{{i + 1}}-{{c.situation}}-{{type(c.field).__name__}}" for i, c in enumerate(CASES)]
)
def test_case(design_env, case):
    run_cases(design_env, SITUATIONS, [case]).assert_ok()
'''


def cases_module(spec: DesignSpec, package: str | None = None) -> str | None:
    """The design's cases as a Python module - ``SITUATIONS`` and ``CASES`` -
    or None for a design without tests."""
    if not spec.tests:
        return None
    situations = design_situations(spec)
    cases = design_cases(spec)
    em = _Emitter(package=package)
    case_lines = [_case(d, {k: _expr(em, ref) for k, ref in refs.items()}) for d, refs in cases]
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


def _expr(em: _Emitter, ref: FieldRef) -> str:
    """``ref`` as code: from the actions module where it names an action."""
    from bluesky_sandbox.interface.fields import actions  # noqa: PLC0415

    return em.field(ref, "act" if hasattr(actions, ref.name) else "obs")


def _case(d: dict[str, Any], exprs: dict[str, str]) -> str:
    tolerance = ", ".join(f"{k}={float(v)!r}" for k, v in sorted(d["tolerance"].items()))
    required = _required(Case)
    extra = "".join(
        f", {f.name}={exprs.get(f.name) or repr(d[f.name])}"
        for f in dataclasses.fields(Case)
        if f.name not in (*required, "_") and d.get(f.name) not in (None, "", f.default)
    )
    return f"    Case({d['situation']!r}, {exprs['field']}, {d['expected']!r}, Tolerance({tolerance}){extra}),\n"
