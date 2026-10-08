"""Test cases: a situation set up by hand, a field, the value it should give.

A field compared with itself - its bulk code against its one-aircraft code -
shows its code agrees; it cannot show the field means what it should: a
wrong sign or frame is written the same way twice. A case can. It places
aircraft where the answer can be worked out on paper, and says it::

    head_on = Situation("head-on", (
        Aircraft("OWN", lat=0.0, lon=0.0, track_deg=90, gs_kts=250, alt_ft=10_000, actype="B744"),
        Aircraft("INTR", relative_to="OWN", distance_nm=10, bearing_deg=90,
                 track_deg=270, gs_kts=250, alt_ft=10_000, actype="B744"),
    ))
    run_cases(env, [head_on], [
        Case("head-on", ClosingRateKts(), 500.0, Tolerance(rel=0.01), of="INTR"),
    ]).assert_ok()

Each case runs in ``env``'s design geometry with none of its traffic
(:func:`~.placement.without_traffic`), in a situation of its own; how a field
gives its value there is the field's to say (``case_value``), so any field -
a built-in, a custom one - is read the same way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np

from bluesky_sandbox.interface.fields.base import EnvBound

from .placement import Situation, place, without_traffic

__all__ = ["Case", "CaseResult", "CasesReport", "Tolerance", "run_cases"]


@dataclass(frozen=True)
class Tolerance:
    """How far a value may be from the one expected: ``abs`` in the field's
    unit, or ``rel`` of the expected value - whichever is wider. Both 0 is
    exact. A case always states its own: what is close enough depends on what
    the field measures."""

    abs: float = 0.0
    rel: float = 0.0

    def holds(self, got: Any, expected: Any) -> bool:
        got_a = np.asarray(got, dtype=np.float64).reshape(-1)
        want = np.asarray(expected, dtype=np.float64).reshape(-1)
        if got_a.shape != want.shape:
            return False
        allowed = np.maximum(float(self.abs), float(self.rel) * np.abs(want))
        return bool(np.all(np.abs(got_a - want) <= allowed))


@dataclass(frozen=True)
class Case:
    """``field``, read in situation ``situation`` for ``aircraft`` (by default
    its ownship) about ``of`` (a pair field's other aircraft), should give
    ``expected`` within ``tolerance``."""

    situation: str
    field: Any
    expected: Any
    tolerance: Tolerance
    _: KW_ONLY
    of: str | None = None
    aircraft: str | None = None
    note: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: Case
    got: Any = None
    ok: bool = False
    #: Why it could not be read at all, where it could not.
    error: str | None = None

    def __str__(self) -> str:
        c = self.case
        what = f"{c.situation}: {type(c.field).__name__}" + (f" of {c.of}" if c.of else "")
        if self.error is not None:
            return f"{what} - {self.error}"
        mark = "ok" if self.ok else "FAILED"
        return f"{what} = {_fmt(self.got)}, expected {_fmt(c.expected)} - {mark}"


@dataclass(frozen=True)
class CasesReport:
    results: tuple[CaseResult, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.ok)

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError(
                f"{len(self.failures)} of {len(self.results)} cases failed:\n"
                + "\n".join(f"  {r}" for r in self.failures)
            )

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.results)


def run_cases(env: Any, situations: Iterable[Situation], cases: Iterable[Case]) -> CasesReport:
    """Each case, in its situation set up afresh in ``env``. ``env`` is left in
    the last situation: reset it before using it again."""
    by_name = {s.name: s for s in situations}
    results = []
    with without_traffic(env):
        for case in cases:
            results.append(_run(env, by_name, case))
    return CasesReport(tuple(results))


def _run(env: Any, situations: dict[str, Situation], case: Case) -> CaseResult:
    situation = situations.get(case.situation)
    if situation is None:
        return CaseResult(case, error=f"no situation is named {case.situation!r}")
    names = [case.aircraft or situation.ownship, *([case.of] if case.of else [])]
    missing = [n for n in names if n not in {a.acid for a in situation.aircraft}]
    if missing:
        return CaseResult(case, error=f"situation {situation.name!r} places no aircraft {missing[0]!r}")
    # A case that cannot be read says why; the rest still run.
    try:
        at = place(env, situation)
    except Exception as e:  # noqa: BLE001
        return CaseResult(case, error=f"situation {situation.name!r} could not be set up: {type(e).__name__}: {e}")
    try:
        field = case.field.bind_env(env) if isinstance(case.field, EnvBound) else case.field
        got = field.case_value(at[names[0]], other=at[case.of] if case.of else None)
    except Exception as e:  # noqa: BLE001
        return CaseResult(case, error=f"{type(e).__name__}: {e}")
    return CaseResult(case, got=got, ok=case.tolerance.holds(got, case.expected))


def _fmt(value: Any) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return f"{array[0]:g}" if array.size == 1 else "[" + ", ".join(f"{v:g}" for v in array) + "]"
