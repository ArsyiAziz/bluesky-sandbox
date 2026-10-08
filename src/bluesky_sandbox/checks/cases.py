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

import bluesky as bs
import numpy as np
from bluesky.stack.stackbase import Stack

from bluesky_sandbox.core import services
from bluesky_sandbox.core.services import value_on_grid
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
    ``expected`` within ``tolerance``.

    First, optionally, action ``apply`` is given ``value`` (in its own unit,
    as a step applies it) and BlueSky takes the command. An action read as
    ``field`` gives what it holds: apply +1,000 ft and read it, for the level
    it commands. Nothing is flown: what the field says is tested, not how
    BlueSky flies it."""

    situation: str
    field: Any
    expected: Any
    tolerance: Tolerance
    _: KW_ONLY
    of: str | None = None
    aircraft: str | None = None
    apply: Any = None
    value: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if (self.apply is None) != (self.value is None):
            raise ValueError("a case applies an action and a value, or neither")


@dataclass(frozen=True)
class CaseResult:
    case: Case
    got: Any = None
    ok: bool = False
    #: Why it could not be read at all, where it could not.
    error: str | None = None
    #: What it saw on the way, in order: the action given and set, the
    #: command BlueSky took, what the action holds, the value raw and
    #: normalized - where the case went wrong, when it did.
    saw: tuple[str, ...] = ()

    def __str__(self) -> str:
        c = self.case
        what = f"{c.situation}: {type(c.field).__name__}" + (f" of {c.of}" if c.of else "")
        if self.error is not None:
            line = f"{what} - {self.error}"
        else:
            mark = "ok" if self.ok else "FAILED"
            line = f"{what} = {_fmt(self.got)}, expected {_fmt(c.expected)} - {mark}"
        return line + (f" ({' · '.join(self.saw)})" if self.saw and not self.ok else "")


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
    saw: list[str] = []
    try:
        if case.apply is not None:
            action = _bound(case.apply, env)
            idx = at[names[0]]
            queued = len(Stack.cmdstack)
            set_to = value_on_grid(action, case.value, idx)
            given = f"gave {_fmt(case.value)}" + ("" if _same(set_to, case.value) else f" → on its grid {_fmt(set_to)}")
            saw.append(given)
            if not env.apply_action(idx, action, case.value):
                return CaseResult(case, error=f"{type(action).__name__} was held back (a mask on it)", saw=tuple(saw))
            saw += [f"command {line}" for line, _sender in Stack.cmdstack[queued:]]
            bs.stack.process()
            held = getattr(action, "held", None)
            if held is not None:
                saw.append(f"holds {_fmt(held(idx))}" if held(idx) is not None else "holds nothing")
        field = _bound(case.field, env)
        own, other = at[names[0]], at[case.of] if case.of else None
        got = field.case_value(own, other=other)
        # Reading the action it applied is what it holds - said above.
        if got is not None and not (case.apply is not None and type(case.field) is type(case.apply)):
            saw.append(_reading(field, got, own))
    except Exception as e:  # noqa: BLE001
        return CaseResult(case, error=f"{type(e).__name__}: {e}", saw=tuple(saw))
    if got is None:
        return CaseResult(case, error=f"{type(case.field).__name__} gives no value here (it holds no clearance)", saw=tuple(saw))
    return CaseResult(case, got=got, ok=case.tolerance.holds(got, case.expected), saw=tuple(saw))


def _reading(field: Any, got: Any, own: int) -> str:
    """``got`` as the field read it - and normalized, for an observation
    with a normalizer (what the policy sees)."""
    line = f"{type(field).__name__} raw {_fmt(got)}"
    if getattr(field, "normalizer", None) is not None and callable(getattr(field, "values_and_bounds", None)):
        line += f" → normalized {_fmt(services._normalize_field_value(field, got, own))}"
    return line


def _same(a: Any, b: Any) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


def _bound(field: Any, env: Any) -> Any:
    return field.bind_env(env) if isinstance(field, EnvBound) else field


def _fmt(value: Any) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return f"{array[0]:g}" if array.size == 1 else "[" + ", ".join(f"{v:g}" for v in array) + "]"
