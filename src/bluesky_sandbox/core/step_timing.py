"""An env step timed phase by phase, as it runs.

Off unless switched on (``env.step_timer.enabled = True``): each timed region
then costs a ``perf_counter`` either side, and next to nothing when off. A
phase's time is its own: a phase run inside another (a field computed while a
hook reads it) counts to the inner one and is taken from the outer. What runs
outside every phase is ``other``, so a step's phases add up to its total. Each
field's raw computation is kept on its own too, by the field.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OTHER", "StepTimer", "StepTiming"]

#: The phase of what runs outside every named phase.
OTHER = "other"


@dataclass
class StepTiming:
    """One step, timed: its total, each phase's own time and each field's raw
    computation (ms) - by the field object's id."""

    total_ms: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    fields: dict[int, float] = field(default_factory=dict)


class _Off:
    """A phase while the timer is off: nothing."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> None:
        return None


_OFF = _Off()


class _Phase:
    """A named region of the step being timed."""

    __slots__ = ("_field", "_name", "_timer")

    def __init__(self, timer: StepTimer, name: str, field_obj: Any = None) -> None:
        self._timer, self._name, self._field = timer, name, field_obj

    def __enter__(self) -> None:
        # [start, time spent in phases within]
        self._timer._open.append([time.perf_counter(), 0.0])

    def __exit__(self, *_exc: Any) -> None:
        timer = self._timer
        start, within = timer._open.pop()
        spent = time.perf_counter() - start
        step = timer._step
        if step is None:
            return
        step.phases[self._name] = step.phases.get(self._name, 0.0) + (spent - within) * 1000.0
        if self._field is not None:
            fields = step.fields
            fields[id(self._field)] = fields.get(id(self._field), 0.0) + spent * 1000.0
        if timer._open:
            timer._open[-1][1] += spent


class StepTimer:
    """Times the phases of an env's steps while ``enabled``; ``last`` is the
    most recent step's timing."""

    def __init__(self) -> None:
        self.enabled = False
        self.last: StepTiming | None = None
        self._step: StepTiming | None = None
        self._started = 0.0
        self._open: list[list[float]] = []
        self._phases: dict[str, _Phase] = {}

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._step = StepTiming()
        self._open.clear()
        self._started = time.perf_counter()

    def end_step(self) -> None:
        step = self._step
        if step is None:
            return
        step.total_ms = (time.perf_counter() - self._started) * 1000.0
        step.phases[OTHER] = max(0.0, step.total_ms - sum(step.phases.values()))
        self.last, self._step = step, None

    def phase(self, name: str) -> _Phase | _Off:
        """Time what runs inside as ``name``, less any phase within it."""
        if self._step is None:
            return _OFF
        phase = self._phases.get(name)
        if phase is None:
            phase = self._phases[name] = _Phase(self, name)
        return phase

    def field(self, field_obj: Any, phase: str) -> _Phase | _Off:
        """Time ``field_obj``'s raw computation, as ``phase`` and on its own."""
        if self._step is None:
            return _OFF
        return _Phase(self, phase, field_obj)
