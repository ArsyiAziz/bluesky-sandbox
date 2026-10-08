"""Masking an action: a 0/1 action that, set to 1, skips another action that
step.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import kwikqdrdist

from .._route import _active_route_waypoint
from ._targets import _issue_crossover_speed
from .._state import (
    _ActionMaskBacked,
    action_lock,
    action_masks,
    clearance_expiries,
    clearance_holds,
    record_action_mask,
    set_action_lock,
    set_clearance_expiry,
)
from ..base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)

__all__ = [
    "AXIS_ERRORS",
    "ActionMask",
    "ClearanceDuration",
    "action_name",
    "check_action_masks",
    "end_clearances",
    "expire_clearances",
    "lock_holds",
    "start_holds",
]

#: What a clearance on each axis leaves to fly, for a mask locking until
#: captured: the autopilot error observation measuring it - the selection
#: against the aircraft.
AXIS_ERRORS: dict[ControlAxis, str] = {
    ControlAxis.HEADING: "ApHdgErrorDeg",
    ControlAxis.ALTITUDE: "ApAltErrorFt",
    ControlAxis.SPEED: "ApCasErrorKts",
}


def action_name(target: Any) -> str:
    """The name of the action ``target`` names: an action field's class or
    instance - its ``meta.name`` - or that name itself."""
    if isinstance(target, str):
        return target
    if isinstance(target, ActionField) or (
        inspect.isclass(target) and issubclass(target, ActionField)
    ):
        meta = target.meta
        if isinstance(meta, ActionMeta):
            return meta.name
    raise TypeError(
        "an action is named by its class, an instance of it or its name, "
        f"got {target!r}."
    )


@dataclass(frozen=True)
class ActionMask(_ActionMaskBacked, SwitchActionMixin, ActionField):
    """Mask another action: 1 skips it this step, 0 lets it through.

    ``target`` is the action masked - its class, an instance of it, or its
    name; it is kept as the name. A masked action is not applied: the aircraft
    keeps the last command it was given, and a masked switch keeps its state,
    even when a switch turned on requires it. The mask itself sends BlueSky
    nothing.

    The policy chooses the mask every step, so it decides when to act as well as
    how: with a mask on
    :class:`~bluesky_sandbox.interface.fields.actions.HdgDeg`, a 1 lets the
    aircraft fly out the last heading it was given - vectoring, one instruction
    at a time.
    ``info["action_applied"]`` says which values of each action took effect, and
    :class:`~bluesky_sandbox.interface.fields.observations.PrevActionMasked`
    shows the policy what it masked last.

    ``lock_until_captured`` makes a clearance a committed unit: once the target
    is applied, it - and every other action commanding its axis, such as a
    switch that takes that axis over - is skipped while the aircraft is still
    flying it: while its autopilot error on that axis (:data:`AXIS_ERRORS`)
    keeps shrinking step on step. The first step it does not - the clearance
    flown, or one the aircraft cannot fly any further, such as a level above
    its ceiling - releases it, so no lock outlives its clearance and there is
    no tolerance to choose. A vector cannot be changed or abandoned mid-turn.

    ``lock_for_duration`` holds the lock for the clearance's whole
    :class:`ClearanceDuration` instead: a vector is not only turned but FLOWN -
    the leg, until it runs out and own navigation takes it back. Nothing on the
    axis, resuming own navigation included, is accepted until then.
    """

    meta = ActionMeta("action_mask", Unit.SWITCH, mode=ActionMode.SWITCH)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the action masked: its name, its class or an instance",
    ] = ""
    lock_until_captured: Annotated[
        bool, "once applied, nothing on the target's axis until it is flown"
    ] = False
    lock_for_duration: Annotated[
        bool, "once applied, nothing on the target's axis until its duration ends"
    ] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        if self.target == self.meta.name:
            raise ValueError("an ActionMask cannot mask another ActionMask.")
        if self.lock_until_captured and self.lock_for_duration:
            raise ValueError(
                "an ActionMask locks until captured or for the duration, not both."
            )
        super().__post_init__()

    def set(self, idx: int, value: float) -> None:
        record_action_mask(idx, self.target, self.switch_command(value))

    def current_switch_state(self, idx: int) -> bool:
        return bool(action_masks(self.target, [idx])[0])

    def locked(self, idx: int) -> bool:
        """Whether the target's axis is locked for the aircraft at ``idx``: its
        last clearance still being flown (:func:`lock_holds`). A lock that no
        longer holds is released; one that does notes the error it read."""
        lock = action_lock(idx, self.target)
        if lock is None:
            return False
        if not lock_holds(idx, self.target):
            set_action_lock(idx, self.target)
            return False
        axis, _last, until = lock
        if until is None:
            set_action_lock(idx, self.target, axis, axis_error(idx, axis))
        return True

    def target_applied(
        self, idx: int, axis: ControlAxis | None, until: float | None = None
    ) -> None:
        """The target was just applied on ``axis``: lock it while it is flown
        (if this mask locks until captured and the axis has an error to watch),
        or until sim time ``until`` - its clearance's expiry (if it locks for the
        duration)."""
        if self.lock_until_captured and axis in AXIS_ERRORS:
            set_action_lock(idx, self.target, axis)
        elif self.lock_for_duration:
            # A clearance counted from capture has no expiry until it is flown:
            # locked until then, and :func:`expire_clearances` dates the lock
            # once its clock starts.
            set_action_lock(
                idx, self.target, axis, until=math.inf if until is None else until
            )


def lock_holds(idx: int, target: str) -> bool:
    """Whether ``target``'s lock still holds for the aircraft at ``idx``: it is
    locked, and its axis error has shrunk since it was last read (or has not
    been read since the clearance). What the dispatcher refuses by, and what
    :class:`~bluesky_sandbox.interface.fields.observations.ActionLocked`
    shows."""
    lock = action_lock(idx, target)
    if lock is None:
        return False
    axis, last, until = lock
    if until is not None:
        return float(bs.sim.simt) < until
    return last is None or axis_error(idx, axis) < last


def axis_error(idx: int, axis: ControlAxis) -> float:
    """How far the aircraft at ``idx`` still has to go on ``axis``: the size of
    its autopilot error there (:data:`AXIS_ERRORS`)."""
    from .. import observations  # noqa: PLC0415 - observations import this module

    return abs(float(getattr(observations, AXIS_ERRORS[axis])().get(idx)))


@dataclass(frozen=True)
class ClearanceDuration(ActionField):
    """How long a clearance of ``target`` lasts, in seconds: a TEMPORARY
    clearance.

    Decided with the clearance - its value only takes effect on a step its
    target is applied, and ``info["action_applied"]`` marks it exactly as the
    target. When it runs out, the aircraft resumes own navigation on that axis:
    once no axis is under a clearance, LNAV and VNAV come back on (and with
    them any RTA); while another still is, the expired axis returns to its
    route value - the heading to the active fix, the fix's level, its speed
    (gate or arrival time). A new clearance of the target restarts the clock;
    a switch that takes the axis over (resuming LNAV+VNAV) ends it.

    So an explored clearance can never strand an aircraft: every deviation
    comes back by itself. :class:`~bluesky_sandbox.interface.fields.observations.
    ClearanceTimeLeftS` shows the time left.

    ``from_capture`` counts the duration from when the clearance is FLOWN rather
    than given: the clock starts once the aircraft has captured the command -
    the first step its autopilot error on the axis (:data:`AXIS_ERRORS`) stops
    shrinking, as a lock until captured releases - and the duration is the hold
    after it. So every clearance is flown in full and held at least ``low``
    seconds, whatever its size: there is no instantaneous one, such as 30 kt
    for a few seconds. Until captured, the time left is the whole hold.
    """

    meta = ActionMeta("clearance_duration", Unit.S)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the temporary clearance: its name, its class or an instance",
    ] = ""
    low: Annotated[float, "shortest clearance, s"] = 0.0
    high: Annotated[float, "longest clearance, s"] = 600.0
    from_capture: Annotated[
        bool, "count the duration from when the command is flown, not given"
    ] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        super().__post_init__()
        if self.low > self.high:
            raise ValueError(
                f"ClearanceDuration needs low <= high, got {self.low}, {self.high}"
            )

    def set(self, idx: int, value: float) -> None:
        """Nothing on its own: the dispatcher starts the clock when the target
        is applied (:meth:`start`)."""

    def bounds(self, idx: int) -> tuple[float, float]:
        return float(self.low), float(self.high)

    def start(self, idx: int, axis: ControlAxis, value: float) -> None:
        """The target was just applied on ``axis``: run it for ``value`` s,
        within the range - one at or below 0 runs out on the next step - from
        now, or from its capture (``from_capture``)."""
        duration = max(0.0, min(max(float(value), float(self.low)), float(self.high)))
        if self.from_capture:
            set_clearance_expiry(idx, self.target, axis, hold_s=duration)
        else:
            set_clearance_expiry(idx, self.target, axis, float(bs.sim.simt) + duration)


#: The axes a temporary clearance can be on: the ones own navigation flies.
RESUMABLE_AXES = (ControlAxis.HEADING, ControlAxis.ALTITUDE, ControlAxis.SPEED)


def end_clearances(idx: int, axes: Iterable[ControlAxis]) -> None:
    """End the aircraft at ``idx``'s running clearances on ``axes`` - own
    navigation has taken them back - with no resume of their own."""
    axes = set(axes)
    for target, (axis, _expires) in clearance_expiries(idx).items():
        if axis in axes:
            set_clearance_expiry(idx, target)


def expire_clearances() -> None:
    """Resume own navigation wherever a temporary clearance has run out.

    Called by the environment each step, before the new actions: a clearance
    counted from capture that has been flown starts its clock (:func:`start_holds`);
    an expired clearance's lock is released, and the axis resumes - fully
    (LNAV+VNAV) once no axis of the aircraft is under a clearance, else to its
    route value.
    """
    now = float(bs.sim.simt)
    for idx, acid in enumerate(list(bs.traf.id)):
        running = clearance_expiries(idx)
        if not running:
            continue
        if any(at is None for _axis, at in running.values()):
            start_holds(idx, now)
            running = clearance_expiries(idx)
        expired = {
            t: axis for t, (axis, at) in running.items() if at is not None and at <= now
        }
        if not expired:
            continue
        for target in expired:
            set_clearance_expiry(idx, target)
            set_action_lock(idx, target)
        if len(expired) == len(running):
            bs.stack.stack(f"LNAV {acid} ON")
            bs.stack.stack(f"VNAV {acid} ON")
            continue
        for axis in expired.values():
            _resume_axis(idx, acid, axis)


def start_holds(idx: int, now: float) -> None:
    """Start the clock of each of the aircraft at ``idx``'s clearances counted
    from capture that it has now flown: its axis error has not shrunk since it
    was last read. One still being flown notes the error it read."""
    for target, (axis, hold, last) in clearance_holds(idx).items():
        error = axis_error(idx, axis)
        if last is None or error < last:
            set_clearance_expiry(idx, target, axis, hold_s=hold, error=error)
            continue
        set_clearance_expiry(idx, target, axis, now + hold)
        lock = action_lock(idx, target)
        if lock is not None and lock[2] is not None:
            set_action_lock(idx, target, axis, until=now + hold)


def _resume_axis(idx: int, acid: str, axis: ControlAxis) -> None:
    """Return one axis to its route value while another is still cleared."""
    fix = _active_route_waypoint(idx)
    if fix is None:
        return
    lat, lon, alt_m, spd_ms = fix
    if axis is ControlAxis.HEADING:
        qdr, _dist = kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), lat, lon)
        bs.stack.stack(f"HDG {acid} {float(qdr) % 360.0:.2f}")
    elif axis is ControlAxis.ALTITUDE and alt_m is not None:
        bs.stack.stack(f"ALT {acid} {alt_m / ft:.1f}")
    elif axis is ControlAxis.SPEED and spd_ms is not None:
        # Mach above the crossover, CAS below - as the crossover actions do.
        _issue_crossover_speed(idx, spd_ms / kts)


def check_action_masks(action_fields: Iterable[Any]) -> None:
    """Refuse masks that name no action of the config, or one ambiguously:
    each :class:`ActionMask` targets exactly one of ``action_fields``, and no
    action is masked twice. Each :class:`ClearanceDuration` times exactly one
    masked action on an axis own navigation flies, and none twice."""
    action_fields = list(action_fields)
    durations = [f for f in action_fields if isinstance(f, ClearanceDuration)]
    action_fields = [f for f in action_fields if not isinstance(f, ClearanceDuration)]
    masks = [f for f in action_fields if isinstance(f, ActionMask)]
    names = [f.meta.name for f in action_fields if not isinstance(f, ActionMask)]
    targets = [mask.target for mask in masks]
    for target in targets:
        if not target:
            raise ValueError(
                f"an ActionMask needs a target: the action it masks, one of {names}."
            )
        count = names.count(target)
        if count == 0:
            raise ValueError(
                f"ActionMask target {target!r} is not an action of this config; "
                f"its actions are {names}."
            )
        if count > 1:
            raise ValueError(
                f"ActionMask target {target!r} names {count} actions of this "
                "config; a mask masks one."
            )
    twice = sorted({t for t in targets if targets.count(t) > 1})
    if twice:
        raise ValueError(f"more than one ActionMask masks {twice}.")
    axes = {f.meta.name: f.meta.control_axis for f in action_fields}
    timed = [d.target for d in durations]
    for target in timed:
        if target not in targets:
            raise ValueError(
                f"ClearanceDuration target {target!r} has no ActionMask: a "
                f"duration times a clearance; the masked actions are {targets}."
            )
        if axes.get(target) not in RESUMABLE_AXES:
            raise ValueError(
                f"ClearanceDuration target {target!r} commands no axis own "
                f"navigation flies ({', '.join(a.value for a in RESUMABLE_AXES)})."
            )
    twice = sorted({t for t in timed if timed.count(t) > 1})
    if twice:
        raise ValueError(f"more than one ClearanceDuration times {twice}.")
    for mask in masks:
        if mask.lock_for_duration and mask.target not in timed:
            raise ValueError(
                f"ActionMask on {mask.target!r} locks for the duration, but no "
                "ClearanceDuration times it."
            )
    for mask in masks:
        if mask.lock_until_captured and axes[mask.target] not in AXIS_ERRORS:
            raise ValueError(
                f"ActionMask on {mask.target!r} locks until captured, but that "
                f"action commands no axis a capture is measured on "
                f"({', '.join(a.value for a in AXIS_ERRORS)})."
            )
