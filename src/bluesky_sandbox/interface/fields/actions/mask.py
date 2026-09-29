"""Masking an action: a 0/1 action that, set to 1, skips another action that
step.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any

from .._state import (
    _ActionMaskBacked,
    action_lock,
    action_masks,
    record_action_mask,
    set_action_lock,
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
    "action_name",
    "check_action_masks",
    "lock_holds",
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
    """

    meta = ActionMeta("action_mask", Unit.SWITCH, mode=ActionMode.SWITCH)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the action masked: its name, its class or an instance",
    ] = ""
    lock_until_captured: Annotated[
        bool, "once applied, nothing on the target's axis until it is flown"
    ] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        if self.target == self.meta.name:
            raise ValueError("an ActionMask cannot mask another ActionMask.")
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
        axis = lock[0]
        set_action_lock(idx, self.target, axis, axis_error(idx, axis))
        return True

    def target_applied(self, idx: int, axis: ControlAxis | None) -> None:
        """The target was just applied on ``axis``: lock it while it is flown,
        if this mask locks and the axis has an error to watch."""
        if self.lock_until_captured and axis in AXIS_ERRORS:
            set_action_lock(idx, self.target, axis)


def lock_holds(idx: int, target: str) -> bool:
    """Whether ``target``'s lock still holds for the aircraft at ``idx``: it is
    locked, and its axis error has shrunk since it was last read (or has not
    been read since the clearance). What the dispatcher refuses by, and what
    :class:`~bluesky_sandbox.interface.fields.observations.ActionLocked`
    shows."""
    lock = action_lock(idx, target)
    if lock is None:
        return False
    axis, last = lock
    return last is None or axis_error(idx, axis) < last


def axis_error(idx: int, axis: ControlAxis) -> float:
    """How far the aircraft at ``idx`` still has to go on ``axis``: the size of
    its autopilot error there (:data:`AXIS_ERRORS`)."""
    from .. import observations  # noqa: PLC0415 - observations import this module

    return abs(float(getattr(observations, AXIS_ERRORS[axis])().get(idx)))


def check_action_masks(action_fields: Iterable[Any]) -> None:
    """Refuse masks that name no action of the config, or one ambiguously:
    each :class:`ActionMask` targets exactly one of ``action_fields``, and no
    action is masked twice."""
    action_fields = list(action_fields)
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
    for mask in masks:
        if mask.lock_until_captured and axes[mask.target] not in AXIS_ERRORS:
            raise ValueError(
                f"ActionMask on {mask.target!r} locks until captured, but that "
                f"action commands no axis a capture is measured on "
                f"({', '.join(a.value for a in AXIS_ERRORS)})."
            )
