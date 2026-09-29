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
    action_locked,
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

__all__ = ["CAPTURE", "ActionMask", "action_name", "check_action_masks"]

#: How a clearance on each axis counts as flown, for a mask locking until
#: captured: the autopilot error observation measuring it - its selection
#: against the aircraft - and the default tolerance, in that field's unit.
CAPTURE: dict[ControlAxis, tuple[str, float]] = {
    ControlAxis.HEADING: ("ApHdgErrorDeg", 2.0),
    ControlAxis.ALTITUDE: ("ApAltErrorFt", 100.0),
    ControlAxis.SPEED: ("ApCasErrorKts", 2.0),
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
    switch that takes that axis over - is skipped until the aircraft has flown
    it, its autopilot error on that axis within ``capture_tolerance``
    (:data:`CAPTURE`). A vector cannot be changed or abandoned mid-turn.
    """

    meta = ActionMeta("action_mask", Unit.SWITCH, mode=ActionMode.SWITCH)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the action masked: its name, its class or an instance",
    ] = ""
    lock_until_captured: Annotated[
        bool, "once applied, nothing on the target's axis until it is flown"
    ] = False
    capture_tolerance: Annotated[
        float | None, "how close counts as flown, in deg / ft / kt; None = default"
    ] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        if self.target == self.meta.name:
            raise ValueError("an ActionMask cannot mask another ActionMask.")
        super().__post_init__()

    def set(self, idx: int, value: float) -> None:
        record_action_mask(idx, self.target, self.switch_command(value))

    def current_switch_state(self, idx: int) -> bool:
        return bool(action_masks(self.target, [idx])[0])

    def locked(self, idx: int, axis: ControlAxis | None) -> bool:
        """Whether the target's axis is locked for the aircraft at ``idx``: its
        last clearance not yet flown. A lock found flown is released."""
        if (
            not self.lock_until_captured
            or axis not in CAPTURE
            or not action_locked(idx, self.target)
        ):
            return False
        if _captured(idx, axis, self.capture_tolerance):
            set_action_lock(idx, self.target, False)
            return False
        return True

    def target_applied(self, idx: int) -> None:
        """The target was just applied: lock it until flown, if this locks."""
        if self.lock_until_captured:
            set_action_lock(idx, self.target, True)


def _captured(idx: int, axis: ControlAxis | None, tolerance: float | None) -> bool:
    """Whether the aircraft has flown its clearance on ``axis``."""
    from .. import observations  # noqa: PLC0415 - observations import this module

    name, default = CAPTURE[axis]
    error = float(getattr(observations, name)().get(idx))
    return abs(error) <= (default if tolerance is None else float(tolerance))


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
        if mask.lock_until_captured and axes[mask.target] not in CAPTURE:
            raise ValueError(
                f"ActionMask on {mask.target!r} locks until captured, but that "
                f"action commands no axis a capture is measured on "
                f"({', '.join(a.value for a in CAPTURE)})."
            )
