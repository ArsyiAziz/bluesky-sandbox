"""Clearances: an action the policy gives when it decides to, for as long as it
decides, the way a controller does.

One :class:`Clearance` per action declares everything that makes it one, and
``EnvConfig`` expands it into the parts the environment runs on:

* the action itself - its value;
* an :class:`~.mask.ActionMask` - the policy says nothing unless it clears it,
  and the aircraft flies on what it was last given;
* with ``duration``, a :class:`~.mask.ClearanceDuration` - how long it lasts,
  after which own navigation takes the axis back;
* with ``lock``, how long nothing else is accepted on its axis: ``"duration"``,
  the whole clearance (a vector is flown, not just turned), or ``"captured"``,
  until the aircraft has flown the command;
* with ``observe``, the observations that keep the state Markov: whether the
  axis is locked, and the time left on the clearance.

::

    action_fields = [
        actions.Clearance(
            actions.ActiveRouteWaypointHdgDeltaDeg(), duration=(0, 600), lock="duration"
        ),
        actions.Clearance(actions.AutopilotLnavVnav()),  # resume own navigation
    ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..base import ActionField
from .mask import ActionMask, ClearanceDuration

__all__ = ["Clearance", "expand_clearances"]

Lock = Literal["duration", "captured"]


@dataclass(frozen=True)
class Clearance:
    """``action`` as a clearance: given when the policy decides, held for
    ``duration`` seconds (a ``(low, high)`` range the policy chooses in; ``None``
    - until changed or resumed), with its axis locked per ``lock``, and its state
    observed unless ``observe`` is off."""

    action: ActionField
    duration: tuple[float, float] | None = None
    lock: Lock | None = None
    observe: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionField) or isinstance(
            self.action, (ActionMask, ClearanceDuration)
        ):
            raise TypeError(
                f"a Clearance wraps an action field, got {self.action!r}."
            )
        if self.lock not in (None, "duration", "captured"):
            raise ValueError(
                f"Clearance lock must be 'duration', 'captured' or None, got "
                f"{self.lock!r}."
            )
        if self.duration is not None:
            low, high = (float(v) for v in self.duration)
            if not 0.0 <= low <= high:
                raise ValueError(
                    f"Clearance duration needs 0 <= low <= high, got {self.duration!r}."
                )
            object.__setattr__(self, "duration", (low, high))
        if self.lock == "duration" and self.duration is None:
            raise ValueError("a Clearance locked for its duration needs a duration.")

    @property
    def name(self) -> str:
        return self.action.meta.name

    def action_fields(self) -> list[ActionField]:
        """The action, its duration if timed, and its mask - in that order."""
        fields: list[ActionField] = [self.action]
        if self.duration is not None:
            low, high = self.duration
            fields.append(ClearanceDuration(target=self.name, low=low, high=high))
        fields.append(
            ActionMask(
                target=self.name,
                lock_until_captured=self.lock == "captured",
                lock_for_duration=self.lock == "duration",
            )
        )
        return fields

    def observation_fields(self) -> list[Any]:
        """What the policy needs to see of it: its lock, and its time left."""
        if not self.observe:
            return []
        from .. import observations  # noqa: PLC0415 - observations import actions

        fields: list[Any] = []
        if self.lock is not None:
            fields.append(observations.ActionLocked(target=self.name))
        if self.duration is not None:
            fields.append(
                observations.ClearanceTimeLeftS(target=self.name, high=self.duration[1])
            )
        return fields


def expand_clearances(
    action_fields: list[Any], obs_fields: list[Any]
) -> tuple[list[Any], list[Any]]:
    """``action_fields`` with each :class:`Clearance` expanded in place into its
    parts, and ``obs_fields`` with each one's observations appended."""
    actions: list[Any] = []
    observed: list[Any] = []
    for field in action_fields:
        if isinstance(field, Clearance):
            actions.extend(field.action_fields())
            observed.extend(field.observation_fields())
        else:
            actions.append(field)
    return actions, [*obs_fields, *observed]
