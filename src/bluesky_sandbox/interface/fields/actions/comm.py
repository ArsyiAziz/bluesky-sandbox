"""Communication actions: a learned message, with no effect on the aircraft."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .._state import record_comm_message
from ..base import ActionField, ActionMeta, Unit


@dataclass(frozen=True)
class CommBroadcast(ActionField):
    """Broadcast one channel of a learned communication message.

    Stores the (squashed, ``[-1, 1]``) action value in the per-process comm
    registry under this aircraft's callsign; other agents read it back on the
    next step through ``IntruderCommMessage`` with the same ``channel``. No
    aircraft-control effect - a pure signaling channel whose meaning the
    shared policy must learn (emergent communication). Exclude these dims from
    any action-magnitude penalty, or the reward will train the channel silent.
    """

    meta = ActionMeta(
        "comm_broadcast",
        Unit.UNITLESS,
    )
    channel: Annotated[int, "message channel index"] = 0
    low: Annotated[float, "message value"] = -1.0
    high: Annotated[float, "message value"] = 1.0

    def set(self, idx: int, value: float) -> None:
        record_comm_message(idx, self.channel, float(value))

    def bounds(self, idx: int) -> tuple[float, float]:
        del idx
        return float(self.low), float(self.high)
