"""A never-reused serial number per aircraft, kept aligned by BlueSky itself.

Its own module, importing nothing of this package, so that code the runtime
depends on - observation fields among it - can read the uids too.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky.core.trafficarrays import TrafficArrays

# Trackers attached to ``bs.traf``, most recent last. One per live env; there
# is one BlueSky per process, so any of them numbers the same aircraft.
_ATTACHED: list[AircraftUids] = []


def live_aircraft_uids() -> np.ndarray | None:
    """Each live aircraft's uid, in ``bs.traf.id`` order; ``None`` without an env.

    For state that must follow an aircraft rather than a callsign. ``None``
    means no runtime has attached a tracker - a field used on its own, or a
    test standing in for BlueSky - and callers fall back to callsigns.
    """
    return _ATTACHED[-1].uid if _ATTACHED else None


def aircraft_keys(ids) -> tuple:
    """A key per aircraft in ``ids`` (``bs.traf.id``) for state that must follow
    the aircraft: its uid, or its callsign when there is no tracker.

    Callsigns are the fallback only - BlueSky reuses one once its aircraft is
    deleted, so callsign-keyed state must be forgotten on despawn. ``ids`` is
    passed in rather than read here so a caller standing in for BlueSky keeps
    the two consistent.
    """
    uids = live_aircraft_uids()
    if uids is not None and len(uids) == len(ids):
        return tuple(uids.tolist())
    return tuple(ids)


class AircraftUids(TrafficArrays):
    """A serial number per aircraft, never reused, that BlueSky keeps aligned.

    BlueSky names an aircraft only by its callsign and reuses a callsign once
    that aircraft is deleted, so anything remembering aircraft by callsign can
    take a new aircraft for an old one. ``uid`` is a BlueSky traffic array:
    BlueSky itself grows it on every ``cre`` and shrinks it on every ``delete``,
    however the aircraft came or went - the spawner, a task hook, a qtgl
    command, a plugin - so ``uid[i]`` always belongs to ``bs.traf.id[i]``.

    ``created`` logs every callsign created since the last ``bs.sim.reset()``,
    reused ones included, for the callsign issuer to keep clear of.
    """

    def __init__(self) -> None:
        # Set before registering: registering with traffic already present
        # calls ``create`` for it straight away.
        self._next_uid = 0
        self.created: list[str] = []
        super().__init__()  # attaches to bs.traf, so only after bs.init
        with self.settrafarrays():
            self.uid = np.array([], dtype=np.int64)
        _ATTACHED.append(self)

    def create(self, n: int = 1) -> None:
        super().create(n)  # appends n zeros
        self.uid[-n:] = np.arange(self._next_uid, self._next_uid + n)
        self._next_uid += n
        # ``cre`` writes the new callsigns before it creates children.
        self.created.extend(bs.traf.id[-n:])

    def reset(self) -> None:
        super().reset()
        self.created.clear()

    def detach(self) -> None:
        """Stop following traffic. BlueSky has no way to unregister a child."""
        if self._parent is not None and self in self._parent._children:
            self._parent._children.remove(self)
        if self in _ATTACHED:
            _ATTACHED.remove(self)
