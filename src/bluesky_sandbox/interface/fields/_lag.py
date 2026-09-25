"""History for lagged (frame-stacked) observation fields."""

from __future__ import annotations

import bluesky as bs
import numpy as np

from bluesky_sandbox.sim.aircraft_uids import aircraft_keys

from ._common import on_reset

# Per-process ring buffers of past raw field values, so ``field.lagged(k)`` can
# report the value from ``k`` steps ago. Same lifecycle as the stores in
# :mod:`._state`: one BlueSky sim per process, the env clears on reset and drops
# an aircraft on despawn.
#
# One ring buffer per INNER field, keyed ``("obs" | "pair", repr(inner))``.
# ``repr`` rather than ``meta.name``: two instances of the same class with
# different kwargs (``ConflictTlosS(rpz_nm=8)`` vs ``rpz_nm=5``) share a name but
# are different signals and must not share a buffer.
#
# All lags of one inner field share its buffer, and the push is guarded by sim
# time, so ``.lagged(1)`` and ``.lagged(2)`` on the same field cost ONE inner
# evaluation per step rather than one each - this sits in the per-agent per-step
# rollout hot path.
#
# Rows follow aircraft by uid (:mod:`~bluesky_sandbox.sim.aircraft_uids`), not
# callsign: BlueSky reuses a deleted aircraft's callsign, and a history keyed by
# callsign would hand the new aircraft the old one's past. A field used without
# a runtime - on its own, or under a test's stand-in for BlueSky - falls back to
# callsigns, and ``on_aircraft_removed`` forgets them.
_LAG_HISTORY: dict[tuple[str, str], _LagRing] = {}
on_reset(lambda _seed: _LAG_HISTORY.clear())


# How deep each inner field's history has to be: the deepest lag anyone
# actually built on it. Derived rather than capped - a fixed ceiling made
# ``.lagged(12)`` a source edit for no benefit. Sizing per field also stops the
# common case over-allocating: with only ``.lagged(1)`` configured a buffer
# holds 2 frames per aircraft instead of a blanket 9.
#
# Keyed by ``repr(inner)`` like the history itself, so sibling lags of one
# field agree on a size. This is derived from CONFIGURATION, not from episode
# state, which is why ``reset_field_state`` clears the history and leaves this
# alone.
_LAG_DEPTH: dict[str, int] = {}


def _register_lag_depth(key: str, steps: int) -> None:
    """Record that ``key``'s history must reach at least ``steps`` back."""
    _LAG_DEPTH[key] = max(_LAG_DEPTH.get(key, 0), int(steps))


class _LagRing:
    """Past values of one inner field, per aircraft or per ordered pair.

    ``values[slot, row]`` (``values[slot, own, other]`` for pairs) is a ring of
    ``depth`` frames. Every entry keeps its own write position and length,
    because entries are pushed independently - only the aircraft (or pairs)
    observed at a sim time. Rows are matched to aircraft by key on every
    access, so an entry follows its aircraft when others spawn or leave.
    """

    def __init__(self, depth: int, *, pair: bool) -> None:
        # Frames kept: the deepest lag registered on the field, plus the latest.
        # Sized once, after configuration built every field; a lag constructed
        # later on the same inner field finds it sized and degrades to a
        # zero-order hold rather than failing.
        self.depth = depth
        self.pair = pair
        self.keys: tuple = ()
        self.row: dict = {}
        grid = (0, 0) if pair else (0,)
        self.values: np.ndarray | None = None  # allocated at the first push
        self.head = np.zeros(grid, dtype=np.intp)
        self.count = np.zeros(grid, dtype=np.intp)
        # When each ownship row was last pushed (pairs); an ownship field is
        # pushed as a whole, once per sim time, by whichever query comes first.
        self.pushed_at = np.full(0, np.nan)
        self.last_push_simt: float | None = None

    def sync(self, keys: tuple) -> None:
        """Match rows to ``keys``; survivors keep their history, newcomers none."""
        if keys == self.keys:
            return
        take = np.fromiter(
            (self.row.get(key, -1) for key in keys), dtype=np.intp, count=len(keys)
        )
        new_rows = np.flatnonzero(take >= 0)
        old_rows = take[new_rows]
        n = len(keys)
        grid = (n, n) if self.pair else (n,)

        def carry(old: np.ndarray, lead: int) -> np.ndarray:
            tail = old.shape[lead + len(grid) :]
            new = np.zeros(old.shape[:lead] + grid + tail, dtype=old.dtype)
            pick = (slice(None),) * lead
            if self.pair:
                new[pick + np.ix_(new_rows, new_rows)] = old[
                    pick + np.ix_(old_rows, old_rows)
                ]
            else:
                new[pick + (new_rows,)] = old[pick + (old_rows,)]
            return new

        self.head = carry(self.head, 0)
        self.count = carry(self.count, 0)
        if self.values is not None:
            self.values = carry(self.values, 1)
        pushed_at = np.full(n, np.nan)
        pushed_at[new_rows] = self.pushed_at[old_rows]
        self.pushed_at = pushed_at
        self.keys = keys
        self.row = {key: i for i, key in enumerate(keys)}

    def _at(self, rows: np.ndarray, cols: np.ndarray | None) -> tuple:
        return (rows[:, None], cols[None, :]) if self.pair else (rows,)

    def push(self, current, rows: np.ndarray, cols: np.ndarray | None = None) -> None:
        """Append ``current`` to the rings of ``rows`` (x ``cols``, for pairs)."""
        current = np.asarray(current, dtype=np.float64)
        at = self._at(rows, cols)
        if self.values is None:
            tail = current.shape[len(at) :]
            self.values = np.zeros((self.depth, *self.head.shape, *tail))
        head = (self.head[at] + 1) % self.depth
        self.head[at] = head
        self.values[(head, *at)] = current
        self.count[at] = np.minimum(self.count[at] + 1, self.depth)

    def read(
        self, steps: int, rows: np.ndarray, cols: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """``(values, missing)``: each entry ``steps`` pushes back, held at its
        oldest frame when the history is shorter.

        Zero-order hold, never zero-fill: a brand-new aircraft has no history,
        and zero is a MEANINGFUL value for these fields (raw 0 on
        ``ConflictTlosS`` means "in LoS right now"), so zero-filling would inject
        a maximal-threat signal on every aircraft that just came into view.
        Repeating the oldest value it has says "no observed change", which is
        the honest reading. An entry with no history at all is ``missing``; the
        caller substitutes its live value.
        """
        at = self._at(rows, cols)
        count = self.count[at]
        missing = count == 0
        if self.values is None:
            return None, missing
        back = np.minimum(int(steps), np.maximum(count - 1, 0))
        slot = (self.head[at] - back) % self.depth
        return self.values[(slot, *at)], missing

    def forget(self, key) -> None:
        """Drop the history of the aircraft keyed ``key``, if it has a row."""
        row = self.row.get(key)
        if row is None:
            return
        if self.pair:
            self.count[row, :] = 0
            self.count[:, row] = 0
        else:
            self.count[row] = 0
        self.pushed_at[row] = np.nan


class _LagHistoryBacked:
    """State hooks for lag/stack wrappers.

    History is pushed lazily on read (see ``get_many``), so there is no
    ``on_step`` here - only the per-aircraft drop.
    """

    def on_aircraft_removed(self, acid: str) -> None:
        # Uid-keyed rows need nothing: the next access drops the departed
        # aircraft. Callsign-keyed ones must forget it here, or a new aircraft
        # given the same callsign would inherit its history.
        for ring in _LAG_HISTORY.values():
            ring.forget(acid)


def _lag_ring(kind: str, key: str, steps: int) -> _LagRing:
    """The shared ring for inner field ``key``, its rows matched to traffic."""
    ring = _LAG_HISTORY.get((kind, key))
    if ring is None:
        depth = _LAG_DEPTH.get(key, int(steps)) + 1
        ring = _LAG_HISTORY[(kind, key)] = _LagRing(depth, pair=kind == "pair")
    ring.sync(aircraft_keys(bs.traf.id))
    return ring
