"""One frame's view of the live traffic, shared by every view that draws it."""

from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft

from bluesky_sandbox.sim.queryables import QueryRegion


class AircraftFrame:
    """Every live aircraft as one frame draws it: positions and motion read
    from ``bs.traf`` once, as arrays (index ``i`` is ``bs.traf`` index ``i``),
    so a view projects them all in one pass; and what is derived per
    aircraft - separation state, background, the query region it is in -
    worked out once, for whichever view asks first.

    A driver makes one per frame (see ``SimDriver.aircraft_frame``); it is
    not updated, so it is read before the sim moves on.
    """

    def __init__(self, driver) -> None:
        traf = bs.traf
        n = int(traf.ntraf)
        self.n = n
        self.ids: list[str] = list(traf.id[:n])
        self.lat = np.array(traf.lat[:n], dtype=float)
        self.lon = np.array(traf.lon[:n], dtype=float)
        self.alt_m = np.array(traf.alt[:n], dtype=float)
        self.alt_ft = self.alt_m / ft
        self.hdg = np.array(traf.hdg[:n], dtype=float)
        self.gs = np.array(traf.gs[:n], dtype=float)
        self.vs = np.array(traf.vs[:n], dtype=float)
        self._driver = driver
        self._query: list[str | None] | None = None

    def state(self, i: int) -> str:
        """Aircraft ``i``'s separation state (``"los"``, ``"conflict"``, ...)."""
        return self._driver._aircraft_state(self.ids[i])

    def background(self, i: int) -> bool:
        """Whether aircraft ``i`` is background traffic (drawn dimmed)."""
        return bool(self._driver._aircraft_snapshot(self.ids[i]).get("background", False))

    def query_color(self, i: int) -> str | None:
        """The color name of the first :class:`QueryRegion` aircraft ``i`` is
        in, or ``None`` if it is in none."""
        if self._query is None:
            self._query = self._query_colors()
        return self._query[i]

    def _query_colors(self) -> list[str | None]:
        env = self._driver._env
        regions = (
            []
            if env is None
            else [q for q in env.episode_queryables.values() if isinstance(q, QueryRegion)]
        )
        colors: list[str | None] = []
        for lat, lon, alt_ft in zip(self.lat, self.lon, self.alt_ft, strict=True):
            colors.append(
                next(
                    (q.color for q in regions if q.shape.contains(lat, lon, alt_ft)),
                    None,
                )
            )
        return colors
