"""Region queryables: whether an aircraft is inside a bounded volume, and
since when."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft

from bluesky_sandbox.interface.task import (
    QueryableTemporalStateUnavailable,
    StepTime,
    UnavailableStepTime,
)
from bluesky_sandbox.sim.bounds import Bounds

from .base import _require_bound_query_result


@dataclass(frozen=True)
class RegionCurrent:
    """Current aircraft-relative region state."""

    inside: bool = False


@dataclass(frozen=True)
class RegionStep:
    """Region facts accumulated during the current env step."""

    inside: bool = False


@dataclass(frozen=True)
class UnavailableRegionStep:
    """Region step placeholder for query results without temporal tracking."""

    @property
    def inside(self) -> bool:
        raise QueryableTemporalStateUnavailable(
            "Region step containment requires track_temporal_state=True "
            "on the region."
        )


@dataclass
class RegionResult:
    """Structured query result for a region."""

    _queryable: Any | None = field(default=None, repr=False, compare=False)
    _acidx: int | None = field(default=None, repr=False, compare=False)
    _current_cache: RegionCurrent | None = field(
        default=None, repr=False, compare=False
    )
    step: RegionStep | UnavailableRegionStep = field(
        default_factory=UnavailableRegionStep
    )
    time: StepTime | UnavailableStepTime = field(default_factory=UnavailableStepTime)

    @classmethod
    def for_aircraft(
        cls,
        queryable: Any,
        acidx: int,
        *,
        current: RegionCurrent | None = None,
        step: RegionStep | UnavailableRegionStep | None = None,
        time: StepTime | UnavailableStepTime | None = None,
    ) -> RegionResult:
        """Build a region result for one aircraft."""
        return cls(
            _queryable=queryable,
            _acidx=acidx,
            _current_cache=current,
            step=step if step is not None else UnavailableRegionStep(),
            time=time if time is not None else UnavailableStepTime(),
        )

    @property
    def current(self) -> RegionCurrent:
        """Current containment state for this aircraft."""
        if self._current_cache is None:
            queryable, acidx = _require_bound_query_result(
                self._queryable, self._acidx, type(self).__name__
            )
            self._current_cache = RegionCurrent(
                inside=queryable.contains_aircraft(acidx)
            )
        return self._current_cache

    def __bool__(self) -> bool:
        return self.current.inside


@dataclass
class QueryRegion:
    """A named spatial region tested for containment per step.

    Parameters
    ----------
    bounds:
        The spatial (and optional altitude) region to test.
    color:
        Display color name recognised by drivers (e.g. ``"orange"``,
        ``"cyan"``, ``"#FF8800"``).  Defaults to ``"orange"``.
    render_shape:
        Whether the region polygon is drawn on the map.  ``True`` by
        default; set ``False`` to keep the region as a query-only
        construct with no visual outline.  When ``False`` the label is
        suppressed too (it has no shape to anchor to).
    render_label:
        Whether the region's name is drawn next to its outline.  ``True``
        by default; set ``False`` to draw the polygon without text.
    track_temporal_state:
        Whether this queryable is sampled every physics substep to provide
        ``during_step``, accumulated ``time``, and step-minimum values.

    Examples
    --------
    ::

        from bluesky_sandbox.sim.bounds import (
            BoxFootprint,
            ConstantAltitudeBand,
            RegionBounds,
        )
        from bluesky_sandbox.sim.queryables import QueryRegion

        goal = QueryRegion(
            RegionBounds(
                footprint=BoxFootprint(
                    lat_min_deg=51.9,
                    lat_max_deg=52.1,
                    lon_min_deg=4.4,
                    lon_max_deg=4.6,
                ),
                altitude=ConstantAltitudeBand(2_000, 8_000),
            ),
            color="cyan",
        )
    """

    bounds: Bounds
    result_type: ClassVar[type[RegionResult]] = RegionResult
    color: str = "orange"
    render_shape: bool = True
    render_label: bool = True
    track_temporal_state: bool = False

    def contains_aircraft(self, acidx: int) -> bool:
        """``True`` iff the aircraft lies inside the region."""
        lat_deg = float(bs.traf.lat[acidx])
        lon_deg = float(bs.traf.lon[acidx])
        alt_ft = float(bs.traf.alt[acidx] / ft)
        return self.bounds.contains(lat_deg, lon_deg, alt_ft)
