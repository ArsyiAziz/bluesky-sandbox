"""The ``Queryable`` protocol and the helpers every queryable kind shares."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import bluesky as bs

_MIN_DYNAMIC_SCALE = 1e-6


def _ensure_navdb_loaded() -> None:
    """Populate ``bs.navdb`` without instantiating Traffic/Simulation.

    Replicates the prefix of :func:`bluesky.init` up to and including the
    navdatabase load so we can resolve waypoint names before the env is
    constructed. Idempotent - does nothing if ``bs.navdb`` is already
    populated. :class:`BlueskyBaseEnvironment` will still run the full
    ``bs.init`` later; this only primes the navdb early.
    """
    if bs.navdb is not None and hasattr(bs.navdb, "wpid"):
        return
    from bluesky import pathfinder, refdata, settings
    from bluesky import tools as _bs_tools
    from bluesky.navdatabase import Navdatabase

    pathfinder.init()
    settings.init()
    _bs_tools.init()
    refdata.RefData()
    bs.navdb = Navdatabase()


@runtime_checkable
class Queryable(Protocol):
    """Marker protocol for scenario resources evaluated through context.query."""

    result_type: type[Any]


def _require_bound_query_result(
    queryable: Any | None,
    acidx: int | None,
    result_name: str,
) -> tuple[Any, int]:
    """Return the bound query context or fail with a clear API error."""
    if queryable is None or acidx is None:
        raise RuntimeError(
            f"{result_name} is not bound to a queryable aircraft. "
            f"Construct it with {result_name}.for_aircraft(...), or provide "
            "explicit current/target/route values."
        )
    return queryable, acidx
