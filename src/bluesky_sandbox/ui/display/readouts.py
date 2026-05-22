"""Helpers for building GUI readout annotations."""

from __future__ import annotations

from collections.abc import Mapping

from bluesky_sandbox.interface.task import (
    WaypointReadoutItem,
    WaypointReadoutKey,
    WaypointReadoutNamespace,
    WaypointReadoutTarget,
)
from bluesky_sandbox.sim.queryables import Waypoint


def waypoint_readouts(
    waypoint: Waypoint,
    target: WaypointReadoutTarget,
    *,
    label: str | None = None,
    target_alt_ft: float | None = None,
    radius_nm: float | None = None,
    alt_tolerance_ft: float | None = None,
    speed_kts: float | None = None,
    speed_min_kts: float | None = None,
    speed_max_kts: float | None = None,
    speed_tolerance_kts: float | None = None,
    include_altitude: bool = True,
    include_radius: bool = True,
    include_speed: bool = False,
    metadata: Mapping[str | WaypointReadoutKey, object] | None = None,
    constraints: Mapping[str | WaypointReadoutKey, object] | None = None,
    custom: Mapping[str | WaypointReadoutKey, object] | None = None,
) -> tuple[WaypointReadoutItem, ...]:
    """Build standard GUI readout items for a waypoint."""
    items: list[WaypointReadoutItem] = []

    def add(
        namespace: WaypointReadoutNamespace,
        key: str | WaypointReadoutKey,
        value: object,
    ) -> None:
        if value is not None:
            items.append(WaypointReadoutItem(target, namespace, key, value))

    add(WaypointReadoutNamespace.METADATA, WaypointReadoutKey.NAME, label)
    if include_altitude:
        add(
            WaypointReadoutNamespace.METADATA,
            WaypointReadoutKey.TARGET_ALT_FT,
            waypoint.alt_ft if target_alt_ft is None else target_alt_ft,
        )
    if include_radius:
        add(
            WaypointReadoutNamespace.CONSTRAINTS,
            WaypointReadoutKey.RADIUS_NM,
            waypoint.reach_radius_nm if radius_nm is None else radius_nm,
        )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.ALT_TOLERANCE_FT,
        alt_tolerance_ft,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_KTS,
        waypoint.speed_kts if include_speed and speed_kts is None else speed_kts,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_MIN_KTS,
        speed_min_kts,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_MAX_KTS,
        speed_max_kts,
    )
    add(
        WaypointReadoutNamespace.CONSTRAINTS,
        WaypointReadoutKey.SPEED_TOLERANCE_KTS,
        speed_tolerance_kts,
    )
    for key, value in (metadata or {}).items():
        add(WaypointReadoutNamespace.METADATA, key, value)
    for key, value in (constraints or {}).items():
        add(WaypointReadoutNamespace.CONSTRAINTS, key, value)
    for key, value in (custom or {}).items():
        add(WaypointReadoutNamespace.CONSTRAINTS, key, value)
    return tuple(items)
