"""Shared TSAS table data model for pygame and Panda3D views."""

from __future__ import annotations

import math
from dataclasses import dataclass

import bluesky as bs
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint


@dataclass(frozen=True)
class TsasRow:
    idx: int
    acid: str
    eta_s: float
    dist_nm: float
    alt_diff_ft: float
    state: str


@dataclass(frozen=True)
class TsasTable:
    name: str
    title: str
    color: str
    waypoint: Waypoint
    rows: list[TsasRow]


class TsasDataMixin:
    """Build toolkit-neutral TSAS tables from env queryables and bs.traf."""

    _MIN_CLOSING_KT = 1.0

    def tsas_tables(self, driver, max_rows: int | None = None) -> list[TsasTable]:
        env = driver._env
        if env is None:
            return []
        tables: list[TsasTable] = []
        for name, waypoint in self.tsas_waypoints(driver):
            rows = self.tsas_aircraft_rows(waypoint, driver)
            if max_rows is not None:
                rows = rows[:max_rows]
            tables.append(
                TsasTable(
                    name=name,
                    title=self.tsas_title(name, waypoint),
                    color=waypoint.color,
                    waypoint=waypoint,
                    rows=rows,
                )
            )
        return tables

    def tsas_waypoints(self, driver) -> list[tuple[str, Waypoint]]:
        env = driver._env
        if env is None:
            return []
        return [
            (name, queryable)
            for name, queryable in env.episode_queryables.items()
            if isinstance(queryable, Waypoint) and self.tsas_show_waypoint(queryable)
        ]

    def tsas_aircraft_rows(self, waypoint: Waypoint, driver) -> list[TsasRow]:
        region = (
            self.tsas_resolve_region(driver, waypoint.tsas_region)
            if waypoint.tsas_region
            else None
        )
        rows: list[TsasRow] = []
        for idx in range(bs.traf.ntraf):
            lat, lon = bs.traf.lat[idx], bs.traf.lon[idx]
            alt_ft = bs.traf.alt[idx] / ft
            if region is not None and not region.bounds.contains(lat, lon, alt_ft):
                continue

            result = waypoint.result_type.for_aircraft(waypoint, idx)
            current = result.current
            gs_kts = bs.traf.gs[idx] / kts
            angle_deg = (current.bearing_deg - bs.traf.hdg[idx] + 540) % 360 - 180
            closing_kts = gs_kts * math.cos(math.radians(angle_deg))
            eta_s = (
                current.distance_nm / closing_kts * 3600.0
                if closing_kts > self._MIN_CLOSING_KT
                else math.inf
            )
            acid = bs.traf.id[idx]
            rows.append(
                TsasRow(
                    idx=idx,
                    acid=acid,
                    eta_s=eta_s,
                    dist_nm=current.distance_nm,
                    alt_diff_ft=current.alt_diff_ft,
                    state=driver._aircraft_state(acid),
                )
            )
        rows.sort(key=lambda row: (row.eta_s, row.dist_nm))
        return rows

    @staticmethod
    def tsas_title(name: str, waypoint: Waypoint) -> str:
        if waypoint.alt_ft is None:
            return name
        return f"{name}  FL{int(round(waypoint.alt_ft / 100)):03d}"

    @staticmethod
    def tsas_show_waypoint(waypoint: Waypoint) -> bool:
        return (
            waypoint.render_shape
            if waypoint.render_tsas is None
            else waypoint.render_tsas
        )

    @staticmethod
    def tsas_eta_text(eta_s: float) -> str:
        if not math.isfinite(eta_s):
            return "--:--"
        secs = max(0, int(round(eta_s)))
        return f"{secs // 60:02d}:{secs % 60:02d}"

    @staticmethod
    def tsas_dtm_text(dist_nm: float) -> str:
        return f"{dist_nm:.0f}NM"

    @staticmethod
    def tsas_resolve_region(driver, name: str) -> QueryRegion:
        queryables = driver._env.episode_queryables
        if name not in queryables:
            raise AssertionError(
                f"Waypoint.tsas_region={name!r} not found in episode queryables "
                f"(known: {sorted(queryables)})"
            )
        region = queryables[name]
        if not isinstance(region, QueryRegion):
            raise AssertionError(
                f"Waypoint.tsas_region={name!r} must point to a QueryRegion, "
                f"got {type(region).__name__}"
            )
        return region
