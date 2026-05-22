"""Shared aircraft, status, and waypoint readouts for GUI drivers."""

from __future__ import annotations

import math

import bluesky as bs
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.interface.task import AircraftReadoutItem, WaypointReadoutItem
from bluesky_sandbox.sim.performance.speeds import crossover_display


class AircraftReadoutMixin:
    """Formatting and route helpers shared by pygame and Panda3D drivers."""

    _INFO_LABEL_WIDTH = 4
    # Above this altitude the marker blob appends Mach (the meaningful/limiting
    # speed near the CAS/Mach crossover); below it, ground speed alone suffices.
    _MACH_LABEL_ALT_FT = 29000.0

    def format_status_line(self) -> str:
        """Return the common two-line runtime status badge."""
        simt = max(0, int(bs.sim.simt))
        clock = f"T+{simt // 3600:02d}:{(simt % 3600) // 60:02d}:{simt % 60:02d}"
        if self.realtime:
            speed = (
                "REAL"
                if self._desired_dtmult == 1.0
                else f"REAL {self._desired_dtmult:g}x"
            )
        else:
            speed = "FAST"
        mode = "PAUSED" if self._paused else "RUNNING"
        line1 = f"{clock}  |  {mode}  |  {speed}"
        if self._env is None:
            return line1
        spawn_progress = self._env.episode_spawn_progress
        live = bs.traf.ntraf
        cap = self._env.episode_spawn.max_aircraft()
        return (
            f"{line1}\nSPAWNED {spawn_progress.spawned}/"
            f"{spawn_progress.scheduled}  |  LIVE {live}/{cap}"
        )

    def format_aircraft_info_lines(
        self,
        acid: str,
        *,
        degree_suffix: str = " DEG",
    ) -> list[str]:
        """Render the selected-aircraft info block as HUD text lines."""
        idx = self._live_index().get(acid)
        if idx is None:
            return []
        fl = int(round(bs.traf.alt[idx] / ft / 100))
        gs = int(round(bs.traf.gs[idx] / kts))
        cas = int(round(bs.traf.cas[idx] / kts))
        mach = float(bs.traf.M[idx])
        hdg = int(round(bs.traf.hdg[idx])) % 360
        info = self._aircraft_snapshot(acid)
        state = self._aircraft_state(acid)
        lines = [
            acid,
            self._info_row("TYPE", bs.traf.type[idx]),
            self._info_row("ALT", f"FL{fl:03d}"),
            self._info_row("GS", f"{gs:>3} KT"),
            self._info_row("CAS", f"{cas:>3} KT"),
            # Mach is the limiting/meaningful speed above the CAS/Mach crossover
            # (cruise), where the envelope is Mach-limited - and it makes the
            # crossover speed action's regime switch visible.
            self._info_row("MACH", f"{mach:.2f}"),
            self._info_row("HDG", f"{hdg:03d}{degree_suffix}"),
        ]
        separation = info.get("separation", {})
        safety_lines: list[str] = []
        if state == "los":
            partners = separation.get("los", {}).get("partners") or []
            safety_lines.append(
                self._info_row("LOS", ", ".join(partners) if partners else "YES")
            )
        elif state == "conflict":
            partners = separation.get("conflict", {}).get("partners") or []
            safety_lines.append(
                self._info_row("CONF", ", ".join(partners) if partners else "YES")
            )
        if safety_lines:
            lines.append(self._separator_row())
            lines.extend(safety_lines)

        task_lines: list[str] = []
        if self._env is not None:
            for item in self._env.define_aircraft_readouts(acid):
                task_lines.append(self._aircraft_readout_row(item))
        if task_lines:
            lines.append(self._separator_row())
            lines.extend(task_lines)
        return lines

    def format_waypoint_speed_lines(
        self,
        idx: int,
        *,
        target_kts: float | None = None,
        min_kts: float | None = None,
        max_kts: float | None = None,
        tolerance_kts: float | None = None,
        alt_ft: float | None = None,
    ) -> list[str]:
        """Waypoint speed-constraint label line(s), CAS below / Mach above crossover.

        Shows the target (or min-max band) in **Mach** when the waypoint lies
        above the aircraft's CAS/Mach crossover altitude - matching how the
        crossover speed action holds the target there - and **CAS** below. Falls
        back to CAS when the waypoint has no altitude (the regime is undecidable).
        ``idx`` is the live aircraft (supplies Mmo); ``alt_ft`` is the waypoint's.
        """
        ref_kts = (
            max_kts if max_kts is not None
            else target_kts if target_kts is not None
            else min_kts
        )
        in_mach = False
        if alt_ft is not None and ref_kts is not None:
            in_mach, _ = crossover_display(idx, float(ref_kts) * kts, float(alt_ft) * ft)

        def to_mach(cas_kts: float) -> float:
            return crossover_display(idx, float(cas_kts) * kts, float(alt_ft) * ft)[1]

        lines: list[str] = []
        if min_kts is not None or max_kts is not None:
            if in_mach:
                lo = "--" if min_kts is None else f"{to_mach(min_kts):.2f}"
                hi = "--" if max_kts is None else f"{to_mach(max_kts):.2f}"
                lines.append(f"MACH {lo}-{hi}")
            else:
                lo = "--" if min_kts is None else int(round(float(min_kts)))
                hi = "--" if max_kts is None else int(round(float(max_kts)))
                lines.append(f"CAS  {lo}-{hi} KT")
        elif target_kts is not None:
            if in_mach:
                lines.append(f"MACH {to_mach(target_kts):.2f}")
            else:
                lines.append(f"CAS  {int(round(float(target_kts))):>3} KT")
            if tolerance_kts is not None:
                if in_mach:
                    # Half the Mach span across the CAS tolerance band. The upper
                    # edge caps at Mmo above the crossover, so a symmetric span
                    # (vs. target+tol alone) still yields a meaningful width.
                    band = abs(
                        to_mach(float(target_kts) + float(tolerance_kts))
                        - to_mach(float(target_kts) - float(tolerance_kts))
                    ) / 2.0
                    # Omit rather than show "+/-0.00 M" when the whole CAS band
                    # sits above Mmo (target pinned at the Mach ceiling).
                    if round(band, 2) > 0.0:
                        lines.append(f"SPD  +/-{band:.2f} M")
                else:
                    lines.append(f"SPD  +/-{int(round(float(tolerance_kts)))} KT")
        return lines

    @classmethod
    def _info_row(cls, label: str, value: object) -> str:
        return f"{label:<{cls._INFO_LABEL_WIDTH}}  {value}"

    @classmethod
    def _separator_row(cls) -> str:
        return "-" * (cls._INFO_LABEL_WIDTH + 2 + 12)

    @classmethod
    def _aircraft_readout_row(cls, item: AircraftReadoutItem) -> str:
        return cls._info_row(item.label, item.value)

    def format_aircraft_marker_label_lines(self, idx: int) -> list[str]:
        """Return compact live-marker label lines for human GUI views."""
        if not (0 <= idx < bs.traf.ntraf):
            return []
        acid = bs.traf.id[idx]
        try:
            actype = str(bs.traf.type[idx] or "")
        except (AttributeError, IndexError):
            actype = ""
        alt_ft = bs.traf.alt[idx] / ft
        fl = int(round(alt_ft / 100.0))
        gs = int(round(bs.traf.gs[idx] / kts))
        speed = f"GS{gs}"
        # Append the air-mass speed the aircraft is controlled in at this altitude:
        # Mach above the crossover threshold (where it's the limiting speed), CAS
        # below - alongside ground speed (map-relevant) either way.
        if alt_ft >= self._MACH_LABEL_ALT_FT:
            speed += f"  M{float(bs.traf.M[idx]):.2f}".replace("M0.", "M.")
        else:
            speed += f"  CAS{int(round(bs.traf.cas[idx] / kts))}"
        return [
            f"{acid}  {actype}" if actype else acid,
            f"FL{fl:03d}  {speed}",
        ]

    def format_aircraft_marker_label(self, idx: int) -> str:
        """Return compact live-marker label text joined for multiline renderers."""
        return "\n".join(self.format_aircraft_marker_label_lines(idx))

    def aircraft_route_waypoints(self, acid: str) -> list[dict]:
        """Return selected-aircraft route/goal waypoints as plain render data."""
        idx = self._live_index().get(acid)
        if idx is None:
            return []
        waypoints: list[dict] = []

        routes = bs.traf.ap.route
        if idx < len(routes):
            route = routes[idx]
            try:
                nwp = int(route.nwp or 0)
            except (TypeError, ValueError):
                nwp = 0
            names = route.wpname
            lats = route.wplat
            lons = route.wplon
            alts = route.wpalt
            speeds = route.wpspd
            active_idx = int(route.iactwp or 0)

            for wp_idx in range(nwp):
                try:
                    lat = float(lats[wp_idx])
                    lon = float(lons[wp_idx])
                except (IndexError, TypeError, ValueError):
                    continue
                if not (math.isfinite(lat) and math.isfinite(lon)):
                    continue
                name = ""
                try:
                    raw_name = names[wp_idx]
                    name = (
                        raw_name.decode()
                        if isinstance(raw_name, bytes)
                        else str(raw_name)
                    )
                    name = name.strip()
                except (IndexError, TypeError, ValueError):
                    pass
                alt_ft = None
                try:
                    alt_m = float(alts[wp_idx])
                    if math.isfinite(alt_m) and alt_m > 0.0:
                        alt_ft = alt_m / ft
                except (IndexError, TypeError, ValueError):
                    pass
                speed_kts = None
                try:
                    speed_ms = float(speeds[wp_idx])
                    if math.isfinite(speed_ms) and speed_ms >= 0.0:
                        speed_kts = speed_ms / kts
                except (IndexError, TypeError, ValueError):
                    pass
                reached = wp_idx < active_idx
                waypoints.append(
                    {
                        "index": wp_idx,
                        "display_index": len(waypoints),
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "alt_ft": alt_ft,
                        "speed_kts": speed_kts,
                        "active": wp_idx == active_idx,
                        "reached": reached,
                        "future": wp_idx >= active_idx and not reached,
                    }
                )

        if waypoints and self._env is not None:
            for item in self._env.define_waypoint_readouts(acid):
                target = self._readout_target_waypoint(waypoints, item)
                if target is None:
                    continue
                bucket = target.setdefault(item.namespace.value, {})
                bucket[item.key] = item.value

        return waypoints

    @staticmethod
    def _readout_target_waypoint(
        waypoints: list[dict],
        item: WaypointReadoutItem,
    ) -> dict | None:
        target = item.target
        matches = [
            wp
            for wp in waypoints
            if AircraftReadoutMixin._readout_target_matches(wp, item)
        ]
        if not matches:
            return None
        if target.name is None and target.index is None and target.future is True:
            return matches[0]
        return matches[0]

    @staticmethod
    def _readout_target_matches(waypoint: dict, item: WaypointReadoutItem) -> bool:
        target = item.target
        if target.index is not None and int(waypoint.get("index", -1)) != target.index:
            return False
        if target.name is not None:
            waypoint_name = str(waypoint.get("name") or "").strip()
            if waypoint_name.upper() != target.name.strip().upper():
                return False
        if (
            target.active is not None
            and bool(waypoint.get("active")) is not target.active
        ):
            return False
        return not (target.future is not None and bool(waypoint.get("future")) is not target.future)
