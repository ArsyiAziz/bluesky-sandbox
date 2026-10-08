"""Shared aircraft, status, and waypoint readouts for GUI drivers."""

from __future__ import annotations

import math
from functools import lru_cache

import bluesky as bs
from bluesky.tools.aero import ft, kts, vcas2mach

from bluesky_sandbox.interface.fields._route import _active_route_waypoint
from bluesky_sandbox.interface.fields._state import arrival_time
from bluesky_sandbox.interface.fields.observations import (
    ActiveRouteWaypointArrivalErrorS,
)
from bluesky_sandbox.interface.task import (
    AircraftReadoutItem,
    WaypointReadoutItem,
    aircraft_readout_items,
)
from bluesky_sandbox.sim.performance.speeds import as_cas_ms, crossover_display, selected_cas_ms


#: Above this altitude a label gives Mach, the speed flown there; below, CAS.
MACH_LABEL_ALT_FT = 29000.0


#: Climbing or descending faster than this (ft/min), a label shows it.
LEVEL_FPM = 200.0
#: Further than this from its selected speed - knots of CAS, or Mach - an
#: aircraft is accelerating or slowing toward it, and a label shows it.
SPEED_TREND_KTS = 2.0
SPEED_TREND_MACH = 0.005
#: The arrows a label shows for up and down; ``ASCII_TRENDS`` where the font
#: has no arrows.
ARROW_TRENDS = ("\u2191", "\u2193")
ASCII_TRENDS = ("^", "v")


def _trend(sign: int, glyphs: tuple[str, str]) -> str:
    return glyphs[0] if sign > 0 else glyphs[1] if sign < 0 else " "


def aircraft_label_lines(
    acid: str,
    actype: str,
    *,
    alt_ft: float,
    gs_kts: float,
    cas_kts: float,
    mach: float,
    alt_trend: int = 0,
    speed_trend: int = 0,
    glyphs: tuple[str, str] = ARROW_TRENDS,
) -> list[str]:
    """An aircraft's marker label: callsign and type, then flight level, ground
    speed and the air-mass speed it is controlled in at that altitude - Mach
    above the crossover threshold, CAS below. An arrow after the level says it
    is climbing (``alt_trend`` > 0) or descending, one after the speed that it
    is speeding up (``speed_trend`` > 0) or slowing down."""
    fl = int(round(alt_ft / 100.0))
    if alt_ft >= MACH_LABEL_ALT_FT:
        air = f"M{mach:.2f}".replace("M0.", "M.")
    else:
        air = f"CAS{int(round(cas_kts))}"
    return [
        f"{acid}  {actype}" if actype else acid,
        f"FL{fl:03d}{_trend(alt_trend, glyphs)} GS{int(round(gs_kts))}  {air}{_trend(speed_trend, glyphs)}".rstrip(),
    ]


@lru_cache(maxsize=256)
def _type_tag_line(actype: str) -> str:
    """A type's tags on one line - ``Fixed-wing · Jet · Wake Heavy`` - under
    the model BlueSky flies now; empty where it has none."""
    from bluesky_sandbox.sim.performance.models import type_info  # noqa: PLC0415

    model = str(getattr(bs.settings, "performance_model", "openap") or "openap")
    try:
        info = type_info(str(actype), model)
    except Exception:  # noqa: BLE001 - a readout never breaks the frame
        return ""
    # The mass is in the info block's own terms elsewhere: the words only.
    return " · ".join(t for t in (info or {}).get("tags", [])[:3])


class AircraftReadoutMixin:
    """Formatting and route helpers shared by pygame and Panda3D drivers."""

    _INFO_LABEL_WIDTH = 4
    # Above this altitude the marker blob appends Mach (the meaningful/limiting
    # speed near the CAS/Mach crossover); below it, ground speed alone suffices.
    _MACH_LABEL_ALT_FT = MACH_LABEL_ALT_FT
    #: The up and down arrows a label shows: a driver whose font lacks them
    #: sets ``ASCII_TRENDS``.
    trend_glyphs: tuple[str, str] = ARROW_TRENDS

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
            # What it is - fixed-wing, helicopter, drone; propulsion, wake -
            # from the performance model's own data.
            *([self._info_row("", tags)] if (tags := _type_tag_line(bs.traf.type[idx])) else []),
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
            for item in aircraft_readout_items(self._env.define_aircraft_readouts(acid)):
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
        scheduled: bool = False,
    ) -> list[str]:
        """Waypoint speed-constraint label line(s), CAS below / Mach above crossover.

        Shows the target (or min-max band) in **Mach** when the waypoint lies
        above the aircraft's CAS/Mach crossover altitude - matching how the
        crossover speed action holds the target there - and **CAS** below. Falls
        back to CAS when the waypoint has no altitude (the regime is undecidable).
        ``idx`` is the live aircraft (supplies Mmo); ``alt_ft`` is the waypoint's.
        ``scheduled`` marks a speed the fix's arrival time implies rather than a
        gate: ``TBO``.
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
        if scheduled and lines:
            lines[0] += " TBO"
        return lines

    @staticmethod
    def format_waypoint_time_lines(wp: dict) -> list[str]:
        """The fix's arrival-time line: how late (+) or early (-) the aircraft
        would be over it flying on as it is; none for a fix without a time."""
        error_s = wp.get("arrival_error_s")
        return [] if error_s is None else [f"TIME {float(error_s):+.0f} S"]

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
        """The aircraft's label lines, as ``aircraft_labels`` says: its full
        data block, its callsign alone, or none - the tracked aircraft's is
        always full."""
        if not (0 <= idx < bs.traf.ntraf):
            return []
        mode = getattr(self, "aircraft_labels", "full")
        if mode != "full":
            tracked = self.tracked_acid() if hasattr(self, "tracked_acid") else None
            if bs.traf.id[idx] != tracked:
                return [bs.traf.id[idx]] if mode == "callsign" else []
        try:
            actype = str(bs.traf.type[idx] or "")
        except (AttributeError, IndexError):
            actype = ""
        return aircraft_label_lines(
            bs.traf.id[idx],
            actype,
            alt_ft=bs.traf.alt[idx] / ft,
            gs_kts=bs.traf.gs[idx] / kts,
            cas_kts=bs.traf.cas[idx] / kts,
            mach=float(bs.traf.M[idx]),
            **self._label_trends(idx),
            glyphs=self.trend_glyphs,
        )

    def _label_trends(self, idx: int) -> dict[str, int]:
        """Whether the aircraft climbs or descends, and speeds up or slows
        down - toward its selected speed, in the quantity its label gives
        (Mach above the crossover threshold, CAS below)."""
        vs_fpm = float(bs.traf.vs[idx]) / ft * 60.0
        alt = 1 if vs_fpm > LEVEL_FPM else -1 if vs_fpm < -LEVEL_FPM else 0
        target_ms = float(selected_cas_ms(idx)[0])
        if float(bs.traf.alt[idx]) / ft >= MACH_LABEL_ALT_FT:
            gap = float(vcas2mach(target_ms, float(bs.traf.alt[idx]))) - float(bs.traf.M[idx])
            threshold = SPEED_TREND_MACH
        else:
            gap = (target_ms - float(bs.traf.cas[idx])) / kts
            threshold = SPEED_TREND_KTS
        speed = 1 if gap > threshold else -1 if gap < -threshold else 0
        return {"alt_trend": alt, "speed_trend": speed}

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
                        # A Mach (a scenario's route) as its CAS at the fix.
                        at_m = float(bs.traf.alt[idx]) if alt_ft is None else alt_ft * ft
                        speed_kts = float(as_cas_ms(speed_ms, at_m)) / kts
                except (IndexError, TypeError, ValueError):
                    pass
                reached = wp_idx < active_idx
                # Ahead of the aircraft, the fix as the library reads it: a
                # speed its arrival time implies where it has no gate, and how
                # late or early the aircraft would be over it.
                scheduled = False
                arrival_error_s = None
                if wp_idx >= active_idx:
                    offset = wp_idx - active_idx
                    fix = _active_route_waypoint(idx, offset)
                    if speed_kts is None and fix is not None and fix[3] is not None:
                        speed_kts, scheduled = fix[3] / kts, True
                    if arrival_time(idx, wp_idx) is not None:
                        arrival_error_s = float(
                            ActiveRouteWaypointArrivalErrorS(route_offset=offset).get(idx)
                        )
                waypoints.append(
                    {
                        "index": wp_idx,
                        "display_index": len(waypoints),
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "alt_ft": alt_ft,
                        "speed_kts": speed_kts,
                        "scheduled": scheduled,
                        "arrival_error_s": arrival_error_s,
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
