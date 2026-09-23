from __future__ import annotations

import math
from collections.abc import Sequence

import bluesky as bs
from bluesky.core import simtime
from bluesky.stack import simstack
from bluesky.tools.aero import ft, kts
from wurlitzer import pipes

from bluesky_sandbox.sim.queryables import WaypointTarget
from bluesky_sandbox.sim.weather import WindField

# BlueSky is a process-global singleton: ``bs.init`` re-imports the plugins and
# re-registers every stack command, and re-registration is not idempotent - the
# command table already holds each name, so the second pass prints an "Attempt to
# reimplement <CMD>" line per command (~60 lines) and keeps the original
# callback. Constructing a second env in one process is routine (eval alongside
# train, a measurement script, the designer preview), so remember whether init
# has run and skip it. Stores the performance model it initialised with, because
# ``bs.settings.performance_model`` is only read *during* init - a later env
# asking for a different one would be silently ignored.
_BLUESKY_PERFORMANCE_MODEL: str | None = None


class BlueSkyRuntime:
    """BlueSky's process simulator interface."""

    def __init__(self, env=None) -> None:
        self.env = env

    def bind_env(self, env) -> None:
        self.env = env

    def _config(self):
        if self.env is None:
            raise RuntimeError("BlueSkyRuntime env has not been set.")
        return self.env.config

    @property
    def agent_ids(self):
        return bs.traf.id

    @property
    def sim_time(self) -> float:
        return bs.sim.simt

    def index(self, acid: str) -> int:
        return bs.traf.id.index(acid)

    def configure(self) -> None:
        global _BLUESKY_PERFORMANCE_MODEL
        config = self._config()
        requested = config.performance_model
        if _BLUESKY_PERFORMANCE_MODEL is None:
            bs.settings.performance_model = requested
            bs.init(mode="sim", group_id="S")
            # bs.init re-reads settings.cfg, which usually pins a model of its
            # own - restore what this env asked for, or BlueSky flies the file's
            # choice while everything else believes the design's.
            bs.settings.performance_model = requested
            _BLUESKY_PERFORMANCE_MODEL = requested
        elif requested != _BLUESKY_PERFORMANCE_MODEL:
            raise RuntimeError(
                f"BlueSky is already initialised with performance model "
                f"{_BLUESKY_PERFORMANCE_MODEL!r}; this env asks for {requested!r}. "
                "The model is fixed at bs.init and cannot be switched in-process - "
                "run the two envs in separate processes."
            )
        self.configure_timestep()

    def configure_timestep(self) -> None:
        config = self._config()
        bs.settings.simdt = config.simdt
        simtime.setdt(config.simdt)

    def reset(self, *, seed: int | None) -> None:
        with pipes():
            bs.sim.reset()
        self.configure_timestep()
        if seed is not None:
            bs.sim.setseed(seed)
        self.configure_conflict_management()
        # ``bs.sim.reset()`` clears the wind field. The environment owns the
        # ``WindField``, so it re-applies it immediately after this returns.

    def operate(self) -> None:
        bs.sim.op()

    def configure_conflict_management(self) -> None:
        # bs.sim.reset() doesn't reliably keep these, so (re-)issue them every
        # episode. Detection (CDMETHOD) always runs; resolution (RESO) defaults
        # off so the agent resolves conflicts itself.
        config = self._config()
        bs.stack.stack(f"CDMETHOD {config.cd_method}")
        bs.stack.stack(f"RESO {config.reso_method or 'OFF'}")
        if config.pz_radius_nm is not None:
            bs.stack.stack(f"ZONER {config.pz_radius_nm}")
        if config.pz_height_ft is not None:
            bs.stack.stack(f"ZONEDH {config.pz_height_ft}")
        if config.lookahead_s is not None:
            bs.stack.stack(f"DTLOOK {config.lookahead_s}")
        simstack.process()

    def apply_wind(self, wind: WindField | None) -> None:
        """Push a wind field's current vector into BlueSky.

        One point yields a spatially uniform field, so the location is
        irrelevant. ``addpointvne`` takes the (north, east) components
        directly - the vector never round-trips through magnitude/bearing and
        back, which the old ``addpoint`` path did only to have BlueSky undo it.

        The field itself decides what "current" means (mean plus any live
        gust); this method owns no wind logic of its own.
        """
        if wind is None or wind.is_still:
            return
        vn, ve = wind.current_ne_ms()
        bs.traf.wind.clear()
        if math.hypot(vn, ve) >= 1e-6:
            bs.traf.wind.addpointvne(0.0, 0.0, vn, ve)

    def create_aircraft(
        self,
        callsign: str,
        actype: str,
        lat_deg: float,
        lon_deg: float,
        heading_deg: float,
        alt_ft: float,
        spd_kts: float,
    ) -> None:
        # Public sandbox inputs use aviation units; BlueSky's Traffic API
        # expects altitude in metres and CAS in metres/second.
        bs.traf.cre(
            callsign,
            actype,
            lat_deg,
            lon_deg,
            heading_deg,
            alt_ft * ft,
            spd_kts * kts,
        )
        # ``cre`` leaves the aircraft in the NA flight phase until the first
        # perf update, and OpenAP maps ``NA -> vmin = 0``. Envelope-speed
        # sampling (feasible_cas_at_alt / feasible_alt_cas) runs immediately
        # after creation, so classify the phase now: ``perf.update`` recomputes
        # the phase and the speed/limit envelope from current state and ignores
        # ``dt`` entirely - no clock advance, no movement, no fuel burn.
        bs.traf.perf.update(0.0)

    def append_aircraft_route(
        self,
        callsign: str,
        route: Sequence[WaypointTarget] | None,
        commit: bool = True,
    ) -> None:
        self._stack_aircraft_route(callsign, route)
        if commit:
            simstack.process()

    def replace_aircraft_route(
        self,
        callsign: str,
        route: Sequence[WaypointTarget] | None,
        commit: bool = True,
    ) -> None:
        bs.stack.stack(f"DELRTE {callsign}")
        self._stack_aircraft_route(callsign, route)
        if commit:
            simstack.process()

    def _stack_aircraft_route(
        self,
        callsign: str,
        route: Sequence[WaypointTarget] | None,
    ) -> None:
        if not route:
            return

        for target in route:
            waypoint_ref = (
                f"{target.lat:.8f},{target.lon:.8f}"
                if target.waypoint is None
                else target.waypoint
            )
            self._stack_addwpt(
                callsign,
                waypoint_ref,
                target.alt_ft,
                target.speed_kts,
            )

    def _stack_addwpt(
        self,
        callsign: str,
        waypoint_ref: str,
        alt_ft: float | None,
        speed_kts: float | None,
    ) -> None:
        if speed_kts is not None:
            if alt_ft is None:
                raise ValueError(
                    "BlueSky route speed constraints require alt_ft. "
                    f"Got speed_kts={speed_kts!r} for waypoint {waypoint_ref!r}."
                )
            bs.stack.stack(
                f"ADDWPT {callsign} {waypoint_ref},{alt_ft:.1f},{speed_kts:.1f}"
            )
        elif alt_ft is not None:
            bs.stack.stack(f"ADDWPT {callsign} {waypoint_ref},{alt_ft:.1f}")
        else:
            bs.stack.stack(f"ADDWPT {callsign} {waypoint_ref}")

    def delete_aircraft(self, acid: str) -> None:
        if acid in bs.traf.id:
            bs.traf.delete(bs.traf.id.index(acid))
