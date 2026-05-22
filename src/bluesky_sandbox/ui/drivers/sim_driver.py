"""Headless sim driver - base class for all BlueSky PettingZoo drivers."""

from __future__ import annotations

import time
from contextlib import contextmanager

import bluesky as bs

from bluesky_sandbox.interface.task import AircraftRenderState

AircraftState = AircraftRenderState


class SimDriver:
    """Headless sim driver - the base class for all sim drivers.

    Owns the sim step loop and exposes no-op hooks for GUI operations so that
    the environment can always call the same interface regardless of whether a
    window is open.

    Parameters
    ----------
    realtime:
        When ``True`` each fixed ``bs.sim.step()`` is followed by wall-clock
        pacing based on ``bs.sim.dtmult``. The simulator timestep remains
        fixed by ``EnvConfig.simdt``; realtime only affects how long the call
        blocks. When ``False`` (default), the simulation runs as fast as the
        CPU allows.
    """

    def __init__(self, realtime: bool = False) -> None:
        self.realtime = realtime
        self._started: bool = False
        # Cached env reference set by bind_env(), so update() and helpers can
        # derive render snapshots without the env passing itself every frame.
        self._env = None
        self._aircraft_snapshot_cache: dict[str, dict] = {}
        self._aircraft_state_cache: dict[str, AircraftState] = {}
        self._aircraft_safety_pairs_cache: (
            dict[str, tuple[tuple[str, str], ...]] | None
        ) = None
        self._live_index_cache: dict[str, int] = {}
        self._sim_wall_target: float = 0.0

    def bind_env(self, env) -> None:
        self._env = env

    def start(self) -> None:
        """Initialise any resources needed by this driver."""
        self._started = True

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        """Block until the driver is ready (no-op in headless mode)."""

    def update(self) -> None:
        """Flush GUI state and process incoming commands (no-op in base)."""

    def step(self) -> None:
        """Flush the GUI, respect pause/FF, then advance the simulation.

        This is the single call the environment makes per substep.  Subclasses
        override ``update()`` to add GUI I/O; the step advancement logic here
        is shared by all drivers.
        """
        with self._aircraft_snapshot_cache_scope():
            self.update()
            was_fastforward = self._fastforward_active()
            bs.sim.step()
            self._finish_fastforward()
            if not was_fastforward:
                self._wait_realtime()

    def _fastforward_active(self) -> bool:
        return bool(getattr(bs.sim, "ffmode", False) and bs.sim.state == bs.OP)

    def _finish_fastforward(self) -> None:
        ffstop = getattr(bs.sim, "ffstop", None)
        if ffstop is not None and bs.sim.simt >= ffstop:
            bs.sim.op()

    def _wait_realtime(self) -> None:
        """Pace a fixed simulator step without letting wall-clock alter simdt."""
        if not self.realtime or self._fastforward_active():
            return

        simdt = float(getattr(bs.sim, "simdt", 0.0) or 0.0)
        dtmult = max(float(getattr(bs.sim, "dtmult", 1.0) or 1.0), 1e-6)
        budget = simdt / dtmult
        if budget <= 0.0:
            return

        now = time.monotonic()
        self._sim_wall_target = max(self._sim_wall_target, now - budget) + budget
        remaining = self._sim_wall_target - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def on_reset(self, env=None) -> None:
        """Called after the environment resets."""
        if env is not None:
            self.bind_env(env)
        self._clear_aircraft_snapshot_cache()

    def on_render(self, env=None) -> None:
        """Called during ``render()`` (no-op in headless mode)."""
        if env is not None:
            self.bind_env(env)
        self._clear_aircraft_snapshot_cache()

    def close(self) -> None:
        """Release any resources held by this driver."""

    # ------------------------------------------------------------------
    # Render snapshot helpers (read-only)
    # ------------------------------------------------------------------

    def _aircraft_snapshot(self, acid: str) -> dict:
        """Return driver-owned display data for *acid*."""
        if not self._aircraft_snapshot_cache:
            self._refresh_aircraft_snapshot_cache()
        return self._aircraft_snapshot_cache.get(acid, {})

    def _clear_aircraft_snapshot_cache(self) -> None:
        self._aircraft_snapshot_cache.clear()
        self._aircraft_state_cache.clear()
        self._aircraft_safety_pairs_cache = None
        self._live_index_cache.clear()

    @contextmanager
    def _aircraft_snapshot_cache_scope(self):
        """Keep driver aircraft caches scoped to one simulator advancement."""
        self._clear_aircraft_snapshot_cache()
        try:
            yield
        finally:
            self._clear_aircraft_snapshot_cache()

    def _live_index(self) -> dict[str, int]:
        if not self._live_index_cache:
            self._live_index_cache = {acid: idx for idx, acid in enumerate(bs.traf.id)}
        return self._live_index_cache

    def _refresh_aircraft_snapshot_cache(self) -> None:
        if self._env is None:
            return
        snapshots: dict[str, dict] = {}
        live_index = self._live_index()
        for acid, acidx in live_index.items():
            control_state = self._env._aircraft_control_state.get(acid)
            control_state_value = None if control_state is None else control_state.value
            snapshots[acid] = {
                "acid": acid,
                "acidx": acidx,
                "controlled": control_state_value == "controlled",
                "background": control_state_value == "background",
                "separation": self._env._traffic_monitor.build_separation_info(
                    acid,
                    acidx,
                ),
            }
        self._aircraft_snapshot_cache = snapshots

    def _aircraft_state(self, acid: str) -> AircraftState:
        """Classify *acid*'s rendered separation state from current driver snapshot.

        Returns ``"los"`` (active loss of separation), ``"conflict"``
        (predicted conflict, separation still intact), or ``"normal"``.
        Subclasses map the returned state to their own palette. Task-specific
        statuses are exposed as aircraft readouts instead of marker states.
        """
        cached = self._aircraft_state_cache.get(acid)
        if cached is not None:
            return cached
        info = self._aircraft_snapshot(acid)
        separation = info.get("separation", {})
        los = separation.get("los", {})
        conflict = separation.get("conflict", {})
        if los.get("current", False) or los.get("during_step", False):
            state = "los"
        elif conflict.get("current", False) or conflict.get("during_step", False):
            state = "conflict"
        else:
            state = "normal"
        self._aircraft_state_cache[acid] = state
        return state

    def _aircraft_safety_pairs(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Return deduplicated current safety pairs from cached env separation info."""
        if self._aircraft_safety_pairs_cache is not None:
            return self._aircraft_safety_pairs_cache
        pairs: dict[str, set[tuple[str, str]]] = {
            "los": set(),
            "conflict": set(),
        }
        live = set(bs.traf.id)
        for acid in bs.traf.id:
            separation = self._aircraft_snapshot(acid).get("separation", {})
            for key in pairs:
                event = separation.get(key, {})
                active = bool(
                    event.get("current", False)
                    or event.get("during_step", False)
                )
                partners = tuple(event.get("partners") or ()) + tuple(
                    event.get("step_partners") or ()
                )
                if not active:
                    continue
                for partner in partners:
                    partner = str(partner)
                    if partner not in live or partner == acid:
                        continue
                    pairs[key].add(tuple(sorted((acid, partner))))

        # LoS is the stronger visual cue; do not also draw the same pair as a
        # predicted conflict.
        pairs["conflict"].difference_update(pairs["los"])
        self._aircraft_safety_pairs_cache = {
            key: tuple(sorted(value))
            for key, value in pairs.items()
        }
        return self._aircraft_safety_pairs_cache
