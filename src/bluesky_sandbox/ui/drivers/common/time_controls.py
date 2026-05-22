"""Shared pause, reset, and simulation-speed controls for GUI drivers."""

from __future__ import annotations

import bluesky as bs


class TimeControlMixin:
    """Reusable time-control state and actions for in-package GUI drivers."""

    def _init_time_controls(self) -> None:
        self._paused: bool = False
        self._desired_dtmult: float = 1.0
        # Monotonic wall-clock deadline accumulator for realtime pacing.
        self._sim_wall_target: float = 0.0

    def toggle_pause(self) -> None:
        """Flip the pause flag. Subclasses' ``step()`` loops honour it."""
        self._paused = not self._paused

    def toggle_realtime(self) -> None:
        """Flip realtime/fast mode and reset dtmult to 1x."""
        self.realtime = not self.realtime
        self._desired_dtmult = 1.0

    def scale_dtmult(self, factor: float) -> None:
        """Scale realtime dtmult, bounded to a practical range."""
        self._desired_dtmult = max(0.125, min(self._desired_dtmult * factor, 1024.0))
        self.realtime = True

    def delete_all_aircraft(self) -> None:
        """Queue every current aircraft for deletion on the next env step."""
        env = getattr(self, "_env", None)
        if env is not None:
            for acid in bs.traf.id:
                env.mark_aircraft_for_deletion(acid)

    def request_episode_reset(self) -> None:
        """Abort the current episode through the normal outer reset path."""
        env = getattr(self, "_env", None)
        if env is None:
            return
        for acid in bs.traf.id:
            env.mark_aircraft_for_deletion(acid)
        env._spawn_queue.clear()
        self._paused = False

    def _advance_sim(self) -> None:
        """Advance ``bs.sim`` one timestep *without* blocking.

        In realtime mode we deliberately use ``bs.sim.step()`` instead of
        ``bs.sim.update()``: ``update()`` does the realtime ``time.sleep()`` on
        this thread, which freezes the window between substeps (badly so at low
        dtmult, e.g. 0.25x). Pacing is handled by :meth:`_wait_realtime`, which
        idles out the per-substep budget while keeping the GUI drawing and
        processing input. (QtGL keeps ``update()``'s sleep via the base
        ``SimDriver`` because its window is a separate process.)
        """
        if self.realtime:
            bs.sim.set_dtmult(self._desired_dtmult)
        bs.sim.step()

    def _wait_realtime(self) -> None:
        """Idle until this substep's realtime budget elapses, keeping the GUI live.

        The budget is ``simdt / dtmult`` wall-seconds (so 0.25x runs 4x slower
        than wall time). Instead of one blocking sleep, we loop redrawing (at
        :attr:`render_fps` via :meth:`_render_due`) and pumping input through
        :meth:`_draw_idle_frame`, so panning, hover, and pause stay responsive
        even in slow motion. No-op in fast mode.
        """
        if not self.realtime:
            return
        import time

        simdt = float(getattr(bs.sim, "simdt", 0.0) or 0.0)
        dtmult = max(float(self._desired_dtmult), 1e-6)
        budget = simdt / dtmult
        if budget <= 0.0:
            return
        now = time.monotonic()
        # Clamp accumulated debt to a single budget so a slow frame or a dtmult
        # change can't make the sim sprint to catch up afterwards.
        self._sim_wall_target = max(self._sim_wall_target, now - budget) + budget
        deadline = self._sim_wall_target
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            self._draw_idle_frame()
            time.sleep(min(remaining, 0.01))

    def _draw_idle_frame(self) -> None:
        """Pump input and redraw one frame while idling in realtime pacing.

        Overridden by each in-process GUI driver; no-op by default.
        """
