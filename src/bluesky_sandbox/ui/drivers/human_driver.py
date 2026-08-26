"""Base class for human-facing sim drivers (qtgl, pygame, panda3d).

Driver specification
--------------------
The environment talks to drivers through the contract defined on
:class:`SimDriver`:

* ``start()`` - open windows / resources.  Sets ``_started = True``.
* ``wait_until_ready(timeout)`` - block until the GUI is connected.
* ``on_render()`` - first ``env.render()`` call after ``start()``.
* ``update()`` - flush GUI I/O without advancing the sim.
* ``step()`` - advance ``bs.sim`` by one substep.
* ``on_reset()`` - fired by ``env.reset()``.
* ``close()`` - tear everything down.

:class:`HumanSimDriver` extends that contract with the surface every
human-facing driver shares:

* trails - ``show_trails`` + :meth:`toggle_trails` + the bookkeeping
  in :meth:`_advance_trails` / :meth:`_clear_trails`.
* render-primitive vocabulary - ``draw_polygon`` / ``draw_point`` /
  ``draw_polyline`` (no-op by default) and the :meth:`draw` dispatcher
  that :meth:`on_reset` already wires up.

Drivers that own their own GUI in this package (pygame, panda3d)
extend :class:`SandboxGUIDriver` instead - that adds the pause /
dtmult / HUD-formatting surface QtGL doesn't share because its
window is BlueSky's own QtGL client.
"""

from __future__ import annotations

import time

import bluesky as bs

from bluesky_sandbox.sim.spawn import expand_route_paths

from .common import PrimitiveDrawMixin, TrailMixin
from .sim_driver import SimDriver


class HumanSimDriver(TrailMixin, PrimitiveDrawMixin, SimDriver):
    """Common scaffolding for drivers that present a human-facing window.

    What this adds on top of :class:`SimDriver`:

    * A default :meth:`on_reset` that caches the env reference and
      dispatches each :meth:`~BlueskyBaseEnvironment.iter_renderables`
      result through :meth:`~SimDriver.draw`.  Subclasses can override
      ``on_reset`` to bracket this with their own setup, calling
      ``super().on_reset()`` at the right point.
    * A construction-time check that every render primitive listed in
      :attr:`_required_draws` has its corresponding ``draw_*`` overridden.
      A no-op ``draw_polygon`` (etc.) means primitives of that kind
      silently disappear in the window - almost never intentional for a
      human view, so we surface it as a warning.

    Subclasses that *don't* render every primitive type can shrink
    :attr:`_required_draws` to silence the warning for the kinds they
    deliberately skip.
    """

    # Wall-clock cap (Hz) for in-process frame drawing. Decouples render cadence
    # from the sim substep loop: ``env.step()`` advances the sim many times per
    # call, but the expensive frame draw should fire at most this often so sim
    # throughput isn't capped by rendering during fast-forward. 0 disables the
    # gate (render every substep).
    render_fps: float = 60.0

    def __init__(self, realtime: bool = True) -> None:
        super().__init__(realtime=realtime)
        self.show_labels = True
        # When True, overlay the design's defined routes; otherwise only the
        # selected aircraft's route is shown.
        self.show_all_routes = False
        # When True, overlay velocity-obstacle cones for the tracked aircraft
        # against its neighbours (conflict-geometry visualization).
        self.show_velocity_obstacles = False
        # VO lookahead as a fraction (0, 1] of the CD detection horizon
        # (``bs.traf.cd.dtlookahead``). 1.0 = the full detector horizon, where the
        # overlay's conflict flags match BlueSky 1:1; drag the plan-view slider
        # down to sweep the cones/conflicts across shorter lookaheads.
        self.vo_horizon_frac = 1.0
        self._defined_routes_cache: list | None = None
        # When True, an aircraft is always tracked (the first live one if none
        # is clicked) — handy for RL eval videos. When False (default) nothing is
        # selected until you click an aircraft, and clicking empty deselects.
        self.auto_track = False
        # Monotonic timestamp (s) of the last frame draw, for the render gate.
        self._last_render_s: float = 0.0
        self._check_draws_implemented()
        self._init_trails()

    def _render_due(self) -> bool:
        """Return ``True`` at most :attr:`render_fps` times per wall-second.

        This is the shared cadence gate used by every in-process human driver
        (pygame, panda3d) so the expensive frame draw is decoupled from the sim
        substep loop. During fast-forward many substeps run per wall-second;
        gating the draw here keeps the simulation advancing flat-out instead of
        stalling once per substep to render. Updates the timestamp when it
        returns ``True`` so callers just do ``if self._render_due(): draw()``.
        """
        if self.render_fps <= 0:
            return True
        now = time.monotonic()
        if now - self._last_render_s >= 1.0 / self.render_fps:
            self._last_render_s = now
            return True
        return False

    def tracked_acid(self) -> str | None:
        """The aircraft whose route/info to display: the clicked one, else (when
        ``auto_track``) the first live aircraft, else ``None``."""
        selected = getattr(self, "_selected", None)
        if selected is not None and selected in bs.traf.id:
            return selected
        if self.auto_track and bs.traf.ntraf > 0:
            return bs.traf.id[0]
        return None

    def toggle_labels(self) -> None:
        """Toggle static overlay labels such as region and waypoint names."""
        self.show_labels = not self.show_labels

    def toggle_all_routes(self) -> None:
        """Toggle between showing every aircraft's route and only the selected one."""
        self.show_all_routes = not self.show_all_routes

    def toggle_velocity_obstacles(self) -> None:
        """Toggle the velocity-obstacle overlay for the tracked aircraft."""
        self.show_velocity_obstacles = not self.show_velocity_obstacles

    # Smallest VO lookahead fraction the slider allows - a sliver so the cones
    # never collapse to nothing (which would leave the handle with no overlay).
    VO_HORIZON_FRAC_MIN = 0.05

    def set_vo_horizon_frac(self, frac: float) -> None:
        """Clamp and store the VO lookahead fraction (see ``vo_horizon_frac``)."""
        self.vo_horizon_frac = max(
            self.VO_HORIZON_FRAC_MIN, min(1.0, float(frac))
        )

    def defined_route_polylines(self) -> list[list[tuple[float, float, float | None]]]:
        """Polylines for the design's *defined* routes (static for the episode).

        These come from the scenario's route definitions + waypoint positions
        (not a live per-aircraft route), so they don't change during the episode
        and are resolved once and cached (cleared on reset). Returns a list of
        routes, each an ordered list of ``(lat, lon, alt_ft)`` points.
        """
        cached = getattr(self, "_defined_routes_cache", None)
        if cached is not None:
            return cached

        polylines: list[list[tuple[float, float, float | None]]] = []
        env = getattr(self, "_env", None)
        if env is not None:
            spawn = env.episode_spawn
            queryables = env.episode_queryables
            routes_lib = dict(spawn.routes or {})
            # Every named route, plus any inline fixed route on a region / global.
            specs = list(routes_lib.values())
            specs.extend(r.route for r in spawn.regions if isinstance(r.route, (list, tuple)))
            if isinstance(spawn.route, (list, tuple)):
                specs.append(spawn.route)

            seen: set[tuple] = set()
            for spec in specs:
                # A branching route (STAR/SID transitions) expands to one path
                # per branch, so the whole network draws, not just one limb.
                try:
                    paths = expand_route_paths(spec, routes_lib)
                except Exception:
                    continue
                for names in paths:
                    pts: list[tuple[float, float, float | None]] = []
                    for name in names:
                        wp = queryables.get(name)
                        lat = getattr(wp, "lat", None)
                        lon = getattr(wp, "lon", None)
                        if lat is None or lon is None:
                            continue
                        pts.append((float(lat), float(lon), getattr(wp, "alt_ft", None)))
                    if len(pts) < 2:
                        continue
                    key = tuple((round(lat, 6), round(lon, 6)) for lat, lon, _ in pts)
                    if key not in seen:
                        seen.add(key)
                        polylines.append(pts)

        self._defined_routes_cache = polylines
        return polylines

    def on_reset(self, env=None) -> None:
        """Cache env, wipe any accumulated trails, dispatch renderables.

        Trails are wiped here so callsigns reused across episodes
        don't inherit a previous run's history.  QtGL keeps its own
        trail layer through BlueSky's ``TRAIL`` command - the
        ``_trails`` dict stays empty there, so the wipe is a no-op.
        """
        super().on_reset(env)
        if self._env is None:
            raise RuntimeError("HumanSimDriver env has not been bound.")
        # Defined routes are episode-static; drop the cache so the next access
        # re-resolves them against this episode's queryables.
        self._defined_routes_cache = None
        self._clear_trails()
        self.draw_renderables(self._env._renderable_builder.iter_renderables())
