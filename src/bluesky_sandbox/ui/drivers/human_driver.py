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

from .common import DISPLAY_TOGGLES, DisplayToggle, PrimitiveDrawMixin, TrailMixin
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
    # The share of wall time drawing may take while the sim runs flat out (not
    # realtime, or fast-forwarding): frames are spaced so the measured draw
    # stays within it, whatever a frame costs - a slow driver draws less often
    # instead of slowing the sim. 1 leaves only ``render_fps``.
    render_budget: float = 0.25

    #: Everything a person can operate in this driver (see ``common.hud``),
    #: and those it cannot do, by name.
    CONTROLS: tuple = DISPLAY_TOGGLES
    UNSUPPORTED_CONTROLS: frozenset[str] = frozenset()
    #: How long a note on what just changed stays up (s).
    NOTE_S = 1.6

    def __init__(self, realtime: bool = True) -> None:
        super().__init__(realtime=realtime)
        # Region and waypoint names (``toggle_labels``); what each aircraft's
        # label shows - its full data block, its callsign or nothing, the
        # tracked aircraft's always full - is ``aircraft_labels``. Each display
        # setting is a toggle (``display_toggles``).
        self.show_labels = True
        self.aircraft_labels = "full"
        self._note: tuple[str, float] | None = None
        # When True, overlay the design's defined routes; otherwise only the
        # selected aircraft's route is shown.
        self.show_all_routes = False
        # When True, overlay velocity-obstacle cones for the tracked aircraft
        # against its neighbors (conflict-geometry visualization).
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
        # Monotonic timestamp (s) of the last frame draw, for the render gate,
        # and what a draw costs (s, smoothed) - for the render budget.
        self._last_render_s: float = 0.0
        self._draw_cost_s: float = 0.0
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
        interval = 1.0 / self.render_fps
        flat_out = not self.realtime or self._fastforward_active()
        if flat_out and self._draw_cost_s > 0.0 and self.render_budget < 1.0:
            interval = max(interval, self._draw_cost_s / max(self.render_budget, 1e-3))
        if now - self._last_render_s >= interval:
            self._last_render_s = now
            return True
        return False

    def _draw_timed(self, draw) -> None:
        """Run ``draw`` (a frame), noting what it cost for the render budget."""
        start = time.monotonic()
        draw()
        cost = time.monotonic() - start
        self._draw_cost_s = cost if self._draw_cost_s == 0.0 else 0.8 * self._draw_cost_s + 0.2 * cost

    def tracked_acid(self) -> str | None:
        """The aircraft whose route/info to display: the clicked one, else (when
        ``auto_track``) the first live aircraft, else ``None``."""
        selected = getattr(self, "_selected", None)
        if selected is not None and selected in bs.traf.id:
            return selected
        if self.auto_track and bs.traf.ntraf > 0:
            return bs.traf.id[0]
        return None

    # ---- controls (the HUD contract, common.hud) -----------------------------
    # Declared once; a driver binds their keys and draws their buttons from
    # :attr:`controls`, runs them with :meth:`activate`, and reacts to a
    # display toggle's change in :meth:`on_toggled`.

    @property
    def controls(self) -> tuple:
        """Everything a person can operate in this driver, in order."""
        return tuple(c for c in self.CONTROLS if c.name not in self.UNSUPPORTED_CONTROLS)

    def control(self, name: str):
        """The control ``name``."""
        for control in self.controls:
            if control.name == name:
                return control
        names = [c.name for c in self.controls]
        raise KeyError(f"no control {name!r} on {type(self).__name__}; controls: {names}")

    def activate(self, name: str) -> None:
        """Run control ``name`` - as its key or its button does - and say what
        it changed."""
        control = self.control(name)
        control.run(self)
        self.notify(control.describe_for(self))

    def control_for_key(self, key: str, shift: bool = False) -> str | None:
        """The control ``key`` runs (as Panda3D names keys: ``"l"``,
        ``"space"``), Shift held or not: one bound with Shift first, else the
        key alone."""
        candidates = (f"shift-{key}", key) if shift else (key,)
        for candidate in candidates:
            for control in self.controls:
                if candidate in control.keys:
                    return control.name
        return None

    def notify(self, text: str) -> None:
        """Show ``text`` for a moment (the HUD's note)."""
        self._note = (text, time.monotonic() + self.NOTE_S)

    @property
    def note(self) -> str | None:
        """The note showing now, if any."""
        if self._note is None or time.monotonic() > self._note[1]:
            return None
        return self._note[0]

    @property
    def display_toggles(self) -> tuple[DisplayToggle, ...]:
        """The display toggles this driver draws, in order."""
        return tuple(c for c in self.controls if isinstance(c, DisplayToggle))

    def display_toggle(self, name: str) -> DisplayToggle:
        """The display toggle ``name``."""
        control = self.control(name)
        if not isinstance(control, DisplayToggle):
            raise KeyError(f"control {name!r} is not a display toggle")
        return control

    def toggle(self, name: str):
        """Step display toggle ``name`` to its next value; return it."""
        try:
            toggle = self.display_toggle(name)
        except KeyError as e:
            raise KeyError(f"no display toggle {name!r}") from e
        value = toggle.after(getattr(self, name))
        setattr(self, name, value)
        self.on_toggled(toggle, value)
        return value

    def on_toggled(self, toggle: DisplayToggle, value) -> None:
        """React to ``toggle`` having changed to ``value``: turning trails
        off drops their points. A driver extends it for what it caches."""
        if toggle.name == "show_trails" and not value:
            self._clear_trails()

    def toggle_labels(self) -> None:
        """Toggle static overlay labels such as region and waypoint names."""
        self.toggle("show_labels")

    def cycle_aircraft_labels(self) -> str:
        """Step the aircraft labels on: full data block, callsign, none."""
        return self.toggle("aircraft_labels")

    def toggle_trails(self) -> None:
        """Flip trail rendering, discarding the points when turning it off."""
        self.toggle("show_trails")

    def toggle_all_routes(self) -> None:
        """Toggle between showing every aircraft's route and only the selected one."""
        self.toggle("show_all_routes")

    def toggle_velocity_obstacles(self) -> None:
        """Toggle the velocity-obstacle overlay for the tracked aircraft."""
        self.toggle("show_velocity_obstacles")

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
