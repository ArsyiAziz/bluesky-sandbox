"""Optimal Reciprocal Collision Avoidance (ORCA) as a policy, flying the Random
Encounters environment (``examples/environments/random_encounters``), or any
other in ``examples/environments`` with the observation and actions it reads.
From the repository root::

    python -m examples.geometric.orca [episodes] [--env NAME] [--render [MODE]]

ORCA (van den Berg et al., 2011; the RVO2 library), computed by each aircraft
from its own observation:

* for each intruder, the velocity obstacle (VO) is the set of relative velocities
  that bring it inside the protected zone within the look-ahead ``tau``. In fact VO is a cone
  from the origin towards the intruder, cut off by a circle of radius
  ``R / tau`` around ``p / tau``;
* ``u`` is the smallest change that takes the relative velocity out of the
  obstacle. In a cooperative setting, the ownship takes half of it, as the intruder is expected to take
  the other half, which leaves a half-plane of allowed velocities;
* the new velocity is the one closest to the current velocity in every
  half-plane and within the speed envelope or, when no velocity is in all of them, the one that violates them
  least.

It flies as ``MVP`` in ``mvp.py`` does with the same observation and actions,
the same ``PastCPA`` resume and with ORCA's resolution in place of MVP's.
This implementation is:
1. Horizontal only: intruders that stay outside the vertical zone through the
look-ahead are left out. 
2. Reciprocal, so every aircraft must be an agent.

Vectors are ``(cross, along)`` in the ownship's track frame, a rotation of
east/north.
"""

from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, vtas2cas

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import observation_layout
from bluesky_sandbox.ui.drivers import WINDOW_RENDER_MODES
from examples.geometric.mvp import MVP

EPS = 1e-9  # below this, two lines are parallel

# What ORCA reads beyond MVP's fields: to leave out vertically separated intruders.
_VERTICAL = ("relative_alt_ft", "rel_vs_ft_min")


def det(a, b):
    """The 2-D cross product ``a x b``: positive when ``b`` points to the left of ``a``."""
    return a[0] * b[1] - a[1] * b[0]


def orca_line(p_rel, v_own, v_other, radius, tau, dt):
    """The ownship's ORCA half-plane against one intruder, as ``(point,
    direction)``: allowed velocities lie to the left of the line through
    ``point`` along ``direction``. As RVO2's ``Agent::computeNewVelocity``.

    ``p_rel`` is the intruder's position minus the ownship's (m), ``v_own`` and
    ``v_other`` their velocities (m/s), ``radius`` the zone to keep (m),
    ``tau`` the look-ahead and ``dt`` the step (s).
    """
    rel_v = v_own - v_other
    dist = math.hypot(*p_rel)

    if dist > radius:
        # The obstacle's cut-off circle, and w from its centre to rel_v.
        c = p_rel / tau
        r = radius / tau
        w = rel_v - c
        w_length = math.hypot(*w)

        # Seen from c, the cut-off arc spans the directions within (90 deg - alpha)
        # of -c, alpha being the cone's half-angle. When w points into that span,
        # the nearest edge is the arc; otherwise it is a leg of the cone.
        w_dot_c = np.dot(w, c)
        if w_dot_c < 0.0 and w_dot_c**2 > (r * w_length) ** 2:
            w_hat = w / w_length
            u = (r - w_length) * w_hat
            direction = np.array([w_hat[1], -w_hat[0]])  # tangent to the circle
        else:
            # The leg on w's side of p: p rotated by +alpha (left) or -alpha (right),
            # with cos(alpha) = leg / |p| and sin(alpha) = radius / |p|. The right
            # leg is reversed, so the outside of the cone is on the left of both.
            leg = math.sqrt(dist**2 - radius**2)
            if det(p_rel, w) > 0.0:
                direction = np.array([p_rel[0] * leg - p_rel[1] * radius,
                                      p_rel[0] * radius + p_rel[1] * leg]) / dist**2
            else:
                direction = -np.array([p_rel[0] * leg + p_rel[1] * radius,
                                       -p_rel[0] * radius + p_rel[1] * leg]) / dist**2
            u = np.dot(rel_v, direction) * direction - rel_v  # to the foot on the leg
    else:
        # Inside the zone already: the same construction with one step for tau,
        # to be out of it by the next.
        c = p_rel / dt
        r = radius / dt
        w = rel_v - c
        w_length = math.hypot(*w)
        w_hat = w / w_length
        u = (r - w_length) * w_hat
        direction = np.array([w_hat[1], -w_hat[0]])

    # Half of u: the intruder is expected to take the other half.
    return v_own + 0.5 * u, direction


def _violated(line, v):
    """Whether ``v`` lies on the forbidden (right) side of ``line``."""
    point, direction = line
    return det(direction, point - v) > 0.0


def _best_on_line(lines, i, v_max, v_pref, toward_direction):
    """The best velocity on line ``i`` that keeps lines ``0..i-1`` and the speed
    limit, or ``None`` if there is none. As RVO2's ``linearProgram1``.

    Along the line, ``point + t * direction``, the speed disc leaves an
    interval of ``t`` and each earlier line cuts it from one side. The answer
    is ``v_pref``'s projection clamped to what remains - or, with
    ``toward_direction``, the end of it furthest along ``v_pref``.
    """
    point, direction = lines[i]
    b = point @ direction
    disc = b * b + v_max * v_max - point @ point
    if disc < 0.0:
        return None  # the line misses the speed disc
    t_left, t_right = -b - math.sqrt(disc), -b + math.sqrt(disc)
    for j in range(i):
        point_j, direction_j = lines[j]
        denom = det(direction, direction_j)
        numer = det(direction_j, point - point_j)
        if abs(denom) <= EPS:
            if numer < 0.0:
                return None  # parallel, and line i wholly forbidden by line j
            continue
        t = numer / denom  # where line j crosses line i
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if toward_direction:
        t = t_right if v_pref @ direction > 0.0 else t_left
    else:
        t = min(max(direction @ (v_pref - point), t_left), t_right)
    return point + t * direction


def _closest_allowed(lines, v_max, v_pref, toward_direction=False):
    """The velocity closest to ``v_pref`` that keeps every line and the speed
    limit, as RVO2's ``linearProgram2``: ``(lines kept, velocity)``.

    Lines are added one at a time. While the current velocity keeps a new line
    it stands; otherwise the new one lies on that line. Fewer lines kept than
    given means no velocity keeps them all.
    """
    if toward_direction:
        v = v_pref * v_max  # v_pref is a unit direction here
    elif v_pref @ v_pref > v_max * v_max:
        v = v_pref / math.hypot(*v_pref) * v_max
    else:
        v = v_pref.copy()
    for i, line in enumerate(lines):
        if _violated(line, v):
            on_line = _best_on_line(lines, i, v_max, v_pref, toward_direction)
            if on_line is None:
                return i, v
            v = on_line
    return len(lines), v


def _least_violation(lines, first_failed, v_max, v):
    """The velocity that violates ``lines`` least - the smallest largest distance
    to the wrong side of any - for when no velocity keeps them all. As RVO2's
    ``linearProgram3``: for each line it does not keep, the furthest velocity
    towards that line's allowed side that keeps the bisectors between it and
    every earlier line.
    """
    worst = 0.0
    for i in range(first_failed, len(lines)):
        point, direction = lines[i]
        if det(direction, point - v) > worst:
            bisectors = []
            for j in range(i):
                point_j, direction_j = lines[j]
                d = det(direction, direction_j)
                if abs(d) <= EPS:
                    if direction @ direction_j > 0.0:
                        continue  # parallel, same way: line j adds nothing
                    p = 0.5 * (point + point_j)  # opposite ways: halfway between
                else:
                    p = point + (det(direction_j, point - point_j) / d) * direction
                diff = direction_j - direction
                bisectors.append((p, diff / math.hypot(*diff)))
            previous = v
            kept, candidate = _closest_allowed(
                bisectors, v_max, np.array([-direction[1], direction[0]]), True
            )
            v = previous if kept < len(bisectors) else candidate
            worst = det(direction, point - v)
    return v


def orca_velocity(lines, v_pref, v_max):
    """The velocity closest to ``v_pref`` in every half-plane of ``lines`` with a
    speed of at most ``v_max``; when there is none, the one violating them least."""
    kept, v = _closest_allowed(lines, v_max, v_pref)
    if kept < len(lines):
        v = _least_violation(lines, kept, v_max, v)
    return v


def vertically_close(rel_alt_ft, rel_vs_fpm, zone_ft, tau_s):
    """Whether an intruder ``rel_alt_ft`` above, climbing ``rel_vs_fpm`` faster,
    comes within ``zone_ft`` vertically in the next ``tau_s`` seconds."""
    if abs(rel_vs_fpm) < 1e-6:
        return abs(rel_alt_ft) < zone_ft
    rate = rel_vs_fpm / 60.0  # ft/s
    t1, t2 = sorted(((-zone_ft - rel_alt_ft) / rate, (zone_ft - rel_alt_ft) / rate))
    return t2 > 0.0 and t1 < tau_s


class ORCA(MVP):
    """ORCA for every agent of an environment built with ``config``, each on its
    own observation and memory; MVP's in all but the resolution.

    ::

        env = RandomEncountersEnv()
        policy = ORCA(env.config, resofach=bs.settings.asas_marh)
        env.reset(seed=0)
        while not env.episode_done:
            env.step({a: policy.act(a, env.raw_observation(a)) for a in env.agents})

    ``resofach`` is BlueSky's ``RFACH``, as for MVP; ``tau_s`` the look-ahead,
    the environment's conflict detection look-ahead unless given.
    """

    def __init__(self, config: EnvConfig, *, resofach: float = 1.01, tau_s: float | None = None) -> None:
        super().__init__(config, resofach=resofach)
        have = {slot.name for slot in observation_layout(config).get("intruders", [])}
        if missing := [name for name in _VERTICAL if name not in have]:
            raise ValueError(f"ORCA reads intruders fields {missing}; the observation has none.")
        self.tau_s = float(tau_s or config.lookahead_s or 300.0)
        self.dt_s = float(config.dt)
        self._zone_ft = float(config.pz_height_ft or 1000.0)
        self._rows = None

    def act(self, agent: str, observation: Mapping[str, Mapping[str, Any]]) -> Any:
        """``agent``'s action for its raw observation (``env.raw_observation``)."""
        self._rows = observation["intruders"]  # for _resolve, which MVP passes no vertical fields
        return super().act(agent, observation)

    def _resolve(self, own, pairs, conflicts):
        """New track (deg) and CAS (kt): ORCA against every intruder that can
        come within the vertical zone in the look-ahead, preferring the
        current velocity."""
        rows = self._rows
        v_own = np.array([0.0, float(own["gs_kts"]) * kts])  # straight ahead, in this frame
        radius = self._rpz_m * self.resofach
        lines = []
        for k, other in enumerate(rows["acid"]):
            if other not in pairs:
                continue
            if not vertically_close(float(rows["relative_alt_ft"][k]), float(rows["rel_vs_ft_min"][k]),
                                    self._zone_ft, self.tau_s):
                continue
            drel, vrel, _tcpa, _in_conf = pairs[other]
            lines.append(orca_line(drel, v_own, v_own + vrel, radius, self.tau_s, self.dt_s))
        v_max = float(own["perf_vmax_kts"]) * kts
        v = orca_velocity(lines, v_own, v_max)
        cross, along = float(v[0]), float(v[1])
        track = (float(own["trk_deg"]) + math.degrees(math.atan2(cross, along))) % 360.0
        # Capped as MVP's: the envelope's speeds against ground speed.
        speed = min(max(math.hypot(cross, along), float(own["perf_vmin_kts"]) * kts), v_max)
        # The autopilot takes a CAS.
        cas = float(vtas2cas(np.array([speed]), np.array([float(own["alt_ft"]) * ft]))[0])
        return track, cas / kts


def main() -> None:
    """Fly an environment with ORCA and count the pairs losing separation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", type=int, nargs="?", default=10)
    parser.add_argument(
        "--env",
        default="random_encounters",
        help="an environment in examples/environments, by package name",
    )
    parser.add_argument(
        "--render",
        nargs="?",
        const="pygame",
        choices=WINDOW_RENDER_MODES,
        help="watch it in a window: pygame unless named",
    )
    args = parser.parse_args()

    try:
        Env = importlib.import_module(f"examples.environments.{args.env}").Env
    except ModuleNotFoundError as e:
        if e.name != f"examples.environments.{args.env}":
            raise  # the environment exists but something it imports does not
        known = sorted(p.parent.name for p in Path(__file__).parents[1].glob("environments/*/__init__.py"))
        parser.error(f"no environment {args.env!r}; there are: {', '.join(known)}")
    # Watched, the episode runs in real time: one second a step.
    env = Env(render_mode=args.render, realtime=args.render is not None)
    try:
        for seed in range(args.episodes):
            env.reset(seed=seed)
            policy = ORCA(env.config, resofach=bs.settings.asas_marh)
            losses = set()
            # Until no aircraft is left nor still to come: in some environments
            # the first spawns after the start.
            while not env.episode_done:
                if args.render:
                    env.render()
                env.step({a: policy.act(a, env.raw_observation(a)) for a in env.agents})
                losses |= {frozenset(pair) for pair in bs.traf.cd.lospairs}
            print(f"episode {seed}: {len(env.spawn_log)} aircraft, {len(losses)} pairs lost separation")
    finally:
        env.close()


if __name__ == "__main__":
    main()
