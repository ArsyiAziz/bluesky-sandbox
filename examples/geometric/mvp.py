"""Modified Voltage Potential (MVP) conflict resolution as a policy, flying the
Random Encounters environment (``examples/environments/random_encounters``),
or any other in ``examples/environments`` with the observation and actions it
reads. From the repository root::

    python -m examples.geometric.mvp [episodes] [--env NAME] [--render [MODE]]

BlueSky's MVP with its default ``PastCPA`` resume, computed by each aircraft
from its own observation: each intruder in conflict adds its MVP velocity
change, flown as a track and speed within the envelope; once past every closest
approach, out of the zone and not bouncing, the aircraft resumes its route at
its earlier speed.

Unlike BlueSky's, it is decentralized: an aircraft keeps its last clearance
until it resumes, where BlueSky recomputes every aircraft while any pair is in
conflict. Horizontal and cooperative, as BlueSky's default, so every aircraft
must be an agent. It needs ``dt == simdt == asas_dt`` (1 s in Random
Encounters) to act on the state BlueSky's would.
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
from bluesky.tools.aero import ft, kts, nm, vtas2cas

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import observation_layout, unflatten_action
from bluesky_sandbox.interface.fields.actions import (
    ActionMask,
    AutopilotLnav,
    HdgDeg,
    SpdKts,
)
from bluesky_sandbox.ui.drivers import WINDOW_RENDER_MODES


_OWNSHIP = (
    "trk_deg",
    "gs_kts",
    "alt_ft",
    "ap_cas_kts",
    "perf_vmin_kts",
    "perf_vmax_kts",
)
_INTRUDERS = (
    "rel_pos_along_track_nm",
    "rel_pos_cross_track_nm",
    "rel_vel_along_track_kts",
    "rel_vel_cross_track_kts",
    "conflict_tcpa_s",
    "in_conf",
)
_ACTIONS = (HdgDeg, SpdKts, AutopilotLnav)


def mvp_dv(drel: np.ndarray, vrel: np.ndarray, tcpa: float, rpz_m: float) -> np.ndarray:
    """MVP's velocity change (m/s) for one conflict pair, as BlueSky's
    ``MVP.MVP``: the intruder at ``drel`` (m), moving at ``vrel`` (m/s) relative
    to the ownship, closest approach in ``tcpa`` (s), resolved to ``rpz_m``. In
    any frame rotated from east/north; the ownship subtracts it."""
    dist = math.hypot(drel[0], drel[1])
    dcpa = drel + vrel * tcpa
    dabs = math.hypot(dcpa[0], dcpa[1])
    intrusion = rpz_m - dabs
    # Head-on: no miss direction to push along, so push sideways.
    if dabs <= 10.0:
        dabs = 10.0
        dcpa = np.array([drel[1] / dist * dabs, -drel[0] / dist * dabs])
    # Outside the zone: aim at the zone's tangent, not through its edge.
    if rpz_m < dist and dabs < dist:
        erratum = math.cos(math.asin(rpz_m / dist) - math.asin(dabs / dist))
        return (rpz_m / erratum - dabs) * dcpa / (abs(tcpa) * dabs)
    return intrusion * dcpa / (abs(tcpa) * dabs)


class MVP:
    """MVP for every agent of an environment built with ``config``, each on its
    own observation and memory.

    ::

        env = RandomEncountersEnv()
        policy = MVP(env.config, resofach=bs.settings.asas_marh)
        env.reset(seed=0)
        policy.reset()
        while env.agents:
            env.step({a: policy.act(a, env.raw_observation(a)) for a in env.agents})

    ``resofach`` is BlueSky's ``RFACH``: the zone is resolved to its radius
    times it.
    """

    def __init__(self, config: EnvConfig, *, resofach: float = 1.01) -> None:
        if config.pz_radius_nm is None:
            raise ValueError(
                "MVP resolves to the protected zone: set EnvConfig.pz_radius_nm."
            )
        layout = observation_layout(config)
        for part, names in (("ownship", _OWNSHIP), ("intruders", _INTRUDERS)):
            have = {slot.name for slot in layout.get(part, [])}
            if missing := [name for name in names if name not in have]:
                raise ValueError(
                    f"MVP reads {part} fields {missing}; the observation has none."
                )
        self._config = config
        self._fields = list(config.action_fields)
        for kind in _ACTIONS:
            field = next((f for f in self._fields if type(f) is kind), None)
            if field is None or getattr(field, "normalizer", None) is not None:
                raise ValueError(
                    f"MVP flies {kind.__name__} in raw values: give it, unnormalized, as a clearance."
                )
        self._rpz_m = float(config.pz_radius_nm) * nm
        self.resofach = float(resofach)
        self.reset()

    def reset(self) -> None:
        """Forget the episode."""
        self._resolving: dict[str, set[str]] = {}
        self._selected_cas_kts: dict[str, float] = {}

    def act(self, agent: str, observation: Mapping[str, Mapping[str, Any]]) -> Any:
        """``agent``'s action for its raw observation (``env.raw_observation``)."""
        own = observation["ownship"]
        rows = observation["intruders"]
        pairs = {
            other: (
                # Cross, along: a rotation of east, north.
                np.array(
                    [rows["rel_pos_cross_track_nm"][k], rows["rel_pos_along_track_nm"][k]]
                )
                * nm,
                np.array(
                    [rows["rel_vel_cross_track_kts"][k], rows["rel_vel_along_track_kts"][k]]
                )
                * kts,
                float(rows["conflict_tcpa_s"][k]),
                bool(rows["in_conf"][k] > 0.5),
            )
            for k, other in enumerate(rows["acid"])
        }
        conflicts = [other for other, (*_, in_conf) in pairs.items() if in_conf]
        resolving = self._resolving.setdefault(agent, set())
        active = self._past_cpa(own, pairs, conflicts, resolving)

        given: dict[str, float] = {}
        if active and conflicts:
            self._selected_cas_kts.setdefault(agent, float(own["ap_cas_kts"]))
            given["hdg_deg"], given["spd_kts"] = self._resolve(own, pairs, conflicts)
        elif active is False:
            given["autopilot_lnav"] = 1.0
            if agent in self._selected_cas_kts:
                given["spd_kts"] = self._selected_cas_kts.pop(agent)
        return self._action(given)

    def _resolve(self, own, pairs, conflicts) -> tuple[float, float]:
        """New track (deg) and CAS (kt), as BlueSky's ``MVP.resolve``."""
        dv = np.zeros(2)
        for other in conflicts:
            drel, vrel, tcpa, _in_conf = pairs[other]
            dv -= mvp_dv(drel, vrel, tcpa, self._rpz_m * self.resofach)
        cross, along = dv[0], float(own["gs_kts"]) * kts + dv[1]
        track = (float(own["trk_deg"]) + math.degrees(math.atan2(cross, along))) % 360.0
        # Capped as BlueSky does: the envelope's speeds against ground speed.
        gs = min(
            max(math.hypot(cross, along), float(own["perf_vmin_kts"]) * kts),
            float(own["perf_vmax_kts"]) * kts,
        )
        # The autopilot takes a CAS.
        cas = float(vtas2cas(np.array([gs]), np.array([float(own["alt_ft"]) * ft]))[0])
        return track, cas / kts

    def _past_cpa(self, own, pairs, conflicts, resolving) -> bool | None:
        """Keep resolving (True), resume (False) or nothing to resolve (None),
        as BlueSky's ``PastCPA.resumenav``; updates ``resolving``."""
        resolving |= set(conflicts)
        if not resolving:
            return None
        active = False
        for other in list(resolving):
            still = False
            # An intruder no longer observed has left.
            if other in pairs:
                drel, vrel, _tcpa, _in_conf = pairs[other]
                hdist = math.hypot(drel[0], drel[1])
                past_cpa = float(drel @ vrel) > 0.0
                # The intruder's track against the own: (0, gs) in this frame.
                relative_trk = math.degrees(
                    math.atan2(vrel[0], float(own["gs_kts"]) * kts + vrel[1])
                )
                bouncing = (
                    abs(relative_trk) < 30.0 and hdist < self._rpz_m * self.resofach
                )
                still = not past_cpa or hdist < self._rpz_m or bouncing
            if still:
                active = True
            else:
                resolving.discard(other)
        return active

    def hold(self) -> Any:
        """The action that gives nothing: an aircraft flies on as it was."""
        return self._action({})

    def _action(self, given: Mapping[str, float]) -> Any:
        """The action clearing ``given``, by action name, and holding the rest."""
        flat = []
        for field in self._fields:
            if isinstance(field, ActionMask):
                flat.append(0.0 if field.target in given else 1.0)
            else:
                flat.append(given.get(field.meta.name, 0.0))
        return unflatten_action(self._config, flat)


def main() -> None:
    """Fly an environment with MVP and count the pairs losing separation."""
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
            policy = MVP(env.config, resofach=bs.settings.asas_marh)
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
