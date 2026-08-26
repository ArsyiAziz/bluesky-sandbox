"""Route specs: the step grammar, validation, resolution and sampling.

A route is a list of steps - a waypoint name, a junction, or a weighted
choice between branches. This module turns that declarative spec into the
concrete waypoint sequence one aircraft flies. Independent of where the
aircraft comes from, which is :mod:`~bluesky_sandbox.sim.spawn.regions`.
"""

from __future__ import annotations

import math

import numpy as np

# A route is an ordered list of steps. A step is one of:
#   * a Waypoint queryable name (``str``) - inherits that waypoint's own
#     altitude/speed constraints;
#   * a constrained waypoint ``{"waypoint": "<name>", "speed_kts": .., "alt_ft": ..}``
#     - a crossing restriction local to this route, overriding the waypoint's
#     own ``speed_kts`` / ``alt_ft``. ``alt_ft`` may be used alone; ``speed_kts``
#     requires ``alt_ft`` or envelope altitude sampling;
#   * a sampled waypoint target ``{"waypoint": "<name>", "sample": Bounds, ...}``
#     - resolved per spawned aircraft into a concrete BlueSky route target;
#   * a subroute reference ``{"route": "<name>"}`` expanded inline;
#   * a branch ``{"choice": [step, ...], "weights": [w, ...]}`` - one branch is
#     taken (weighted; uniform if ``weights`` omitted). Branches let a procedure
#     diverge (a SID core that splits into transitions) or merge (entry
#     transitions that share an arrival trunk): a branch whose endpoint is a
#     waypoint also used by the trunk joins it there, and consecutive duplicate
#     waypoints across that junction are collapsed.
# Resolution comes in two flavours: :func:`sample_route_path` picks one concrete
# path (sampling each choice), preserving per-step constraints for the aircraft's
# ADDWPT commands, and :func:`expand_route_paths` enumerates every distinct path
# as plain waypoint names for validation and network visualisation.
RouteStep = str | dict
RouteSpec = list[RouteStep] | tuple[RouteStep, ...]
_WAYPOINT_STEP_KEYS = (
    "waypoint",
    "speed_kts",
    "alt_ft",
    "sample",
    "sample_alt_from_envelope",
    "sample_speed_from_envelope",
    "envelope_alt_floor_ft",
    "reachable_from_spawn",
    "reachable_vs_fraction",
)


def route_step_name(step) -> str:
    """The waypoint queryable name a resolved route step refers to.

    Accepts a bare name (``str``) or a constrained ``{"waypoint": name, ...}``
    step (the form :func:`sample_route_path` produces).
    """
    if isinstance(step, str):
        return step
    if isinstance(step, dict) and isinstance(step.get("waypoint"), str):
        return step["waypoint"]
    raise ValueError(f"cannot derive a waypoint name from route step {step!r}.")


def route_step_names(steps) -> list[str]:
    """Names-only view of a resolved route (drops per-step constraints)."""
    return [route_step_name(step) for step in steps]


def _validate_waypoint_step(step: dict) -> str:
    """Validate a ``{"waypoint": name, ...}`` step; return its waypoint name."""
    name = step.get("waypoint")
    if not isinstance(name, str) or not name:
        raise ValueError(f"route waypoint step needs a 'waypoint' name, got {step!r}.")
    extra = set(step) - set(_WAYPOINT_STEP_KEYS)
    if extra:
        raise ValueError(
            f"route waypoint step {step!r} has unknown keys {sorted(extra)}; "
            f"allowed: {list(_WAYPOINT_STEP_KEYS)}."
        )
    for key in ("speed_kts", "alt_ft", "envelope_alt_floor_ft", "reachable_vs_fraction"):
        value = step.get(key)
        if value is not None and not (
            isinstance(value, (int, float)) and math.isfinite(value)
        ):
            raise ValueError(
                f"route waypoint step {key!r} must be a finite number, got {value!r}."
            )
    if (
        step.get("speed_kts") is not None
        and step.get("alt_ft") is None
        and not bool(step.get("sample_alt_from_envelope", False))
    ):
        raise ValueError(
            "route waypoint step speed_kts requires alt_ft or "
            f"sample_alt_from_envelope=True, got {step!r}."
        )
    if bool(step.get("reachable_from_spawn", False)) and not bool(
        step.get("sample_alt_from_envelope", False)
    ):
        raise ValueError(
            "route waypoint step reachable_from_spawn=True requires "
            f"sample_alt_from_envelope=True, got {step!r}."
        )
    frac = step.get("reachable_vs_fraction")
    if frac is not None:
        if not bool(step.get("reachable_from_spawn", False)):
            raise ValueError(
                "route waypoint step reachable_vs_fraction requires "
                f"reachable_from_spawn=True, got {step!r}."
            )
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError(
                "route waypoint step reachable_vs_fraction must be in (0, 1], "
                f"got {frac!r}."
            )
    return name


def _collapse_junctions(names: list[str]) -> list[str]:
    """Drop consecutive duplicate waypoints so shared junctions join cleanly."""
    out: list[str] = []
    for name in names:
        if not out or out[-1] != name:
            out.append(name)
    return out


def _collapse_steps(steps: list) -> list:
    """Collapse consecutive steps for the same waypoint (a shared junction).

    Keeps the more specific step at the junction - a constrained
    ``{"waypoint": ...}`` step wins over a bare name - so a crossing restriction
    survives a junction merge. A step that ``sample``s a position is a *distinct*
    target, not a shared junction, so two such steps are never collapsed even
    when they reference the same waypoint queryable - this is what lets a route
    chain several sampled fixes off one waypoint template (e.g. merge fix then
    exit).
    """
    out: list = []
    for step in steps:
        prev = out[-1] if out else None
        prev_sampled = isinstance(prev, dict) and prev.get("sample") is not None
        step_sampled = isinstance(step, dict) and step.get("sample") is not None
        if (
            prev is not None
            and route_step_name(prev) == route_step_name(step)
            and not prev_sampled
            and not step_sampled
        ):
            if isinstance(step, dict) and not isinstance(prev, dict):
                out[-1] = step
            continue
        out.append(step)
    return out


def _branch_steps(branch) -> list:
    """A choice branch may be a single step or an inline list of steps."""
    return list(branch) if isinstance(branch, (list, tuple)) else [branch]


def _validate_choice(step: dict) -> tuple[list, list[float] | None]:
    """Return ``(branches, weights)`` for a validated ``choice`` step."""
    branches = step["choice"]
    if not isinstance(branches, (list, tuple)) or len(branches) == 0:
        raise ValueError(f"route 'choice' must list >= 1 branch, got {branches!r}.")
    weights = step.get("weights")
    if weights is None:
        return list(branches), None
    if len(weights) != len(branches):
        raise ValueError(
            f"route 'choice' has {len(weights)} weights for {len(branches)} "
            f"branches; lengths must match."
        )
    w = [float(x) for x in weights]
    if any(x < 0 for x in w) or sum(w) <= 0.0:
        raise ValueError(
            f"route 'choice' weights must be non-negative with a positive sum, "
            f"got {weights!r}."
        )
    return list(branches), w


def _resolve_pick(steps, routes, seen, pick) -> list[str]:
    """Resolve ``steps`` to one flat path, using ``pick`` to choose branches."""
    out: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            if "choice" in step:
                branches, weights = _validate_choice(step)
                out.extend(_resolve_pick(_branch_steps(pick(branches, weights)), routes, seen, pick))
            elif "route" in step:
                name = step["route"]
                if name in seen:
                    raise ValueError(f"route composition cycle through {name!r}.")
                if name not in routes:
                    raise ValueError(
                        f"subroute {name!r} not found; available routes: {list(routes)}."
                    )
                out.extend(_resolve_pick(routes[name], routes, seen + (name,), pick))
            elif "waypoint" in step:
                _validate_waypoint_step(step)
                out.append(step)  # constrained leaf, preserved for ADDWPT
            else:
                raise ValueError(
                    f"route step dict must have a 'waypoint', 'route', or 'choice' "
                    f"key, got {step!r}."
                )
        else:
            out.append(step)
    return out


def _resolve_all(steps, routes, seen) -> list[list[str]]:
    """Enumerate every concrete path through ``steps`` (cartesian over choices)."""
    paths: list[list[str]] = [[]]
    for step in steps:
        if isinstance(step, dict):
            if "choice" in step:
                branches, _ = _validate_choice(step)
                branch_paths: list[list[str]] = []
                for branch in branches:
                    branch_paths.extend(_resolve_all(_branch_steps(branch), routes, seen))
                paths = [p + bp for p in paths for bp in branch_paths]
            elif "route" in step:
                name = step["route"]
                if name in seen:
                    raise ValueError(f"route composition cycle through {name!r}.")
                if name not in routes:
                    raise ValueError(
                        f"subroute {name!r} not found; available routes: {list(routes)}."
                    )
                sub = _resolve_all(routes[name], routes, seen + (name,))
                paths = [p + sp for p in paths for sp in sub]
            elif "waypoint" in step:
                name = _validate_waypoint_step(step)
                paths = [p + [name] for p in paths]  # names only for viz/validation
            else:
                raise ValueError(
                    f"route step dict must have a 'waypoint', 'route', or 'choice' "
                    f"key, got {step!r}."
                )
        else:
            paths = [p + [step] for p in paths]
    return paths


def resolve_route(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
) -> list[str]:
    """Flatten a route deterministically, taking the first branch of any choice.

    Expands ``{"route": name}`` subroutes and collapses junction duplicates;
    raises on a missing subroute or a reference cycle. Returns plain waypoint
    names (per-step constraints dropped). For sampling a random branch with
    constraints use :func:`sample_route_path`; to enumerate every branch use
    :func:`expand_route_paths`.
    """
    steps_out = _resolve_pick(steps, routes, (), lambda b, _w: b[0])
    return _collapse_junctions(route_step_names(steps_out))


def sample_route_path(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
    rng: np.random.Generator,
) -> list[RouteStep]:
    """Resolve ``steps`` to one concrete path, sampling each ``choice`` branch.

    Constrained ``{"waypoint": ...}`` steps are preserved (so the per-step
    crossing restriction reaches ADDWPT); bare names stay strings. Use
    :func:`route_step_names` for a names-only view.
    """

    def pick(branches, weights):
        if weights is None:
            return branches[int(rng.integers(len(branches)))]
        w = np.asarray(weights, dtype=float)
        return branches[int(rng.choice(len(branches), p=w / w.sum()))]

    return _collapse_steps(_resolve_pick(steps, routes, (), pick))


def expand_route_paths(
    steps: RouteSpec,
    routes: dict[str, RouteSpec],
) -> list[list[str]]:
    """Enumerate every distinct concrete path through a (possibly branching) route.

    Used for validation (every branch's waypoints must exist) and network
    visualisation (draw all of a procedure's transitions). Duplicate paths and
    junction duplicates are collapsed.
    """
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for path in _resolve_all(steps, routes, ()):
        collapsed = _collapse_junctions(path)
        key = tuple(collapsed)
        if key not in seen:
            seen.add(key)
            out.append(collapsed)
    return out
