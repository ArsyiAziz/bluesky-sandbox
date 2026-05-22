"""Compile a :class:`~bluesky_sandbox.ui.designer.spec.DesignSpec` into live objects.

A spec is the design document; the builder turns it into the runtime objects
the env consumes:

* :func:`build_scenario` - a :class:`~bluesky_sandbox.sim.sampling.Scenario` over the
  airspace / spawn / queryables. Because the spawn config carries its own
  distributions (counts, params), per-episode randomisation happens inside
  ``SpawnConfig.iter_spawns`` at reset time, so ``sample()`` and ``support()``
  return the same schema-stable :class:`EpisodeSpec`.
* :func:`build_design_config` - a static :class:`~bluesky_sandbox.config.EnvConfig`,
  resolving field references against the field modules.

The split mirrors the design seam: structured data is materialised via
:func:`~bluesky_sandbox.ui.designer.spec.load`; logic is resolved by import.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import keyword
import math
import sys
import textwrap
from collections.abc import Callable
from types import ModuleType
from typing import Any

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.interface.fields import actions as _actions
from bluesky_sandbox.interface.fields import observations as _observations
from bluesky_sandbox.interface.fields import queryables as _queryable_fields
from bluesky_sandbox.interface.fields.base import (
    ActionField,
    ObsField,
    PairObsField,
    QueryableFieldCardinality,
    QueryableFieldRequirement,
)
from bluesky_sandbox.interface.wrappers.observations import normalizer as _normalizers
from bluesky_sandbox.sim.bounds import Bounds, RegionBounds, union_footprints
from bluesky_sandbox.sim.bounds.base import Footprint
from bluesky_sandbox.sim.queryables import Queryable
from bluesky_sandbox.sim.scenario import RandomizedScenario
from bluesky_sandbox.sim.scenario import transforms as _t
from bluesky_sandbox.sim.spawn import SpawnConfig

from . import spec as _spec
from .spec import DesignSpec, FieldRef, TaskInfoSpec


class BuildError(ValueError):
    """Raised when a spec cannot be compiled into runtime objects."""


# Namespace under which a design's editable code modules are registered, so a
# ref like "task:reward" resolves to the user's in-designer source without
# colliding with real top-level modules.
CODE_NS = "_bsx_designer_code"


def install_code_modules(code: dict[str, str]) -> None:
    """Register a design's editable code (``{"task.py": "..."}``) as importable.

    Each ``name.py`` becomes module ``name`` (and ``_bsx_designer_code.name``),
    so spec references like ``"task:reward"`` or ``"custom_fields:MyField"``
    resolve to the user's code without writing files to disk. Re-running
    replaces the previous source - this is how the live designer validates
    edited reward/termination and custom-field code.
    """
    if not code:
        return
    pkg = sys.modules.get(CODE_NS)
    if pkg is None:
        pkg = ModuleType(CODE_NS)
        pkg.__path__ = []  # mark as a package
        sys.modules[CODE_NS] = pkg
    for filename, source in code.items():
        if not filename.endswith(".py") or not isinstance(source, str):
            continue
        stem = filename[:-3]
        module = ModuleType(stem)
        module.__file__ = f"<designer:{filename}>"
        # Register before exec so class creation (e.g. dataclasses looking up
        # sys.modules[cls.__module__]) can find the module being defined.
        sys.modules[stem] = module
        sys.modules[f"{CODE_NS}.{stem}"] = module
        setattr(pkg, stem, module)
        try:
            exec(compile(source, f"<designer:{filename}>", "exec"), module.__dict__)
        except Exception as e:  # surface user syntax/runtime errors clearly
            sys.modules.pop(stem, None)
            sys.modules.pop(f"{CODE_NS}.{stem}", None)
            raise BuildError(f"error in {filename}: {e}") from e


# --------------------------------------------------------------------------- #
# Code-reference resolution ("module:attr")                                   #
# --------------------------------------------------------------------------- #
def resolve_callable(ref: str) -> Callable[..., Any]:
    """Import a ``"package.module:attr"`` reference and return the attribute.

    This is how the spec points at code-tab logic (reward / termination /
    truncation functions and task-info providers) without serialising it.
    """
    if not isinstance(ref, str) or ":" not in ref:
        raise BuildError(
            f"code reference must be of the form 'module:attr', got {ref!r}."
        )
    module_name, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise BuildError(f"cannot import module {module_name!r} for ref {ref!r}: {e}") from e
    try:
        obj = getattr(module, attr)
    except AttributeError as e:
        raise BuildError(f"{module_name!r} has no attribute {attr!r} (ref {ref!r}).") from e
    if not callable(obj):
        raise BuildError(f"code reference {ref!r} resolved to a non-callable.")
    return obj


def build_inline_task_info_providers(
    setup: str,
    providers: list[TaskInfoSpec],
) -> list[Callable[..., Any]]:
    """Compile designer-authored task-info providers into shared module state."""
    module = ModuleType(f"{CODE_NS}.task_info_inline")
    module.__dict__["np"] = __import__("numpy")
    # Register the synthetic module while the setup runs. ``@dataclass`` looks
    # its owning class up via ``sys.modules[cls.__module__]`` (to spot InitVar
    # and resolve string annotations), so a class DEFINED in this setup - which
    # the task-info scaffolds now do, since providers ship with the task rather
    # than the library - crashes on a module that was never registered.
    sys.modules[module.__name__] = module
    try:
        if setup.strip():
            exec(
                compile(setup, "<designer task_info setup>", "exec"),
                module.__dict__,
            )
    except Exception as e:
        raise BuildError(f"error in task-info setup: {e}") from e
    finally:
        sys.modules.pop(module.__name__, None)

    compiled: list[Callable[..., Any]] = []
    seen: set[str] = set()
    for provider in providers:
        name = provider.name.strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise BuildError(f"task-info provider name must be a Python identifier, got {name!r}.")
        if name in seen:
            raise BuildError(f"duplicate task-info provider name {name!r}.")
        seen.add(name)
        body = provider.body.rstrip() or "pass"
        if body.isidentifier() and body in module.__dict__:
            obj = module.__dict__[body]
            if not callable(obj):
                raise BuildError(f"task-info provider {name!r} references non-callable {body!r}.")
            compiled.append(obj)
            continue
        indented = textwrap.indent(body, "    ")
        source = f"def {name}(obs, action, info, context, rng):\n{indented}\n"
        try:
            exec(compile(source, f"<designer task_info:{name}>", "exec"), module.__dict__)
        except Exception as e:
            raise BuildError(f"error in task-info provider {name!r}: {e}") from e
        compiled.append(module.__dict__[name])
    return compiled


def validate_hook_setup(setup: str) -> None:
    """Compile and execute hook setup so bad imports fail during validation."""
    if not setup.strip():
        return
    module = ModuleType(f"{CODE_NS}.hook_setup")
    try:
        exec(
            compile(setup, "<designer hook setup>", "exec"),
            module.__dict__,
        )
    except Exception as e:
        raise BuildError(f"error in hook setup: {e}") from e


# --------------------------------------------------------------------------- #
# Field resolution                                                            #
# --------------------------------------------------------------------------- #
def _resolve_normalizer(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("type") != "normalizer":
        return value
    name = value.get("name")
    cls = getattr(_normalizers, str(name), None)
    if cls is None:
        raise BuildError(f"unknown normalizer {name!r}.")
    kwargs = dict(value.get("kwargs", {}))
    try:
        return cls(**kwargs)
    except Exception as e:
        raise BuildError(f"failed to construct normalizer {name!r}: {e}") from e


def _resolve_constructor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _resolve_normalizer(value) if key == "normalizer" else value
        for key, value in kwargs.items()
    }


def _resolve_field(ref: FieldRef, modules, kind: str):
    # A name containing ':' is a custom field referenced by import path
    # ("package.module:ClassName") - this is how user-coded observation/action
    # fields plug in alongside the built-ins.
    if ":" in ref.name:
        cls = resolve_callable(ref.name)
    else:
        cls = None
        searched = []
        for module in modules:
            searched.append(module.__name__)
            cls = getattr(module, ref.name, None)
            if cls is not None:
                break
    if cls is None:
        raise BuildError(
            f"unknown {kind} field {ref.name!r}; not found in "
            f"{', '.join(searched)}."
        )
    kwargs = _resolve_constructor_kwargs(ref.kwargs)
    try:
        field_obj = cls(**kwargs)
    except Exception as e:
        raise BuildError(
            f"failed to construct {kind} field {ref.name!r} with kwargs "
            f"{ref.kwargs!r}: {e}"
        ) from e
    if ref.transform:
        method = getattr(field_obj, ref.transform, None)
        if method is None or not callable(method):
            raise BuildError(
                f"{kind} field {ref.name!r} has no transform {ref.transform!r}."
            )
        field_obj = method(**_resolve_constructor_kwargs(ref.transform_kwargs))
    return field_obj


def resolve_obs_field(
    ref: FieldRef,
) -> ObsField | PairObsField | list[ObsField | PairObsField]:
    """Resolve one field ref, or a LIST of them.

    A transform may expand one ref into several channels - ``stacked(depth=n)``
    is the frame-stacking case - so this returns a list for those.
    ``EnvConfig.__post_init__`` flattens one level, which is what every caller
    here feeds into.
    """
    field_obj = _resolve_field(
        ref,
        (_observations, _queryable_fields),
        "observation",
    )
    candidates = field_obj if isinstance(field_obj, (list, tuple)) else [field_obj]
    for candidate in candidates:
        if not isinstance(candidate, (ObsField, PairObsField)):
            raise BuildError(
                f"{ref.name!r} did not resolve to an ObsField/PairObsField."
            )
    return field_obj


def resolve_action_field(ref: FieldRef) -> ActionField:
    field_obj = _resolve_field(ref, (_actions,), "action")
    if not isinstance(field_obj, ActionField):
        raise BuildError(f"{ref.name!r} did not resolve to an ActionField.")
    return field_obj


# The scenario itself lives in the core API (bluesky_sandbox.sim.scenario) so that
# generated task packages depend only on the main library, not the designer.
# Re-exported here under the historical name for the designer's own callers.
DesignScenario = RandomizedScenario


# --------------------------------------------------------------------------- #
# Public builders                                                             #
# --------------------------------------------------------------------------- #
def _field_queryable_spec(ref: FieldRef):
    if ":" in ref.name:
        return None
    cls = getattr(_queryable_fields, ref.name, None)
    if cls is None:
        return None
    return getattr(cls, "queryable_spec", None)


def _temporal_queryable_names(spec: DesignSpec) -> set[str]:
    names: set[str] = set()
    fields = list(spec.env.obs_fields)
    if spec.env.intruder_obs_fields:
        fields.extend(spec.env.intruder_obs_fields)
    if spec.env.critic_obs_fields:
        fields.extend(spec.env.critic_obs_fields)
    if spec.env.critic_intruder_obs_fields:
        fields.extend(spec.env.critic_intruder_obs_fields)
    for ref in fields:
        queryable_spec = _field_queryable_spec(ref)
        if queryable_spec is None:
            continue
        requirements = set(queryable_spec.requirements)
        if not (
            QueryableFieldRequirement.STEP in requirements
            or QueryableFieldRequirement.TIME in requirements
        ):
            continue
        kwargs = ref.kwargs
        if kwargs.get("query_name"):
            names.add(str(kwargs["query_name"]))
            continue
        if "query_names" in kwargs:
            names.update(str(name) for name in kwargs["query_names"])
            continue
        if queryable_spec.cardinality in (
            QueryableFieldCardinality.MULTIPLE,
            QueryableFieldCardinality.ACTIVE,
        ):
            names.update(spec.queryables)
    return names


def with_inferred_temporal_tracking(spec: DesignSpec) -> DesignSpec:
    """Return a copy with temporal queryables marked from field requirements."""
    temporal_names = _temporal_queryable_names(spec)
    if not temporal_names:
        return DesignSpec.from_dict(copy.deepcopy(spec.to_dict()))
    out = DesignSpec.from_dict(copy.deepcopy(spec.to_dict()))
    for name in temporal_names:
        queryable = out.queryables.get(name)
        if isinstance(queryable, dict) and queryable.get("type") in (
            "query_region",
            "waypoint",
        ):
            queryable["track_temporal_state"] = True
    return out


def _region_resolver(spec: DesignSpec) -> Callable[[Any], Any]:
    """Return a function inlining ``{"ref": name}`` bounds from ``spec.regions``."""
    regions = spec.regions or {}

    def resolve(bounds: Any) -> Any:
        if isinstance(bounds, dict) and set(bounds) == {"ref"}:
            name = bounds["ref"]
            if name not in regions:
                raise BuildError(f"region ref {name!r} not found in spec.regions.")
            return copy.deepcopy(regions[name])
        return bounds

    return resolve


def _resolved_geometry(
    spec: DesignSpec,
) -> tuple[Any, dict[str, dict[str, Any]], Any]:
    """Return (airspace, queryables, spawn) spec dicts with region refs inlined."""
    resolve = _region_resolver(spec)
    airspace = resolve(spec.airspace) if spec.airspace is not None else None
    queryables: dict[str, dict[str, Any]] = {}
    route_sampling: dict[str, dict[str, Any]] = {}
    for name, q in spec.queryables.items():
        if isinstance(q, dict):
            q = dict(q)
            if q.get("type") == "query_region" and "bounds" in q:
                q["bounds"] = resolve(q["bounds"])
            if q.get("type") == "waypoint":
                if q.get("sample") is not None:
                    q["sample"] = resolve(q["sample"])
                    if q.get("sample_per") == "aircraft":
                        route_sampling.setdefault(name, {})["sample"] = q["sample"]
                # Distribution-valued constraint fields become their support scalar
                # here; the per-episode draw happens in the scenario.
                q, _ = _spec.extract_waypoint_field_dists(q)
                if _spec.is_envelope_value(q.get("alt_ft")):
                    route_sampling.setdefault(name, {})["sample_alt_from_envelope"] = True
                    q["alt_ft"] = None
                if _spec.is_envelope_value(q.get("speed_kts")):
                    route_sampling.setdefault(name, {})["sample_speed_from_envelope"] = True
                    q["speed_kts"] = None
                if "envelope_alt_floor_ft" in q and name in route_sampling:
                    route_sampling[name]["envelope_alt_floor_ft"] = q[
                        "envelope_alt_floor_ft"
                    ]
                # Bound the per-aircraft envelope altitude draw to what the
                # aircraft can climb/descend to before reaching the fix. Only
                # meaningful alongside an envelope-sampled altitude.
                reachable = q.pop("reachable_from_spawn", False)
                vs_fraction = q.pop("reachable_vs_fraction", None)
                if reachable and route_sampling.get(name, {}).get(
                    "sample_alt_from_envelope"
                ):
                    route_sampling[name]["reachable_from_spawn"] = True
                    if vs_fraction is not None:
                        route_sampling[name]["reachable_vs_fraction"] = vs_fraction
        queryables[name] = q
    spawn = spec.spawn
    if isinstance(spawn, dict):
        spawn = dict(spawn)
        spawn = _move_waypoint_sampling_to_routes(spawn, route_sampling)
        spawn["regions"] = [
            {**r, "bounds": resolve(r["bounds"])} if isinstance(r, dict) and "bounds" in r else r
            for r in spawn.get("regions", [])
        ]
    return airspace, queryables, spawn


def _move_waypoint_sampling_to_routes(
    spawn: dict[str, Any],
    route_sampling: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Copy per-aircraft waypoint sampling metadata onto route steps."""
    if not route_sampling:
        return spawn

    def step_with_sampling(step):
        if isinstance(step, str):
            metadata = route_sampling.get(step)
            return {"waypoint": step, **metadata} if metadata else step
        if not isinstance(step, dict):
            return step
        if "waypoint" in step:
            metadata = route_sampling.get(step["waypoint"])
            return {**step, **metadata} if metadata else step
        if "choice" in step:
            return {
                **step,
                "choice": [
                    [step_with_sampling(s) for s in branch]
                    if isinstance(branch, list)
                    else step_with_sampling(branch)
                    for branch in step["choice"]
                ],
            }
        return step

    def route_with_sampling(route):
        if not isinstance(route, list):
            return route
        return [step_with_sampling(step) for step in route]

    out = dict(spawn)
    out["route"] = route_with_sampling(out.get("route"))
    out["routes"] = {
        name: route_with_sampling(route)
        for name, route in out.get("routes", {}).items()
    }
    out["regions"] = [
        {**region, "route": route_with_sampling(region.get("route"))}
        if isinstance(region, dict)
        else region
        for region in out.get("regions", [])
    ]
    return out


def _materialise(
    spec: DesignSpec,
) -> tuple[Bounds | None, dict[str, Queryable], SpawnConfig]:
    airspace_d, queryables_d, spawn_d = _resolved_geometry(spec)
    airspace = _spec.load(airspace_d) if airspace_d is not None else None
    queryables = {name: _spec.load(q) for name, q in queryables_d.items()}
    if not isinstance(spawn_d, dict):
        raise BuildError("DesignSpec.spawn must be a spawn_config spec dict.")
    spawn = _spec.load(spawn_d)
    if not isinstance(spawn, SpawnConfig):
        raise BuildError("DesignSpec.spawn did not resolve to a SpawnConfig.")
    _load_route_step_sample_bounds(spawn)
    return airspace, queryables, spawn


def _load_route_step_sample_bounds(spawn: SpawnConfig) -> None:
    """Convert designer route-step sample dicts into runtime Bounds objects."""

    def load_step(step):
        if not isinstance(step, dict):
            return step
        if "sample" in step and isinstance(step["sample"], dict):
            step = {**step, "sample": _load_sample_bounds(step["sample"])}
        if "choice" in step:
            step = {
                **step,
                "choice": [
                    [load_step(s) for s in branch]
                    if isinstance(branch, list)
                    else load_step(branch)
                    for branch in step["choice"]
                ],
            }
        return step

    def load_route(route):
        if not isinstance(route, list):
            return route
        return [load_step(step) for step in route]

    spawn.route = load_route(spawn.route)
    spawn.routes = {
        name: load_route(route)
        for name, route in spawn.routes.items()
    }
    for region in spawn.regions:
        region.route = load_route(region.route)


def _load_sample_bounds(d: dict[str, Any]) -> Bounds:
    """Load a waypoint ``sample`` region, accepting a bare footprint or a Bounds."""
    obj = _spec.load(d)
    return RegionBounds(obj) if isinstance(obj, Footprint) else obj






def _sampled_waypoint_regions(spec: DesignSpec) -> dict[str, Bounds]:
    """Per-episode sampled-waypoint regions (DesignScenario resamples each reset).

    Per-aircraft sampled waypoints (``sample_per == "aircraft"``) are excluded:
    their region is copied onto route steps and drawn per aircraft at spawn,
    not per episode here.
    """
    resolve = _region_resolver(spec)
    out: dict[str, Bounds] = {}
    for name, q in spec.queryables.items():
        if (
            isinstance(q, dict)
            and q.get("type") == "waypoint"
            and q.get("sample")
            and q.get("sample_per") != "aircraft"
        ):
            out[name] = _load_sample_bounds(resolve(q["sample"]))
    return out


def _region_param_dists(spec: DesignSpec) -> dict[str, dict[str, Any]]:
    """Sampled footprint params of named regions: {region: {param_path: value}}.

    Values are loaded samplers (a ``(low, high)`` tuple or a scipy dist),
    ready for :func:`~bluesky_sandbox.sim.scenario.transforms.sample_scalar`.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, region in spec.regions.items():
        fp = region.get("footprint") if isinstance(region, dict) else None
        if not isinstance(fp, dict):
            continue
        dists = _spec.footprint_param_dists(fp)
        if dists:
            out[name] = {path: _spec.load_value(v) for path, v in dists.items()}
    return out


def _value_endpoints(value: Any) -> tuple[float, float]:
    """Finite (low, high) endpoints of a loaded sampled value."""
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    lo, hi = value.support()
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise BuildError(
            "sampled region params require finite support; wrap the "
            "distribution in Bounded or use a range"
        )
    return float(lo), float(hi)


def _support_substituted_spec(
    spec: DesignSpec, dists: dict[str, dict[str, Any]]
) -> DesignSpec:
    """Copy of the spec with each sampled region widened to its support union.

    Each sampled footprint is replaced by the shapely union of the shapes at
    every parameter-endpoint combination, so the static geometry (what
    ``support()`` reports, and what the scenario holds before the first
    ``sample()``) covers every episode the sampler can draw. Sound for params
    that grow/shrink the shape monotonically (radii, half-angles, box edges);
    positional params such as a sampled bearing are only covered at their
    endpoints - orient with ``transform.rotation`` instead.
    """
    out = copy.deepcopy(spec)
    for name, params in dists.items():
        paths = sorted(params)
        if len(paths) > 8:
            raise BuildError(
                f"region {name!r} samples {len(paths)} footprint params; "
                "the endpoint-union support caps at 8"
            )
        base_fp = out.regions[name]["footprint"]
        variants: list[Footprint] = []
        for combo in itertools.product(
            *(_value_endpoints(params[p]) for p in paths)
        ):
            fp_dict = copy.deepcopy(base_fp)
            for path, val in zip(paths, combo):
                _spec.set_footprint_param(fp_dict, path, val)
            variants.append(
                _spec.load({"type": "region", "footprint": fp_dict}).footprint
            )
        union = union_footprints(variants)
        out.regions[name]["footprint"] = _spec.dump(
            RegionBounds(union, None)
        )["footprint"]
    return out


def _load_named_regions(spec: DesignSpec) -> dict[str, Bounds]:
    """Resolved Bounds for every named region (sampled params representative)."""
    return {
        name: _spec.load(d)
        for name, d in spec.regions.items()
        if isinstance(d, dict)
    }


def _make_episode_geometry_fn(
    spec: DesignSpec,
    dists: dict[str, dict[str, Any]],
    region_sink: dict[str, Bounds],
) -> Callable[[Any], dict[str, Any]]:
    """Per-episode geometry rebuild for sampled region params.

    Draws every sampled param, substitutes the values into a copy of the spec,
    and re-materialises the resolved geometry - so every element referencing a
    sampled region (spawn bounds, route sample steps, sampled waypoints, the
    airspace) picks up the episode's shape through the normal ref resolution.
    ``region_sink`` is refreshed with the drawn named-region bounds each
    episode, so tooling (the designer preview) can show them.
    """

    def rebuild(rng) -> dict[str, Any]:
        sub = copy.deepcopy(spec)
        for region, params in dists.items():
            fp = sub.regions[region]["footprint"]
            for path, value in params.items():
                _spec.set_footprint_param(fp, path, _t.sample_scalar(value, rng))
        airspace, queryables, spawn = _materialise(sub)
        region_sink.clear()
        region_sink.update(_load_named_regions(sub))
        return {
            "airspace_bounds": airspace,
            "queryables": queryables,
            "spawn": spawn,
            "sampled_waypoints": _sampled_waypoint_regions(sub),
        }

    return rebuild


def compile_scenario_hooks(spec: DesignSpec) -> dict[str, Callable[..., Any]]:
    """Compile ``scenario_setup`` + ``scenario_hooks`` into callables.

    The designer's live preview builds a scenario straight from the spec rather
    than from generated code, so without this the hooks would run in the
    generated package and nowhere else - preview and `Generate task structure`
    would silently disagree about what the environment is. Compiling the same
    two strings here keeps the two paths on one definition.

    ``scenario_setup`` is exec'd into a bare namespace that each hook then
    closes over, mirroring how codegen emits it at module scope. That namespace
    is deliberately bare: a hook body that relies on an import the setup did not
    make will fail here exactly as it would in the generated module, rather than
    picking up a name this module happens to have.
    """
    from .spec import SCENARIO_HOOKS

    hooks = {k: v for k, v in (spec.scenario_hooks or {}).items() if v.strip()}
    if not hooks:
        return {}
    namespace: dict[str, Any] = {}
    if spec.scenario_setup.strip():
        exec(compile(spec.scenario_setup, "<scenario_setup>", "exec"), namespace)
    out: dict[str, Callable[..., Any]] = {}
    for name, body in hooks.items():
        args = SCENARIO_HOOKS[name][0]
        source = f"def _hook({', '.join(args)}):\n" + textwrap.indent(body, "    ")
        exec(compile(source, f"<scenario_hook:{name}>", "exec"), namespace)
        out[name] = namespace.pop("_hook")
    return out


def _parse_rotation(spec: DesignSpec) -> dict[str, Any] | None:
    """Parse ``spec.transform.rotation`` into a sampler dict, or ``None``."""
    transform = spec.transform or {}
    rot = transform.get("rotation")
    if not rot:
        return None
    angle = _spec.load_value(rot.get("angle_deg", 0.0))
    pivot = rot.get("pivot")
    pivot = tuple(pivot) if pivot else None
    return {"angle": angle, "pivot": pivot}


def elements_for_region(spec: DesignSpec, region_name: str) -> list[str]:
    """Element ids whose geometry comes from the named bounds ``region_name``.

    Group membership is expressed as **bounds** (named regions); rotating a
    bounds rotates every element that references it. Ids: ``"airspace"``,
    ``"q:<name>"`` (queryable), ``"s:<name>"`` (spawn region).
    """
    ids: list[str] = []
    air = spec.airspace
    if isinstance(air, dict) and air.get("ref") == region_name:
        ids.append("airspace")
    for qname, q in spec.queryables.items():
        if not isinstance(q, dict):
            continue
        bounds = q.get("bounds")
        sample = q.get("sample")
        if (isinstance(bounds, dict) and bounds.get("ref") == region_name) or (
            isinstance(sample, dict) and sample.get("ref") == region_name
        ):
            ids.append(f"q:{qname}")
    spawn = spec.spawn
    regions = spawn.get("regions", []) if isinstance(spawn, dict) else []
    for r in regions:
        if isinstance(r, dict):
            bounds = r.get("bounds")
            if isinstance(bounds, dict) and bounds.get("ref") == region_name:
                ids.append(f"s:{r.get('name')}")
    return ids


def expand_region_members(spec: DesignSpec, region_names: list[str]) -> list[str]:
    """Flatten group members into the element ids the runtime transforms.

    A member is either a bounds (region) name — expanded to every element that
    references it — or ``"wp:<name>"`` naming a waypoint queryable directly (so a
    fixed lat/lon waypoint can be grouped even though it has no bounds).
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(eid: str) -> None:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)

    for member in region_names:
        if isinstance(member, str) and member.startswith("wp:"):
            name = member[3:]
            if name in spec.queryables:
                add(f"q:{name}")
            continue
        for eid in elements_for_region(spec, member):
            add(eid)
    return out


def _parse_groups(spec: DesignSpec) -> tuple[dict[str, Any], ...] | None:
    """Parse ``spec.transform.groups`` into runtime rotation groups, or ``None``.

    Group ``members`` are bounds (region) names in the spec; they're translated
    here into the element ids the core scenario rotates.
    """
    transform = spec.transform or {}
    groups = transform.get("groups")
    if not groups:
        return None

    def _parse_translation(t: Any) -> dict[str, Any] | None:
        """Per-episode east/north offset (nm), each a sampled value, or None."""
        if not t:
            return None
        east = _spec.load_value(t.get("east_nm", 0.0))
        north = _spec.load_value(t.get("north_nm", 0.0))
        if not east and not north:
            return None
        return {"east": east, "north": north}

    out: list[dict[str, Any]] = []
    for g in groups:
        pivot = g.get("pivot")
        out.append(
            {
                "id": g["id"],
                "angle": _spec.load_value(g.get("angle_deg", 0.0)),
                "translation": _parse_translation(g.get("translation")),
                "scale": _spec.load_value(g.get("scale", 1.0)),
                "pivot": tuple(pivot) if pivot else None,
                "members": expand_region_members(spec, list(g.get("members", []))),
                "parent": g.get("parent"),
            }
        )
    return tuple(out)


def _waypoint_field_dists(spec: DesignSpec) -> dict[str, dict[str, Any]]:
    """Per-episode resampled waypoint constraint/target fields, keyed by name."""
    out: dict[str, dict[str, Any]] = {}
    for name, q in spec.queryables.items():
        if isinstance(q, dict) and q.get("type") == "waypoint":
            _, dists = _spec.extract_waypoint_field_dists(q)
            if dists:
                out[name] = {f: _spec.load_value(v) for f, v in dists.items()}
    return out


def build_scenario(spec: DesignSpec) -> DesignScenario:
    """Compile the spec's geometry/spawn/queryables into a runnable scenario."""
    from bluesky_sandbox.config import apply_performance_model

    # Before any sampling: spawn altitudes and speeds are drawn from the
    # aircraft's flight envelope, which is read from whichever performance
    # model BlueSky is set to. This path never builds an EnvConfig (the
    # designer previews geometry without one), so nothing else would set it.
    apply_performance_model(getattr(spec.env, "performance_model", None))
    spec = with_inferred_temporal_tracking(spec)
    region_dists = _region_param_dists(spec)
    # Named-region bounds for tooling (the designer preview): starts canonical
    # (representative shapes); the episode hook refreshes it with each sample's
    # drawn shapes. Exposed on the scenario as ``design_regions``.
    region_sink = _load_named_regions(spec)
    scenario_hooks = compile_scenario_hooks(spec)
    if region_dists or scenario_hooks:
        # Static geometry = endpoint-union support of the sampled shapes, so
        # support() covers every episode; the hook rebuilds per episode.
        support_spec = (
            _support_substituted_spec(spec, region_dists) if region_dists else spec
        )
        airspace, queryables, spawn = _materialise(support_spec)
        sampled_waypoints = _sampled_waypoint_regions(support_spec)
        episode_geometry_fn = _make_episode_geometry_fn(
            spec, region_dists, region_sink
        )
    else:
        airspace, queryables, spawn = _materialise(spec)
        sampled_waypoints = _sampled_waypoint_regions(spec)
        episode_geometry_fn = None
    # Chain the design's own hook after the structured rebuild, exactly as
    # codegen does. With no region params there is nothing to rebuild,
    # so the hook is handed the static geometry instead.
    design_hook = scenario_hooks.get("episode_geometry")
    if design_hook is not None:
        structured = episode_geometry_fn
        static_geometry = {
            "airspace_bounds": airspace,
            "queryables": queryables,
            "spawn": spawn,
            "sampled_waypoints": sampled_waypoints,
        }

        def episode_geometry_fn(rng, _structured=structured, _static=static_geometry):
            base = dict(_structured(rng)) if _structured else copy.deepcopy(_static)
            return design_hook(base, rng)
    scenario = DesignScenario(
        airspace_bounds=airspace,
        spawn=spawn,
        queryables=queryables,
        rotation=_parse_rotation(spec),
        groups=_parse_groups(spec),
        sampled_waypoints=sampled_waypoints,
        waypoint_fields=_waypoint_field_dists(spec),
        episode_geometry_fn=episode_geometry_fn,
    )
    object.__setattr__(scenario, "design_regions", region_sink)
    object.__setattr__(scenario, "design_region_group_chains", _region_group_chains(spec))
    return scenario


def _region_group_chains(spec: DesignSpec) -> dict[str, list[str]]:
    """Group-id chain (inner-most first) for each named region under groups.

    Group ``members`` name regions in the spec; a region moves with its group
    and that group's ancestors. Used by the preview to place named-region
    geometry in the sampled episode's frame (mirrors the runtime's element
    chains, at region rather than element granularity).
    """
    transform = spec.transform or {}
    groups = {g["id"]: g for g in transform.get("groups") or []}
    if not groups:
        return {}

    def chain(gid: str | None) -> list[str]:
        out: list[str] = []
        while gid is not None and gid in groups:
            out.append(gid)
            gid = groups[gid].get("parent")
        return out

    out: dict[str, list[str]] = {}
    for gid, g in groups.items():
        for member in g.get("members", ()):
            if isinstance(member, str) and not member.startswith("wp:"):
                out.setdefault(member, chain(gid))
    return out


def build_design_config(spec: DesignSpec) -> EnvConfig:
    """Compile the static :class:`EnvConfig` from a spec.

    Field references and code references are resolved here, so this is where a
    bad import string or unknown field surfaces as a :class:`BuildError`.
    """
    # Make the design's editable code (reward/termination, custom fields)
    # importable before resolving any references to it.
    install_code_modules(spec.code)

    env = spec.env
    validate_hook_setup(env.hook_setup)
    intruder_fields = (
        None
        if env.intruder_obs_fields is None
        else [resolve_obs_field(f) for f in env.intruder_obs_fields]
    )
    critic_obs_fields = (
        None
        if env.critic_obs_fields is None
        else [resolve_obs_field(f) for f in env.critic_obs_fields]
    )
    critic_intruder_fields = (
        None
        if env.critic_intruder_obs_fields is None
        else [resolve_obs_field(f) for f in env.critic_intruder_obs_fields]
    )
    try:
        config = EnvConfig(
            obs_fields=[resolve_obs_field(f) for f in env.obs_fields],
            intruder_obs_fields=intruder_fields,
            critic_obs_fields=critic_obs_fields,
            critic_intruder_obs_fields=critic_intruder_fields,
            action_fields=[resolve_action_field(f) for f in env.action_fields],
            allowed_aircraft=list(env.allowed_aircraft),
            dt=env.dt,
            simdt=env.simdt,
            cd_method=env.cd_method,
            reso_method=env.reso_method,
            pz_radius_nm=env.pz_radius_nm,
            pz_height_ft=env.pz_height_ft,
            lookahead_s=env.lookahead_s,
            performance_model=env.performance_model,
            wind_dir_deg=env.wind_dir_deg,
            wind_kts=env.wind_kts,
            turbulence_kts=env.turbulence_kts,
            gust_tau_s=env.gust_tau_s,
            task_info_providers=build_inline_task_info_providers(
                env.task_info_setup,
                env.task_info,
            ) + [
                resolve_callable(p) for p in env.task_info_providers
            ],
        )
        return config
    except BuildError:
        raise
    except (ValueError, TypeError) as e:
        # EnvConfig.__post_init__ validates aggressively; surface it cleanly.
        raise BuildError(f"EnvConfig validation failed: {e}") from e


# Re-exported for callers that want to apply derived bounds etc. themselves.
__all__ = [
    "BuildError",
    "DesignScenario",
    "build_design_config",
    "build_scenario",
    "resolve_action_field",
    "resolve_callable",
    "resolve_obs_field",
]
