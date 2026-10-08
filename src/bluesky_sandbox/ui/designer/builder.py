"""Compile a :class:`~bluesky_sandbox.ui.designer.spec.DesignSpec` into live objects.

A spec is the design document; the builder turns it into the runtime objects
the env consumes:

* :func:`build_scenario` - a :class:`~bluesky_sandbox.sim.sampling.Scenario` over the
  airspace / spawn / queryables. Because the spawn config carries its own
  distributions (counts, params), per-episode randomization happens inside
  ``SpawnConfig.plan_episode`` at reset time, so ``sample()`` and ``support()``
  return the same schema-stable :class:`EpisodeSpec`.
* :func:`build_design_config` - a static :class:`~bluesky_sandbox.config.EnvConfig`,
  resolving field references against the field modules.

The split mirrors the design seam: structured data is materialized via
:func:`~bluesky_sandbox.ui.designer.spec.load`; logic is resolved by import.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import math
import sys
import textwrap
import traceback
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

from bluesky_sandbox.config import EnvConfig, apply_performance_model
from bluesky_sandbox.env import BATCHABLE_HOOKS
from bluesky_sandbox.interface.fields import actions as _actions
from bluesky_sandbox.interface.fields.actions import Clearance, Grid
from bluesky_sandbox.interface.fields import observations as _observations
from bluesky_sandbox.interface.fields.base import (
    ActionField,
    ObsField,
    PairObsField,
    QueryableFieldCardinality,
    QueryableFieldRequirement,
)
from bluesky_sandbox.interface.wrappers.observations import normalizer as _normalizers
from bluesky_sandbox.sim.bounds import Bounds, RegionBounds, generate_regions, union_footprints
from bluesky_sandbox.sim.bounds.base import Footprint
from bluesky_sandbox.sim.queryables import Queryable
from bluesky_sandbox.sim.scenario import GeometryDict, RandomizedScenario
from bluesky_sandbox.sim.scenario import transforms as _t
from bluesky_sandbox.sim.spawn import PlannedSource, SpawnConfig, SpawnSource, nearest_entry

from . import setup_code
from . import spec as _spec
from .spec import SCENARIO_HOOKS, DesignSpec, EnvSpec, FieldRef, SpecError


class BuildError(ValueError):
    """Raised when a spec cannot be compiled into runtime objects.

    ``block`` and ``line`` place an error in a code block, when it has one: the
    editor's block key (``"hook_setup"``, ``"task_info:waypoint_outcome"``,
    ``"code:custom_fields.py"``) and the line within that block's own text.
    """

    def __init__(
        self, message: str, *, block: str | None = None, line: int | None = None
    ) -> None:
        super().__init__(message)
        self.block = block
        self.line = line


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
            line = _error_line(e, f"<designer:{filename}>")
            raise BuildError(
                f"error in {filename}: {e}", block=f"code:{filename}", line=line
            ) from e


# --------------------------------------------------------------------------- #
# Code-reference resolution ("module:attr")                                   #
# --------------------------------------------------------------------------- #
def resolve_callable(ref: str) -> Callable[..., Any]:
    """Import a ``"package.module:attr"`` reference and return the attribute.

    This is how the spec points at code-tab logic (reward / termination /
    truncation functions and task-info providers) without serializing it.
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


def run_setup_module(env: EnvSpec, config: EnvConfig) -> ModuleType:
    """Run the design's setup code in one module, as the generated ``setup.py``.

    The code is :func:`.setup_code.setup_source` - the task-info setup, the
    inline task-info entries as functions, the hook setup - with ``CONFIG``
    bound to ``config``, so a helper one block defines is visible to the others
    exactly as it is in a generated package. A line that fails is reported in
    the block it came from.
    """
    try:
        source, starts = setup_code.located_setup_source(
            env.task_info_setup, env.task_info, env.hook_setup
        )
    except ValueError as e:
        raise BuildError(str(e)) from e
    module = ModuleType(f"{CODE_NS}.setup")
    module.__dict__["CONFIG"] = config
    # Registered while the code runs: ``@dataclass`` looks a class's module up
    # in ``sys.modules`` to resolve its annotations.
    sys.modules[module.__name__] = module
    try:
        code = compile(
            "from __future__ import annotations\n" + source,
            _SETUP_FILENAME,
            "exec",
        )
        exec(code, module.__dict__)
    except Exception as e:
        raise _located_setup_error(e, starts) from e
    finally:
        sys.modules.pop(module.__name__, None)
    return module


_SETUP_FILENAME = "<designer setup>"


def _located_setup_error(
    error: Exception, starts: list[tuple[setup_code.Part, int]]
) -> BuildError:
    """``error`` as a BuildError placed on the setup block and line it came from."""
    lineno = _error_line(error, _SETUP_FILENAME)
    if lineno is None or not starts:
        return BuildError(f"error in setup code: {error}")
    part, line = setup_code.locate(starts, lineno - 1)  # the __future__ line
    return BuildError(
        f"error in {part.where}, line {line}: {error}", block=part.block, line=line
    )


def _error_line(error: Exception, filename: str) -> int | None:
    """The line of ``filename`` an error was raised on, if it came from there."""
    if isinstance(error, SyntaxError) and error.filename == filename:
        return error.lineno
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if frame.filename == filename
    ]
    return frames[-1].lineno if frames else None


def setup_providers(env: EnvSpec, module: ModuleType) -> list[Callable[..., Any]]:
    """The design's inline task-info providers, from its setup module."""
    names = setup_code.provider_names(
        env.task_info, env.task_info_setup, env.hook_setup
    )
    providers = []
    for spec, name in zip(env.task_info, names):
        provider = module.__dict__.get(name)
        if not callable(provider):
            raise BuildError(
                f"task-info provider {spec.name!r} references non-callable {name!r}."
            )
        providers.append(provider)
    return providers


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


def _resolve_grid(value: Any) -> Any:
    """``{"type": "grid", "step": 1000, "on": "target"}`` as a :class:`Grid`."""
    if not isinstance(value, dict) or value.get("type") != "grid":
        return value
    try:
        return Grid(value.get("step"), on=value.get("on") or "value")
    except (TypeError, ValueError) as e:
        raise BuildError(f"grid: {e}") from e


_RESOLVERS: dict[str, Callable[[Any], Any]] = {
    "normalizer": _resolve_normalizer,
    "grid": _resolve_grid,
}


def _resolve_constructor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _RESOLVERS.get(key, lambda value: value)(value)
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
        (_observations,),
        "observation",
    )
    candidates = field_obj if isinstance(field_obj, (list, tuple)) else [field_obj]
    for candidate in candidates:
        if not isinstance(candidate, (ObsField, PairObsField)):
            raise BuildError(
                f"{ref.name!r} did not resolve to an ObsField/PairObsField."
            )
    return field_obj


def resolve_action_field(ref: FieldRef) -> ActionField | Clearance:
    field_obj = _resolve_field(ref, (_actions,), "action")
    if not isinstance(field_obj, ActionField):
        raise BuildError(f"{ref.name!r} did not resolve to an ActionField.")
    if ref.clearance is None:
        return field_obj
    duration = ref.clearance.get("duration")
    try:
        return Clearance(
            field_obj,
            duration=None if duration is None else tuple(duration),
            lock=ref.clearance.get("lock"),
            duration_normalizer=_resolve_normalizer(
                ref.clearance.get("duration_normalizer")
            ),
            duration_from=ref.clearance.get("duration_from") or "issued",
            duration_grid=_resolve_grid(ref.clearance.get("duration_grid")),
        )
    except (TypeError, ValueError) as e:
        raise BuildError(f"clearance on {ref.name!r}: {e}") from e


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
    cls = getattr(_observations, ref.name, None)
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
    fields.extend(spec.env.state_fields or ())
    fields.extend(spec.env.intruder_state_fields or ())
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
    """Return a function inlining ``{"ref": name}`` shapes from ``spec.shapes``."""
    regions = spec.shapes or {}

    def resolve(bounds: Any) -> Any:
        if isinstance(bounds, dict) and set(bounds) == {"ref"}:
            name = bounds["ref"]
            if name in regions:
                return copy.deepcopy(regions[name])
            # One of a partition's shapes, before it is drawn: where any of
            # them can be - the partition (its envelope). An episode's draw is
            # written into the regions under this name, and resolves above.
            base = name.rpartition(".")[0]
            if base in regions and name in _spec.partition_names(base, regions[base]):
                return copy.deepcopy(regions[base])
            raise BuildError(f"shape {name!r} is not in the design.")
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
            if q.get("type") == "query_region" and _spec.shape_of(q) is not None:
                q["shape"] = resolve(_spec.shape_of(q))
                q.pop("bounds", None)
            if q.get("type") == "waypoint":
                # At a point: the point itself, named so the episode's draw of
                # it is the one the waypoint is at.
                shape = _spec.shape_of(q)
                if isinstance(shape, dict) and "ref" in shape:
                    q["shape"] = {**resolve(shape), "name": shape["ref"]}
                    q.pop("bounds", None)
                if q.get("sample") is not None:
                    q["sample"] = resolve(q["sample"])
                    if q.get("sample_per") == "aircraft":
                        route_sampling.setdefault(name, {})["sample"] = q["sample"]
                # Distribution-valued constraint fields become their support scalar
                # here; the per-episode draw happens in the scenario.
                q, _ = _spec.extract_waypoint_field_dists(q)
                if _spec.is_envelope_value(q.get("alt_ft")):
                    route_sampling.setdefault(name, {})["sample_alt_from_envelope"] = True
                    # Levels: the envelope marker's grid, for the per-aircraft draw.
                    if (step := _spec.envelope_alt_step(q["alt_ft"])) is not None:
                        route_sampling[name]["alt_step_ft"] = step
                    q["alt_ft"] = None
                elif _spec.is_start_value(q.get("alt_ft")):
                    # Level flight: the altitude the leg starts at, per aircraft.
                    route_sampling.setdefault(name, {})["alt_from_start"] = True
                    if (step := _spec.envelope_alt_step(q["alt_ft"])) is not None:
                        route_sampling[name]["alt_step_ft"] = step
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
                # A target arrival time over this fix, for every aircraft whose
                # route crosses it: nominal plus a slack drawn per aircraft.
                slack = q.pop("arrival_slack_s", None)
                if slack is not None:
                    route_sampling.setdefault(name, {})["arrival_slack_s"] = (
                        _spec.load_value(slack)
                    )
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
            {**{k: v for k, v in r.items() if k != "bounds"}, "shape": resolve(_spec.shape_of(r))}
            if isinstance(r, dict) and _spec.shape_of(r) is not None
            else r
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


def _materialize(
    spec: DesignSpec,
    sources: Sequence[Any] = (),
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
    # The design's spawn sources (compile_scenario_code): its aircraft beyond
    # the regions'.
    spawn.sources = list(sources)
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
    for name, region in spec.shapes.items():
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
        base_fp = out.shapes[name]["footprint"]
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
        out.shapes[name]["footprint"] = _spec.dump(
            RegionBounds(union, None)
        )["footprint"]
    return out


def _load_named_regions(spec: DesignSpec) -> dict[str, Bounds]:
    """Resolved Bounds for every named region (sampled params representative,
    a generated one its generator's envelope)."""
    try:
        return {
            name: _spec.load(d)
            for name, d in spec.shapes.items()
            if isinstance(d, dict)
        }
    except (SpecError, ValueError, TypeError) as e:
        raise BuildError(str(e)) from e


def _generated_regions(spec: DesignSpec) -> list[str]:
    """The named regions drawn anew each episode: a generator draws their
    shape, or a placement their spot - or both."""
    return [
        name
        for name, d in (spec.shapes or {}).items()
        if isinstance(d, dict)
        and (
            (d.get("footprint") or {}).get("type") == "generated"
            or d.get("placement")
            or d.get("motion")
        )
    ]


def _inline_generator_refs(spec: DesignSpec) -> DesignSpec:
    """``spec`` with each ``{"ref": name}`` in a generator's or a placement's
    params - a parent, a region to stay within or avoid - replaced by that
    region, as other refs are."""
    names = _generated_regions(spec)
    if not names:
        return spec
    resolve = _region_resolver(spec)
    out = copy.deepcopy(spec)

    def inline(value: Any, chain: tuple[str, ...]) -> Any:
        """``value`` with every ref in it inlined - and the refs inside what
        they name, so a region placed clear of a generated one gets that
        one's parent too."""
        if isinstance(value, dict) and set(value) == {"ref"}:
            if value["ref"] in chain:
                through = f", through {' -> '.join((*chain, value['ref']))}" if len(chain) > 1 else ""
                raise BuildError(f"region {chain[0]!r} cannot refer to itself{through}")
            # Named, so an episode's draw of it can stand in for it.
            return inline_region({**resolve(value), "name": value["ref"]}, (*chain, value["ref"]))
        if isinstance(value, list):
            return [inline(v, chain) for v in value]
        return value

    def inline_region(region: Any, chain: tuple[str, ...]) -> Any:
        if not isinstance(region, dict):
            return region
        footprint = region.get("footprint") or {}
        if footprint.get("type") == "generated":
            params = footprint.setdefault("params", {})
            for key, value in list(params.items()):
                params[key] = inline(value, chain)
        for layer in [region.get("placement"), *(region.get("motion") or [])]:
            if isinstance(layer, dict):
                for key, value in list(layer.items()):
                    if key != "type":
                        layer[key] = inline(value, chain)
        return region

    for name in names:
        inline_region(out.shapes[name], (name,))
    return out


def _draw_generated(sub: DesignSpec, rng) -> dict[str, Bounds]:
    """Draw ``sub``'s generated regions for an episode, writing each drawn
    shape back into ``sub.shapes`` - so every ref to it resolves to the draw -
    and a partition's ``<name>.<i>`` beside it. Returns the named regions."""
    named = _load_named_regions(sub)
    try:
        drawn = generate_regions(named, rng)
    except ValueError as e:
        raise BuildError(str(e)) from e
    for name, bounds in drawn.items():
        if named.get(name) is not bounds:
            sub.shapes[name] = _spec.dump(bounds)
    return drawn


def _make_episode_geometry_fn(
    spec: DesignSpec,
    dists: dict[str, dict[str, Any]],
    region_sink: dict[str, Bounds],
    sources: Sequence[Any] = (),
) -> Callable[[Any], dict[str, Any]]:
    """Per-episode geometry rebuild for sampled region params and generated
    regions.

    Draws every sampled param, substitutes the values into a copy of the spec,
    draws each generated region (its shape written back into the copy),
    and re-materializes the resolved geometry - so every element referencing a
    sampled region (spawn bounds, route sample steps, sampled waypoints, the
    airspace) picks up the episode's shape through the normal ref resolution.
    ``region_sink`` is refreshed with the drawn named-region bounds each
    episode, so tooling (the designer preview) can show them.
    """

    def rebuild(rng) -> dict[str, Any]:
        sub = copy.deepcopy(spec)
        for region, params in dists.items():
            fp = sub.shapes[region]["footprint"]
            for path, value in params.items():
                _spec.set_footprint_param(fp, path, _t.sample_scalar(value, rng))
        # Generated regions: drawn, and written back so refs resolve to them.
        drawn = _draw_generated(sub, rng)
        airspace, queryables, spawn = _materialize(sub, sources)
        region_sink.clear()
        region_sink.update(drawn)
        return {
            "airspace_bounds": airspace,
            "queryables": queryables,
            "spawn": spawn,
            "sampled_waypoints": _sampled_waypoint_regions(sub),
            "shapes": dict(region_sink),
        }

    return rebuild


def compile_scenario_code(
    spec: DesignSpec,
) -> tuple[dict[str, Callable[..., Any]], list[SpawnSource]]:
    """Compile ``scenario_setup``, ``scenario_hooks`` and the spawn sources'
    ``plan`` code into callables - the hooks by name, and each source as a
    :class:`PlannedSource` with its policies.

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
    hooks = {k: v for k, v in (spec.scenario_hooks or {}).items() if v.strip()}
    sources = spawn_sources_of(spec)
    if not hooks and not sources:
        return {}, []
    # The library names the generated scenario.py imports, so a hook runs here
    # with what it has there (setup_code.SCENARIO_API).
    namespace: dict[str, Any] = dict(setup_code.scenario_api_names())
    if spec.scenario_setup.strip():
        try:
            exec(compile(spec.scenario_setup, "<scenario_setup>", "exec"), namespace)
        except Exception as e:
            raise _scenario_code_error(e, "<scenario_setup>", "scenario setup") from e
    out: dict[str, Callable[..., Any]] = {}
    for name, body in hooks.items():
        args = SCENARIO_HOOKS[name][0]
        filename = f"<scenario_hook:{name}>"
        source = f"def _hook({', '.join(args)}):\n" + textwrap.indent(body, "    ")
        try:
            exec(compile(source, filename, "exec"), namespace)
        except Exception as e:
            raise _scenario_code_error(e, filename, f"scenario hook {name}", shift=1) from e
        out[name] = _reporting(namespace.pop("_hook"), filename, f"scenario hook {name}")
    planned: list[SpawnSource] = []
    for d in sources:
        name = d["name"]
        filename = f"<spawn_source:{name}>"
        body = d.get("plan") or "return []"
        source = f"def _plan({', '.join(SPAWN_SOURCE_ARGS)}):\n" + textwrap.indent(body, "    ")
        try:
            exec(compile(source, filename, "exec"), namespace)
        except Exception as e:
            raise _scenario_code_error(e, filename, f"spawn source {name}", shift=1) from e
        plan = _reporting(namespace.pop("_plan"), filename, f"spawn source {name}")
        try:
            planned.append(PlannedSource(plan, **spawn_source_policies(d)))
        except (TypeError, ValueError) as e:
            raise BuildError(f"spawn source {name!r}: {e}") from e
    return out, planned


def compile_scenario_hooks(spec: DesignSpec) -> dict[str, Callable[..., Any]]:
    """The design's scenario hooks, compiled (see :func:`compile_scenario_code`)."""
    return compile_scenario_code(spec)[0]


#: A spawn source's ``plan`` parameters, in the design and the generated package.
SPAWN_SOURCE_ARGS = ("rng", "ctx")


def spawn_sources_of(spec: DesignSpec) -> list[dict[str, Any]]:
    """The design's spawn sources (``spawn["sources"]``), each named."""
    spawn = spec.spawn if isinstance(spec.spawn, dict) else {}
    out = []
    for i, d in enumerate(spawn.get("sources") or []):
        if not isinstance(d, dict):
            raise BuildError(f"spawn source {i} must be a dict, got {type(d).__name__}.")
        name = str(d.get("name") or f"source{i + 1}")
        if not name.isidentifier():
            raise BuildError(f"spawn source name {name!r} must be a Python identifier.")
        if any(o["name"] == name for o in out):
            raise BuildError(f"two spawn sources are named {name!r}.")
        out.append({**d, "name": name})
    return out


def spawn_source_policies(d: dict[str, Any]) -> dict[str, Any]:
    """A design spawn source's policies, as :class:`SpawnSource` takes them."""
    out: dict[str, Any] = {"name": d["name"]}
    if d.get("conflict_free"):
        out["conflict_free"] = True
    if d.get("when_blocked"):
        out["when_blocked"] = d["when_blocked"]
    if d.get("route"):
        out["route"] = d["route"]
    if d.get("assign_route") == "nearest_entry":
        out["assign_route"] = nearest_entry
    elif d.get("assign_route"):
        raise BuildError(f"spawn source {d['name']!r}: assign_route must be 'nearest_entry' or empty.")
    if d.get("max_aircraft") is not None:
        out["max_aircraft"] = int(d["max_aircraft"])
    return out


def _scenario_code_error(
    error: Exception, filename: str, where: str, shift: int = 0
) -> BuildError:
    """``error``, raised by the design's scenario code, as a BuildError naming
    the block and its line - so a mistake in it is reported, not a crash."""
    lineno = _error_line(error, filename)
    if lineno is None:
        return BuildError(f"error in {where}: {type(error).__name__}: {error}")
    line = max(lineno - shift, 1)  # past the ``def _hook(...)`` line
    return BuildError(f"error in {where}, line {line}: {type(error).__name__}: {error}")


def _reporting(hook: Callable[..., Any], filename: str, where: str) -> Callable[..., Any]:
    """``hook``, with whatever it raises reported as its block and line."""

    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return hook(*args, **kwargs)
        except BuildError:
            raise
        except Exception as e:
            raise _scenario_code_error(e, filename, where, shift=1) from e

    return run


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
    bounds rotates every element that references it - one of its shapes, for a
    partition (``sectors.1``), included - and every waypoint anchored to it.
    Ids: ``"airspace"``, ``"q:<name>"`` (queryable), ``"s:<name>"`` (spawn
    region).
    """
    shapes = set(_spec.partition_names(region_name, (spec.shapes or {}).get(region_name)))

    def refers(b: Any) -> bool:
        return isinstance(b, dict) and (b.get("ref") == region_name or b.get("ref") in shapes)

    ids: list[str] = []
    if refers(spec.airspace):
        ids.append("airspace")
    for qname, q in spec.queryables.items():
        if not isinstance(q, dict):
            continue
        if refers(_spec.shape_of(q)) or refers(q.get("sample")) or q.get("anchor") == region_name:
            ids.append(f"q:{qname}")
    spawn = spec.spawn
    regions = spawn.get("regions", []) if isinstance(spawn, dict) else []
    for r in regions:
        if isinstance(r, dict) and refers(_spec.shape_of(r)):
            ids.append(f"s:{r.get('name')}")
    return ids


def expand_region_members(spec: DesignSpec, region_names: list[str]) -> list[str]:
    """Flatten group members into the element ids the runtime transforms.

    A member is either a bounds (region) name — expanded to every element that
    references it, and to the named bounds itself (``"b:<name>"``, so code
    reading it gets it in the episode's frame) — or ``"wp:<name>"`` naming a
    waypoint queryable directly (so a fixed lat/lon waypoint can be grouped even
    though it has no bounds).
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
        if member in spec.shapes:
            add(f"b:{member}")
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

    resolve = _region_resolver(spec)

    def motion(m: dict[str, Any]) -> Any:
        """One of a group's motions, its region refs (a drift's ``within``)
        inlined."""
        m = {k: resolve(v) if isinstance(v, dict) else v for k, v in m.items()}
        try:
            return _spec._placement_load(m, _spec.motion_class)
        except _spec.SpecError as e:
            raise BuildError(f"group {g.get('name') or g['id']!r}: {e}") from e

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
                # How the group moves during the episode, as one.
                "motion": tuple(motion(m) for m in g.get("motion") or ()),
                "motion_update": g.get("motion_update") or "step",
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


def _point_center(region: dict[str, Any]) -> dict[str, float]:
    """A point region's position: its ``center``, or its navdb ``fix``'s."""
    footprint = region.get("footprint") or {}
    center = footprint.get("center")
    if isinstance(center, dict):
        return center
    fix = footprint.get("fix")
    if not fix:
        raise BuildError("a point needs a position: a center or a navdb fix")
    from .nav import resolve_waypoint  # noqa: PLC0415 - navdb loads on demand

    try:
        found = resolve_waypoint(str(fix))
    except ValueError as e:
        raise BuildError(str(e)) from e
    return {"lat_deg": found.lat_deg, "lon_deg": found.lon_deg}


def lower_waypoints(spec: DesignSpec) -> DesignSpec:
    """``spec`` with its waypoints on point bounds (``"bounds": {"ref": p}``)
    ready to build: each point's position settled (a navdb ``fix`` with no
    ``center`` looked up), and a waypoint drawn *for each aircraft* - on a
    placed point, ``sample_per: "aircraft"`` - in the form its spawn routes
    draw from: each aircraft its own spot in the point's placement region.

    Any other waypoint keeps its point: the scenario builds it ``at`` the
    point, so it is wherever the point is each episode - placed, in a group.
    A waypoint on anything but a point is refused. Waypoints in the older
    forms (lat/lon, ``waypoint``, ``sample``) are left as they are."""
    regions = spec.shapes or {}
    lowered = {
        name
        for name, q in spec.queryables.items()
        if isinstance(q, dict) and q.get("type") == "waypoint" and isinstance(_spec.shape_of(q), dict)
    }
    unplaced = [
        name
        for name, r in regions.items()
        if isinstance(r, dict)
        and (r.get("footprint") or {}).get("type") == "point"
        and not isinstance((r.get("footprint") or {}).get("center"), dict)
    ]
    if not lowered and not unplaced:
        return spec
    out = copy.deepcopy(spec)
    for name in unplaced:
        out.shapes[name]["footprint"]["center"] = _point_center(out.shapes[name])
    for name in lowered:
        q = out.queryables[name]
        ref = _spec.shape_of(q).get("ref")
        region = out.shapes.get(ref) if isinstance(ref, str) else None
        if ((region or {}).get("footprint") or {}).get("type") != "point":
            what = "a region's shape" if ref not in regions else "an area"
            raise BuildError(f"waypoint {name!r} needs a point; {ref!r} is {what}")
        for key in ("lat", "lon", "waypoint", "sample", "anchor"):
            q.pop(key, None)
        placement = region.get("placement")
        within = placement.get("within") if isinstance(placement, dict) else None
        if q.get("sample_per") == "aircraft" and within is not None:
            center = region["footprint"]["center"]
            q.pop("shape", None)
            q.pop("bounds", None)
            q["lat"], q["lon"] = float(center["lat_deg"]), float(center["lon_deg"])
            q["sample"] = copy.deepcopy(within)
        else:
            q.pop("sample_per", None)
    return out


def _check_points(spec: DesignSpec) -> None:
    """Refuse a point bounds wherever an area is needed: the airspace, a
    query region, a generator's region, a placement's or a drift's region to
    stay within. A point is fine where a position is: a spawn, a waypoint's
    sample, a placement's region to keep clear of."""
    regions = spec.shapes or {}

    def point(ref: Any) -> str | None:
        name = ref.get("ref") if isinstance(ref, dict) else None
        region = regions.get(name) if isinstance(name, str) else None
        footprint = region.get("footprint") if isinstance(region, dict) else None
        return name if isinstance(footprint, dict) and footprint.get("type") == "point" else None

    def refuse(ref: Any, what: str) -> None:
        if name := point(ref):
            raise BuildError(f"{what} needs an area; {name!r} is a point")

    refuse(spec.airspace, "the airspace")
    for qname, q in spec.queryables.items():
        if isinstance(q, dict) and q.get("type") == "query_region":
            refuse(_spec.shape_of(q), f"query region {qname!r}")

    def refs(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    def layers(owner: str, footprint: Any, placement: Any, motions: Any) -> None:
        if isinstance(footprint, dict) and footprint.get("type") == "generated":
            for key, value in (footprint.get("params") or {}).items():
                for ref in refs(value):
                    refuse(ref, f"{owner}'s generator {key!r}")
        if isinstance(placement, dict):
            refuse(placement.get("within"), f"{owner}'s placement")
        for m in motions or []:
            if isinstance(m, dict):
                for key, value in m.items():
                    for ref in refs(value) if key != "type" else []:
                        refuse(ref, f"{owner}'s {m.get('type', 'motion')} {key!r}")

    for name, region in regions.items():
        if isinstance(region, dict):
            layers(f"region {name!r}", region.get("footprint"), region.get("placement"), region.get("motion"))
    for g in (spec.transform or {}).get("groups") or []:
        if isinstance(g, dict):
            layers(f"group {g.get('name') or g.get('id')!r}", None, None, g.get("motion"))


def build_scenario(spec: DesignSpec) -> DesignScenario:
    """Compile the spec's geometry/spawn/queryables into a runnable scenario."""
    _check_points(spec)
    spec = lower_waypoints(spec)
    # Before any sampling: spawn altitudes and speeds are drawn from the
    # aircraft's flight envelope, which is read from whichever performance
    # model BlueSky is set to. This path never builds an EnvConfig (the
    # designer previews geometry without one), so nothing else would set it.
    apply_performance_model(getattr(spec.env, "performance_model", None))
    spec = _inline_generator_refs(with_inferred_temporal_tracking(spec))
    region_dists = _region_param_dists(spec)
    generated = _generated_regions(spec)
    # Named-region bounds for tooling (the designer preview): starts canonical
    # (representative shapes); the episode hook refreshes it with each sample's
    # drawn shapes. Exposed on the scenario as ``design_regions``.
    region_sink = _load_named_regions(spec)
    scenario_hooks, sources = compile_scenario_code(spec)
    if region_dists or scenario_hooks or generated:
        # Static geometry = endpoint-union support of the sampled shapes, so
        # support() covers every episode; the hook rebuilds per episode.
        support_spec = (
            _support_substituted_spec(spec, region_dists) if region_dists else spec
        )
        airspace, queryables, spawn = _materialize(support_spec, sources)
        sampled_waypoints = _sampled_waypoint_regions(support_spec)
        named_bounds = _load_named_regions(support_spec)
        episode_geometry_fn = _make_episode_geometry_fn(
            spec, region_dists, region_sink, sources
        )
    else:
        airspace, queryables, spawn = _materialize(spec, sources)
        sampled_waypoints = _sampled_waypoint_regions(spec)
        named_bounds = _load_named_regions(spec)
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
            "shapes": named_bounds,
        }

        def episode_geometry_fn(rng, _structured=structured, _static=static_geometry):
            base = GeometryDict(_structured(rng)) if _structured else GeometryDict(copy.deepcopy(_static))
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
        shapes=named_bounds,
    )
    object.__setattr__(scenario, "design_regions", region_sink)
    # Generated regions, by name, as their envelope: for the preview to show
    # where every draw lies.
    object.__setattr__(
        scenario,
        "design_generated",
        {name: named_bounds[name] for name in generated if name in named_bounds},
    )
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
    for hook in BATCHABLE_HOOKS:
        if (env.hooks.get(hook) or "").strip() and (
            env.hooks.get(f"{hook}_batch") or ""
        ).strip():
            raise BuildError(
                f"the design defines both {hook} and {hook}_batch; keep one - "
                "the per-agent hook or its batched counterpart."
            )
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
    state_fields = (
        None
        if env.state_fields is None
        else [resolve_obs_field(f) for f in env.state_fields]
    )
    intruder_state_fields = (
        None
        if env.intruder_state_fields is None
        else [resolve_obs_field(f) for f in env.intruder_state_fields]
    )
    try:
        config = EnvConfig(
            obs_fields=[resolve_obs_field(f) for f in env.obs_fields],
            intruder_obs_fields=intruder_fields,
            critic_obs_fields=critic_obs_fields,
            critic_intruder_obs_fields=critic_intruder_fields,
            state_fields=state_fields,
            intruder_state_fields=intruder_state_fields,
            action_fields=[resolve_action_field(f) for f in env.action_fields],
            allowed_aircraft=list(env.allowed_aircraft),
            dt=env.dt,
            simdt=env.simdt,
            asas_dt=env.asas_dt,
            cd_method=env.cd_method,
            reso_method=env.reso_method,
            intruder_obs_bounds=env.intruder_obs_bounds,
            pz_radius_nm=env.pz_radius_nm,
            pz_height_ft=env.pz_height_ft,
            lookahead_s=env.lookahead_s,
            performance_model=env.performance_model,
            wind_dir_deg=env.wind_dir_deg,
            wind_kts=env.wind_kts,
            turbulence_kts=env.turbulence_kts,
            gust_tau_s=env.gust_tau_s,
            fly_arrival_times=env.fly_arrival_times,
        )
    except BuildError:
        raise
    except (ValueError, TypeError) as e:
        # EnvConfig.__post_init__ validates aggressively; surface it cleanly.
        raise BuildError(f"EnvConfig validation failed: {e}") from e
    # The setup code may read CONFIG, as the generated setup.py's does, so it
    # runs once the config exists; its providers join the config after.
    module = run_setup_module(env, config)
    config.task_info_providers.extend(setup_providers(env, module))
    config.task_info_providers.extend(
        resolve_callable(ref) for ref in env.task_info_providers
    )
    return config


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
