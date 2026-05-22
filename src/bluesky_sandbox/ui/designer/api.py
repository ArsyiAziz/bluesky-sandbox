"""FastAPI server for the Environment Designer.

Exposes the backend foundation over HTTP so the web frontend (map + code tabs)
can drive it:

* ``GET  /api/health``                 - liveness.
* ``GET  /api/catalog``                - palette of footprints / bands /
  queryables / obs & action fields / aircraft types.
* ``POST /api/nav/features``           - navdb features within a bounds window.
* ``GET  /api/nav/waypoint/{ident}``   - resolve one fix.
* ``GET  /api/nav/airport/{icao}``     - resolve one airport (+ runways).
* ``POST /api/spec/validate``          - build the spec, report ok/errors + a
  summary (the map tab calls this on every edit).
* ``POST /api/spec/preview``           - renderable geometry + sampled traffic.
* ``GET/PUT/DELETE /api/specs[/{name}]`` - persist named designs.

If a built frontend exists at ``designer/web/dist`` it is served at ``/``;
otherwise run Vite's dev server and point it at this API.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_type_hints

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import catalog as _catalog
from . import codegen as _codegen
from . import nav as _nav
from . import runner as _runner
from . import spec as _spec
from .builder import BuildError, build_design_config, build_scenario
from .preview import airspace_warnings, scenario_preview
from .spec import DesignSpec, SpecError
from .store import SpecStore

_WEB_DIST = Path(__file__).parent / "web" / "dist"


def _parse_spec(body: dict[str, Any]) -> DesignSpec:
    try:
        return DesignSpec.from_dict(body)
    except (SpecError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"invalid spec: {e}") from e


def _spec_summary(spec: DesignSpec) -> dict[str, Any]:
    cfg = build_design_config(spec)
    support = build_scenario(spec).support()
    return {
        "obs_fields": [f.meta.name for f in cfg.obs_fields],
        "intruder_obs_fields": (
            None
            if cfg.intruder_obs_fields is None
            else [f.meta.name for f in cfg.intruder_obs_fields]
        ),
        "critic_obs_fields": (
            None
            if cfg.critic_obs_fields is None
            else [f.meta.name for f in cfg.critic_obs_fields]
        ),
        "critic_intruder_obs_fields": (
            None
            if cfg.critic_intruder_obs_fields is None
            else [f.meta.name for f in cfg.critic_intruder_obs_fields]
        ),
        "action_fields": [f.meta.name for f in cfg.action_fields],
        "allowed_aircraft": list(cfg.allowed_aircraft),
        "max_aircraft": spec_max_aircraft(spec),
        "has_airspace": support.airspace_bounds is not None,
        "queryables": list(support.queryables),
    }


def _type_label(value: Any) -> str:
    if value is None or value is inspect.Signature.empty:
        return ""
    if isinstance(value, str):
        return value
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    if module == "builtins" or not module:
        return name or str(value)
    return f"{module}.{name}" if name else str(value)


def _completion_symbol(
    name: str,
    *,
    kind: str = "value",
    detail: str = "",
    doc: str = "",
    insert: str | None = None,
    color: str | None = None,
    access: str | None = None,
) -> dict[str, Any]:
    item = {"name": name, "kind": kind, "detail": detail, "doc": doc}
    if insert is not None:
        item["insert"] = insert
    if color:
        item["color"] = color
    if access:
        item["access"] = access
    return item


def _member_doc(obj: Any) -> str:
    return inspect.getdoc(obj) or ""


def _setup_completion_context(source: str) -> dict[str, Any]:
    symbols: dict[str, dict[str, str]] = {}
    imports: dict[str, str] = {}

    def add_symbol(
        name: str,
        *,
        kind: str = "variable",
        detail: str = "",
        module: str | None = None,
    ) -> None:
        if not name.isidentifier():
            return
        symbols[name] = _completion_symbol(name, kind=kind, detail=detail)
        if module:
            imports[name] = module

    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return {"symbols": [], "imports": {}}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", 1)[0]
                add_symbol(name, kind="module", detail=alias.name, module=alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                add_symbol(
                    name,
                    kind="value",
                    detail=f"{node.module}.{alias.name}",
                    module=f"{node.module}.{alias.name}",
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_symbol(node.name, kind="function", detail=f"{node.name}(...)")
        elif isinstance(node, ast.ClassDef):
            add_symbol(node.name, kind="class", detail=node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _assigned_names(target):
                    add_symbol(name, kind="variable")
        elif isinstance(node, ast.AugAssign):
            for name in _assigned_names(node.target):
                add_symbol(name, kind="variable")

    return {"symbols": sorted(symbols.values(), key=lambda item: item["name"]), "imports": imports}


def _assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            names.extend(_assigned_names(elt))
        return names
    return []


def _hook_contexts_by_name() -> dict[str, dict[str, Any]]:
    from bluesky_sandbox.env import BlueskyEnv
    from bluesky_sandbox.interface.task import RewardFn, TerminationFn, TruncationFn

    protocol_hooks = {
        "reward": RewardFn,
        "terminated": TerminationFn,
        "truncated": TruncationFn,
    }

    def _protocol_hints(hook_name: str) -> dict[str, Any]:
        protocol = protocol_hooks.get(hook_name)
        if protocol is None:
            return {}
        try:
            return get_type_hints(protocol.__call__)
        except (NameError, TypeError):
            return {}

    contexts: dict[str, dict[str, Any]] = {}
    for cls in inspect.getmro(BlueskyEnv):
        for hook_name, fn in vars(cls).items():
            if not getattr(fn, "__overridable__", False) or hook_name in contexts:
                continue
            sig = inspect.signature(fn)
            try:
                function_hints = get_type_hints(fn)
            except (NameError, TypeError):
                function_hints = {}
            protocol_hints = _protocol_hints(hook_name)
            hook_params: list[dict[str, str]] = []
            hook_members: dict[str, list[dict[str, Any]]] = {}
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                clean = name.removeprefix("_")
                annotation = protocol_hints.get(clean, function_hints.get(name, param.annotation))
                detail = _type_label(annotation)
                hook_params.append(
                    _completion_symbol(clean, kind="variable", detail=detail or "parameter")
                )
                members = _members_for_annotation(annotation)
                if members:
                    hook_members[clean] = members
            contexts[hook_name] = {"params": hook_params, "members": hook_members}
    return contexts


def _typed_dict_members(cls: type) -> list[dict[str, str]]:
    try:
        annotations = get_type_hints(cls)
    except (NameError, TypeError):
        annotations = getattr(cls, "__annotations__", {})
    return [
        _completion_symbol(
            name,
            kind="field",
            detail=_type_label(annotation),
            doc=f"{cls.__name__}[{name!r}]",
            access="item",
        )
        for name, annotation in annotations.items()
    ]


def _dataclass_members(cls: type) -> list[dict[str, str]]:
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError):
        hints = {}
    fields = [
        _completion_symbol(
            field.name,
            kind="property",
            detail=_type_label(hints.get(field.name, field.type)),
            doc=f"{cls.__name__}.{field.name}",
        )
        for field in dataclasses.fields(cls)
        if not field.name.startswith("_")
    ]
    methods: list[dict[str, str]] = []
    for name, value in vars(cls).items():
        if name.startswith("_") or not callable(value):
            continue
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = "()"
        methods.append(
            _completion_symbol(
                name,
                kind="function",
                detail=f"{name}{signature}",
                doc=_member_doc(value),
            )
        )
    return fields + methods


def _class_members(cls: type) -> list[dict[str, str]]:
    if dataclasses.is_dataclass(cls):
        members = _dataclass_members(cls)
        for name, value in vars(cls).items():
            if name.startswith("_") or not isinstance(value, property):
                continue
            detail = ""
            try:
                detail = _type_label(
                    get_type_hints(value.fget or (lambda: None)).get("return")
                )
            except (NameError, TypeError):
                pass
            members.append(
                _completion_symbol(
                    name,
                    kind="property",
                    detail=detail,
                    doc=_member_doc(value),
                )
            )
        return sorted(
            {member["name"]: member for member in members}.values(),
            key=lambda item: item["name"],
        )
    if hasattr(cls, "__annotations__") and hasattr(cls, "__total__"):
        return _typed_dict_members(cls)

    members: list[dict[str, str]] = []
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError):
        hints = getattr(cls, "__annotations__", {})
    for name, annotation in hints.items():
        if name.startswith("_"):
            continue
        members.append(
            _completion_symbol(
                name,
                kind="property",
                detail=_type_label(annotation),
                doc=f"{cls.__name__}.{name}",
            )
        )
    class_values = dict(vars(cls))
    for name in dir(cls):
        if name.startswith("_") or name in class_values:
            continue
        try:
            class_values[name] = getattr(cls, name)
        except Exception:
            continue
    for name, value in class_values.items():
        if name.startswith("_"):
            continue
        if isinstance(value, property):
            detail = ""
            try:
                detail = _type_label(get_type_hints(value.fget or (lambda: None)).get("return"))
            except (NameError, TypeError):
                pass
            members.append(
                _completion_symbol(
                    name,
                    kind="property",
                    detail=detail,
                    doc=_member_doc(value),
                )
            )
        elif callable(value):
            try:
                signature = str(inspect.signature(value))
            except (TypeError, ValueError):
                signature = "()"
            members.append(
                _completion_symbol(
                    name,
                    kind="function",
                    detail=f"{name}{signature}",
                    doc=_member_doc(value),
                )
            )
    return sorted({member["name"]: member for member in members}.values(), key=lambda item: item["name"])


def _members_for_annotation(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None or annotation is inspect.Signature.empty:
        return []
    if inspect.isclass(annotation):
        return _class_members(annotation)

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    class_args = [arg for arg in args if inspect.isclass(arg)]
    if len(class_args) == 1:
        return _class_members(class_args[0])
    return []


def _query_return_type(queryable: Any) -> type | None:
    result = getattr(queryable, "result_type", None)
    return result if inspect.isclass(result) else None


def _member_type_map(cls: type | None) -> dict[str, type]:
    if cls is None:
        return {}
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError):
        hints = getattr(cls, "__annotations__", {})
    out = {
        name: annotation
        for name, annotation in hints.items()
        if isinstance(name, str)
        and not name.startswith("_")
        and inspect.isclass(annotation)
    }
    for name, value in vars(cls).items():
        if name.startswith("_") or not isinstance(value, property):
            continue
        try:
            annotation = get_type_hints(value.fget or (lambda: None)).get("return")
        except (NameError, TypeError):
            annotation = None
        if inspect.isclass(annotation):
            out[name] = annotation
    return out


def _spec_completion_context(spec: DesignSpec) -> dict[str, Any]:
    from bluesky_sandbox.interface.task import (
        AgentStepContext,
        BaseAgentInfo,
        TaskInfoProvider,
    )
    from bluesky_sandbox.sim.queryables import RegionResult

    cfg = build_design_config(spec)
    support = build_scenario(spec).support()
    hook_catalog = _catalog.hooks()
    hook_contexts_by_name = _hook_contexts_by_name()

    queryables = []
    query_calls = []
    for name, queryable in support.queryables.items():
        queryable_type = type(queryable)
        queryable_color = getattr(queryable, "color", None)
        result_type = _query_return_type(queryable)
        result_detail = _type_label(result_type) if result_type else "Any"
        queryables.append(
            _completion_symbol(
                f'"{name}"',
                kind="value",
                detail=queryable_type.__name__,
                doc=_member_doc(queryable_type),
                insert=f'"{name}"',
                color=queryable_color,
            )
        )
        query_calls.append(
            _completion_symbol(
                f'context.query("{name}")',
                kind="function",
                detail=f"{name}: {result_detail}",
                doc=_member_doc(result_type) or _member_doc(queryable_type),
                insert=f'context.query("{name}")',
                color=queryable_color,
            )
        )
    obs_fields = [
        _completion_symbol(
            field.meta.name,
            kind="field",
            detail=field.__class__.__name__,
        )
        for field in cfg.obs_fields
    ]
    intruder_obs_fields = [
        _completion_symbol(
            field.meta.name,
            kind="field",
            detail=field.__class__.__name__,
        )
        for field in cfg.intruder_obs_fields or []
    ]
    action_fields = [
        _completion_symbol(
            field.meta.name,
            kind="property",
            detail=field.__class__.__name__,
        )
        for field in cfg.action_fields
    ]
    query_result_members: dict[str, list[dict[str, str]]] = {}
    query_result_nested_members: dict[str, dict[str, list[dict[str, str]]]] = {}
    queryable_members: dict[str, list[dict[str, str]]] = {}
    for name, queryable in support.queryables.items():
        result_type = _query_return_type(queryable)
        query_result_members[name] = _class_members(result_type) if result_type else []
        query_result_nested_members[name] = {
            member_name: _class_members(member_type)
            for member_name, member_type in _member_type_map(result_type).items()
        }
        queryable_members[name] = []

    context_members = {
        member["name"]: member
        for member in _dataclass_members(AgentStepContext)
    }
    context_members["airspace"] = _completion_symbol(
        "airspace",
        kind="property",
        detail=_type_label(RegionResult),
        doc="Airspace query result for the current aircraft.",
    )

    airspace_result_members = _class_members(RegionResult)
    airspace_result_nested_members = {
        member_name: _class_members(member_type)
        for member_name, member_type in _member_type_map(RegionResult).items()
    }

    base_members = {
        "context": sorted(context_members.values(), key=lambda item: item["name"]),
        "info": _typed_dict_members(BaseAgentInfo),
        "self": [
            _completion_symbol("episode_queryables", kind="property", detail="Mapping[str, Queryable]"),
            _completion_symbol("config", kind="property", detail=cfg.__class__.__name__),
            _completion_symbol("scenario", kind="property", detail=support.__class__.__name__),
        ],
    }
    hook_contexts: dict[str, dict[str, Any]] = {}
    for hook in hook_catalog:
        hook_completion = hook_contexts_by_name.get(hook["name"], {"params": [], "members": {}})
        params = hook_completion["params"]
        members = dict(hook_completion["members"])
        members["self"] = base_members["self"]
        hook_contexts[hook["name"]] = {
            "params": params,
            "members": members,
        }

    try:
        task_info_hints = get_type_hints(TaskInfoProvider.__call__)
    except (NameError, TypeError):
        task_info_hints = {}
    task_info_params = [
        _completion_symbol(name, kind="variable", detail=_type_label(task_info_hints.get(name)))
        for name in ("obs", "action", "info", "context", "rng")
    ]
    task_info_members = {
        name: members
        for name in ("obs", "action", "info", "context", "rng")
        if (members := _members_for_annotation(task_info_hints.get(name)))
    }

    return {
        "ok": True,
        "hook_setup": _setup_completion_context(spec.env.hook_setup),
        "task_info_setup": _setup_completion_context(spec.env.task_info_setup),
        "queryables": queryables,
        "query_calls": query_calls,
        "airspace_result_members": airspace_result_members,
        "airspace_result_nested_members": airspace_result_nested_members,
        "query_result_members": query_result_members,
        "query_result_nested_members": query_result_nested_members,
        "queryable_members": queryable_members,
        "obs_fields": obs_fields,
        "intruder_obs_fields": intruder_obs_fields,
        "action_fields": action_fields,
        "hooks": hook_contexts,
        "task_info": {
            "params": task_info_params,
            "members": task_info_members,
        },
    }


def spec_max_aircraft(spec: DesignSpec) -> int:
    from .builder import build_scenario

    return int(build_scenario(spec).support().max_aircraft)


def _airspace_validation_errors(spec: DesignSpec) -> list[str]:
    from .builder import build_scenario

    return airspace_warnings(build_scenario(spec).support())


@lru_cache(maxsize=128)
def _python_module_members(module_name: str) -> dict[str, Any]:
    """Return public members for a Python module used in designer setup code."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise HTTPException(status_code=404, detail=f"cannot import {module_name!r}: {e}") from e

    members: list[dict[str, str]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        if inspect.ismodule(obj):
            kind = "module"
        elif inspect.isclass(obj):
            kind = "class"
        elif callable(obj):
            kind = "function"
        else:
            kind = "value"
        detail = type(obj).__name__
        try:
            if callable(obj):
                sig = str(inspect.signature(obj))
                detail = f"{name}{sig}"
        except (TypeError, ValueError):
            pass
        doc = inspect.getdoc(obj) or ""
        members.append(
            {
                "name": name,
                "kind": kind,
                "detail": detail,
                "doc": doc.splitlines()[0] if doc else "",
            }
        )
    members.sort(key=lambda item: item["name"].lower())
    return {"module": module_name, "members": members}


def create_app() -> FastAPI:
    app = FastAPI(title="bluesky-sandbox Environment Designer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = SpecStore()

    # ----------------------------------------------------------------- health
    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---------------------------------------------------------------- catalog
    @app.get("/api/catalog")
    def get_catalog() -> dict[str, Any]:
        return _catalog.catalog()

    @app.get("/api/python/module-members")
    def python_module_members(module: str) -> dict[str, Any]:
        if not module or not module.replace(".", "").replace("_", "").isalnum():
            raise HTTPException(status_code=422, detail="invalid module name")
        return _python_module_members(module)

    # -------------------------------------------------------------------- nav
    @app.post("/api/nav/features")
    def nav_features(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        bounds_spec = body.get("bounds")
        if bounds_spec is None:
            raise HTTPException(status_code=422, detail="body must include 'bounds'.")
        try:
            bounds = _spec.load(bounds_spec)
        except SpecError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        payload = _nav.features_in_bounds(
            bounds,
            margin_frac=body.get("margin_frac", 0.1),
            waypoint_limit=body.get("waypoint_limit", 2000),
            airport_limit=body.get("airport_limit", 500),
            airway_limit=body.get("airway_limit", 4000),
        )
        return {
            "window": payload["window"],
            "waypoints": [dataclasses.asdict(w) for w in payload["waypoints"]],
            "airports": [dataclasses.asdict(a) for a in payload["airports"]],
            "airways": [dataclasses.asdict(a) for a in payload["airways"]],
        }

    @app.get("/api/nav/waypoint/{ident}")
    def nav_waypoint(ident: str) -> dict[str, Any]:
        try:
            return dataclasses.asdict(_nav.resolve_waypoint(ident))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/api/nav/airport/{icao}")
    def nav_airport(icao: str) -> dict[str, Any]:
        try:
            return dataclasses.asdict(_nav.resolve_airport(icao))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/api/nav/search")
    def nav_search(q: str, limit: int = 30) -> dict[str, Any]:
        result = _nav.search(q, limit=limit)
        return {
            "waypoints": [dataclasses.asdict(w) for w in result["waypoints"]],
            "airports": [dataclasses.asdict(a) for a in result["airports"]],
        }

    # ------------------------------------------------------------------- spec
    @app.post("/api/spec/validate")
    def validate_spec(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body)
        try:
            summary = _spec_summary(spec)
            warnings = _airspace_validation_errors(spec)
        except (BuildError, ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}
        if warnings:
            return {"ok": False, "error": f"outside airspace: {', '.join(warnings)}"}
        return {"ok": True, "summary": summary}

    @app.post("/api/spec/completions")
    def completion_context(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        try:
            return _spec_completion_context(spec)
        except (BuildError, ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/spec/preview")
    def preview_spec(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        seed = int(body.get("seed", 0)) if isinstance(body, dict) else 0
        try:
            return scenario_preview(spec, seed=seed)
        except (BuildError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # --------------------------------------------------------------- codegen
    @app.post("/api/spec/generate")
    def generate_task(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        package_name = body.get("package_name") or spec.metadata.get("name") or "designed_task"
        try:
            files = _codegen.generate_task(spec, package_name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {"package": next(iter(files)).split("/", 1)[0], "files": files}

    @app.post("/api/spec/run")
    def run_design(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Launch the design in a live driver window (pygame / panda3d / qtgl).

        Runs on the machine hosting the API (the local designer), in a detached
        subprocess; only one runs at a time. Returns immediately with the pid.
        """
        spec = _parse_spec(body.get("spec", body))
        render_mode = body.get("render_mode", "pygame")
        views = body.get("views")
        show_all_routes = bool(body.get("show_all_routes", False))
        auto_track = bool(body.get("auto_track", False))
        seed = int(body.get("seed", 0))
        action_mode = "zero" if body.get("action_mode") == "zero" else "random"
        try:
            info = _runner.launch_design(
                spec,
                render_mode,
                views=views,
                show_all_routes=show_all_routes,
                auto_track=auto_track,
                seed=seed,
                action_mode=action_mode,
            )
        except (BuildError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {"ok": True, **info}

    @app.get("/api/spec/run/status")
    def run_status() -> dict[str, Any]:
        return _runner.run_status()

    @app.post("/api/spec/run/stop")
    def run_stop() -> dict[str, Any]:
        return _runner.stop_run()

    @app.post("/api/spec/sample")
    def sample_obs(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Build the env, reset, and return labeled observations + a sampled
        action for a few agents - so you can inspect the exact obs layout
        (field order, normalization) the policy receives.
        """
        spec = _parse_spec(body.get("spec", body))
        seed = int(body.get("seed", 0))
        max_agents = int(body.get("max_agents", 3))
        max_intruders = int(body.get("max_intruders", 25))
        try:
            return _runner.sample_design(
                spec,
                seed=seed,
                max_agents=max_agents,
                max_intruders=max_intruders,
            )
        except (BuildError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/api/spec/generate/zip")
    def generate_zip(body: dict[str, Any] = Body(...)) -> Response:
        spec = _parse_spec(body.get("spec", body))
        package_name = body.get("package_name") or spec.metadata.get("name") or "designed_task"
        try:
            files = _codegen.generate_task(spec, package_name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        pkg = next(iter(files)).split("/", 1)[0]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, content in files.items():
                zf.writestr(path, content)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{pkg}.zip"'},
        )

    # ----------------------------------------------------------------- store
    @app.get("/api/specs")
    def list_specs() -> list[dict[str, str]]:
        return store.list()

    @app.get("/api/specs/{name}")
    def get_spec(name: str) -> dict[str, Any]:
        try:
            return store.load(name).to_dict()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.put("/api/specs/{name}")
    def put_spec(name: str, body: dict[str, Any] = Body(...)) -> dict[str, str]:
        spec = _parse_spec(body)
        saved = store.save(name, spec)
        return {"name": saved}

    @app.delete("/api/specs/{name}")
    def delete_spec(name: str) -> dict[str, str]:
        store.delete(name)
        return {"name": name}

    # ----------------------------------------------------- static frontend
    if _WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="web")
    else:

        @app.get("/")
        def _no_frontend() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "API is running. Frontend not built; run the Vite "
                    "dev server in designer/web, or `npm run build` to serve it here.",
                    "docs": "/docs",
                }
            )

    return app


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Environment Designer API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "bluesky_sandbox.ui.designer.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
