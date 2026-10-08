"""FastAPI server for the Environment Designer.

Exposes the backend foundation over HTTP so the web frontend (map + code tabs)
can drive it:

* ``GET  /api/health``                 - liveness.
* ``POST /api/refresh``                - forget what the server has cached.
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

import argparse
import dataclasses
import importlib
import inspect
import io
import os
import threading
import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bluesky_sandbox import __version__

from . import catalog as _catalog
from . import codegen as _codegen
from . import nav as _nav
from . import runner as _runner
from . import spec as _spec
from .builder import BuildError, build_design_config, build_scenario
from .code_intel import code_intel, describe_type, forget_type_checking_names
from .diagnostics import diagnostics
from .mdp import mdp_summary
from .preview import airspace_warnings, alert_hued_colors, scenario_preview
from .evaluation import evaluation_catalog
from .recording import record_catalog
from .spec import DesignSpec, SpecError
from .store import SpecStore
from .trail import forget_call_trails

_WEB_DIST = Path(__file__).parent / "web" / "dist"
#: Set by :func:`main`: the server starts the sampling worker on startup.
WARM_WORKER_ENV = "BSD_WARM_WORKER"


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
        "state_fields": [f.meta.name for f in cfg.state_fields or ()],
        "intruder_state_fields": [
            f.meta.name for f in cfg.intruder_state_fields or ()
        ],
        "action_fields": [f.meta.name for f in cfg.action_fields],
        # Every type the spawn regions name.
        "aircraft_types": sorted({
            str(t).upper()
            for region in support.spawn.regions
            for t in _type_names(region.aircraft_type)
        }),
        "max_aircraft": spec_max_aircraft(spec),
        "has_airspace": support.airspace_bounds is not None,
        "queryables": list(support.queryables),
    }


def _type_names(types: Any) -> list[str]:
    if types is None:
        return []
    if isinstance(types, str):
        return [types]
    return list(getattr(types, "weights", {}) or [])


def spec_max_aircraft(spec: DesignSpec) -> int:
    return int(build_scenario(spec).support().max_aircraft)


def _airspace_validation_errors(spec: DesignSpec) -> list[str]:
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
    # The code intel runs to a few hundred KiB of repetitive JSON.
    app.add_middleware(GZipMiddleware, minimum_size=4096)
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
        return {"status": "ok", "version": __version__}

    @app.post("/api/refresh")
    def refresh() -> dict[str, bool]:
        # What was read from Python code: a module's members, the names it
        # imports for type checking, a field's call trail. Library code itself
        # is not re-imported; that takes a restart, or --reload.
        importlib.invalidate_caches()
        _python_module_members.cache_clear()
        forget_type_checking_names()
        forget_call_trails()
        return {"ok": True}

    # ---------------------------------------------------------------- catalog
    @app.get("/api/catalog")
    def get_catalog() -> dict[str, Any]:
        return _catalog.catalog()

    @app.get("/api/python/module-members")
    def python_module_members(module: str) -> dict[str, Any]:
        if not module or not module.replace(".", "").replace("_", "").isalnum():
            raise HTTPException(status_code=422, detail="invalid module name")
        return _python_module_members(module)

    @app.get("/api/python/type")
    def python_type(key: str) -> dict[str, Any]:
        """A type the code editor reached, described one level down."""
        if not key or not key.replace(".", "").replace("_", "").isalnum():
            raise HTTPException(status_code=422, detail="invalid type key")
        try:
            return {"types": describe_type(key)}
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

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
    def nav_search(
        q: str, limit: int = 30, lat: float | None = None, lon: float | None = None
    ) -> dict[str, Any]:
        near = (lat, lon) if lat is not None and lon is not None else None
        result = _nav.search(q, limit=limit, near=near)
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
        # Outside the airspace is worth knowing - an exit fix, a feeder - but
        # the design builds: a warning, not an error.
        out: dict[str, Any] = {"ok": True, "summary": summary}
        if warnings:
            out["warnings"] = [f"outside the airspace: {', '.join(warnings)}"]
        shifted = alert_hued_colors(spec)
        if shifted:
            out.setdefault("warnings", []).append(
                "near an alert's color: "
                + ", ".join(f"{where} ({color} as {drawn})" for where, color, drawn in shifted)
            )
        return out

    @app.get("/api/design/schema")
    def design_schema_json() -> dict[str, Any]:
        """What a design file may hold, as a JSON Schema (see schema.py)."""
        from .schema import design_schema

        return design_schema()

    @app.post("/api/spec/code-intel")
    def spec_code_intel(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        try:
            return code_intel(spec)
        except (BuildError, ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/spec/diagnostics")
    def spec_diagnostics(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        return {"ok": True, "problems": diagnostics(spec)}

    @app.post("/api/spec/mdp")
    def spec_mdp(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        spec = _parse_spec(body.get("spec", body))
        try:
            return mdp_summary(spec)
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
        notes = _codegen.sb3_notes(spec) if _codegen.template_of(spec) == "sb3" else []
        return {
            "package": next(iter(files)).split("/", 1)[0],
            "files": files,
            "notes": notes,
            # For choosing how many processes to train in: this machine's cores.
            "cpus": os.cpu_count() or 1,
            # For choosing how clips are recorded: each driver's views, and defaults.
            "recording": record_catalog(),
            # For choosing the window evaluate.py draws the trained policy in.
            "evaluation": evaluation_catalog(),
        }

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
        """What the newest aircraft observe at ``at_s`` into the seeded
        episode, raw and normalized, with each field's range - see
        :func:`runner.sample_design`.
        """
        spec = _parse_spec(body.get("spec", body))
        seed = int(body.get("seed", 0))
        max_agents = int(body.get("max_agents", 3))
        max_intruders = int(body.get("max_intruders", 25))
        at_s = float(body.get("at_s", 0.0))
        acid = body.get("acid") or None
        try:
            return _runner.sample_design(
                spec,
                seed=seed,
                max_agents=max_agents,
                max_intruders=max_intruders,
                at_s=at_s,
                acid=acid,
            )
        except (BuildError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/api/spec/episode")
    def episode(body: dict[str, Any] = Body(...)) -> StreamingResponse:
        """Each aircraft of the seeded episode as it is created, one JSON line
        each as the run goes, then a ``done`` line - see
        :func:`runner.iter_episode_spawns`. A broken design is refused before
        the stream starts; a run that fails midway ends with an ``error`` line."""
        spec = _parse_spec(body.get("spec", body))
        try:
            build_design_config(spec)
        except (BuildError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        run = _runner.iter_episode_spawns(
            spec,
            seed=int(body.get("seed", 0)),
            until_s=float(body.get("until_s", 3600.0)),
        )

        def lines():
            try:
                for item in run:
                    yield json.dumps(item) + "\n"
            except (BuildError, ValueError, TypeError) as e:
                yield json.dumps({"error": str(e)}) + "\n"
            finally:
                run.close()

        # Declared unencoded, so the gzip middleware passes each line through
        # as it comes instead of holding the stream to compress it.
        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Content-Encoding": "identity", "Cache-Control": "no-store"},
        )

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

    if os.environ.get(WARM_WORKER_ENV) == "1":
        # Off the request path: the worker's imports take a few seconds.
        threading.Thread(target=_runner.warm, daemon=True).start()
    return app


app = create_app()


def main() -> None:
    # optional extra: [designer]
    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run the Environment Designer API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    # The server's process - uvicorn's own, under --reload - starts the
    # sampling worker as it comes up, so the first sample is as quick as the rest.
    os.environ[WARM_WORKER_ENV] = "1"
    uvicorn.run(
        "bluesky_sandbox.ui.designer.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
