"""Launch a *real* driver on the current design.

The designer's Map tab shows a static preview (geometry + one sampled episode).
This module instead spins up the actual environment in a chosen renderer
(pygame / panda3d / qtgl) so the design can be watched stepping live.

How it works: the spec is code-generated into a throwaway package (exactly the
same output as "Generate task"/"Download .zip", so hooks and custom fields are
included and there are no ``bluesky_sandbox.ui.designer`` imports), written to a
temp dir, and run in a **subprocess**. A subprocess is used deliberately —
BlueSky keeps process-global state (``bs.traf`` …) and the GUI drivers own a
blocking event loop, so running in-process would both block the API and corrupt
later validate/preview calls.

Only **one** driver runs at a time: launching a new one terminates the previous
(so repeated clicks don't pile up windows). The child writes a ``ready`` marker
after its first render, which :func:`run_status` reports so the UI can show a
loading state until the window is up.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import codegen
from .builder import BuildError, build_design_config
from .spec import DesignSpec

# Render modes that open a window. (``None`` — headless — is not offered here:
# there's nothing to watch.)
VALID_RENDER_MODES = ("pygame", "panda3d", "qtgl")

# Per-driver view layout options. ``style`` says how a selection is passed to
# ``Env(views=...)``: pygame takes a tuple of view *classes*, panda3d a list
# of view *instances*, and qtgl takes no views. ``import`` is the line the
# generated runner uses to bring the chosen view names into scope.
DRIVER_VIEWS: dict[str, dict[str, Any]] = {
    "pygame": {
        "options": ["VerticalView", "HorizontalView", "TSASView"],
        "default": ["HorizontalView"],
        "style": "classes",
        "import": "from bluesky_sandbox.ui.drivers import HorizontalView, TSASView, VerticalView",
    },
    "panda3d": {
        "options": ["WorldView", "TSASView"],
        "default": ["WorldView"],
        "style": "instances",
        "import": "from bluesky_sandbox.ui.drivers.panda3d.views import TSASView, WorldView",
    },
    "qtgl": {"options": [], "default": [], "style": "none", "import": ""},
}

def _repo_root() -> Path:
    """Parent of the ``bluesky_sandbox`` package.

    Put on the subprocess's ``PYTHONPATH`` so it can import the package when
    that package isn't pip-installed - the child runs with ``cwd`` set to a
    temp workdir, so it inherits nothing useful from the parent's ``sys.path``.

    Found by walking up to the package directory rather than counting parents:
    a fixed ``parents[N]`` silently points somewhere else the moment this module
    moves a level (it did, when ``designer`` moved under ``ui``), and the only
    symptom is ``ModuleNotFoundError`` inside a subprocess.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "bluesky_sandbox":
            return parent.parent
    return here.parents[2]


_REPO_ROOT = _repo_root()

# The single live run (at most one at a time).
_active: dict[str, Any] | None = None

_RUNNER_TEMPLATE = '''\
"""Auto-generated live runner for a designed environment."""
from __future__ import annotations

from pathlib import Path

{views_import}
from bluesky_sandbox import zero_action
from {pkg} import Env

RENDER_MODE = {render_mode!r}
SEED = {seed!r}
MAX_STEPS = {max_steps!r}
VIEWS = {views_expr}
SHOW_ALL_ROUTES = {show_all_routes!r}
AUTO_TRACK = {auto_track!r}
# "random" = sample the action space each step; "zero" = the null action (all
# zeros), which in the waypoint-relative action frame is "fly the nominal route
# directly" - useful for watching what the *un-controlled* dynamics do.
ACTION_MODE = {action_mode!r}
READY = Path(__file__).with_name("ready")


def main() -> None:
    env = Env(render_mode=RENDER_MODE, realtime=True, views=VIEWS)
    driver = env.unwrapped._driver
    if SHOW_ALL_ROUTES and hasattr(driver, "show_all_routes"):
        driver.show_all_routes = True
    if AUTO_TRACK and hasattr(driver, "auto_track"):
        driver.auto_track = True
    env.reset(seed=SEED)
    env.render()
    READY.write_text("1")  # signal the designer that the window is up
    try:
        for _ in range(MAX_STEPS):
            if env.unwrapped.episode_done:
                env.reset()
                env.render()
                continue
            if ACTION_MODE == "zero":
                actions = {{acid: zero_action(env.action_space(acid)) for acid in env.agents}}
            else:
                actions = {{acid: env.action_space(acid).sample() for acid in env.agents}}
            env.step(actions)
            env.render()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def _views_code(render_mode: str, views: list[str] | None) -> tuple[str, str]:
    """Return ``(import_line, views_expr)`` for the runner's Env constructor."""
    meta = DRIVER_VIEWS[render_mode]
    selected = [v for v in (views or meta["default"]) if v in meta["options"]]
    if meta["style"] == "none" or not selected:
        return "", "None"
    if meta["style"] == "instances":
        return meta["import"], "[" + ", ".join(f"{v}()" for v in selected) + "]"
    # classes: a (possibly singleton) tuple of view classes
    return meta["import"], "(" + ", ".join(selected) + ",)"


def _terminate_active() -> None:
    """Stop the current live run, if any."""
    global _active
    if _active is None:
        return
    proc = _active.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    _active = None


def launch_design(
    spec: DesignSpec,
    render_mode: str,
    *,
    views: list[str] | None = None,
    show_all_routes: bool = False,
    auto_track: bool = False,
    seed: int = 0,
    max_steps: int = 1_000_000,
    action_mode: str = "random",
) -> dict[str, Any]:
    """Generate, write, and launch the design in a live driver subprocess.

    Terminates any previous run first (one window at a time). Returns a dict
    with the subprocess ``pid``, temp ``workdir``, and ``log`` path. Raises
    :class:`BuildError` (bad design) or ``ValueError`` (bad render mode) before
    spawning anything.
    """
    if render_mode not in VALID_RENDER_MODES:
        raise ValueError(
            f"render_mode must be one of {VALID_RENDER_MODES}, got {render_mode!r}"
        )

    # Surface a broken design as a clean error *before* spawning a subprocess.
    build_design_config(spec)

    pkg = "designed_run"
    files = codegen.generate_task(spec, pkg)

    workdir = Path(tempfile.mkdtemp(prefix="bsd_run_"))
    for rel, source in files.items():
        path = workdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    views_import, views_expr = _views_code(render_mode, views)
    runner_path = workdir / "_run_design.py"
    runner_path.write_text(
        _RUNNER_TEMPLATE.format(
            pkg=pkg,
            render_mode=render_mode,
            seed=seed,
            max_steps=max_steps,
            views_import=views_import,
            views_expr=views_expr,
            show_all_routes=show_all_routes,
            auto_track=auto_track,
            action_mode=action_mode,
        )
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(workdir), str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )

    # Only one driver at a time — repeated launches replace the previous window.
    _terminate_active()

    ready_path = workdir / "ready"
    log_path = workdir / "run.log"
    log_file = open(log_path, "w")  # noqa: SIM115 - handed to the child process
    proc = subprocess.Popen(
        [sys.executable, str(runner_path)],
        cwd=str(workdir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    global _active
    _active = {
        "proc": proc,
        "pid": proc.pid,
        "render_mode": render_mode,
        "workdir": str(workdir),
        "log": str(log_path),
        "ready": str(ready_path),
    }
    return {
        "pid": proc.pid,
        "package": pkg,
        "render_mode": render_mode,
        "workdir": str(workdir),
        "log": str(log_path),
    }


_SAMPLE_MARKER = "__SAMPLE_JSON__"

# One-shot script: build the env, reset, and dump labeled observations + a
# sampled action to stdout as JSON. Run in a subprocess (BlueSky is a
# process-singleton) like the live runner, but it exits immediately.
_SAMPLE_TEMPLATE = '''\
"""Auto-generated one-shot observation/action sampler for a designed env."""
from __future__ import annotations

import json
import math

import bluesky as bs
import numpy as np

from bluesky_sandbox import flatten_action, observation_layout, zero_action
from bluesky_sandbox.core.layout import observation_parts, slots
from bluesky_sandbox.interface.fields.observations import LaggedObs, LaggedPair
from {pkg} import Env

SEED = {seed!r}
MAX_AGENTS = {max_agents!r}
MAX_INTRUDERS = {max_intruders!r}
AT_S = {at_s!r}
ACID = {acid!r}
MARKER = {marker!r}
#: Steps to wait for the first aircraft, for designs that spawn them over time.
MAX_WAIT_STEPS = 2000


def _plain(value):
    """A JSON value: arrays as lists, and None for what is not a finite number."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return str(value)


def _bounds(field, idx):
    """The range ``field`` scales against for aircraft ``idx``."""
    try:
        low, high = field.bounds(idx)
        return _plain(low), _plain(high)
    except Exception:
        return None, None


def main() -> None:
    env = Env(render_mode=None)
    try:
        obs, _ = env.reset(seed=SEED)
        # Step to AT_S, holding every agent still, and on until an aircraft is
        # up - traffic that spawns over time has none at reset. With ACID, on
        # until that aircraft is up.
        steps = 0
        while steps < MAX_WAIT_STEPS:
            if obs and bs.sim.simt >= AT_S and (ACID is None or ACID in obs):
                break
            obs, *_ = env.step({{a: zero_action(env.action_space(a)) for a in env.agents}})
            steps += 1
        focus = ACID
        base = env.unwrapped
        cfg = base.config
        parts = observation_parts(cfg)
        layout = observation_layout(cfg)
        action_slots = slots(cfg.action_fields)
        agents = []
        ranges = {{part: {{}} for part in parts}}
        ranges["action"] = {{}}
        # The aircraft asked for first, then the newest.
        order = list(reversed(list(obs)))
        if focus in order:
            order.remove(focus)
            order.insert(0, focus)
        for n, acid in enumerate(order):
            idx = bs.traf.id2idx(acid)
            actype = str(bs.traf.type[idx])
            # Every agent's ranges: the spread a per-aircraft range takes.
            for part, fields in parts.items():
                for field, slot in zip(fields, layout[part]):
                    low, high = _bounds(field, idx)
                    ranges[part].setdefault(slot.name, []).append([acid, actype, low, high])
            for field, slot in zip(cfg.action_fields, action_slots):
                low, high = _bounds(field, idx)
                ranges["action"].setdefault(slot.name, []).append([acid, actype, low, high])
            if n >= MAX_AGENTS:
                continue
            o = obs[acid]
            raw = base.raw_observation(acid)
            agent_parts = {{}}
            for part, fields in parts.items():
                if not isinstance(o, dict) or part not in o:
                    continue
                values = np.asarray(o[part], dtype=float)
                per_intruder = values.ndim == 2
                if per_intruder:
                    values = values[:MAX_INTRUDERS]
                rows = []
                for field, slot in zip(fields, layout[part]):
                    cols = values[..., slot.columns]
                    value = raw[part].get(slot.name)
                    if per_intruder and value is not None:
                        value = np.asarray(value)[:MAX_INTRUDERS]
                    low, high = _bounds(field, idx)
                    unit = str(getattr(field.meta, "unit", "") or "")
                    rows.append({{
                        "name": slot.name,
                        "unit": "" if unit == "unitless" else unit,
                        "lag": int(field.steps) if isinstance(field, (LaggedObs, LaggedPair)) else None,
                        "raw": _plain(value),
                        "obs": _plain(cols),
                        "low": low,
                        "high": high,
                    }})
                entry = {{"fields": rows}}
                if per_intruder:
                    entry["acids"] = _plain(np.asarray(raw[part].get("acid", []))[:MAX_INTRUDERS])
                agent_parts[part] = entry
            action = flatten_action(cfg, env.action_space(acid).sample())
            agents.append({{
                "acid": acid,
                "type": actype,
                "parts": agent_parts,
                "action": [
                    {{"name": slot.name, "value": _plain(action[slot.columns])}}
                    for slot in action_slots
                ],
            }})
        out = {{"seed": SEED, "sim_time_s": float(bs.sim.simt), "agents": agents, "ranges": ranges}}
        print(MARKER + json.dumps(out))
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def _run_script(
    spec: DesignSpec, pkg: str, source: str, timeout_s: float
) -> dict[str, Any]:
    """Generate ``spec`` as ``pkg``, run ``source`` against it in a subprocess,
    and return the JSON it prints after the marker."""
    files = codegen.generate_task(spec, pkg)
    workdir = Path(tempfile.mkdtemp(prefix="bsd_sample_"))
    try:
        for rel, text in files.items():
            path = workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        script = workdir / "_sample_design.py"
        script.write_text(source)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(workdir), str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
        )
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        for line in proc.stdout.splitlines():
            if line.startswith(_SAMPLE_MARKER):
                return json.loads(line[len(_SAMPLE_MARKER):])
        raise BuildError(
            "sampling produced no output; "
            + (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "see logs")
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def sample_design(
    spec: DesignSpec,
    *,
    seed: int = 0,
    max_agents: int = 3,
    max_intruders: int = 25,
    at_s: float = 0.0,
    acid: str | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Build the env in a subprocess, reset, step to ``at_s`` (and on until an
    aircraft is up - with ``acid``, until that one is), and return what that
    aircraft and the newest others, ``max_agents`` in all, observe - raw and normalized, with each field's range for that aircraft -
    and a sampled action; plus every aircraft's ranges and type.
    """
    build_design_config(spec)  # surface a broken design before spawning anything
    pkg = "designed_sample"
    source = _SAMPLE_TEMPLATE.format(
        pkg=pkg,
        seed=seed,
        max_agents=max_agents,
        max_intruders=max_intruders,
        at_s=float(at_s),
        acid=acid,
        marker=_SAMPLE_MARKER,
    )
    return _run_script(spec, pkg, source, timeout_s)


# One-shot script: run the seeded episode, every agent held still, until its
# scheduled spawns are all up (or ``until_s``), printing each aircraft as it is
# created - with the label the pygame view gives it and its route's resolved
# targets - so a caller can show them as they come.
_SPAWNS_TEMPLATE = '''\
"""Auto-generated spawn log of one seeded episode of a designed env."""
from __future__ import annotations

import dataclasses
import json
import math

import bluesky as bs

from bluesky_sandbox import zero_action
from bluesky_sandbox.ui.drivers.common.readouts import aircraft_label_lines
from {pkg} import Env

SEED = {seed!r}
UNTIL_S = {until_s!r}
MAX_STEPS = {max_steps!r}
MARKER = {marker!r}


def _plain(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _row(record, queryables):
    row = {{k: _plain(v) for k, v in record._asdict().items() if k != "targets"}}
    row["route"] = list(record.route) if record.route else None
    row["label"] = aircraft_label_lines(
        record.callsign, record.actype, alt_ft=record.alt_ft, gs_kts=record.gs_kts,
        cas_kts=record.cas_kts, mach=record.mach,
    )
    # A per-aircraft sampled target carries no name; its route step does.
    names = list(record.route or ())
    row["targets"] = []
    for i, t in enumerate(record.targets):
        name = t.waypoint or (names[i] if i < len(names) else None)
        row["targets"].append({{
            **{{k: _plain(v) for k, v in dataclasses.asdict(t).items()}},
            "waypoint": name,
            "color": getattr(queryables.get(name), "color", None),
        }})
    return row


def _emit(kind, payload):
    print(MARKER + json.dumps({{kind: payload}}), flush=True)


def main() -> None:
    env = Env(render_mode=None)
    try:
        env.reset(seed=SEED)
        base = env.unwrapped
        queryables = dict(getattr(base.episode_spec, "queryables", {{}}) or {{}})
        sent = 0
        steps = 0
        while True:
            log = base.spawn_log
            for record in log[sent:]:
                _emit("aircraft", _row(record, queryables))
            sent = len(log)
            progress = base.episode_spawn_progress
            done = progress.scheduled and progress.spawned >= progress.scheduled
            if done or base.episode_done or steps >= MAX_STEPS or bs.sim.simt >= UNTIL_S:
                break
            env.step({{a: zero_action(env.action_space(a)) for a in env.agents}})
            steps += 1
        progress = base.episode_spawn_progress
        _emit("done", {{
            "seed": SEED,
            "sim_time_s": float(bs.sim.simt),
            "complete": progress.spawned >= progress.scheduled,
            "scheduled": progress.scheduled,
        }})
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def iter_episode_spawns(
    spec: DesignSpec,
    *,
    seed: int = 0,
    until_s: float = 3600.0,
    max_steps: int = 2000,
    timeout_s: float = 180.0,
) -> Iterator[dict[str, Any]]:
    """Run the seeded episode in a subprocess, every agent held still, until
    its scheduled aircraft are all up (or ``until_s``), yielding each aircraft
    as it is created - ``{"aircraft": {...}}``: callsign, type, time, position,
    heading, speeds, its pygame label and its route's resolved targets - and
    last ``{"done": {...}}``. Closing the iterator stops the run."""
    build_design_config(spec)
    pkg = "designed_spawns"
    files = codegen.generate_task(spec, pkg)
    workdir = Path(tempfile.mkdtemp(prefix="bsd_spawns_"))
    proc = None
    try:
        for rel, text in files.items():
            path = workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        script = workdir / "_episode_spawns.py"
        script.write_text(
            _SPAWNS_TEMPLATE.format(
                pkg=pkg,
                seed=seed,
                until_s=float(until_s),
                max_steps=int(max_steps),
                marker=_SAMPLE_MARKER,
            )
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(workdir), str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
        )
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + timeout_s
        finished = False
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.monotonic() > deadline:
                break
            if line.startswith(_SAMPLE_MARKER):
                item = json.loads(line[len(_SAMPLE_MARKER):])
                yield item
                if "done" in item:
                    finished = True
                    break
        if not finished:
            proc.wait(timeout=5)
            err = proc.stderr.read().strip() if proc.stderr else ""
            raise BuildError(
                "the episode run stopped early; "
                + (err.splitlines()[-1] if err else "see logs")
            )
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(workdir, ignore_errors=True)


def episode_spawns(spec: DesignSpec, **kwargs: Any) -> dict[str, Any]:
    """:func:`iter_episode_spawns`, collected: the ``done`` summary with the
    aircraft under ``aircraft``."""
    aircraft: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    for item in iter_episode_spawns(spec, **kwargs):
        if "aircraft" in item:
            aircraft.append(item["aircraft"])
        else:
            out = item["done"]
    return {**out, "aircraft": aircraft}


def run_status() -> dict[str, Any]:
    """Report the live run's state so the UI can track it start to finish.

    Tracking continues after the window comes up: when the child exits we report
    its ``returncode`` and the tail of its captured stdout/stderr ``log`` - so a
    crash *after* the window opened (not just during startup) is surfaced, and
    ``error`` carries the log for any abnormal exit.
    """
    if _active is None:
        return {"active": False, "alive": False, "ready": False}
    proc = _active["proc"]
    alive = proc.poll() is None
    ready = Path(_active["ready"]).exists()
    out: dict[str, Any] = {
        "active": True,
        "alive": alive,
        "ready": ready,
        "pid": _active["pid"],
        "render_mode": _active["render_mode"],
    }
    # The child's captured stdout+stderr, tail-limited, available throughout the
    # run so the UI can show output and (on failure) the traceback.
    try:
        out["log"] = Path(_active["log"]).read_text()[-4000:]
    except OSError:
        out["log"] = ""
    if not alive:
        out["returncode"] = proc.returncode
        # Non-zero exit = crash/build error. Negative = killed by a signal
        # (e.g. terminate); 0 = clean finish or window closed. Surface the log
        # as ``error`` only for an abnormal exit so the UI can flag it.
        if proc.returncode not in (0, None):
            out["error"] = out["log"]
    return out


def stop_run() -> dict[str, Any]:
    """Terminate the live run, if any."""
    _terminate_active()
    return {"ok": True}


__all__ = [
    "DRIVER_VIEWS",
    "VALID_RENDER_MODES",
    "BuildError",
    "launch_design",
    "run_status",
    "stop_run",
]
