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

import numpy as np

{views_import}
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
                actions = {{
                    acid: np.zeros(
                        env.action_space(acid).shape,
                        dtype=env.action_space(acid).dtype,
                    )
                    for acid in env.agents
                }}
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
import numpy as np

from {pkg} import Env

SEED = {seed!r}
MAX_AGENTS = {max_agents!r}
MAX_INTRUDERS = {max_intruders!r}
MARKER = {marker!r}


def _output_size(field) -> int:
    norm = getattr(field, "normalizer", None)
    if norm is not None and hasattr(norm, "output_size"):
        try:
            return int(norm.output_size(field))
        except Exception:
            pass
    size = getattr(field, "output_size", None)
    return int(size()) if callable(size) else 1


def _labels(fields) -> list[str]:
    out: list[str] = []
    for f in fields or ():
        name = getattr(getattr(f, "meta", None), "name", type(f).__name__)
        n = _output_size(f)
        if n == 1:
            out.append(name)
        else:
            out.extend(f"{{name}}[{{i}}]" for i in range(n))
    return out


def main() -> None:
    env = Env(render_mode=None)
    try:
        obs, _ = env.reset(seed=SEED)
        cfg = env.unwrapped.config
        own_labels = _labels(cfg.obs_fields)
        intr_labels = _labels(cfg.intruder_obs_fields)
        act_labels = _labels(cfg.action_fields)
        agents = []
        for acid in list(obs)[:MAX_AGENTS]:
            o = obs[acid]
            if isinstance(o, dict):
                ownship = np.asarray(o["ownship"], dtype=float).reshape(-1).tolist()
                intr = np.asarray(o.get("intruders", []), dtype=float)
                intruders = intr.tolist()[:MAX_INTRUDERS] if intr.ndim == 2 else []
            else:
                ownship = np.asarray(o, dtype=float).reshape(-1).tolist()
                intruders = []
            action = np.asarray(
                env.action_space(acid).sample(), dtype=float
            ).reshape(-1).tolist()
            agents.append({{
                "acid": acid,
                "ownship": [
                    {{"name": n, "value": v}}
                    for n, v in zip(own_labels, ownship)
                ],
                "intruder_fields": intr_labels,
                "intruders": intruders,
                "n_intruders": len(intruders),
                "action": [
                    {{"name": n, "value": v}}
                    for n, v in zip(act_labels, action)
                ],
            }})
        print(MARKER + json.dumps({{"seed": SEED, "agents": agents}}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def sample_design(
    spec: DesignSpec,
    *,
    seed: int = 0,
    max_agents: int = 3,
    max_intruders: int = 25,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Build the env in a subprocess, reset, and return labeled obs + a sampled
    action for a few agents - so the designer can inspect the exact observation
    layout/values (column order, normalization) the policy would receive.
    """
    build_design_config(spec)  # surface a broken design before spawning anything

    pkg = "designed_sample"
    files = codegen.generate_task(spec, pkg)
    workdir = Path(tempfile.mkdtemp(prefix="bsd_sample_"))
    try:
        for rel, source in files.items():
            path = workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        script = workdir / "_sample_design.py"
        script.write_text(
            _SAMPLE_TEMPLATE.format(
                pkg=pkg,
                seed=seed,
                max_agents=max_agents,
                max_intruders=max_intruders,
                marker=_SAMPLE_MARKER,
            )
        )
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
