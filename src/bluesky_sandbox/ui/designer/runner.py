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

import atexit
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import codegen
from . import worker as _worker
from .builder import BuildError, build_design_config, config_fields_of
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


# ---- running a design: in the warm worker, or a process of its own ---------- #


class _Worker:
    """The warm process designs are sampled in (see :mod:`.worker`): started
    on first use, reused while it flies the same performance model, and
    started again after an error, a timeout or :attr:`MAX_JOBS` jobs."""

    MAX_JOBS = 200
    #: How long a job the caller left is given to finish (s) before the
    #: worker is started afresh instead.
    DRAIN_S = 5.0

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._model: str | None = None
        self._jobs = 0
        self._log = Path(tempfile.gettempdir()) / f"bsd_worker_{os.getpid()}.log"

    def _start(self, model: str) -> None:
        self.stop()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p)
        log = open(self._log, "a")  # noqa: SIM115 - the process holds it
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "bluesky_sandbox.ui.designer.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        log.close()
        self._lines = queue.Queue()
        stdout = self._proc.stdout

        def pump() -> None:
            for line in stdout:
                self._lines.put(line.rstrip("\n"))
            self._lines.put(None)  # it exited

        threading.Thread(target=pump, daemon=True).start()
        self._model, self._jobs = model, 0
        if not self._wait_for(_worker.READY, 120.0):
            self.stop()
            raise BuildError("the sampling process did not start; see " + str(self._log))

    def _wait_for(self, marker: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while (left := deadline - time.monotonic()) > 0:
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                return False
            if line is None:
                return False
            if line == marker:
                return True
        return False

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def run(self, model: str, job: dict[str, Any], timeout_s: float) -> Iterator[str]:
        """The job's output lines, until it is done. Must hold :attr:`lock`."""
        if (
            self._proc is None
            or self._proc.poll() is not None
            or self._model != model
            or self._jobs >= self.MAX_JOBS
        ):
            self._start(model)
        assert self._proc is not None and self._proc.stdin is not None
        self._jobs += 1
        self._proc.stdin.write(json.dumps(job) + "\n")
        self._proc.stdin.flush()
        deadline = time.monotonic() + timeout_s
        finished = False
        try:
            while (left := deadline - time.monotonic()) > 0:
                try:
                    line = self._lines.get(timeout=left)
                except queue.Empty:
                    break
                if line is None:  # it died
                    break
                if line == _worker.DONE:
                    finished = True
                    return
                if line.startswith(_worker.ERROR):
                    error = json.loads(line[len(_worker.ERROR):])["error"]
                    self._wait_for(_worker.DONE, 10.0)
                    finished = True
                    raise BuildError(error)
                yield line
            raise BuildError("the sample did not finish in time; see " + str(self._log))
        finally:
            if not finished:
                # The caller went away mid-job (an episode stream closed): let
                # the job run out - most take well under a second - and keep
                # the worker. Timed out or died: start afresh next time.
                alive = self._proc is not None and self._proc.poll() is None
                if not (alive and self._wait_for(_worker.DONE, self.DRAIN_S)):
                    self.stop()


_WORKER = _Worker()
atexit.register(_WORKER.stop)


def warm(model: str = "openap") -> None:
    """Start the sampling worker ahead of the first sample, if it is free."""
    if _WORKER.lock.acquire(blocking=False):
        try:
            if _WORKER._proc is None or _WORKER._proc.poll() is not None:
                _WORKER._start(model)
        except BuildError:
            pass  # the first sample will try again, and report it
        finally:
            _WORKER.lock.release()
_JOB_IDS = itertools.count(1)


def _job_lines(
    spec: DesignSpec,
    prefix: str,
    template: str,
    params: dict[str, Any],
    timeout_s: float,
    *,
    fresh: bool = False,
) -> Iterator[str]:
    """Generate ``spec`` as a package of its own and run ``template`` against
    it - in the warm worker when it is free (unless ``fresh``: a job that must
    start from a clean simulator), else in a process of its own - yielding
    what the script prints."""
    package = f"{prefix}_{os.getpid()}_{next(_JOB_IDS)}"
    files = codegen.generate_task(spec, package)
    workdir = Path(tempfile.mkdtemp(prefix=f"bsd_{prefix}_"))
    try:
        for rel, text in files.items():
            path = workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        script = workdir / f"_{prefix}.py"
        script.write_text(template.format(pkg=package, marker=_SAMPLE_MARKER, **params))
        model = str(getattr(spec.env, "performance_model", None) or "openap").lower()
        if not fresh and _WORKER.lock.acquire(blocking=False):
            try:
                job = {"workdir": str(workdir), "package": package, "script": str(script)}
                yield from _WORKER.run(model, job, timeout_s)
            finally:
                _WORKER.lock.release()
            return
        # The worker is busy - an episode streaming, say: a process of its own.
        yield from _own_process(workdir, script, timeout_s)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _own_process(workdir: Path, script: Path, timeout_s: float) -> Iterator[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(workdir), str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(script)!r}, run_name='__main__')"],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout_s
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.monotonic() > deadline:
                raise BuildError("the sample did not finish in time")
            yield line.rstrip("\n")
        proc.wait(timeout=10)
        if proc.returncode:
            err = proc.stderr.read().strip() if proc.stderr else ""
            raise BuildError(err.splitlines()[-1] if err else "the sample failed; see logs")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _run_script(spec: DesignSpec, prefix: str, template: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """The JSON the script prints after the marker - the job read to its end,
    so the worker finishes it and stays warm for the next."""
    out = None
    for line in _job_lines(spec, prefix, template, params, timeout_s):
        if out is None and line.startswith(_SAMPLE_MARKER):
            out = json.loads(line[len(_SAMPLE_MARKER):])
    if out is None:
        raise BuildError("sampling produced no output; see logs")
    return out


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
    params = {"seed": seed, "max_agents": max_agents, "max_intruders": max_intruders, "at_s": float(at_s), "acid": acid}
    return _run_script(spec, "designed_sample", _SAMPLE_TEMPLATE, params, timeout_s)


# One-shot script: fly the seeded episode to AT_S, every agent held still,
# recording each stacked field's history as it goes, and probe one field there.
_PROBE_TEMPLATE = """\
\"\"\"Auto-generated: one field of a designed env, probed at one moment.\"\"\"
from __future__ import annotations

import json

import bluesky as bs
import numpy as np

from bluesky_sandbox import zero_action
from bluesky_sandbox.checks.probe import LagRecorder, Override, probe
from {pkg} import Env

SEED = {seed!r}
AT_S = {at_s!r}
ACID = {acid!r}
OTHER = {other!r}
LIST = {list_key!r}
INDEX = {index!r}
OVERRIDES = {overrides!r}
GIVE = {give!r}
MARKER = {marker!r}
#: Steps to wait for the first aircraft, for designs that spawn them over time.
MAX_WAIT_STEPS = 2000


def _nearest(own):
    lat, lon = np.radians(bs.traf.lat), np.radians(bs.traf.lon)
    d = np.hypot((lat - lat[own]) * 1.0, (lon - lon[own]) * np.cos(lat[own]))
    d[own] = np.inf
    return str(bs.traf.id[int(np.argmin(d))])


def main() -> None:
    env = Env(render_mode=None)
    try:
        obs, _ = env.reset(seed=SEED)
        base = env.unwrapped
        history = LagRecorder(base)
        history.record()
        steps = 0
        while steps < MAX_WAIT_STEPS:
            if obs and bs.sim.simt >= AT_S and (ACID is None or ACID in obs):
                break
            obs, *_ = env.step({{a: zero_action(env.action_space(a)) for a in env.agents}})
            history.record()
            steps += 1
        ids = [str(acid) for acid in bs.traf.id]
        acid = ACID if ACID in ids else (list(obs)[-1] if obs else None)
        field = getattr(base.config, LIST)[INDEX]
        out = {{"seed": SEED, "sim_time_s": float(bs.sim.simt), "aircraft": ids, "acid": acid}}
        if acid is None:
            out["error"] = "no aircraft is in the air"
        else:
            other = OTHER
            if other is None and hasattr(field, "get_pair_matrix") and len(ids) > 1:
                other = _nearest(ids.index(acid))
            try:
                result = probe(
                    base, field, acid, other=other, give=GIVE, history=history,
                    overrides=[Override(**o) for o in OVERRIDES],
                )
                out["result"] = result.as_dict()
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                out["error"] = str(e)
        print(MARKER + json.dumps(out))
    finally:
        env.close()


if __name__ == "__main__":
    main()
"""


def probe_design(
    spec: DesignSpec,
    *,
    list_key: str,
    entry: int,
    part: int = 0,
    seed: int = 0,
    at_s: float = 0.0,
    acid: str | None = None,
    other: str | None = None,
    overrides: list[dict[str, Any]] | None = None,
    give: float | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Fly the seeded episode to ``at_s`` and probe the spec's entry ``entry``
    of ``list_key`` - its field ``part``, for an entry that is several (a frame
    stack's live field and its lags) - for ``acid`` (the newest aircraft
    when not named): see :func:`bluesky_sandbox.checks.probe.probe`."""
    build_design_config(spec)  # surface a broken design before spawning anything
    start, count = config_fields_of(spec, list_key, entry)
    if not 0 <= part < count:
        raise ValueError(f"{list_key} entry {entry} has no part {part}")
    params = {
        "seed": int(seed),
        "at_s": float(at_s),
        "acid": acid,
        "other": other,
        "list_key": list_key,
        "index": start + part,
        "overrides": [dict(o) for o in overrides or ()],
        "give": None if give is None else float(give),
    }
    return _run_script(spec, "designed_probe", _PROBE_TEMPLATE, params, timeout_s)


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
    params = {"seed": seed, "until_s": float(until_s), "max_steps": int(max_steps)}
    finished = False
    for line in _job_lines(spec, "designed_spawns", _SPAWNS_TEMPLATE, params, timeout_s):
        if line.startswith(_SAMPLE_MARKER):
            item = json.loads(line[len(_SAMPLE_MARKER):])
            yield item
            if "done" in item:
                finished = True
    if not finished:
        raise BuildError("the episode run stopped early; see logs")


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


# One-shot script: every field of the design checked against itself, then
# each of its test cases, a line per result as it is known.
_TESTS_TEMPLATE = '''\
"""Auto-generated: a designed env's field checks and test cases."""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from bluesky_sandbox.checks import check_fields, run_cases
from {pkg} import Env

MARKER = {marker!r}
try:
    from {pkg}.cases import CASES, SITUATIONS
except ImportError:
    CASES, SITUATIONS = (), ()


def _plain(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    out = [v if math.isfinite(v) else None for v in array.tolist()]
    return out[0] if np.ndim(value) == 0 else out


def _emit(kind, payload):
    print(MARKER + json.dumps({{"kind": kind, **payload}}), flush=True)


def main() -> None:
    env = Env(render_mode=None)
    base = env.unwrapped
    try:
        report = check_fields(base)
        for finding in report.episode_findings:
            _emit("field", {{"field": "(episode)", "ok": False, "findings": [finding], "notes": []}})
        for r in report.results:
            _emit("field", {{"field": r.field, "ok": r.ok, "findings": list(r.findings[:5]), "notes": list(r.notes)}})
        for i, case in enumerate(CASES):
            r = run_cases(base, SITUATIONS, [case]).results[0]
            _emit("case", {{"index": i, "ok": r.ok, "got": _plain(r.got), "error": r.error, "saw": list(r.saw)}})
    finally:
        env.close()
    _run_test_files()


def _run_test_files() -> None:
    """The design's own test files, a line per test: pytest, as the package's
    tests folder runs them (its fixtures included)."""
    tests = pathlib.Path(__file__).parent / "{pkg}" / "tests"
    files = sorted(str(p) for p in tests.glob("test_*.py") if p.name != "test_design.py")
    if not files:
        return
    try:
        import pytest
    except ImportError:
        _emit("test", {{"name": "(test files)", "ok": False, "error": "pytest is not installed"}})
        return

    class _Report:
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or not report.passed:
                crash = getattr(report.longrepr, "reprcrash", None)
                failed = not (report.passed or report.skipped)
                _emit("test", {{
                    "name": report.nodeid.split("tests/", 1)[-1],
                    "ok": not failed,
                    "skipped": report.skipped,
                    "error": (crash.message if crash else str(report.longrepr)) if failed else None,
                    "line": crash.lineno if failed and crash else None,
                }})

    # Its own report is each test's line here: no terminal output of pytest's.
    pytest.main(["-p", "no:terminal", "-p", "no:cacheprovider", "--rootdir", str(tests), *files], plugins=[_Report()])


if __name__ == "__main__":
    main()
'''


def iter_design_tests(spec: DesignSpec, *, timeout_s: float = 300.0) -> Iterator[dict[str, Any]]:
    """The design's field checks, then its test cases, then its own test
    files - each result as it is known: ``{"kind": "field", "field", "ok",
    "findings"}``, ``{"kind": "case", "index", "ok", "got", "error"}`` or
    ``{"kind": "test", "name", "ok", "skipped", "error"}``. In a process of
    its own: a test sets the simulator up as it needs, which no later job
    should inherit."""
    build_design_config(spec)  # surface a broken design before starting anything
    for line in _job_lines(spec, "designed_tests", _TESTS_TEMPLATE, {}, timeout_s, fresh=True):
        # Anywhere in the line: what a test prints can share it.
        at = line.find(_SAMPLE_MARKER)
        if at >= 0:
            yield json.loads(line[at + len(_SAMPLE_MARKER):])


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
