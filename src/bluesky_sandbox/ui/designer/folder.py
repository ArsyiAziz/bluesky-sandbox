"""A design as a folder: ``design.json`` for its structure, real Python for its
code.

::

    my_design/
      design.json          the structure: shapes, elements, spawn, spaces, config
      design.schema.json   what design.json may hold - any JSON editor checks it
      code/
        hooks.py           the env hooks, as functions, after their setup
        task_info.py       the task-info entries, as functions, after their setup
        scenario.py        scenario setup, scenario hooks, spawn sources' plans
        custom_fields.py   the design's own modules, as they are

Each code file is a module: its functions named for a hook (``reward``), an
entry ``design.json`` lists, a scenario hook (``episode_geometry``) or a spawn
source's plan (``plan_<source>``) are those; everything else in it is its
setup - imports, constants, helpers. A block at the top of each, between the
``in scope`` markers, imports what the design has in scope there, so an editor
or a linter reads the file as the design runs it; it is written each time and
never read back.

A design is still one ``.json`` file where that is easier to share:
:func:`load_design` reads either, :func:`save_design` writes either.
"""

from __future__ import annotations

import ast
import json
import shutil
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import setup_code
from .spec import SCENARIO_HOOKS, DesignSpec

__all__ = [
    "DESIGN_FILE",
    "FolderError",
    "load_design",
    "read_folder",
    "save_design",
    "write_folder",
]

DESIGN_FILE = "design.json"
SCHEMA_FILE = "design.schema.json"
CODE_DIR = "code"
HOOKS_FILE = "hooks.py"
TASK_INFO_FILE = "task_info.py"
SCENARIO_FILE = "scenario.py"
#: The code files the design's own blocks are written to; its modules take any other name.
RESERVED = frozenset({HOOKS_FILE, TASK_INFO_FILE, SCENARIO_FILE})

_IN_SCOPE = "# >>> in scope, as the design runs it - written each save, never read back"
_END_IN_SCOPE = "# <<< in scope"
_TASK_INFO_ARGS = "(obs, action, info, context, rng)"
_SOURCE_ARGS = "(rng, ctx)"


class FolderError(ValueError):
    """A design folder that cannot be read or written as one."""


# ---- one entry point for either form ------------------------------------- #


def load_design(path: str | Path) -> DesignSpec:
    """The design at ``path``: a folder (or its ``design.json``), or one ``.json`` file."""
    path = Path(path)
    if path.is_dir():
        return read_folder(path)
    if path.name == DESIGN_FILE and (path.parent / CODE_DIR).is_dir():
        return read_folder(path.parent)
    return DesignSpec.from_json(path.read_text())


def save_design(spec: DesignSpec, path: str | Path) -> Path:
    """Write ``spec`` to ``path``: one file when it ends in ``.json``, else a folder."""
    path = Path(path)
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec.to_json(indent=2) + "\n")
        return path
    write_folder(spec, path)
    return path


# ---- writing ------------------------------------------------------------- #


def write_folder(spec: DesignSpec, folder: str | Path) -> None:
    """Write ``spec`` as a design folder (see the module). Files it no longer
    has - a module the design dropped - are removed from ``code/``."""
    folder = Path(folder)
    clash = RESERVED & set(spec.code)
    if clash:
        raise FolderError(
            f"a design module named {sorted(clash)[0]!r} would clash with the folder's own "
            f"code files ({', '.join(sorted(RESERVED))}): rename it"
        )
    files = folder_files(spec)
    code_dir = folder / CODE_DIR
    code_dir.mkdir(parents=True, exist_ok=True)
    for stale in code_dir.glob("*.py"):
        if f"{CODE_DIR}/{stale.name}" not in files:
            stale.unlink()
    for rel, text in files.items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    pycache = code_dir / "__pycache__"
    if pycache.is_dir():
        shutil.rmtree(pycache)


def folder_files(spec: DesignSpec) -> dict[str, str]:
    """The folder's files, by path within it."""
    from .schema import design_schema  # noqa: PLC0415 - the catalog is heavy

    files = {
        DESIGN_FILE: json.dumps(_structure(spec), indent=2) + "\n",
        SCHEMA_FILE: json.dumps(design_schema(), indent=2) + "\n",
        f"{CODE_DIR}/{HOOKS_FILE}": _hooks_module(spec),
        f"{CODE_DIR}/{TASK_INFO_FILE}": _task_info_module(spec),
        f"{CODE_DIR}/{SCENARIO_FILE}": _scenario_module(spec),
    }
    for name, source in spec.code.items():
        files[f"{CODE_DIR}/{name}"] = source if source.endswith("\n") or not source else source + "\n"
    return files


def _structure(spec: DesignSpec) -> dict[str, Any]:
    """``design.json``: the design without its code - which entries, hooks and
    sources there are stays, by name, so the code files are read back as them."""
    d = spec.to_dict()
    env = d["env"]
    env["hooks"] = sorted(name for name, body in env.get("hooks", {}).items() if body.strip())
    env["task_info"] = [entry["name"] for entry in env.get("task_info", [])]
    env.pop("hook_setup", None)
    env.pop("task_info_setup", None)
    d["scenario_hooks"] = sorted(name for name, body in d.get("scenario_hooks", {}).items() if body.strip())
    d.pop("scenario_setup", None)
    d["code"] = sorted(d.get("code", {}))
    spawn = d.get("spawn")
    if isinstance(spawn, dict) and spawn.get("sources"):
        spawn["sources"] = [{k: v for k, v in s.items() if k != "plan"} for s in spawn["sources"]]
    return {"$schema": f"./{SCHEMA_FILE}", **d}


def _module(header: str, setup: str, functions: list[tuple[str, str, str]]) -> str:
    """A code file: its in-scope block, its setup, then each ``(name, args, body)``."""
    parts = [f"{_IN_SCOPE}\n{header.rstrip()}\n{_END_IN_SCOPE}\n"]
    if setup.strip():
        parts.append(setup.strip("\n") + "\n")
    for name, args, body in functions:
        body = body.strip("\n") or "pass"
        parts.append(f"def {name}{args}:\n{textwrap.indent(body, '    ')}\n")
    return "\n\n".join(parts)


@lru_cache(maxsize=1)
def _hook_signatures() -> dict[str, str]:
    from .catalog import hooks  # noqa: PLC0415 - the catalog is heavy

    return {h["name"]: h["def_signature"] for h in hooks()}


def _env_header() -> str:
    names = textwrap.fill(
        ", ".join(sorted(setup_code.hook_api_names())) + ",",
        width=88,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    return (
        "from __future__ import annotations\n\n"
        "import numpy as np\n\n"
        "from bluesky_sandbox import (  # noqa: F401\n"
        f"{names}\n)\n"
        "from bluesky_sandbox.config import EnvConfig\n\n"
        "CONFIG: EnvConfig  # the design's config"
    )


def _hooks_module(spec: DesignSpec) -> str:
    signatures = _hook_signatures()
    hooks = [
        (name, signatures.get(name, "(self, *args, **kwargs)"), body)
        for name, body in sorted(spec.env.hooks.items())
        if body.strip()
    ]
    return _module(_env_header(), spec.env.hook_setup, hooks)


def _task_info_module(spec: DesignSpec) -> str:
    entries = [(e.name, _TASK_INFO_ARGS, e.body) for e in spec.env.task_info]
    return _module(_env_header(), spec.env.task_info_setup, entries)


def _scenario_module(spec: DesignSpec) -> str:
    header = "from __future__ import annotations\n\n" + setup_code.scenario_api_imports()
    functions = [
        (name, f"({', '.join(SCENARIO_HOOKS[name][0])})" if name in SCENARIO_HOOKS else "(*args)", body)
        for name, body in sorted(spec.scenario_hooks.items())
        if body.strip()
    ]
    spawn = spec.spawn if isinstance(spec.spawn, dict) else {}
    for source in spawn.get("sources") or []:
        functions.append((f"plan_{source['name']}", _SOURCE_ARGS, source.get("plan") or "return []"))
    return _module(header, spec.scenario_setup, functions)


# ---- reading ------------------------------------------------------------- #


def read_folder(folder: str | Path) -> DesignSpec:
    """The design in ``folder`` (see the module)."""
    folder = Path(folder)
    design = folder / DESIGN_FILE
    if not design.is_file():
        raise FolderError(f"{folder} has no {DESIGN_FILE}")
    d = json.loads(design.read_text())
    d.pop("$schema", None)
    code_dir = folder / CODE_DIR

    def code(name: str) -> str:
        path = code_dir / name
        return path.read_text() if path.is_file() else ""

    env = d.setdefault("env", {})
    hook_names = set(env.get("hooks") or []) | set(_hook_signatures())
    setup, bodies = split_module(code(HOOKS_FILE), hook_names, HOOKS_FILE)
    env["hook_setup"] = setup
    env["hooks"] = {name: bodies[name] for name in sorted(bodies)}

    entries = [e if isinstance(e, str) else e.get("name") for e in env.get("task_info") or []]
    setup, bodies = split_module(code(TASK_INFO_FILE), set(entries), TASK_INFO_FILE)
    env["task_info_setup"] = setup
    env["task_info"] = [{"name": name, "body": bodies.get(name, "")} for name in entries]

    spawn = d.get("spawn") if isinstance(d.get("spawn"), dict) else {}
    sources = spawn.get("sources") or []
    plans = {f"plan_{s['name']}" for s in sources}
    setup, bodies = split_module(code(SCENARIO_FILE), set(SCENARIO_HOOKS) | set(d.get("scenario_hooks") or []) | plans, SCENARIO_FILE)
    d["scenario_setup"] = setup
    d["scenario_hooks"] = {name: body for name, body in sorted(bodies.items()) if name not in plans}
    for source in sources:
        source["plan"] = bodies.get(f"plan_{source['name']}", "")

    modules = d.get("code") or []
    if isinstance(modules, dict):  # a design.json written whole
        modules = list(modules)
    d["code"] = {name: code(name) for name in modules}
    return DesignSpec.from_dict(d)


def split_module(source: str, names: set[str], filename: str = "<code>") -> tuple[str, dict[str, str]]:
    """``source`` as ``(setup, {name: body})``: the body of each top-level
    function named in ``names``, and the rest - its in-scope block left out -
    as the setup, in order."""
    source = _without_in_scope(source)
    if not source.strip():
        return "", {}
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise FolderError(f"{filename}, line {e.lineno}: {e.msg}") from e
    lines = source.splitlines()
    taken: list[tuple[int, int]] = []
    bodies: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
        header_end = _header_end(node)
        end = node.end_lineno
        # Comments indented under the function after its last statement are its own.
        while end < len(lines) and (not lines[end].strip() or lines[end][:1] in (" ", "\t")):
            end += 1
        while end > header_end and not lines[end - 1].strip():
            end -= 1
        bodies[node.name] = textwrap.dedent("\n".join(lines[header_end:end])).strip("\n") + "\n"
        taken.append((start, end))
    # The setup: what is left, its own blank lines kept; those that set a
    # function apart go with it, and setup either side of one is two blank
    # lines apart.
    chunks, at = [], 0
    for start, end in taken:
        chunks.append("\n".join(lines[at:start]).strip("\n"))
        at = end
    chunks.append("\n".join(lines[at:]).strip("\n"))
    setup = "\n\n\n".join(chunk for chunk in chunks if chunk.strip())
    return (setup + "\n" if setup else ""), bodies


def _header_end(node: ast.FunctionDef) -> int:
    """The line (1-based) the ``def`` header ends on: its body starts after it."""
    parts = [node.lineno]
    if node.returns is not None:
        parts.append(node.returns.end_lineno)
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg):
        if arg is not None:
            parts.append(arg.end_lineno)
    for default in (*node.args.defaults, *node.args.kw_defaults):
        if default is not None:
            parts.append(default.end_lineno)
    end = max(parts)
    # A body on the header's own line (``def f(): return 1``) starts there.
    if node.body and node.body[0].lineno == end:
        raise FolderError(f"put the body of {node.name}() on the lines below its def")
    return end


def _without_in_scope(source: str) -> str:
    """``source`` with its in-scope block - and the blank lines after it - left out."""
    start = source.find(_IN_SCOPE)
    if start < 0:
        return source
    end = source.find(_END_IN_SCOPE, start)
    if end < 0:
        return source[:start]
    rest = source[end + len(_END_IN_SCOPE):]
    return source[:start] + rest.lstrip("\n")
