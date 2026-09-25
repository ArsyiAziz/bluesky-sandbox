"""A design's module-level code, composed once for both ways it is built.

A design carries two setup blocks - ``task_info_setup`` and ``hook_setup`` -
and its inline task-info entries. A generated package writes them into one
``setup.py``; the in-process builder runs them in one module. Both take the
code from :func:`setup_source`, so a helper one block defines is visible to the
other the same way either way: a task-info entry may call a function from the
hook setup, as the generated package has always allowed.
"""

from __future__ import annotations

import ast
import keyword
import textwrap
from typing import NamedTuple

from .spec import TaskInfoSpec

#: The parameters of every inline task-info entry.
PROVIDER_SIGNATURE = "(obs, action, info, context, rng)"

#: What the generated ``setup.py`` binds before the setup code runs; the
#: builder binds the same names. The setup code's own copies are dropped.
PRELUDE = "from __future__ import annotations\n\nfrom .config import CONFIG\n"


def setup_names(source: str) -> set[str]:
    """Top-level names ``source`` binds: what the setup module exports.

    Underscore-prefixed names are included deliberately: most setup helpers are
    private by convention, and the hooks depend on exactly those.
    """
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def direct_provider(provider: TaskInfoSpec, defined: set[str]) -> str | None:
    """The setup name an entry's body refers to, when the body is only that name.

    ``AUTO_COST_PROVIDER`` as a body means "this object is the provider", not a
    function whose body evaluates the name and discards it.
    """
    body = provider.body.strip()
    if body.isidentifier() and not keyword.iskeyword(body) and body in defined:
        return body
    return None


def provider_names(
    task_info: list[TaskInfoSpec], task_info_setup: str, hook_setup: str
) -> list[str]:
    """The name each task-info entry is bound to in the setup module, in order."""
    defined = _defined(task_info_setup, hook_setup)
    return [direct_provider(p, defined) or p.name.strip() for p in task_info]


class Part(NamedTuple):
    """One block of the setup module: ``where`` names it in messages -
    ``"task info 'waypoint_outcome'"`` - and ``block`` in the editor -
    ``"task_info:waypoint_outcome"``."""

    where: str
    block: str
    code: str


def setup_parts(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> list[Part]:
    """The setup module's code in order: the task-info setup, each inline
    task-info entry as a function, then the hook setup."""
    defined = _defined(task_info_setup, hook_setup)
    setup = dedupe_imports(task_info_setup, PRELUDE).rstrip()
    parts = [Part("task-info setup", "task_info_setup", setup)]
    seen: set[str] = set()
    for provider in task_info:
        name = provider.name.strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f"task-info provider name must be a Python identifier, got {name!r}."
            )
        if name in seen:
            raise ValueError(f"duplicate task-info provider name {name!r}.")
        seen.add(name)
        if direct_provider(provider, defined) is not None:
            continue
        body = textwrap.indent(provider.body.rstrip() or "pass", "    ")
        code = f"def {name}{PROVIDER_SIGNATURE} -> None:\n{body}\n"
        parts.append(Part(f"task info {name!r}", f"task_info:{name}", code))
    hook_setup = dedupe_imports(hook_setup, PRELUDE + task_info_setup)
    parts.append(Part("hook setup", "hook_setup", hook_setup.rstrip()))
    return [part for part in parts if part.code.strip()]


def setup_source(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> str:
    """The setup module's code after :data:`PRELUDE`: :func:`setup_parts`, joined."""
    return located_setup_source(task_info_setup, task_info, hook_setup)[0]


def located_setup_source(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> tuple[str, list[tuple[Part, int]]]:
    """:func:`setup_source`, and the line (1-based) each part starts on."""
    parts = setup_parts(task_info_setup, task_info, hook_setup)
    setup = [p for p in parts if p.block == "task_info_setup"]
    entries = [p for p in parts if p.block.startswith("task_info:")]
    hooks = [p for p in parts if p.block == "hook_setup"]
    functions = ("\n\n".join(p.code for p in entries) + "\n") if entries else ""
    source = "\n".join(
        code
        for code in (*(p.code for p in setup), functions, *(p.code for p in hooks))
        if code.strip()
    )

    starts: list[tuple[Part, int]] = []
    line = 1
    for part in setup:
        starts.append((part, line))
        line += part.code.count("\n") + 1
    for part in entries:
        starts.append((part, line))
        line += part.code.count("\n") + 2  # two blank lines follow each entry
    for part in hooks:
        starts.append((part, line))
    return source, starts


def locate(starts: list[tuple[Part, int]], lineno: int) -> tuple[Part, int]:
    """Which part a line of the setup code is in, and its line in that part's
    own text - for an entry, counted from its body, below the def line."""
    part, first = starts[0]
    for candidate, start in starts:
        if start > lineno:
            break
        part, first = candidate, start
    line = lineno - first + 1
    return part, line - 1 if part.block.startswith("task_info:") else line


def dedupe_imports(setup: str, existing_source: str = "") -> str:
    """``setup`` without the import lines ``existing_source`` already has."""
    seen = {line.strip() for line in existing_source.splitlines() if _is_import(line)}
    out: list[str] = []
    for line in setup.rstrip().splitlines():
        stripped = line.strip()
        if _is_import(line):
            if stripped in seen:
                continue
            seen.add(stripped)
        out.append(line)
    return "\n".join(out)


def _defined(task_info_setup: str, hook_setup: str) -> set[str]:
    return setup_names(task_info_setup or "") | setup_names(hook_setup or "")


def _is_import(line: str) -> bool:
    return line.strip().startswith(("import ", "from "))
