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


def setup_parts(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> list[tuple[str, str]]:
    """The setup module's code in order, as ``(where, code)``: the task-info
    setup, each inline task-info entry as a function, then the hook setup.

    ``where`` names what the design calls that code - ``"task-info setup"``,
    ``"task info 'waypoint_outcome'"``, ``"hook setup"`` - so an error in the
    module can say which block it came from.
    """
    defined = _defined(task_info_setup, hook_setup)
    parts = [("task-info setup", dedupe_imports(task_info_setup, PRELUDE).rstrip())]
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
        parts.append((f"task info {name!r}", code))
    hook_setup = dedupe_imports(hook_setup, PRELUDE + task_info_setup)
    parts.append(("hook setup", hook_setup.rstrip()))
    return [(where, code) for where, code in parts if code.strip()]


def setup_source(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> str:
    """The setup module's code after :data:`PRELUDE`: :func:`setup_parts`, joined."""
    return located_setup_source(task_info_setup, task_info, hook_setup)[0]


def located_setup_source(
    task_info_setup: str, task_info: list[TaskInfoSpec], hook_setup: str
) -> tuple[str, list[tuple[str, int]]]:
    """:func:`setup_source`, and the line (1-based) each part starts on."""
    parts = setup_parts(task_info_setup, task_info, hook_setup)
    setup = [code for where, code in parts if where == "task-info setup"]
    entries = [(where, code) for where, code in parts if where.startswith("task info ")]
    hooks = [code for where, code in parts if where == "hook setup"]
    functions = (
        ("\n\n".join(code for _where, code in entries) + "\n") if entries else ""
    )
    source = "\n".join(part for part in (*setup, functions, *hooks) if part.strip())

    starts: list[tuple[str, int]] = []
    line = 1
    if setup:
        starts.append(("task-info setup", line))
        line += setup[0].count("\n") + 1
    for where, code in entries:
        starts.append((where, line))
        line += code.count("\n") + 2  # two blank lines follow each entry
    if hooks:
        starts.append(("hook setup", line))
    return source, starts


def locate(starts: list[tuple[str, int]], lineno: int) -> tuple[str, int]:
    """Which part a line of the setup code is in, and its line within that part."""
    where, first = starts[0]
    for part, start in starts:
        if start > lineno:
            break
        where, first = part, start
    return where, lineno - first + 1


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
