"""Problems in a design's code, placed on their block and line, for the editor.

Each code block is compiled as it runs - a hook body as a method of the
generated environment, a task-info entry as a function of the setup module, the
setup blocks and custom modules as modules - and the names it looks up globally
are read from the compiled code, so Python's own scoping decides what is local.
Each must be one its module has, or a builtin: what would raise NameError when
the code runs is flagged as it is typed. Errors raised while the setup code or a
custom module runs are placed on their block and line too.

Returns ``{block: [{"line", "column", "end_column", "message", "severity"}]}``,
with lines and columns 1-based in the block's own text.
"""

from __future__ import annotations

import ast
import builtins
import dis
import importlib.util
import inspect
import textwrap
from collections.abc import Iterator
from types import CodeType
from typing import Any, NamedTuple

from bluesky_sandbox.interface.task import TaskInfoProvider

from . import setup_code
from .builder import BuildError, build_design_config
from .code_intel import BlueskyEnv, _hooks
from .spec import SCENARIO_HOOKS, DesignSpec

__all__ = ["diagnostics"]

_BUILTINS = frozenset(dir(builtins))


class Block(NamedTuple):
    """One code block: its key, text, and - for a body - the function it is in."""

    key: str
    source: str
    names: frozenset[str]
    params: tuple[str, ...] | None = None


def diagnostics(spec: DesignSpec) -> dict[str, list[dict[str, Any]]]:
    """Every block's problems, keyed by block."""
    out: dict[str, list[dict[str, Any]]] = {}
    for block in _blocks(spec):
        problems = _check(block)
        if problems:
            out[block.key] = problems
    try:
        build_design_config(spec)
    except BuildError as error:
        if error.block is not None and not out.get(error.block):
            out.setdefault(error.block, []).append(
                _problem(error.line or 1, 1, None, str(error), "error")
            )
    return out


def _blocks(spec: DesignSpec) -> Iterator[Block]:
    env = spec.env
    try:
        setup = setup_code.setup_source(
            env.task_info_setup, env.task_info, env.hook_setup
        )
        setup_names = frozenset({"CONFIG"} | _names(setup))
    except (ValueError, SyntaxError):
        setup_names = frozenset(
            {"CONFIG"} | _names(env.task_info_setup) | _names(env.hook_setup)
        )
    yield Block("task_info_setup", env.task_info_setup, setup_names)
    yield Block("hook_setup", env.hook_setup, setup_names)
    entry_params = _param_names(TaskInfoProvider.__call__)
    defined = setup_code._defined(env.task_info_setup, env.hook_setup)
    for entry in env.task_info:
        if setup_code.direct_provider(entry, defined) is None:
            yield Block(
                f"task_info:{entry.name.strip()}", entry.body, setup_names, entry_params
            )
    hooks = dict(_hooks(BlueskyEnv))
    for name, body in (env.hooks or {}).items():
        if name in hooks and body.strip():
            params = ("self", *_param_names(hooks[name], clean=True))
            yield Block(f"hook:{name}", body, setup_names, params)
    scenario_names = frozenset(_names(spec.scenario_setup or ""))
    yield Block("scenario_setup", spec.scenario_setup or "", scenario_names)
    for name, body in (spec.scenario_hooks or {}).items():
        if name in SCENARIO_HOOKS and body.strip():
            params = tuple(SCENARIO_HOOKS[name][0])
            yield Block(f"scenario:{name}", body, scenario_names, params)
    for filename, source in (spec.code or {}).items():
        if filename.endswith(".py"):
            yield Block(f"code:{filename}", source, frozenset())


def _check(block: Block) -> list[dict[str, Any]]:
    """The block's syntax errors, or else the names it cannot find."""
    if not block.source.strip():
        return []
    if block.params is None:
        source, line_shift, column_shift = block.source, 0, 0
    else:
        body = textwrap.indent(block.source, "    ")
        source = f"def _block({', '.join(block.params)}):\n{body}\n"
        line_shift, column_shift = 1, 4
    try:
        code = compile(source, "<block>", "exec")
    except SyntaxError as error:
        line = max((error.lineno or 1) - line_shift, 1)
        column = max((error.offset or 1) - column_shift, 1)
        return [_problem(line, column, None, error.msg, "error")]
    available = block.names | _BUILTINS | _module_bindings(code)
    problems = _missing_imports(block.source, line_shift=0)
    for name, position in _global_lookups(code):
        # Dunder names (__annotations__, __name__...) the runtime provides.
        dunder = name.startswith("__") and name.endswith("__")
        if name in available or dunder or position is None or position.lineno is None:
            continue
        line = position.lineno - line_shift
        start = (position.col_offset or 0) - column_shift + 1
        end = (position.end_col_offset or 0) - column_shift + 1
        problems.append(_problem(line, start, end, f"{name!r} is not defined", "error"))
    return problems


def _missing_imports(source: str, line_shift: int) -> list[dict[str, Any]]:
    """Imports of modules that cannot be found - each would raise when it runs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            if not _importable(module):
                problems.append(
                    _problem(
                        node.lineno - line_shift,
                        node.col_offset + 1,
                        node.end_col_offset + 1 if node.end_col_offset else None,
                        f"no module named {module!r}",
                        "error",
                    )
                )
    return problems


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _global_lookups(code: CodeType) -> Iterator[tuple[str, Any]]:
    """Each name ``code`` or a function in it looks up outside its locals.

    A class body's own names shadow globals within it, so those are left out.
    """
    for co in _code_objects(code):
        class_locals = _stores(co) if _is_class_body(co) else set()
        for instruction in dis.get_instructions(co):
            if (
                instruction.opname
                in (
                    "LOAD_GLOBAL",
                    "LOAD_NAME",
                    "LOAD_FROM_DICT_OR_GLOBALS",
                )
                and instruction.argval not in class_locals
            ):
                yield instruction.argval, instruction.positions


def _module_bindings(code: CodeType) -> set[str]:
    """Every name the module binds at its top level, or a function declares global."""
    bound = _stores(code)
    for co in _code_objects(code):
        bound |= {
            i.argval for i in dis.get_instructions(co) if i.opname == "STORE_GLOBAL"
        }
    return bound


def _stores(co: CodeType) -> set[str]:
    return {
        i.argval
        for i in dis.get_instructions(co)
        if i.opname in ("STORE_NAME", "DELETE_NAME", "IMPORT_NAME", "IMPORT_FROM")
    } | {
        i.argval.split(".")[0]
        for i in dis.get_instructions(co)
        if i.opname == "IMPORT_NAME"
    }


def _is_class_body(co: CodeType) -> bool:
    return any(
        i.opname == "STORE_NAME" and i.argval == "__qualname__"
        for i in dis.get_instructions(co)
    )


def _code_objects(code: CodeType) -> Iterator[CodeType]:
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from _code_objects(const)


def _names(source: str) -> set[str]:
    try:
        return setup_code.setup_names(source or "")
    except SyntaxError:
        return set()


def _param_names(fn: Any, clean: bool = False) -> tuple[str, ...]:
    names = [
        name for name in inspect.signature(fn).parameters if name not in ("self", "cls")
    ]
    return tuple(name.removeprefix("_") if clean else name for name in names)


def _problem(
    line: int, column: int, end_column: int | None, message: str, severity: str
) -> dict[str, Any]:
    problem = {"line": line, "column": column, "message": message, "severity": severity}
    if end_column is not None:
        problem["end_column"] = end_column
    return problem
