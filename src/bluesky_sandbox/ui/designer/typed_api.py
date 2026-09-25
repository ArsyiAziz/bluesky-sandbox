"""Types for a generated task's code, so an IDE completes and checks its keys.

A generated package gets a ``task_types.py``: for each class a hook receives that
has ``DesignKeys`` members - the step context, the step batch - a subclass typed
for this design. Its members that hold design keys become ``TypedDict``s of
those keys; a method taking a key gets one overload per key, returning what that
key gives. The hooks and task-info functions are annotated with them, so
``context.raw_obs["ownship"]["alt_ft"]`` completes, hovers and is checked in any
editor that reads annotations. Nothing here runs: the annotations are strings,
and the objects a hook receives are the library's own.

Which members are retyped, and to what, is read from the classes' annotations
and the design (:mod:`.design_keys`); nothing here names a member.
"""

from __future__ import annotations

import importlib
import inspect
import keyword
import sys
import types as pytypes
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.task import DesignKeys, TaskInfoProvider

from .code_intel import _hooks, hints
from .design_keys import Key, design_keys

__all__ = ["TaskTypes"]

#: The generated module's name, beside env.py and setup.py.
MODULE = "task_types"


class Renderer:
    """Annotations as source text, collecting the imports they need.

    Classes this task specializes render as their local name; a module the
    code already imports under an alias (``np`` for numpy) is used through it.
    """

    def __init__(self, local: dict[type, str], aliases: dict[str, str]) -> None:
        self.local = local
        self.aliases = aliases
        # Named type aliases (``BaseObs``) by identity, as ``(module, name)``.
        self.named: dict[int, tuple[str, str]] = {}
        self.imports: dict[str, set[str]] = {}
        self.local_used: set[str] = set()
        self.aliases_used: set[str] = set()

    def __call__(self, annotation: Any) -> str:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if annotation is None or annotation is type(None):
            return "None"
        if annotation is Any or isinstance(annotation, TypeVar):
            return self._name("typing", "Any")
        if annotation is Ellipsis:
            return "..."
        if isinstance(annotation, str):
            return annotation
        if id(annotation) in self.named:
            return self._name(*self.named[id(annotation)])
        origin = get_origin(annotation)
        if origin in (Union, pytypes.UnionType):
            return " | ".join(self(arg) for arg in get_args(annotation))
        if origin is Literal:
            args = ", ".join(repr(arg) for arg in get_args(annotation))
            return f"{self._name('typing', 'Literal')}[{args}]"
        if origin is not None:
            args = ", ".join(self(arg) for arg in get_args(annotation))
            return f"{self(origin)}[{args}]" if args else self(origin)
        if inspect.isclass(annotation):
            return self._class(annotation)
        return self._name("typing", "Any")

    def know_aliases(self, module_name: str) -> None:
        """Learn the type aliases ``module_name`` binds, to render them by name."""
        module = sys.modules.get(module_name)
        for name, value in vars(module or {}).items():
            if not name.startswith("_") and get_origin(value) in (
                Union,
                pytypes.UnionType,
            ):
                self.named.setdefault(id(value), (_defining_module(value, name), name))

    def library(self, cls: type) -> str:
        """``cls`` itself, never this task's typed subclass of it."""
        return self._class(cls, local=False)

    def _class(self, cls: type, local: bool = True) -> str:
        if local and cls in self.local:
            self.local_used.add(self.local[cls])
            return self.local[cls]
        if cls.__module__ == "builtins":
            return cls.__qualname__
        module = _public_module(cls)
        for imported, alias in self.aliases.items():
            if module == imported or module.startswith(imported + "."):
                self.aliases_used.add(imported)
                return f"{alias}{module[len(imported) :]}.{cls.__qualname__}"
        return self._name(module, cls.__qualname__)

    def _name(self, module: str, name: str) -> str:
        self.imports.setdefault(module, set()).add(name.split(".")[0])
        return name

    def import_lines(self, indent: str = "") -> list[str]:
        """The imports the rendered text needs."""
        lines = [
            f"{indent}import {module} as {self.aliases[module]}"
            for module in sorted(self.aliases_used)
        ]
        for module, names in sorted(self.imports.items()):
            lines.append(f"{indent}from {module} import {', '.join(sorted(names))}")
        return lines


def _defining_module(value: Any, name: str) -> str:
    """The shortest module path that binds ``name`` to ``value``."""
    for module_name in sorted(sys.modules, key=len):
        if not module_name.startswith("bluesky_sandbox"):
            continue
        if getattr(sys.modules[module_name], name, None) is value:
            return module_name
    return "typing"


def _public_module(cls: type) -> str:
    """The shortest module path ``cls`` can be imported from."""
    parts = cls.__module__.split(".")
    for end in range(1, len(parts) + 1):
        path = ".".join(parts[:end])
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        if getattr(module, cls.__qualname__, None) is cls:
            return path
    return cls.__module__


class TaskTypes:
    """The task's typed classes, and the signatures of its hooks in their terms."""

    def __init__(
        self, config: EnvConfig, support: Any, aliases: dict[str, str]
    ) -> None:
        self._config = config
        self._support = support
        self._aliases = aliases
        self._roots = _hook_classes()
        self.local = {
            cls: f"Task{cls.__name__}" for cls in self._roots if _design_members(cls)
        }

    def renderer(self) -> Renderer:
        """A renderer for one generated file: its imports are that file's."""
        return Renderer(self.local, self._aliases)

    # --- the module -----------------------------------------------------------
    def module_source(self) -> str:
        """``task_types.py``."""
        render = Renderer(self.local, self._aliases)
        dicts: list[str] = []
        named: dict[tuple[str, bool], str] = {}
        classes = []
        for cls, local in self.local.items():
            body = []
            for name, marker, kind, member in _design_members(cls):
                if kind == "attr":
                    typed = self._typed_dict(named, dicts, render, marker, _camel(name))
                    body.append(f"    {name}: {typed}")
                else:
                    body.extend(self._overloads(render, name, marker, member))
            for name, member in _returning(cls, self.local):
                body.extend(self._method(render, name, member))
            doc = (
                f'    """``{cls.__qualname__}``, typed for this task: the members that '
                f'hold\n    the design\'s keys, as this design has them."""'
            )
            classes.append(
                f"class {local}({render.library(cls)}):\n{doc}\n\n"
                + "\n".join(body)
                + "\n"
            )
        header = [
            '"""Types for this task\'s code, generated from its design.',
            "",
            "Editors and type checkers read these to complete and check the design's",
            "keys - observation parts and fields, action fields, queryables. Nothing",
            "here runs: hooks receive the library's objects, and these subclasses only",
            "describe what those objects hold for this task. Regenerate the package",
            "to update them.",
            '"""',
            "",
            "# Narrowing the library's types, in methods without bodies, is the point:",
            "# say so to type checkers.",
            "# pyright: reportIncompatibleVariableOverride=false, "
            "reportIncompatibleMethodOverride=false",
            '# mypy: disable-error-code="assignment, override, empty-body"',
            "",
            "from __future__ import annotations",
            "",
            *render.import_lines(),
        ]
        return "\n".join(header) + "\n\n\n" + "\n\n".join(dicts + classes)

    def _typed_dict(
        self,
        named: dict[tuple[str, bool], str],
        dicts: list[str],
        render: Renderer,
        marker: DesignKeys,
        stem: str,
    ) -> str:
        """The TypedDict of ``marker``'s keys, written once and named by its member."""
        key = (marker.source, marker.batched)
        if key not in named:
            name = stem + ("Batch" if marker.batched else "")
            named[key] = name
            keys = design_keys(marker, self._config, self._support)
            self._write_dict(dicts, render, name, keys)
        return named[key]

    def _write_dict(
        self, dicts: list[str], render: Renderer, name: str, keys: list[Key]
    ) -> None:
        fields = []
        for k in keys:
            if k.keys is not None:
                inner = name + _camel(k.name)
                self._write_dict(dicts, render, inner, k.keys)
                fields.append((k.name, inner, k.detail))
            else:
                fields.append(
                    (
                        k.name,
                        render(k.value if k.value is not None else Any),
                        _field_note(k),
                    )
                )
        typed_dict = render._name("typing", "TypedDict")
        if all(n.isidentifier() and not keyword.iskeyword(n) for n, _t, _d in fields):
            lines = [f"class {name}({typed_dict}):"]
            for field_name, typed, note in fields:
                lines.append(f"    {field_name}: {typed}")
                if note:
                    lines.append(f'    """{note}"""')
            if not fields:
                lines.append("    pass")
            dicts.append("\n".join(lines) + "\n")
        else:
            # Keys that are not identifiers (a repeated field's "alt_ft#2").
            items = ", ".join(f"{n!r}: {t}" for n, t, _d in fields)
            dicts.append(f"{name} = {typed_dict}({name!r}, {{{items}}})\n")

    def _overloads(
        self,
        render: Renderer,
        name: str,
        marker: DesignKeys,
        member: Callable[..., Any],
    ) -> list[str]:
        """A method taking one of the design's keys: an overload per key."""
        keys = design_keys(marker, self._config, self._support)
        if not keys:
            return []
        signature = inspect.signature(member)
        member_hints = hints(member)
        declared = member_hints.get("return", Any)
        param = next(p for p in signature.parameters.values() if p.name != "self")
        overload = render._name("typing", "overload")
        literal = render._name("typing", "Literal")
        lines = []
        if declared is Any:
            for k in keys:
                lines += [
                    f"    @{overload}",
                    f"    def {name}(self, {param.name}: {literal}[{k.name!r}]) -> "
                    f"{render(k.value)}: ...",
                ]
            lines.append(
                f"    def {name}(self, {param.name}: str) -> {render(Any)}: ..."
            )
        else:
            names = ", ".join(repr(k.name) for k in keys)
            lines.append(
                f"    def {name}(self, {param.name}: {literal}[{names}]) -> "
                f"{render(declared)}: ..."
            )
        return lines

    def _method(
        self, render: Renderer, name: str, member: Callable[..., Any]
    ) -> list[str]:
        """A method re-declared to return this task's typed class."""
        return [f"    def {name}{self.signature(member, render)}: ..."]

    # --- signatures -----------------------------------------------------------
    def signature(
        self,
        fn: Callable[..., Any],
        render: Renderer,
        clean: bool = False,
        self_param: bool = True,
    ) -> str:
        """``fn``'s parameter list and return, annotated in this task's types."""
        params, tail = self.signature_parts(fn, render, clean, self_param)
        return f"({', '.join(params)}){tail}"

    def signature_parts(
        self,
        fn: Callable[..., Any],
        render: Renderer,
        clean: bool = False,
        self_param: bool = True,
    ) -> tuple[list[str], str]:
        """``fn``'s annotated parameters, and its `` -> return``."""
        render.local = self.local
        render.know_aliases(fn.__module__)
        fn_hints = hints(fn)
        params = []
        for param in inspect.signature(fn).parameters.values():
            name = param.name.removeprefix("_") if clean else param.name
            if param.name == "self":
                if self_param:
                    params.append("self")
                continue
            annotation = fn_hints.get(param.name, param.annotation)
            text = (
                name
                if annotation is inspect.Parameter.empty
                else f"{name}: {render(annotation)}"
            )
            if param.default is not inspect.Parameter.empty:
                text += f" = {param.default!r}"
            params.append(text)
        returns = fn_hints.get("return", inspect.Signature.empty)
        tail = "" if returns is inspect.Signature.empty else f" -> {render(returns)}"
        return params, tail

    def hook_signature(self, hook: str, render: Renderer) -> str | None:
        """A hook's signature, without the leading underscores the base class
        marks its unused parameters with; None for a name that is no hook."""
        fn = dict(_hooks(BlueskyEnv)).get(hook)
        if fn is None:
            return None
        params, tail = self.signature_parts(fn, render, clean=True)
        return wrap_signature(params, tail, indent="    ", name_width=len(hook) + 8)

    def provider_signature(self, render: Renderer) -> str:
        """A task-info entry's signature: the provider protocol's, without self.

        One line, however long: the builder places a task-info error by line,
        and counts an entry's def as one.
        """
        return self.signature(TaskInfoProvider.__call__, render, self_param=False)


def wrap_signature(params: list[str], tail: str, indent: str, name_width: int) -> str:
    """``(params) -> return`` on one line when it fits in 88 columns after
    ``def name``, else one parameter per line."""
    one_line = f"({', '.join(params)}){tail}"
    if len(indent) + name_width + len(one_line) + 1 <= 88:
        return one_line
    inner = indent + "    "
    return "(\n" + "".join(f"{inner}{p},\n" for p in params) + f"{indent}){tail}"


def _hook_classes() -> list[type]:
    """The classes the hooks and task-info providers are given."""
    found: list[type] = []
    functions = [fn for _name, fn in _hooks(BlueskyEnv)] + [TaskInfoProvider.__call__]
    for fn in functions:
        for annotation in hints(fn).values():
            for cls in _classes_in(annotation):
                if cls not in found:
                    found.append(cls)
    return found


def _classes_in(annotation: Any) -> Iterable[type]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if inspect.isclass(annotation) and get_origin(annotation) is None:
        yield annotation
    for arg in get_args(annotation):
        yield from _classes_in(arg)


def _design_members(cls: type) -> list[tuple[str, DesignKeys, str, Any]]:
    """``(name, marker, "attr" | "method", member)`` for each member of ``cls``
    that holds or takes design keys."""
    out = []
    for name, annotation in hints(cls).items():
        marker = _marker(annotation)
        if marker is not None and not name.startswith("_"):
            out.append((name, marker, "attr", annotation))
    for name, member in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_"):
            continue
        for param, annotation in hints(member).items():
            marker = _marker(annotation)
            if param != "return" and marker is not None:
                out.append((name, marker, "method", member))
                break
    return out


def _returning(cls: type, local: dict[type, str]) -> list[tuple[str, Any]]:
    """Public methods of ``cls`` that return a class this task specializes."""
    out = []
    keyed = {name for name, _m, kind, _x in _design_members(cls) if kind == "method"}
    for name, member in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_") or name in keyed:
            continue
        if hints(member).get("return") in local:
            out.append((name, member))
    return out


def _marker(annotation: Any) -> DesignKeys | None:
    if get_origin(annotation) is Annotated:
        return next(
            (m for m in get_args(annotation)[1:] if isinstance(m, DesignKeys)), None
        )
    return None


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("#", "_").split("_"))


def _field_note(key: Key) -> str:
    note = key.detail
    first = key.doc.split("\n\n", 1)[0].replace("`", "")
    if first:
        note = f"{note} - {first}" if note else first
    return note.replace('"""', "'''")
