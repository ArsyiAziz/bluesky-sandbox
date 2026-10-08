"""What a design's code can use - types, scopes and design keys - by introspection.

The designer's code editor completes, colors, explains and checks code from what
this returns. Nothing here names a library member:

- a type's members come from the class - its annotations, properties and
  signatures - with names only type checkers import resolved the same way;
- the keys a design fixes (observation parts and fields, action fields,
  queryables) come from ``DesignKeys`` annotations, filled from the design;
- each code block's names come from what runs there: a hook's signature, the
  setup module the builder runs, a custom code module.

``types`` maps a key to ``{"doc", "attrs", "items"}``; a member names its
``type`` by key, so the editor resolves ``context.obs["intruders"]["acid"]``
one step at a time. ``scopes`` maps each code block to its ``params`` and the
key of its ``names`` in ``names`` - blocks sharing a module share one list.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import sys
import types as pytypes
import typing
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Union, get_args, get_origin

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.task import DesignKeys, TaskInfoProvider

from . import scenario_api, setup_code
from .builder import build_design_config, build_scenario, run_setup_module
from .design_keys import Key, design_keys
from .spec import SCENARIO_HOOKS, DesignSpec

__all__ = ["code_intel", "describe_type", "hints"]

#: Values whose members say nothing a task needs: numbers, strings, containers.
_OPAQUE = (int, float, complex, str, bytes, bool, type(None), object, type, Any)

#: How deep member types are followed from a scope's names up front; a deeper
#: one is named and marked ``partial``, and the editor asks for it
#: (:func:`describe_type`) when it gets there.
_DEPTH = 4


def describe_type(key: str) -> dict[str, dict[str, Any]]:
    """The type ``key`` (``"module.Qualname"``, as the types are keyed),
    described one level down - its members' own types named, ``partial`` -
    for the editor to fill in a type as completion reaches it. Raises
    ``LookupError`` for a key that names no class."""
    cls = _class_of(key)
    table = TypeTable(None, None)
    table.ref(cls, depth=1)
    return table.types


def _class_of(key: str) -> type:
    """The class keyed ``key``: the longest importable module prefix, then
    the qualname's parts on it."""
    parts = key.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for name in parts[cut:]:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if inspect.isclass(obj):
            return obj
    raise LookupError(f"no class {key!r}")


def code_intel(spec: DesignSpec) -> dict[str, Any]:
    """Types and scopes for every code block of ``spec``."""
    config = build_design_config(spec)
    support = build_scenario(spec).support()
    table = TypeTable(config, support)
    names: dict[str, list[dict[str, Any]]] = {}
    scopes = _scopes(spec, config, table, names)
    return {"ok": True, "scopes": scopes, "names": names, "types": table.types}


# --------------------------------------------------------------------------- #
# Annotations                                                                 #
# --------------------------------------------------------------------------- #
def hints(obj: Any) -> dict[str, Any]:
    """``obj``'s type hints with ``Annotated`` kept, names that only type
    checkers import (``if TYPE_CHECKING:``) resolved as they would be."""
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except Exception:
        pass
    owners = inspect.getmro(obj) if inspect.isclass(obj) else (obj,)
    localns: dict[str, Any] = {}
    for owner in owners:
        module = sys.modules.get(getattr(owner, "__module__", None) or "")
        if module is not None:
            localns.update(_type_checking_names(module))
    try:
        return typing.get_type_hints(obj, localns=localns, include_extras=True)
    except Exception:
        return {}


def forget_type_checking_names() -> None:
    """Read each module's type-checking imports again on next use."""
    _type_checking_names.cache_clear()


@functools.cache
def _type_checking_names(module: pytypes.ModuleType) -> dict[str, Any]:
    """The names ``module`` imports under ``if TYPE_CHECKING:``."""
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return {}
    namespace: dict[str, Any] = {
        "__name__": module.__name__,
        "__package__": module.__package__,
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
            continue
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                code = compile(ast.Module([statement], []), module.__name__, "exec")
                try:
                    exec(code, namespace)
                except Exception:
                    continue
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _unwrap(annotation: Any) -> tuple[Any, DesignKeys | None]:
    """``annotation`` without ``Annotated`` or ``| None``, and its DesignKeys."""
    marker = None
    if get_origin(annotation) is Annotated:
        annotation, *extras = get_args(annotation)
        marker = next((e for e in extras if isinstance(e, DesignKeys)), None)
    if get_origin(annotation) in (Union, pytypes.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            inner, inner_marker = _unwrap(options[0])
            return inner, marker or inner_marker
    return annotation, marker


def label(annotation: Any) -> str:
    """A short, readable name for a type annotation."""
    if annotation is None or annotation is inspect.Signature.empty:
        return ""
    annotation, _marker = _unwrap(annotation)
    if isinstance(annotation, str):
        return annotation
    if inspect.isclass(annotation) and not get_args(annotation):
        return annotation.__qualname__
    text = inspect.formatannotation(annotation)
    for prefix in ("typing.", "collections.abc.", "numpy.random._generator.", "numpy."):
        text = text.replace(prefix, "np." if prefix.startswith("numpy") else "")
    return text


def _doc(obj: Any) -> str:
    """The first paragraph of ``obj``'s docstring."""
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n", 1)[0].strip()


# --------------------------------------------------------------------------- #
# Types                                                                       #
# --------------------------------------------------------------------------- #
class TypeTable:
    """Types by key, described from classes and from the design's keys."""

    def __init__(self, config: EnvConfig | None, support: Any) -> None:
        self.types: dict[str, dict[str, Any]] = {}
        self._depth: dict[str, int] = {}
        self._config = config
        self._support = support

    # --- references -----------------------------------------------------------
    def ref(self, annotation: Any, depth: int = _DEPTH) -> str | None:
        """The key of ``annotation``'s type, described down to ``depth``."""
        annotation, marker = _unwrap(annotation)
        if marker is not None:
            return self.design(marker, depth)
        cls = get_origin(annotation) or annotation
        # A union of several types names no single type to describe.
        if (
            cls in (Union, pytypes.UnionType)
            or not inspect.isclass(cls)
            or cls in _OPAQUE
        ):
            return None
        key = f"{cls.__module__}.{cls.__qualname__}"
        if self._depth.get(key, -1) >= depth:
            return key
        self._depth[key] = depth
        self.types[key] = {"name": cls.__qualname__, "doc": _doc(cls), "attrs": []}
        if depth > 0:
            self.types[key].update(self._members(cls, depth - 1))
        else:
            # Named, not described: the editor asks for it when it gets there.
            self.types[key]["partial"] = True
        return key

    def function(self, name: str, fn: Callable[..., Any], depth: int) -> dict[str, Any]:
        """A callable as a member: its parameters, return type and doc."""
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return _member(name, "function", detail=f"{name}(...)", doc=_doc(fn))
        fn_hints = hints(fn)
        params = []
        keyed = None
        for param in signature.parameters.values():
            if param.name in ("self", "cls") or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            annotation = fn_hints.get(param.name, param.annotation)
            base, marker = _unwrap(annotation)
            entry = {
                "name": param.name,
                "detail": label(annotation),
                "type": self.ref(base, depth),
            }
            if param.default is not param.empty:
                entry["default"] = repr(param.default)
            if marker is not None:
                entry["keys"] = keyed = self.design(marker, depth)
            params.append(entry)
        returns = fn_hints.get("return", signature.return_annotation)
        member = _member(
            name,
            "function",
            detail=f"{name}({', '.join(_param_text(p) for p in params)})"
            + (f" -> {label(returns)}" if label(returns) else ""),
            doc=_doc(fn),
            type=self.ref(returns, depth),
            params=params,
        )
        # A keyed argument picks the result when the return type does not say.
        if keyed is not None and _unwrap(returns)[0] in (
            Any,
            None,
            inspect.Signature.empty,
        ):
            member["returns_by_key"] = keyed
        return member

    def _members(self, cls: type, depth: int) -> dict[str, Any]:
        cls_hints = hints(cls)
        attrs: dict[str, dict[str, Any]] = {}
        items = None
        if _is_typed_dict(cls):
            items = [
                _member(n, "field", detail=label(a), type=self.ref(a, depth))
                for n, a in cls_hints.items()
            ]
        else:
            for name, annotation in cls_hints.items():
                if (
                    not name.startswith("_")
                    and get_origin(annotation) is not typing.ClassVar
                ):
                    attrs[name] = _member(
                        name,
                        "property",
                        detail=label(annotation),
                        type=self.ref(annotation, depth),
                    )
        for name in dir(cls):
            if name.startswith("_") or name in attrs:
                continue
            try:
                raw = inspect.getattr_static(cls, name)
            except AttributeError:
                continue
            if isinstance(raw, (property, functools.cached_property)):
                fget = raw.fget if isinstance(raw, property) else raw.func
                returns = hints(fget).get("return") if fget else None
                attrs[name] = _member(
                    name,
                    "property",
                    detail=label(returns),
                    doc=_doc(raw),
                    type=self.ref(returns, depth),
                )
            elif isinstance(raw, (staticmethod, classmethod)) or inspect.isroutine(raw):
                attrs[name] = self.function(name, getattr(cls, name), depth)
            elif not inspect.isclass(raw):
                attrs[name] = _member(name, "property", detail=type(raw).__name__)
        described: dict[str, Any] = {"attrs": sorted(attrs.values(), key=_by_name)}
        if items is not None:
            described["items"] = items
        return described

    # --- the design's keys ----------------------------------------------------
    def design(self, marker: DesignKeys, depth: int) -> str:
        """The key of a synthetic type whose items are the design's keys."""
        suffix = ":batched" if marker.batched else ""
        key = f"design:{marker.source}{suffix}"
        # Without a design (a type described on its own) the keys are the
        # design's, already described with it.
        if key not in self.types and self._config is not None:
            keys = design_keys(marker, self._config, self._support)
            self._closed(key, marker.source, keys, depth)
        return key

    def _closed(self, key: str, name: str, keys: list[Key], depth: int) -> None:
        """A type whose items are ``keys`` - closed: the design has no others."""
        self.types[key] = {"name": name, "doc": "", "attrs": [], "closed": True}
        items = []
        for k in keys:
            if k.keys is not None:
                inner = f"{key}:{k.name}"
                self._closed(inner, k.name, k.keys, depth)
                item = _member(k.name, "field", detail=k.detail, type=inner)
            else:
                item = _member(
                    k.name,
                    "field",
                    detail=k.detail,
                    doc=k.doc,
                    type=self.ref(k.value, depth),
                )
            if k.color:
                item["color"] = k.color
            items.append(item)
        self.types[key]["items"] = items


def _is_typed_dict(cls: type) -> bool:
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__total__")


def _member(name: str, kind: str, **info: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        **{k: v for k, v in info.items() if v not in (None, "")},
    }


def _param_text(param: dict[str, Any]) -> str:
    text = param["name"]
    if param.get("detail"):
        text += f": {param['detail']}"
    if "default" in param:
        text += f" = {param['default']}"
    return text


def _by_name(member: dict[str, Any]) -> str:
    return member["name"]


# --------------------------------------------------------------------------- #
# Scopes                                                                      #
# --------------------------------------------------------------------------- #
def _scopes(
    spec: DesignSpec,
    config: EnvConfig,
    table: TypeTable,
    names: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Each code block's parameters, and the key of its module's names."""
    env = spec.env
    module = run_setup_module(env, config)
    setup_source = setup_code.setup_source(
        env.task_info_setup, env.task_info, env.hook_setup
    )
    names["setup"] = _module_names(
        module, ["CONFIG", *sorted(setup_code.setup_names(setup_source))], table
    )
    setup = "setup"
    task = BlueskyEnv
    scopes: dict[str, Any] = {
        "hook_setup": {"params": [], "names": setup},
        "task_info_setup": {"params": [], "names": setup},
        "task_info": {
            "params": _params(TaskInfoProvider.__call__, table),
            "names": setup,
        },
    }
    # A hook body sees the setup's names and the library types the hooks name.
    api = setup_code.hook_api_names()
    names["hook"] = names["setup"] + _module_names(
        importlib.import_module("bluesky_sandbox"), sorted(set(api) - {m["name"] for m in names["setup"]}), table
    )
    for name, hook in _hooks(task):
        params = _params(hook, table, clean=True)
        params.insert(
            0, _member("self", "parameter", detail=task.__name__, type=table.ref(task))
        )
        scopes[f"hook:{name}"] = {"params": params, "names": "hook"}
    # Scenario code: its setup's names, and the library names it has in scope
    # (setup_code.SCENARIO_API); each hook's parameters typed.
    defined = {m["name"] for m in _source_names(spec.scenario_setup or "")}
    names["scenario"] = _source_names(spec.scenario_setup or "") + [
        _describe_value(name, value, table)
        for name, value in sorted(setup_code.scenario_api_names().items())
        if name not in defined
    ]
    scenario_names = "scenario"
    scopes["scenario_setup"] = {"params": [], "names": scenario_names}
    for name, (args, *_rest) in SCENARIO_HOOKS.items():
        signature = scenario_api.SIGNATURES.get(name)
        params = _params(signature, table) if signature else [_member(arg, "parameter") for arg in args]
        scopes[f"scenario:{name}"] = {"params": params, "names": scenario_names}
    for filename, source in (spec.code or {}).items():
        stem = filename.removesuffix(".py")
        code_module = sys.modules.get(stem)
        defined = sorted(setup_code.setup_names(source)) if source.strip() else []
        names[f"code:{filename}"] = (
            _module_names(code_module, defined, table) if code_module else []
        )
        scopes[f"code:{filename}"] = {"params": [], "names": f"code:{filename}"}
    return scopes


def _hooks(task: type) -> Iterable[tuple[str, Callable[..., Any]]]:
    """The task's overridable hooks, by name."""
    seen = set()
    for cls in inspect.getmro(task):
        for name, fn in vars(cls).items():
            if getattr(fn, "__overridable__", False) and name not in seen:
                seen.add(name)
                yield name, fn


def _params(
    fn: Callable[..., Any], table: TypeTable, clean: bool = False
) -> list[dict[str, Any]]:
    """``fn``'s parameters as a scope's names - hooks name theirs without the
    leading underscore that marks them unused in the base class."""
    params = table.function("", fn, _DEPTH)["params"] if fn else []
    out = []
    for param in params:
        name = param["name"].removeprefix("_") if clean else param["name"]
        out.append({**param, "name": name, "kind": "parameter"})
    return out


def _module_names(
    module: Any, names: list[str], table: TypeTable
) -> list[dict[str, Any]]:
    """Members for ``names`` in ``module``, described from the objects themselves."""
    out = []
    namespace = vars(module) if module is not None else {}
    for name in names:
        if name not in namespace:
            continue
        out.append(_describe_value(name, namespace[name], table))
    return out


def _describe_value(name: str, value: Any, table: TypeTable) -> dict[str, Any]:
    if inspect.ismodule(value):
        return _member(
            name,
            "module",
            detail=value.__name__,
            module=value.__name__,
            doc=_doc(value),
        )
    if inspect.isclass(value):
        member = table.function(name, value, _DEPTH)
        return {
            **member,
            "kind": "class",
            "type": table.ref(value),
            "returns": table.ref(value),
        }
    if inspect.isroutine(value):
        return table.function(name, value, _DEPTH)
    return _member(
        name,
        "variable",
        detail=type(value).__qualname__,
        type=table.ref(type(value)),
        doc=_doc(type(value)) if not isinstance(value, _OPAQUE) else "",
    )


def _source_names(source: str) -> list[dict[str, Any]]:
    """Names a source defines, without running it."""
    try:
        names = setup_code.setup_names(source)
    except SyntaxError:
        return []
    return [_member(name, "variable") for name in sorted(names)]
