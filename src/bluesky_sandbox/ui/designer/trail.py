"""What a field computes with: the functions its value is built from.

A field's class source is often a single call - ``DistToOwnNm._pairs`` is one
line into ``_pair_qdr_dist`` - so the designer shows, beneath it, the trail of
calls that produce the value: every function and class of this package the
field's value methods call, recursively, in call order, each with its source.
BlueSky functions end the trail as named leaves.

Derived from the code, so it cannot drift: rename a helper and its entry
follows. What static analysis cannot follow is linked explicitly (see
:data:`_DYNAMIC`), and plumbing a reader does not need is left out (see
:data:`_SKIP`).
"""

from __future__ import annotations

import ast
import functools
import inspect
import pathlib
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import bluesky_sandbox
from bluesky_sandbox.core.services import QueryBatch
from bluesky_sandbox.interface.fields import _common, _pairs, base
from bluesky_sandbox.interface.fields.observations import kinematics as _kinematics
from bluesky_sandbox.interface.fields.observations import queryable as _queryable_fields

__all__ = ["call_trail"]

# The methods that produce a field's value or bounds - where a trail starts.
_ENTRY = (
    "_si_values",
    "_values",
    "_pairs",
    "get",
    "get_many",
    "get_pair",
    "get_pairs",
    "get_pair_matrix",
    "bounds",
    "set",
)

# Framework classes whose methods are wiring, not the computation.
_PLUMBING_CLASSES = (
    object,
    base._BoundedField,
    base.ObsField,
    base.PairObsField,
    base.ActionField,
    base.EnvObsField,
    base.EnvPairObsField,
    _common._BroadcastObs,
    _pairs._BroadcastPairs,
    _kinematics._UnitField,
)

# Helpers too small or too generic to explain anything: index coercion,
# array reads, configured-bounds lookups, inner-field accessors.
_SKIP = frozenset(
    {
        "_indices_array",
        "_traf_array",
        "_per_own",
        "_per_aircraft",
        "_pair_index_grid",
        "_configured_bounds",
        "_dynamic_or_configured_bounds",
        "_validate_bound_policy",
        "_field",
        "_fields",
        "_clip",
        "_unavailable",
        "_convert",
    }
)

_MAX_DEPTH = 5
_PACKAGE_DIR = pathlib.Path(bluesky_sandbox.__file__).parent


def _queryable_batch_methods(cls: type) -> list[Callable]:
    """The :class:`QueryBatch` methods that compute a queryable field's value.

    Queryable fields read ``env.query_batch(name, ...).values(path)``; the
    batch is reached through the env, which no call graph can see, so the
    field's ``queryable_spec`` names the path and this maps it to the method.
    """
    spec = getattr(cls, "queryable_spec", None)
    if spec is None:
        return []
    paths = spec.path.split("|")
    if issubclass(cls, _queryable_fields.ActiveWaypointObsField):
        paths += ["route.active", "route.future"]
    if issubclass(cls, _queryable_fields._WaypointRouteFlag):
        paths = [f"route.{cls.flag_name}"]
    region = spec.kind.value == "region"
    methods: list[Callable] = []
    for path in paths:
        group = path.split(".", 1)[0]
        if region:
            method = QueryBatch._region
        elif group == "current":
            method = QueryBatch._waypoint_current
        elif group == "route":
            method = QueryBatch._waypoint_route
        else:
            method = QueryBatch._waypoint
        if method not in methods:
            methods.append(method)
    if any(p.split(".", 1)[0] in ("step", "time") for p in paths):
        methods.append(QueryBatch._table_cells)
    return methods


# Calls no static analysis can follow, and where they really lead.
_DYNAMIC: dict[Any, Callable[[type], list[Callable]]] = {
    _queryable_fields.QueryableObsField._column: _queryable_batch_methods,
}


def _location(obj: Any) -> str:
    try:
        path = pathlib.Path(inspect.getsourcefile(obj) or "")
        line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return getattr(obj, "__module__", "")
    try:
        path = path.relative_to(_PACKAGE_DIR)
    except ValueError:
        pass
    return f"{path.as_posix()}:{line}"


def _source(obj: Any) -> str | None:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return None


def _calls(fn: Callable) -> list[tuple[str, str]]:
    """``(kind, name)`` of each call in ``fn``, in source order: ``"global"``
    for a bare name, ``"self"`` for a method on ``self``."""
    source = _source(fn)
    if source is None:
        return []
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.append((node.lineno, node.col_offset, "global", func.id))
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            found.append((node.lineno, node.col_offset, "self", func.attr))
    return [(kind, name) for _line, _col, kind, name in sorted(found)]


def _is_ours(obj: Any) -> bool:
    return getattr(obj, "__module__", "").startswith("bluesky_sandbox")


def _is_computation_class(obj: type) -> bool:
    """A class worth showing: one that computes, not a record of results."""
    return any(
        inspect.isfunction(value)
        and name not in ("__init__", "__post_init__", "__repr__")
        for name, value in vars(obj).items()
    )


def _defining_class(cls: type, name: str) -> tuple[type, Any] | None:
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass, vars(klass)[name]
    return None


def _unit_mixin(cls: type) -> type | None:
    """The mixin that sets a unit field's SI-to-unit ``_scale``, if any."""
    for klass in cls.__mro__:
        if "_scale" in vars(klass) and klass not in _PLUMBING_CLASSES:
            return klass
    return None


def call_trail(cls: type) -> list[dict[str, Any]]:
    """The functions ``cls``'s value is computed with - see :func:`_call_trail`."""
    return [dict(entry) for entry in _call_trail(cls)]


@functools.cache
def _call_trail(cls: type) -> tuple[dict[str, Any], ...]:
    """The functions ``cls``'s value is computed with, in call order.

    Each entry: ``name`` (``Owner.method`` or ``module.function``),
    ``location`` (``path:line`` within the package, or the BlueSky module),
    ``depth`` (call nesting below the field), ``source`` (``None`` for a
    BlueSky function) and ``external`` (true for BlueSky).
    """
    trail: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(obj: Any, name: str, depth: int, *, external: bool = False) -> None:
        trail.append(
            {
                "name": name,
                "location": obj.__module__ if external else _location(obj),
                "depth": depth,
                "source": None if external else _source(obj),
                "external": external,
            }
        )

    def visit(fn: Callable, self_cls: type, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        dynamic = _DYNAMIC.get(fn)
        for kind, name in _calls(fn):
            if name in _SKIP:
                continue
            if kind == "self":
                found = _defining_class(self_cls, name)
                if found is None:
                    continue
                owner, target = found
                if not inspect.isfunction(target) or owner in _PLUMBING_CLASSES:
                    continue
                if id(target) in seen:
                    continue
                seen.add(id(target))
                if owner is cls:
                    visit(target, self_cls, depth)  # already in the class source
                    continue
                add(target, f"{owner.__name__}.{name}", depth)
                visit(target, self_cls, depth + 1)
                continue
            target = sys.modules[fn.__module__].__dict__.get(name)
            if target is None or id(target) in seen:
                continue
            if not _is_ours(target):
                module = getattr(target, "__module__", "") or ""
                if callable(target) and module.startswith("bluesky."):
                    seen.add(id(target))
                    add(
                        target,
                        f"{module.rsplit('.', 1)[-1]}.{name}",
                        depth,
                        external=True,
                    )
                continue
            if inspect.isclass(target):
                if not _is_computation_class(target) or issubclass(
                    target, _PLUMBING_CLASSES[1:]
                ):
                    continue
                seen.add(id(target))
                add(target, target.__name__, depth)
                for method in vars(target).values():
                    if inspect.isfunction(method):
                        visit(method, target, depth + 1)
                continue
            if inspect.isfunction(target):
                seen.add(id(target))
                add(target, f"{target.__module__.rsplit('.', 1)[-1]}.{name}", depth)
                visit(target, self_cls, depth + 1)
        if dynamic is not None:
            for method in dynamic(cls):
                if id(method) in seen:
                    continue
                seen.add(id(method))
                owner = method.__qualname__.split(".")[0]
                add(method, f"{owner}.{method.__name__}", depth)
                visit(method, QueryBatch, depth + 1)

    # A unit variant reports its SI quantity scaled by its unit mixin - the
    # mixin is read (``self._scale``), never called, so name it explicitly.
    unit = _unit_mixin(cls)
    if unit is not None:
        seen.add(id(unit))
        add(unit, unit.__name__, 0)

    for name in _ENTRY:
        found = _defining_class(cls, name)
        if found is None:
            continue
        owner, method = found
        if not inspect.isfunction(method) or owner in _PLUMBING_CLASSES:
            continue
        if id(method) in seen:
            continue
        seen.add(id(method))
        if owner is not cls:
            add(method, f"{owner.__name__}.{name}", 0)
            visit(method, cls, 1)
        else:
            visit(method, cls, 0)
    # Cached per class: source does not change within a process.
    return tuple(trail)
