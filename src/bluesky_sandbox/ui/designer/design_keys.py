"""The keys a design fixes, for what reads a ``DesignKeys`` annotation.

A ``DesignKeys`` marker says where a value's keys come from - the observation's
parts and fields, the action fields, the queryables. :func:`design_keys` lists
them for one design: each key's name, the type of its value, a one-line detail
and a doc. The code editor (``code_intel``) and the generated package's types
(``typed_api``) are both built from this, so they cannot disagree.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.layout import observation_parts
from bluesky_sandbox.core.step_values import ACID, unique_names_of
from bluesky_sandbox.interface.fields.base import ActionKind
from bluesky_sandbox.interface.task import DesignKeys

__all__ = ["Key", "design_keys"]


@dataclass(frozen=True)
class Key:
    """One key: ``value`` is the type of what it holds - ``float``,
    ``np.ndarray``, a queryable or result class - or ``None`` when ``keys``
    holds its own keys, as an observation part does."""

    name: str
    value: Any
    detail: str = ""
    doc: str = ""
    color: str | None = None
    keys: list[Key] | None = field(default=None)


def design_keys(marker: DesignKeys, config: EnvConfig, support: Any) -> list[Key]:
    """The keys ``marker`` stands for in this design."""
    build = _SOURCES.get(marker.source)
    if build is None:
        raise ValueError(f"no DesignKeys source {marker.source!r} is described")
    return build(marker.batched, config, support)


def _observation(batched: bool, config: EnvConfig, _support: Any) -> list[Key]:
    parts = []
    for part, fields in observation_parts(config).items():
        per_intruder = part.endswith("intruders")
        keys = [
            _field_key(name, f, batched, per_intruder)
            for name, f in zip(unique_names_of(fields), fields)
        ]
        if per_intruder:
            shape = _shape(batched, True, 1)
            keys.insert(
                0,
                Key(ACID, np.ndarray, f"{shape} str", "Each intruder row's callsign."),
            )
        parts.append(Key(part, None, f"{len(fields)} fields", keys=keys))
    return parts


def _action(batched: bool, config: EnvConfig, _support: Any) -> list[Key]:
    fields = list(config.action_fields)
    keys = []
    for name, f in zip(unique_names_of(fields), fields):
        values = "0 or 1" if f.kind is ActionKind.BINARY else _unit(f)
        shape = "ndarray (n_agents,)" if batched else "float"
        keys.append(
            Key(
                name,
                np.ndarray if batched else float,
                " · ".join(p for p in (shape, values) if p),
                _field_doc(f),
            )
        )
    return keys


def _queryable(batched: bool, _config: EnvConfig, support: Any) -> list[Key]:
    return _queryables(support, type)


def _queryable_result(batched: bool, _config: EnvConfig, support: Any) -> list[Key]:
    return _queryables(support, lambda q: getattr(q, "result_type", None))


_SOURCES = {
    "observation": _observation,
    "action": _action,
    "queryable": _queryable,
    "queryable_result": _queryable_result,
}


def _queryables(support: Any, value: Any) -> list[Key]:
    keys = []
    for name, queryable in support.queryables.items():
        result = value(queryable)
        keys.append(
            Key(
                name,
                result,
                getattr(result, "__qualname__", ""),
                _doc(type(queryable)),
                color=getattr(queryable, "color", None) or None,
            )
        )
    return keys


def _field_key(name: str, f: Any, batched: bool, per_intruder: bool) -> Key:
    width = f.output_size() if callable(getattr(f, "output_size", None)) else 1
    shape = _shape(batched, per_intruder, width)
    array = batched or per_intruder or width > 1
    return Key(
        name,
        np.ndarray if array else float,
        " · ".join(p for p in (shape, _unit(f)) if p),
        _field_doc(f),
    )


def _shape(batched: bool, per_intruder: bool, width: int) -> str:
    axes = (["n_agents"] if batched else []) + (["n_intruders"] if per_intruder else [])
    axes += [str(width)] if width > 1 else []
    if not axes:
        return "float"
    return f"ndarray ({', '.join(axes)}{',' if len(axes) == 1 else ''})"


def _unit(f: Any) -> str:
    unit = str(getattr(getattr(f, "meta", None), "unit", "") or "")
    return "" if unit in ("", "unitless") else unit


def _field_doc(f: Any) -> str:
    doc = f"`{type(f).__name__}`: {_doc(type(f))}"
    normalizer = getattr(f, "normalizer", None)
    if normalizer is not None:
        observed = type(normalizer).__name__
        doc += f"\n\nObserved through `{observed}`; this is the raw value."
    return doc


def _doc(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n", 1)[0].strip()
