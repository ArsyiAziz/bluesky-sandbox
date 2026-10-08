"""What a design file may hold, as a JSON Schema - so any JSON editor completes
and checks a ``design.json``, the designer's own included.

Read from the library, not written by hand: the fields, actions, normalizers,
footprint and queryable kinds it offers are the catalog's, and the environment
settings :class:`~.spec.EnvSpec`'s - so a new field or setting is in the schema
with nothing else to update. It describes both forms of a design: one ``.json``
file, its code inline; and a folder's ``design.json``, its code in ``code/``
(see :mod:`.folder`) - the code's places then naming what ``code/`` holds.
"""

from __future__ import annotations

import dataclasses
import types
from functools import lru_cache
from typing import Any, Union, get_args, get_origin, get_type_hints

from .spec import SCENARIO_HOOKS, EnvSpec

__all__ = ["design_schema"]

_SCHEMA_ID = "https://bluesky-sandbox.dev/schemas/design.schema.json"
_FOOTPRINTS = ["box", "disk", "point", "polygon", "sector", "annular_sector", "boolean", "generated"]


def _json_type(hint: Any) -> dict[str, Any]:
    """A setting's JSON type, from its annotation."""
    if hasattr(hint, "__metadata__"):
        hint = get_args(hint)[0]
    if get_origin(hint) in (Union, types.UnionType):
        options = [_json_type(a) for a in get_args(hint)]
        kinds = [o["type"] for o in options if "type" in o]
        flat = [k for kind in kinds for k in (kind if isinstance(kind, list) else [kind])]
        return {"type": sorted(set(flat))} if len(kinds) == len(options) else {}
    return {
        bool: {"type": "boolean"},
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        type(None): {"type": "null"},
    }.get(hint, {})


def _names_or(code: dict[str, Any], what: str) -> dict[str, Any]:
    """Inline code, or - in a design folder - the names of what ``code/`` holds."""
    return {
        "anyOf": [
            code,
            {"type": "array", "items": {"type": "string"}, "description": f"the {what} in code/, by name"},
        ]
    }


def _dataclass_object(cls: type, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """``cls``'s fields as a JSON object: each typed from its annotation,
    unless ``overrides`` says otherwise; required where it has no default."""
    hints = get_type_hints(cls)
    fields = [f for f in dataclasses.fields(cls) if f.name != "_"]
    overrides = overrides or {}
    out: dict[str, Any] = {
        "type": "object",
        "properties": {f.name: overrides.get(f.name, _json_type(hints[f.name])) for f in fields},
        "additionalProperties": False,
    }
    required = [
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if required:
        out["required"] = required
    return out


def _tests_schema() -> dict[str, Any]:
    """A design's test cases (see :mod:`.design_tests`), from the
    :mod:`bluesky_sandbox.checks` classes they become."""
    from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance  # noqa: PLC0415

    number_or_numbers = {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}}]}
    aircraft = _dataclass_object(Aircraft)
    # Placed exactly one way (Aircraft.PLACEMENTS).
    aircraft["oneOf"] = [{"required": list(group)} for group in Aircraft.PLACEMENTS]
    situation = _dataclass_object(Situation, {"aircraft": {"type": "array", "items": aircraft}})
    case = _dataclass_object(
        Case,
        {
            # An observation, or an action - read for what it holds.
            "field": {"anyOf": [{"$ref": "#/$defs/obs_ref"}, {"$ref": "#/$defs/action_ref"}]},
            "apply": {"$ref": "#/$defs/action_ref"},
            "expected": number_or_numbers,
            "tolerance": _dataclass_object(Tolerance),
        },
    )
    return {
        "type": "object",
        "properties": {
            "situations": {"type": "array", "items": situation},
            "cases": {"type": "array", "items": case},
        },
        "additionalProperties": False,
    }


@lru_cache(maxsize=1)
def design_schema() -> dict[str, Any]:
    """The design file's JSON Schema (draft 2020-12)."""
    from . import catalog  # noqa: PLC0415 - the catalog is heavy

    obs = sorted(f["name"] for f in catalog.obs_fields())
    actions = sorted(f["name"] for f in catalog.action_fields())
    normalizers = sorted(n["name"] for n in catalog.normalizers())
    hooks = sorted(h["name"] for h in catalog.hooks())

    normalizer = {
        "type": "object",
        "required": ["type", "name"],
        "properties": {
            "type": {"const": "normalizer"},
            "name": {"enum": normalizers},
            "kwargs": {"type": "object"},
        },
        "additionalProperties": False,
    }
    grid = {
        "type": "object",
        "required": ["type", "step"],
        "properties": {
            "type": {"const": "grid"},
            "step": {"type": "number", "exclusiveMinimum": 0},
            "on": {"enum": ["value", "target"]},
        },
        "additionalProperties": False,
    }
    crossover = {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"const": "crossover"},
            "cas_kts": {"type": "number", "exclusiveMinimum": 0},
            "mach": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "margin_ft": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    mach_regime = {
        "type": "object",
        "required": ["type"],
        "properties": {
            "type": {"const": "mach_regime"},
            "crossover": {"$ref": "#/$defs/crossover"},
            "low": {"type": ["number", "null"]},
            "high": {"type": ["number", "null"]},
            "normalizer": {"anyOf": [{"$ref": "#/$defs/normalizer"}, {"type": "null"}]},
            "grid": {"anyOf": [{"$ref": "#/$defs/grid"}, {"type": "null"}]},
            "handover": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

    def field_ref(names: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["field"],
            "properties": {
                "field": {
                    "anyOf": [
                        {"enum": names},
                        {"type": "string", "pattern": r"^[\w.]+:[A-Za-z_]\w*$", "description": "module:Class"},
                    ]
                },
                "kwargs": {
                    "type": "object",
                    "properties": {
                        "normalizer": {"$ref": "#/$defs/normalizer"},
                        "grid": {"$ref": "#/$defs/grid"},
                        "crossover": {"$ref": "#/$defs/crossover"},
                        "above_crossover": {"$ref": "#/$defs/mach_regime"},
                    },
                },
                "transform": {"type": "string"},
                "transform_kwargs": {"type": "object"},
                "clearance": {"type": ["object", "null"]},
            },
            "additionalProperties": False,
        }

    def field_list(ref: str) -> dict[str, Any]:
        return {"type": "array", "items": {"$ref": ref}}

    hints = get_type_hints(EnvSpec)
    env_properties: dict[str, Any] = {}
    for f in dataclasses.fields(EnvSpec):
        if f.name.endswith("_fields"):
            ref = "#/$defs/action_ref" if f.name == "action_fields" else "#/$defs/obs_ref"
            listed = field_list(ref)
            env_properties[f.name] = listed if f.name in ("obs_fields", "action_fields") else {
                "anyOf": [listed, {"type": "null"}]
            }
        else:
            env_properties[f.name] = _json_type(hints.get(f.name))
    env_properties["allowed_aircraft"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": "an earlier design's types: read onto each spawn region",
    }
    env_properties["task_info_providers"] = {"type": "array", "items": {"type": "string"}}
    env_properties["hooks"] = _names_or(
        {
            "type": "object",
            "propertyNames": {"enum": hooks},
            "additionalProperties": {"type": "string"},
        },
        "hooks",
    )
    env_properties["task_info"] = _names_or(
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "body": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        "task-info entries",
    )
    for setup in ("hook_setup", "task_info_setup"):
        env_properties[setup] = {"type": "string"}
    env_properties["reward_fn"] = {"description": "an earlier design's reward reference; read as hooks"}

    source = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string", "pattern": r"^[A-Za-z_]\w*$"},
            "plan": {"type": "string", "description": "the body of plan(rng, ctx) -> list[SpawnRequest]"},
            "aircraft_type": {"description": "the types its requests naming none fly: an ICAO type, or a categorical"},
            "conflict_free": {"type": "boolean"},
            "when_blocked": {"enum": ["resample", "defer", "skip", "allow"]},
            "route": {"type": ["string", "null"]},
            "assign_route": {"enum": ["nearest_entry", None]},
            "max_aircraft": {"type": ["integer", "null"], "minimum": 0},
        },
        "additionalProperties": False,
    }
    element = {
        "type": "object",
        "properties": {"type": {"type": "string"}},
    }
    region = {
        "type": "object",
        "properties": {
            "type": {"const": "region"},
            "footprint": {
                "type": "object",
                "required": ["type"],
                "properties": {"type": {"enum": _FOOTPRINTS}},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _SCHEMA_ID,
        "title": "BlueSky Sandbox design",
        "type": "object",
        "required": ["env", "spawn"],
        "properties": {
            "$schema": {"type": "string"},
            "version": {"type": "integer"},
            "env": {
                "type": "object",
                "properties": env_properties,
                "additionalProperties": False,
            },
            "spawn": {
                "type": "object",
                "properties": {
                    "type": {"const": "spawn_config"},
                    "regions": {"type": "array", "items": {"type": "object"}},
                    "routes": {"type": "object"},
                    "sources": {"type": "array", "items": {"$ref": "#/$defs/spawn_source"}},
                    "aircraft_cap": {"type": ["integer", "null"], "minimum": 1},
                    "spawn_max_tries": {"type": ["integer", "null"], "minimum": 1},
                    "spawn_warn_after": {"type": ["integer", "null"], "minimum": 1},
                    "conflict_free_spawn": {"type": ["boolean", "null"]},
                },
            },
            "airspace": {"type": ["object", "null"]},
            "shapes": {"type": "object", "additionalProperties": {"$ref": "#/$defs/region"}},
            "regions": {"type": "object", "description": "an earlier design's shapes"},
            "queryables": {"type": "object", "additionalProperties": element},
            "transform": {"type": ["object", "null"]},
            "code": _names_or(
                {"type": "object", "additionalProperties": {"type": "string"}}, "modules"
            ),
            "scenario_setup": {"type": "string"},
            "scenario_hooks": _names_or(
                {
                    "type": "object",
                    "propertyNames": {"enum": sorted(SCENARIO_HOOKS)},
                    "additionalProperties": {"type": "string"},
                },
                "scenario hooks",
            ),
            "tests": _tests_schema(),
            "nav_cycle": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
        },
        "additionalProperties": False,
        "$defs": {
            "normalizer": normalizer,
            "grid": grid,
            "crossover": crossover,
            "mach_regime": mach_regime,
            "obs_ref": field_ref(obs),
            "action_ref": field_ref(actions),
            "spawn_source": source,
            "region": region,
        },
    }
