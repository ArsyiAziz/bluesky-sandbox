from . import actions, observations
from .base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    EnvObsField,
    EnvPairObsField,
    ObsField,
    ObsMeta,
    ObsQuantity,
    PairObsField,
    QueryableFieldCardinality,
    QueryableFieldRequirement,
    QueryableFieldSpec,
    QueryableKind,
    SwitchActionMixin,
    TaskContextObsField,
    TaskContextPairObsField,
    Unit,
)


def __getattr__(name: str):
    if name == "queryables":
        from importlib import import_module

        module = import_module(f"{__name__}.queryables")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ActionField",
    "ActionMeta",
    "ActionMode",
    "ControlAxis",
    "EnvObsField",
    "EnvPairObsField",
    "ObsField",
    "ObsMeta",
    "ObsQuantity",
    "PairObsField",
    "QueryableFieldCardinality",
    "QueryableFieldRequirement",
    "QueryableFieldSpec",
    "QueryableKind",
    "SwitchActionMixin",
    "TaskContextObsField",
    "TaskContextPairObsField",
    "Unit",
    "actions",
    "observations",
    "queryables",
]
