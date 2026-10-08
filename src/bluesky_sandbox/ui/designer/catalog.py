"""Palette catalogs: what primitives the GUI can offer and their parameters.

Introspects the simulation primitives so the map tab can present footprint /
altitude-band / queryable builders, and the field pickers can list available
observation and action fields, without hard-coding a parallel list that drifts
from the code.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import re
import sys
import textwrap
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from bluesky.tools.aero import ft, nm
from scipy import stats as ss

from bluesky_sandbox.config import _available_aircraft
from bluesky_sandbox.env import BATCHABLE_HOOKS, BlueskyEnv
from bluesky_sandbox.interface.fields import actions as _actions
from bluesky_sandbox.interface.fields import observations as _observations
from bluesky_sandbox.config import bluesky_simdt_s
from bluesky_sandbox.interface.fields.base import ActionField, ObsField, PairObsField
from bluesky_sandbox.interface.wrappers.observations import normalizer as _normalizers
from bluesky_sandbox.sim import bounds as _bounds
from bluesky_sandbox.sim.geometry.conflict import (
    bluesky_asas_dt_s,
    cd_hpz_m,
    cd_lookahead_s,
    cd_rpz_m,
)
from bluesky_sandbox.sim.performance.models import MODELS, spawnable_types, type_info
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.spawn import SpawnConfig

from .emit import _normalizer_import_line
from .spec import DEFAULT_HOOKS, SCENARIO_HOOKS
from .trail import call_trail


def _doc(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    return doc.splitlines()[0] if doc else ""


def _paragraph(obj: Any) -> str:
    """The first paragraph of ``obj``'s docstring, on one line: a sentence
    wrapped over several lines is not cut at the first."""
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _dataclass_params(cls) -> list[dict[str, Any]]:
    """Describe a dataclass's constructor params: name, type, default."""
    out: list[dict[str, Any]] = []
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        default = f.default if f.default is not dataclasses.MISSING else None
        out.append(
            {
                "name": f.name,
                "type": _type_name(f.type),
                "required": not has_default,
                "default": default if isinstance(default, (int, float, str, bool)) else None,
            }
        )
    return out


def _type_name(t: Any) -> str:
    if isinstance(t, str):
        return t
    if hasattr(t, "__metadata__"):  # Annotated[T, ...]: T's name
        t = get_args(t)[0]
    return getattr(t, "__name__", str(t))


def _concrete_subclasses(module, base) -> list[type]:
    """The usable subclasses of ``base`` in ``module``: concrete, and - for a
    field - one that says what it is. A base class a family of fields shares
    (``QueryableObsField``) defines no ``meta``; every field does."""
    out = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isclass(obj) and issubclass(obj, base) and obj is not base):
            continue
        if inspect.isabstract(obj):
            continue
        is_field = issubclass(obj, (ObsField, PairObsField, ActionField))
        if is_field and not any("meta" in vars(cls) for cls in obj.__mro__):
            continue
        out.append(obj)
    return out


def _param_kind(hint: Any) -> str | None:
    """What input a generator's or placement's param takes, from its
    annotation: ``latlon`` (a point), ``points`` (several), ``region`` (a named
    region), ``regions`` (several), ``value`` (a number, range or
    distribution), ``int``, ``float``, ``bool`` - else None (not offered)."""
    text = str(hint)
    if "Sequence[" in text and "LatLon" in text:
        return "points"
    if "Sequence[" in text and ("Bounds" in text or "Footprint" in text):
        return "regions"
    if "LatLon" in text:
        return "latlon"
    if "Bounds" in text or "Footprint" in text:
        return "region"
    if text.startswith(("Value", "Any", "typing.Any")):
        return "value"
    if text in ("bool", "<class 'bool'>"):
        return "bool"
    if text in ("int", "<class 'int'>"):
        return "int"
    if text in ("float", "float | None", "<class 'float'>"):
        return "float"
    return None


def _params_of(cls: type) -> list[dict[str, Any]]:
    """A generator's or placement's params: each one's name, kind, default."""
    params = []
    for f in dataclasses.fields(cls):
        kind = _param_kind(f.type)
        if not f.init or kind is None:
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        if isinstance(default, _bounds.LatLon):
            default = {"lat_deg": default.lat_deg, "lon_deg": default.lon_deg}
        elif not isinstance(default, (int, float, str, bool, type(None))):
            default = None
        params.append({"name": f.name, "kind": kind, "default": default})
    return params


def _summary(cls: type) -> str:
    """A docstring's first paragraph, as one line."""
    doc = inspect.getdoc(cls) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _concrete(module: Any, base: type) -> list[tuple[str, type]]:
    return [
        (name, cls)
        for name, cls in sorted(vars(module).items())
        if inspect.isclass(cls) and issubclass(cls, base) and not inspect.isabstract(cls)
    ]


def generators() -> list[dict[str, Any]]:
    """The shape generators: each one's class name, doc, and params (see
    :func:`_param_kind`); a partition's ``partition`` names the param its
    number of shapes is set by."""
    from bluesky_sandbox.sim.bounds import generators as _gen  # noqa: PLC0415

    return [
        {"name": name, "doc": _summary(cls), "params": _params_of(cls), "partition": cls.partition}
        for name, cls in _concrete(_gen, _gen.ShapeGenerator)
    ]


def placements() -> list[dict[str, Any]]:
    """The placements - where a region sits each episode: each one's class
    name, doc, and params (see :func:`_param_kind`)."""
    from bluesky_sandbox.sim.bounds import placement as _place  # noqa: PLC0415

    return [
        {"name": name, "doc": _summary(cls), "params": _params_of(cls)}
        for name, cls in _concrete(_place, _place.Placement)
    ]


def motions() -> list[dict[str, Any]]:
    """The motions - how a region moves during an episode: each one's class
    name, doc, and params (see :func:`_param_kind`)."""
    from bluesky_sandbox.sim.bounds import motion as _motion  # noqa: PLC0415

    return [
        {"name": name, "doc": _summary(cls), "params": _params_of(cls)}
        for name, cls in _concrete(_motion, _motion.Motion)
    ]


def footprints() -> list[dict[str, Any]]:
    """Available footprint primitives with their constructor parameters."""
    names = [
        "BoxFootprint",
        "DiskFootprint",
        "PolygonFootprint",
        "SectorFootprint",
        "AnnularSectorFootprint",
    ]
    out = []
    for name in names:
        cls = getattr(_bounds, name)
        out.append(
            {
                "name": name,
                "doc": _doc(cls),
                "params": _dataclass_params(cls),
                "composable": True,
            }
        )
    # Boolean composition is a binary op over two footprints, surfaced specially.
    out.append(
        {
            "name": "BooleanFootprint",
            "doc": _doc(_bounds.BooleanFootprint),
            "params": [{"name": "op", "type": "str", "required": True, "default": None,
                        "choices": ["union", "intersection", "difference"]}],
            "composable": True,
            "binary": True,
        }
    )
    return out


def altitude_bands() -> list[dict[str, Any]]:
    """Available altitude-band primitives with their constructor parameters."""
    names = [
        "ConstantAltitudeBand",
        "LinearAltitudeBand",
        "RadialAltitudeBand",
        "VertexAltitudeBand",
    ]
    return [
        {"name": name, "doc": _doc(getattr(_bounds, name)),
         "params": _dataclass_params(getattr(_bounds, name))}
        for name in names
    ]


def queryables() -> list[dict[str, Any]]:
    """Built-in queryable kinds. Custom queryables come via code references."""
    return [
        {"name": "QueryRegion", "doc": _doc(QueryRegion), "params": _dataclass_params(QueryRegion)},
        {"name": "Waypoint", "doc": _doc(Waypoint), "params": _dataclass_params(Waypoint)},
    ]


_SKIP_FIELD_PARAMS = {"env", "meta"}


def _optional_scalar(hint: Any) -> type | None:
    """Return the scalar type of an ``Optional[scalar]`` annotation, else ``None``.

    Recognizes ``X | None`` (optionally wrapped in ``Annotated[...]``) where ``X``
    is ``int``/``float``/``str``/``bool`` - e.g. a dynamic-bounds field's
    ``low``/``high`` typed ``float | None``. Lets those be exposed as optional
    overrides that fall back to the runtime/dynamic default when left blank.
    """
    if hint is None:
        return None
    if hasattr(hint, "__metadata__"):  # unwrap Annotated[T, ...]
        hint = get_args(hint)[0]
    if get_origin(hint) in (Union, types.UnionType):
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1 and non_none[0] in (int, float, str, bool):
            return non_none[0]
    return None


#: What a parameter can name, by the type it is annotated with: a parameter
#: that takes one of these (its class or an instance) is offered the design's
#: own - ``ActionMask.target`` the design's actions.
_REFERABLE: dict[type, str] = {ActionField: "action"}


def _refers(hint: Any) -> str | None:
    """What a parameter annotated ``hint`` names, from :data:`_REFERABLE`."""
    if hint is None:
        return None
    if hasattr(hint, "__metadata__"):  # unwrap Annotated[T, ...]
        hint = get_args(hint)[0]
    options = get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)
    for option in options:
        if get_origin(option) is type:  # type[X]
            option = get_args(option)[0]
        if not inspect.isclass(option):
            continue
        for base, kind in _REFERABLE.items():
            if issubclass(option, base):
                return kind
    return None


def _blank(hint: Any) -> str | None:
    """What leaving an optional parameter blank (``None``) means, as its
    annotation says it: the text after ``None =`` in ``Annotated[T, "...;
    None = the grid's step"]``."""
    for note in getattr(hint, "__metadata__", ()):
        if isinstance(note, str) and "None =" in note:
            return note.rsplit("None =", 1)[1].strip().rstrip(".") or None
    return None


def _field_params(cls) -> list[dict[str, Any]]:
    """Simple-typed constructor params of a field (e.g. ``low`` / ``high``).

    Exposes scalar-defaulted params, plus optional-scalar params (``float |
    None`` etc.) whose default is ``None`` - flagged ``optional`` so the designer
    can override them or leave them blank to keep the field's dynamic bounds.
    """
    out: list[dict[str, Any]] = []
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}
    def _append(name: str, annotation: Any, default: Any) -> None:
        entry = {"name": name, "type": _type_name(annotation), "default": default}
        refers = _refers(hints.get(name))
        if refers is not None:
            entry["refers"] = refers
        if isinstance(default, (int, float, str, bool)):
            out.append(entry)
        elif default is None and _optional_scalar(hints.get(name, annotation)) is not None:
            # Optional scalar (e.g. dynamic-bounds ``low``/``high``): expose it so
            # the designer can override, leaving blank (``None``) to keep the
            # field's runtime/dynamic bounds.
            entry["optional"] = True
            blank = _blank(hints.get(name, annotation))
            if blank is not None:
                entry["blank"] = blank
            out.append(entry)
        # else: a non-scalar object the uniform Picker can't edit - skip.

    try:
        fields = dataclasses.fields(cls)
    except TypeError:
        # Not a dataclass - a Normalizer, say. Its constructor annotations live
        # on ``__init__``, not on the class, and PEP 563 leaves them as strings
        # there. Resolving them off the class alone leaves every ``X | None``
        # parameter looking like an un-editable object, so it drops silently
        # out of the palette.
        try:
            init_hints = get_type_hints(cls.__init__, include_extras=True)
        except Exception:
            init_hints = {}
        for name, p in inspect.signature(cls).parameters.items():
            if name in {"self", * _SKIP_FIELD_PARAMS} or name.startswith("_"):
                continue
            default = None if p.default is inspect.Parameter.empty else p.default
            _append(name, init_hints.get(name, p.annotation), default)
        return out
    for f in fields:
        if not f.init or f.name in _SKIP_FIELD_PARAMS or f.name.startswith("_"):
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()
            except Exception:
                default = None
        else:
            default = None
        _append(f.name, f.type, default)
    return out


def _profile(cls) -> dict[str, Any]:
    meta = getattr(cls, "meta", None)
    meta_dict = {}
    if meta is not None:
        for name in ("name", "unit", "quantity", "control_axis", "mode", "is_pair", "circular", "dynamic_bounds", "requires_on", "suppresses_when_on"):
            if hasattr(meta, name):
                value = getattr(meta, name)
                if isinstance(value, tuple):
                    value = [getattr(v, "value", v) for v in value]
                else:
                    value = getattr(value, "value", value)
                meta_dict[name] = value
    try:
        source = inspect.getsource(cls)
    except OSError:
        source = ""
    return {
        "module": cls.__module__,
        "class_name": cls.__name__,
        "signature": str(inspect.signature(cls)),
        "meta": meta_dict,
        "queryable_spec": _queryable_spec(cls),
        "source": source,
        # What the value is computed with, beneath the class source: fields
        # are often one call into a shared helper.
        "trail": (
            call_trail(cls)
            if issubclass(cls, (ObsField, PairObsField, ActionField))
            else []
        ),
    }


def _queryable_spec(cls) -> dict[str, Any] | None:
    spec = getattr(cls, "queryable_spec", None)
    if spec is None:
        return None
    return {
        "kind": spec.kind.value,
        "path": spec.path,
        "label": spec.label,
        "description": spec.description,
        "requirements": [requirement.value for requirement in spec.requirements],
        "cardinality": spec.cardinality.value,
        "allow_empty_selection": spec.allow_empty_selection,
    }


def _wraps_a_field(cls: type) -> bool:
    """Whether the constructor takes another field (``Difference``'s ``left``,
    ``LaggedObs``'s ``inner``): built by a transform on the field it wraps -
    ``relative_to_own``, ``stacked`` - not picked on its own."""
    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        hint = hints.get(f.name)
        options = get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)
        if f.init and any(
            inspect.isclass(t) and issubclass(t, (ObsField, PairObsField)) for t in options
        ):
            return True
    return False


def _takes_normalizer(cls: type) -> bool:
    """Whether the constructor accepts a ``normalizer`` - a switch does not."""
    return any(f.name == "normalizer" and f.init for f in dataclasses.fields(cls))


def _takes_grid(cls: type) -> bool:
    """Whether the constructor accepts a ``grid`` - a switch does not."""
    return any(f.name == "grid" and f.init for f in dataclasses.fields(cls))


def _takes(cls: type, name: str) -> bool:
    """Whether the constructor accepts ``name``."""
    return any(f.name == name and f.init for f in dataclasses.fields(cls))


def _category(cls: type) -> dict[str, str]:
    """The module a field is defined in, which the picker groups it under.

    Read from the package layout - ``observations/kinematics.py`` files its
    fields under "kinematics" - with the module docstring's first paragraph as
    the category's description, so a new module is a new category.
    """
    module = sys.modules[cls.__module__]
    doc = inspect.getdoc(module) or ""
    return {
        "category": cls.__module__.rpartition(".")[2],
        "category_doc": " ".join(doc.split("\n\n", 1)[0].split()),
    }


def obs_fields() -> list[dict[str, Any]]:
    """Observation fields available to ``obs_fields`` / ``intruder_obs_fields``."""
    out = []
    for cls in _concrete_subclasses(_observations, (ObsField, PairObsField)):
        if _wraps_a_field(cls):
            continue
        out.append(
            {
                "name": cls.__name__,
                "doc": _doc(cls),
                **_category(cls),
                "normalizable": _takes_normalizer(cls),
                # Takes a speed schedule's crossover (``crossover``).
                "crossover": _takes(cls, "crossover"),
                "pair_only": issubclass(cls, PairObsField),
                "params": _field_params(cls),
                "profile": _profile(cls),
                "queryable_spec": _queryable_spec(cls),
            }
        )
    return sorted(out, key=lambda d: d["name"])


def action_fields() -> list[dict[str, Any]]:
    """Action fields available to ``action_fields``."""
    out = [
        {
            "name": cls.__name__,
            "doc": _doc(cls),
            **_category(cls),
            "kind": cls.kind.value,
            "normalizable": _takes_normalizer(cls),
            "griddable": _takes_grid(cls),
            # Acts in Mach above a crossover, given one (``above_crossover``).
            "mach_regime": _takes(cls, "above_crossover"),
            "params": _field_params(cls),
            "profile": _profile(cls),
        }
        for cls in _concrete_subclasses(_actions, ActionField)
    ]
    return sorted(out, key=lambda d: d["name"])


def normalizers() -> list[dict[str, Any]]:
    """Normalizer strategies constructible from a spec.

    Derived by introspection like the field catalogs: every concrete
    ``Normalizer`` subclass whose constructor has no required parameters, so a
    strategy needing hand-built arguments drops out of the palette naturally.
    """
    out = []
    for cls in _concrete_subclasses(_normalizers, _normalizers.Normalizer):
        sig = inspect.signature(cls.__init__)
        required = [
            p
            for name, p in sig.parameters.items()
            if name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        if required:
            continue
        out.append(
            {
                "name": cls.__name__,
                "doc": _doc(cls),
                "params": _field_params(cls),
                "profile": _profile(cls),
            }
        )
    return sorted(out, key=lambda d: d["name"])


# Scaffolds inserted into a custom code module when the user adds a custom field.
OBS_FIELD_SCAFFOLD = '''

@dataclass(frozen=True)
class {name}(ObsField):
    """Custom observation: one scalar value per aircraft.

    The designer constructs this class from the Spaces tab. Constructor values
    such as ``low``, ``high``, and ``normalizer`` can be configured there.
    """

    meta = ObsMeta(
        "{snake}",
        Unit.UNITLESS,
        ObsQuantity.DISTANCE,
    )
    low: float = -1.0   # lower bound used by spaces/normalizers
    high: float = 1.0   # upper bound used by spaces/normalizers

    def get(self, idx: int):
        # Common BlueSky arrays:
        #   bs.traf.lat[idx], bs.traf.lon[idx]        # degrees
        #   bs.traf.alt[idx] / ft                     # feet
        #   bs.traf.cas[idx] / kts                    # knots
        #   bs.traf.hdg[idx], bs.traf.trk[idx]        # degrees
        #
        # Return one scalar for the aircraft at traffic index `idx`.
        return 0.0

    def bounds(self, idx: int):
        # Use constructor bounds. For dynamic bounds, compute and return a
        # (low, high) tuple here instead.
        return self._configured_bounds()
'''

ACTION_FIELD_SCAFFOLD = '''

@dataclass(frozen=True)
class {name}(ActionField):
    """Custom action: maps one agent action scalar to a BlueSky command.

    The designer constructs this class from the Spaces tab. Constructor values
    such as ``low``, ``high``, and ``normalizer`` can be configured there.
    """

    meta = ActionMeta(
        "{snake}",
        Unit.UNITLESS,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.ABSOLUTE,
    )
    low: float = 0.0    # physical command lower bound
    high: float = 1.0   # physical command upper bound

    def set(self, idx: int, value: float) -> None:
        value = min(max(float(value), self.low), self.high)
        acid = bs.traf.id[idx]
        # Examples:
        #   bs.stack.stack(f"HDG {acid} {value:.6f}")
        #   bs.stack.stack(f"SPD {acid} {value:.6f}")
        #   bs.stack.stack(f"ALT {acid} {value:.6f}")
        bs.stack.stack(f"HDG {acid} {value:.6f}")

    def bounds(self, idx: int):
        # Use constructor bounds. For dynamic bounds, compute and return a
        # (low, high) tuple here instead.
        return self._configured_bounds()
'''

CUSTOM_MODULE_HEADER = '''"""Custom observation/action fields for this design.

Classes in this module are referenced as import strings, e.g.
``custom_fields:MyField``. Use the designer's field configuration modal to set
constructor bounds and normalizers for each referenced class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bluesky as bs
from bluesky.tools.aero import ft, kts

from bluesky.tools.aero import ft, nm
from bluesky_sandbox.sim.geometry.conflict import (
    cd_hpz_m,
    cd_lookahead_s,
    cd_rpz_m,
)
from bluesky_sandbox.interface.fields.base import (
    ActionField, ActionMeta, ActionMode, ControlAxis,
    ObsField, ObsMeta, ObsQuantity, PairObsField,
    QueryableFieldCardinality, QueryableFieldRequirement, QueryableFieldSpec,
    QueryableKind, Unit,
)
'''


def _custom_module_header() -> str:
    """Scaffold header with the normalizer import derived by introspection
    (same helper generated task code uses), so a new normalizer is available
    in custom-field modules without editing this template."""
    return CUSTOM_MODULE_HEADER + _normalizer_import_line() + "\n"


def scaffolds() -> dict[str, str]:
    """Code templates the UI uses when adding a custom field/module."""
    return {
        "module_header": _custom_module_header(),
        "obs_field": OBS_FIELD_SCAFFOLD,
        "action_field": ACTION_FIELD_SCAFFOLD,
    }


# Always-present task-outcome hooks (never removable in the GUI).


def _hook_category(name: str) -> str:
    """Bucket a hook for the GUI picker — derived from its name, not hard-coded."""
    if name.removesuffix("_batch") in BATCHABLE_HOOKS:
        return "task outcome"
    if name.startswith("define_"):
        return "definitions"
    if name.startswith("on_"):
        return "lifecycle events"
    return "other"


def _hook_default(fn) -> str | None:
    """The base implementation's literal ``return`` value, for display.

    Parses the method source rather than hard-coding defaults, so it stays in
    sync if a base hook's no-op return changes.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, SyntaxError):
        return None
    func = tree.body[0]
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                return ast.unparse(node.value)
            except Exception:  # pragma: no cover - unparse is total in 3.9+
                return None
    return None


# Per-hook starter bodies for hooks where a blank scaffold isn't obvious. Keyed
# by hook name; everything else falls back to the generic scaffold in the GUI.
_HOOK_SCAFFOLDS: dict[str, str] = {
    "reward_batch": (
        "# Every agent's reward at once, one per batch.acids, in that order.\n"
        "# batch.obs[\"ownship\"][name] is (n_agents,); batch.obs[\"intruders\"]\n"
        "# [name] is (n_agents, n_intruders). batch.terminated / .truncated are set.\n"
        "# Replaces reward: a design defines one of the two.\n"
        "return [0.0] * len(batch)"
    ),
    "terminated_batch": (
        "# Every agent's termination at once, one bool per batch.acids.\n"
        "# Replaces terminated: a design defines one of the two.\n"
        "return [False] * len(batch)"
    ),
    "truncated_batch": (
        "# Every agent's truncation at once, one bool per batch.acids.\n"
        "# Replaces truncated: a design defines one of the two.\n"
        "return [False] * len(batch)"
    ),
    "define_aircraft_readouts": (
        "# Rows for this aircraft's readout in the drivers, in order: {label: value}.\n"
        "# acid is its callsign; self.live_info.get(acid) its latest info\n"
        "# (acidx, type, phase, ...), None for one that is not an agent.\n"
        "return {\"ACID\": acid}"
    ),
    "define_agent_context": (
        "# Build the per-aircraft `context.data` payload (any object).\n"
        "# Available: self.episode_queryables, acid (callsign), acidx (traffic index).\n"
        "# Prefer context.query(\"goal\") in agent hooks when you need query results.\n"
        "return {\"acid\": acid}"
    ),
}


def hooks() -> list[dict[str, Any]]:
    """Overridable environment hooks, discovered by introspection.

    Returns each ``@overridable`` method's name, full signature, one-line doc,
    derived ``category`` (for grouping the picker), base ``default`` return, and
    an optional richer ``scaffold`` — so the designer can offer and describe them
    without a hard-coded list.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cls in inspect.getmro(BlueskyEnv):
        for name, fn in vars(cls).items():
            if name in seen or not getattr(fn, "__overridable__", False):
                continue
            seen.add(name)
            sig = inspect.signature(fn)

            # Base hooks prefix unused params with "_" (a marker). An override
            # that *uses* them wants natural names, so strip the underscore. Also
            # drop annotations so the generated package has no undefined forward
            # refs. The full signature is kept for display.
            def _clean_name(pname: str) -> str:
                return pname[1:] if pname.startswith("_") and pname != "self" else pname

            clean = sig.replace(
                parameters=[
                    p.replace(name=_clean_name(p.name), annotation=inspect.Parameter.empty)
                    for p in sig.parameters.values()
                ],
                return_annotation=inspect.Signature.empty,
            )
            params = [_clean_name(p) for p in sig.parameters if p != "self"]
            out.append(
                {
                    "name": name,
                    "signature": str(sig),
                    "def_signature": str(clean),
                    "params": params,
                    "returns": _type_name(sig.return_annotation)
                    if sig.return_annotation is not inspect.Signature.empty
                    else "",
                    "returns_none": str(sig.return_annotation) in ("None", "<class 'NoneType'>"),
                    "default": _hook_default(fn),
                    "category": _hook_category(name),
                    "always_present": name in DEFAULT_HOOKS,
                    "scaffold": _HOOK_SCAFFOLDS.get(name),
                    "doc": _paragraph(fn),
                }
            )
    return sorted(out, key=lambda d: d["name"])


def distributions() -> list[dict[str, Any]]:
    """Every ``scipy.stats`` distribution with its parameter names.

    Introspected (shape params from ``.shapes`` + ``loc``/``scale``) so the
    designer's value editor can show the right named fields for whichever
    distribution is picked, instead of asking for a freeform ``k=v`` string.
    """
    out: list[dict[str, Any]] = []
    for name in dir(ss):
        obj = getattr(ss, name, None)
        is_continuous = isinstance(obj, ss.rv_continuous)
        is_discrete = isinstance(obj, ss.rv_discrete)
        if not (is_continuous or is_discrete):
            continue
        shapes = [s.strip() for s in obj.shapes.split(",")] if obj.shapes else []
        params = [*shapes, "loc"] + (["scale"] if is_continuous else [])
        signature = "{}({})".format(
            name,
            ", ".join([*shapes, "loc=0"] + (["scale=1"] if is_continuous else [])),
        )
        # Infinite support (e.g. poisson, nbinom, norm) can't size an observation
        # space, so such a pick needs an explicit ``Bounded(...)`` wrapper. The
        # GUI uses this flag to surface the bounds control before validation fails.
        # Distributions with a custom ``_get_support`` (randint, betabinom,
        # loguniform, ...) have arg-dependent - hence finite once configured -
        # support, so they are not flagged despite class-level infinite ``a``/``b``.
        if "_get_support" in type(obj).__dict__:
            unbounded = False
        else:
            try:
                unbounded = math.isinf(float(obj.a)) or math.isinf(float(obj.b))
            except Exception:
                unbounded = False
        out.append(
            {
                "name": name,
                "params": params,
                "shapes": shapes,
                "discrete": is_discrete,
                "unbounded": unbounded,
                "signature": signature,
            }
        )
    return sorted(out, key=lambda d: d["name"])


def conflict_methods() -> dict[str, list[str]]:
    """Common BlueSky conflict-detection and -resolution method names."""
    return {
        "cd_methods": ["CSTATEBASED", "STATEBASED"],
        "reso_methods": ["OFF", "MVP"],
    }


def bluesky_defaults() -> dict[str, float | None]:
    """BlueSky's own values for the settings a design may leave unset.

    Read from ``bs.settings`` rather than written into the GUI, so the designer
    shows what BlueSky will actually run. The designer has no ``bs.init``, so
    this reads settings.cfg first. ``asas_dt`` is ``None`` when settings.cfg does
    not set it: BlueSky's built-in default is registered only by importing its
    traffic module, which would pre-empt the performance-model choice a later
    ``bs.init`` in this process makes.
    """
    return {"simdt": bluesky_simdt_s(), "asas_dt": bluesky_asas_dt_s()}


def spawn_defaults() -> dict[str, int]:
    """``SpawnConfig``'s own spawn-retry defaults, for the designer's placeholders.

    Read from the dataclass rather than written into the GUI, so the number
    shown for an empty field is the one a design that leaves it empty gets.
    """
    return {
        f.name: f.default
        for f in dataclasses.fields(SpawnConfig)
        if f.name in ("spawn_max_tries", "spawn_warn_after")
    }


def colors() -> dict[str, str]:
    """Named display colors → hex, for the GUI color picker.

    The renderers accept either a palette name or a ``#rrggbb`` literal, so the
    picker offers the named swatches plus a custom hex. Sourced from the driver
    palette (guarded so the catalog never hard-depends on pygame).
    """
    try:
        # optional extra: [pygame] - catalog must not hard-depend on it
        from bluesky_sandbox.ui.drivers.pygame.colors import (  # noqa: PLC0415
            NAMED_COLORS as _named,
        )
    except Exception:
        _named = {
            "red": (220, 20, 60), "green": (30, 150, 30), "blue": (30, 80, 200),
            "cyan": (0, 200, 200), "yellow": (235, 235, 30), "orange": (255, 140, 0),
            "purple": (160, 70, 200), "magenta": (220, 60, 200),
            "white": (255, 255, 255), "black": (0, 0, 0), "gray": (80, 80, 80),
        }
    from bluesky_sandbox.ui.drivers.common.palette import overlay_rgb  # noqa: PLC0415

    # As drawn: a name near an alert's hue is drawn clear of it.
    return {
        name: "#%02x%02x%02x" % overlay_rgb(tuple(rgb))
        for name, rgb in _named.items()
        if name != "violation"  # internal status color, not a design choice
    }


def alert_colors() -> list[str]:
    """The named colors drawn in another hue than their name - near an
    alert's: the picker leaves them out (see ``common.palette``)."""
    from bluesky_sandbox.ui.drivers.common.palette import overlay_rgb  # noqa: PLC0415

    try:
        from bluesky_sandbox.ui.drivers.pygame.colors import NAMED_COLORS as _named  # noqa: PLC0415
    except Exception:
        return ["red", "orange", "purple"]
    return sorted(n for n, rgb in _named.items() if n != "violation" and overlay_rgb(tuple(rgb)) != tuple(rgb))


def reserved_hues() -> list[list[float]]:
    """The hue bands (deg) kept for alerts, for the map to draw custom colors
    as the drivers do."""
    from bluesky_sandbox.ui.drivers.common.palette import reserved_hues as _bands  # noqa: PLC0415

    return [[float(a), float(b)] for a, b in _bands()]


def drivers() -> list[dict[str, Any]]:
    """Live-run render modes and the view layouts each one offers."""
    # cycle: catalog -> runner -> codegen -> catalog
    from .runner import DRIVER_VIEWS, VALID_RENDER_MODES  # noqa: PLC0415

    return [
        {
            "render_mode": mode,
            "views": DRIVER_VIEWS[mode]["options"],
            "default_views": DRIVER_VIEWS[mode]["default"],
        }
        for mode in VALID_RENDER_MODES
    ]


def aircraft_types(model: str | None = None) -> list[str]:
    """ICAO aircraft types available in the active performance model."""
    return sorted(t.upper() for t in _available_aircraft(model))


def aircraft_by_model() -> dict[str, Any]:
    """Available aircraft per performance model.

    Each value is a list of ``{"type", "name", "tags"}`` - sorted by type, as
    the model BlueSky flies carries them - or ``{"error": msg}`` when that
    model's database can't be loaded (e.g. BADA not installed).
    """
    out: dict[str, Any] = {}
    for model in MODELS:
        try:
            # Offer only types with envelope bounds: a design that picks one
            # without them looks fine until a spawn tries to sample an altitude.
            types = sorted(t.upper() for t in spawnable_types(model))
        except Exception as e:  # model unavailable -> surface the reason
            out[model] = {"error": str(e)}
            continue
        out[model] = [_described(t, model) for t in types]
    return out


def _described(actype: str, model: str) -> dict[str, Any]:
    """A type as the picker offers it: its ICAO code, name, and tags - what
    it is, from the model's own data (``type_info``)."""
    info = type_info(actype, model) or {}
    return {"type": actype, "name": info.get("name"), "tags": info.get("tags") or []}


def scenario_hooks() -> list[dict[str, Any]]:
    """Scenario-side hooks, derived from :data:`~.spec.SCENARIO_HOOKS`.

    The scenario twin of :func:`hooks`. These run on the generated ``Scenario``
    rather than the env, and are the escape hatch for per-episode sampling the
    structured design cannot express. Derived from the spec's own table so the
    designer never carries a second, drifting copy of the hook list.
    """
    return [
        {
            "name": name,
            "args": list(args),
            "signature": f"({', '.join(args)})",
            "doc": purpose,
            "scaffold": f"# {purpose}\nreturn {args[0]}\n",
        }
        for name, (args, purpose) in sorted(SCENARIO_HOOKS.items())
    ]


def separation_defaults() -> dict[str, float]:
    """The protected zone and lookahead a conflict-free spawn check resolves to.

    Sourced from BlueSky's conflict detection rather than hardcoded in the GUI,
    so the designer shows the values a spawn is actually cleared against. These
    are what a design's ``spawn_sep_nm`` / ``spawn_sep_ft`` /
    ``spawn_lookahead_s`` fall back to when left unset.
    """
    return {
        "pz_radius_nm": cd_rpz_m() / nm,
        "pz_height_ft": cd_hpz_m() / ft,
        "lookahead_s": cd_lookahead_s(),
    }


def catalog(model: str | None = None) -> dict[str, Any]:
    """Full palette payload for the GUI in one call."""
    return {
        "footprints": footprints(),
        "generators": generators(),
        "placements": placements(),
        "motions": motions(),
        "altitude_bands": altitude_bands(),
        "queryables": queryables(),
        "obs_fields": obs_fields(),
        "action_fields": action_fields(),
        "normalizers": normalizers(),
        "aircraft_types": aircraft_types(model),
        "aircraft": aircraft_by_model(),
        "hooks": hooks(),
        "scenario_hooks": scenario_hooks(),
        "drivers": drivers(),
        "colors": colors(),
        "alert_colors": alert_colors(),
        "reserved_hues": reserved_hues(),
        "distributions": distributions(),
        "conflict": conflict_methods(),
        "bluesky_defaults": bluesky_defaults(),
        "separation": separation_defaults(),
        "spawn_defaults": spawn_defaults(),
        "scaffolds": scaffolds(),
    }
