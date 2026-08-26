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
import pathlib
import re
import textwrap
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from scipy import stats as ss

from bluesky_sandbox.config import _available_aircraft
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as _actions
from bluesky_sandbox.interface.fields import observations as _observations
from bluesky_sandbox.interface.fields import queryables as _queryable_fields
from bluesky_sandbox.interface.fields.base import ActionField, ObsField, PairObsField
from bluesky_sandbox.interface.wrappers.observations import normalizer as _normalizers
from bluesky_sandbox.sim import bounds as _bounds
from bluesky_sandbox.sim.performance.models import spawnable_types
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint

from .emit import _normalizer_import_line
from .spec import SCENARIO_HOOKS


def _doc(obj: Any) -> str:
    doc = inspect.getdoc(obj)
    return doc.splitlines()[0] if doc else ""


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
    return getattr(t, "__name__", str(t))


def _concrete_subclasses(module, base) -> list[type]:
    out = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isclass(obj) and issubclass(obj, base) and obj is not base):
            continue
        if inspect.isabstract(obj):
            continue
        out.append(obj)
    return out


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
        if isinstance(default, (int, float, str, bool)):
            out.append(entry)
        elif default is None and _optional_scalar(hints.get(name, annotation)) is not None:
            # Optional scalar (e.g. dynamic-bounds ``low``/``high``): expose it so
            # the designer can override, leaving blank (``None``) to keep the
            # field's runtime/dynamic bounds.
            entry["optional"] = True
            out.append(entry)
        # else: a non-scalar object the uniform Picker can't edit - skip.

    try:
        fields = dataclasses.fields(cls)
    except TypeError:
        for name, p in inspect.signature(cls).parameters.items():
            if name in {"self", * _SKIP_FIELD_PARAMS} or name.startswith("_"):
                continue
            default = None if p.default is inspect.Parameter.empty else p.default
            _append(name, p.annotation, default)
        return out
    for f in fields:
        if f.name in _SKIP_FIELD_PARAMS or f.name.startswith("_"):
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


def obs_fields() -> list[dict[str, Any]]:
    """Observation fields available to ``obs_fields`` / ``intruder_obs_fields``."""
    out = []
    for module in (_observations, _queryable_fields):
        classes = _concrete_subclasses(module, (ObsField, PairObsField))
        for cls in classes:
            out.append(
                {
                    "name": cls.__name__,
                    "doc": _doc(cls),
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
        {"name": cls.__name__, "doc": _doc(cls), "params": _field_params(cls), "profile": _profile(cls)}
        for cls in _concrete_subclasses(_actions, ActionField)
    ]
    return sorted(out, key=lambda d: d["name"])


def normalizers() -> list[dict[str, Any]]:
    """Normalizer strategies constructible from a spec.

    Derived by introspection like the field catalogs: every concrete
    ``Normalizer`` subclass whose constructor has no required parameters.
    (``PerFieldNormalizer`` needs a field map, so it is code-composition only
    and drops out of the palette naturally.)
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

    The designer constructs this class from the Fields tab. Constructor values
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

    The designer constructs this class from the Fields tab. Constructor values
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
_OUTCOME_HOOKS = ("reward", "terminated", "truncated")


def _hook_category(name: str) -> str:
    """Bucket a hook for the GUI picker — derived from its name, not hard-coded."""
    if name in _OUTCOME_HOOKS:
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
                    "always_present": name in _OUTCOME_HOOKS,
                    "scaffold": _HOOK_SCAFFOLDS.get(name),
                    "doc": _doc(fn),
                }
            )
    return sorted(out, key=lambda d: d["name"])


# The task-info providers the designer can scaffold. The provider SOURCE is
# emitted into the task package rather than imported from the library: what
# counts as a cost, how many channels there are and what their limits mean is
# the task's business, so the code shaping it ships with the task and stays
# editable. The library keeps only the protocols these are written against.
def _autocost_source() -> str:
    """The provider class the designer writes into a generated task.

    Kept as a real .py template rather than a string literal in this module:
    it carries docstrings of its own, so embedding it would fight the quoting,
    and as a file it stays lintable and editable like ordinary source.
    """
    return (
        pathlib.Path(__file__).parent / "templates" / "autocost_provider.py"
    ).read_text().rstrip()


_TASK_INFO_TYPES: list[dict[str, Any]] = [
    {
        "name": "AutoCostConstraintTaskInfoProvider",
        "doc": (
            "Constraint costs derived from a cost function: an extrinsic term "
            "for true violations and an optional intrinsic term for dense risk "
            "before one occurs. The class is written into your task package so "
            "you can change how costs are shaped."
        ),
        "params": [
            {"name": "names", "type": "tuple[str, ...]", "required": True, "default": None},
            {"name": "limits", "type": "np.ndarray", "required": True, "default": None},
            {"name": "extrinsic_cost_fn", "type": "ConstraintFn", "required": True, "default": None},
            {"name": "intrinsic_cost_fn", "type": "ConstraintFn | None", "required": False, "default": None},
        ],
        "category": "task info",
    },
]


def task_info_types() -> list[dict[str, Any]]:
    """Task-info providers the designer can scaffold into a task package.

    A static list, not a scan of the library: the providers are no longer part
    of :mod:`bluesky_sandbox.interface.task`, which now declares only the
    protocols. Each entry carries the source the designer writes into the
    generated task.
    """
    return [dict(item, scaffold=_task_info_scaffold(item["name"])) for item in _TASK_INFO_TYPES]


def _snake_name(name: str) -> str:
    name = name.removesuffix("TaskInfoProvider")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower() or "task_info"


def _task_info_scaffold(cls_name: str) -> dict[str, str]:
    name = _snake_name(cls_name)
    provider_var = f"{name.upper()}_TASK_INFO_PROVIDER"
    if cls_name == "AutoCostConstraintTaskInfoProvider":
        setup = f'''from dataclasses import dataclass
from typing import Any

import numpy as np

from bluesky_sandbox.interface.task import (
    AgentStepContext,
    BaseAgentInfo,
    BaseObs,
    ConstraintFn,
)


# The cost -> constraint-info adapter, defined here rather than imported: what
# counts as a cost is the task\'s business. Edit freely.
{_autocost_source()}

def {name}_extrinsic_cost(obs, action, info, context, rng):
    # True task violation cost. This is what defines constraint violations.
    return np.array([0.0], dtype=np.float32)


def {name}_intrinsic_cost(obs, action, info, context, rng):
    # Optional dense nonnegative risk before the true violation occurs.
    return np.array([0.0], dtype=np.float32)


{provider_var} = AutoCostConstraintTaskInfoProvider(
    names=("constraint",),
    limits=np.array([0.0], dtype=np.float32),
    extrinsic_cost_fn={name}_extrinsic_cost,
    intrinsic_cost_fn={name}_intrinsic_cost,
)
'''
    else:
        raise ValueError(f"no scaffold for task-info provider {cls_name!r}")
    return {
        "name": name,
        "provider_var": provider_var,
        "setup": setup,
        "body": provider_var,
    }


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


def colors() -> dict[str, str]:
    """Named display colors → hex, for the GUI colour picker.

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
            "cyan": (0, 200, 200), "yellow": (240, 220, 30), "orange": (255, 140, 0),
            "purple": (160, 70, 200), "magenta": (220, 60, 200),
            "white": (255, 255, 255), "black": (0, 0, 0), "gray": (80, 80, 80),
        }
    return {
        name: "#%02x%02x%02x" % tuple(rgb)
        for name, rgb in _named.items()
        if name != "violation"  # internal status colour, not a design choice
    }


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

    Each value is either a sorted list of ICAO types or ``{"error": msg}`` when
    that model's database can't be loaded (e.g. BADA not installed).
    """
    out: dict[str, Any] = {}
    for model in ("openap", "bada"):
        try:
            # Offer only types with envelope bounds: a design that picks one
            # without them looks fine until a spawn tries to sample an altitude.
            out[model] = sorted(t.upper() for t in spawnable_types(model))
        except Exception as e:  # model unavailable -> surface the reason
            out[model] = {"error": str(e)}
    return out


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


def catalog(model: str | None = None) -> dict[str, Any]:
    """Full palette payload for the GUI in one call."""
    return {
        "footprints": footprints(),
        "altitude_bands": altitude_bands(),
        "queryables": queryables(),
        "obs_fields": obs_fields(),
        "action_fields": action_fields(),
        "normalizers": normalizers(),
        "aircraft_types": aircraft_types(model),
        "aircraft": aircraft_by_model(),
        "hooks": hooks(),
        "scenario_hooks": scenario_hooks(),
        "task_info_types": task_info_types(),
        "drivers": drivers(),
        "colors": colors(),
        "distributions": distributions(),
        "conflict": conflict_methods(),
        "scaffolds": scaffolds(),
    }
