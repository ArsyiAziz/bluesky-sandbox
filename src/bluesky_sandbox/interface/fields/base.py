from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import bluesky as bs
import numpy as np

from ._consistency import beyond_rounding, differs
from .grid import Grid

if TYPE_CHECKING:
    from bluesky_sandbox.interface.task import AgentStepContext

ContextT = TypeVar("ContextT")


# Bump whenever a field's ``get()``/``bounds()``/normalization semantics
# change in a way that shifts the observation VALUE for an unchanged
# physical state, even though the field's name/dtype/width stays the same.
FIELD_SEMANTICS_REVISION = 1


class ObsQuantity(StrEnum):
    """Framework-owned observation semantic tags."""

    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    HEADING = "heading"
    TRACK = "track"
    ALTITUDE = "altitude"
    SPEED = "speed"
    VERTICAL_SPEED = "vertical_speed"
    DISTANCE = "distance"
    BEARING = "bearing"
    TIME = "time"
    AUTOPILOT = "autopilot"
    PHASE = "phase"
    RISK = "risk"
    COUNT = "count"
    ACTION = "action"
    INDICATOR = "indicator"
    MASS = "mass"


class Unit(StrEnum):
    """Framework-owned physical unit tags."""

    DEG = "deg"
    DEG_PER_SEC = "deg/s"
    FT = "ft"
    M = "m"
    KTS = "kts"
    M_PER_S = "m/s"
    FT_PER_MIN = "ft/min"
    S = "s"
    NM = "nm"
    SWITCH = "switch"
    UNITLESS = "unitless"
    T = "t"


class QueryableKind(StrEnum):
    """Queryable resource families understood by field/catalog metadata."""

    ANY = "any"
    REGION = "region"
    WAYPOINT = "waypoint"


class QueryableFieldRequirement(StrEnum):
    """Queryable capabilities required by a queryable-backed field."""

    ALTITUDE = "altitude"
    SPEED = "speed"
    ROUTE = "route"
    TOLERANCE = "tolerance"
    STEP = "step"
    TIME = "time"


class QueryableFieldCardinality(StrEnum):
    """How a queryable-backed field binds to queryable names."""

    SINGLE = "single"
    MULTIPLE = "multiple"
    ACTIVE = "active"


@dataclass(frozen=True)
class QueryableFieldSpec:
    """Designer/runtime metadata for fields backed by context queryables.

    ``kind`` and ``requirements`` are enums because the designer branches on
    them. ``path`` remains a dotted string so it can describe evolving result
    objects without expanding a registry for every exposed attribute.
    """

    kind: QueryableKind
    path: str
    label: str
    description: str = ""
    requirements: tuple[QueryableFieldRequirement, ...] = ()
    cardinality: QueryableFieldCardinality = QueryableFieldCardinality.SINGLE
    allow_empty_selection: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QueryableKind):
            raise TypeError(
                f"QueryableFieldSpec.kind must be QueryableKind, got {self.kind!r}."
            )
        if not isinstance(self.cardinality, QueryableFieldCardinality):
            raise TypeError(
                "QueryableFieldSpec.cardinality must be QueryableFieldCardinality, "
                f"got {self.cardinality!r}."
            )
        if not self.path:
            raise ValueError("QueryableFieldSpec.path must be non-empty.")
        if not self.label:
            raise ValueError("QueryableFieldSpec.label must be non-empty.")
        if not isinstance(self.allow_empty_selection, bool):
            raise TypeError(
                "QueryableFieldSpec.allow_empty_selection must be bool, got "
                f"{self.allow_empty_selection!r}."
            )
        for requirement in self.requirements:
            if not isinstance(requirement, QueryableFieldRequirement):
                raise TypeError(
                    "QueryableFieldSpec.requirements entries must be "
                    f"QueryableFieldRequirement, got {requirement!r}."
                )


class ControlAxis(StrEnum):
    """Aircraft control channel commanded by an action field."""

    HEADING = "heading"
    SPEED = "speed"
    ALTITUDE = "altitude"
    AUTOPILOT = "autopilot"

    @property
    def period(self) -> float | None:
        """Where the axis wraps round to 0 - a heading's 360 deg - or None for
        one that does not."""
        return 360.0 if self is ControlAxis.HEADING else None


class ActionMode(StrEnum):
    """How an action value should be interpreted."""

    ABSOLUTE = "absolute"
    DELTA = "delta"
    SWITCH = "switch"


class ActionKind(StrEnum):
    """What values an action takes, which sets the part of the action space
    it is in: ``continuous`` a range, ``binary`` 0 or 1, ``discrete`` one of a
    few choices (an index: a ``MultiDiscrete`` entry)."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    DISCRETE = "discrete"


def action_kind(field: Any) -> ActionKind:
    """The part of the action space ``field`` is in: ``discrete`` when its
    normalizer makes it a choice among a few values (a ``StepNormalizer``),
    else its class's :attr:`ActionField.kind`."""
    if getattr(getattr(field, "normalizer", None), "discrete", False):
        return ActionKind.DISCRETE
    return field.kind


def _validate_bounds(
    owner: str,
    low: float,
    high: float,
) -> tuple[float, float]:
    lo, hi = float(low), float(high)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{owner} bounds must be finite, got ({low!r}, {high!r}).")
    if lo > hi:
        raise ValueError(
            f"{owner} bounds must satisfy low <= high, got ({low!r}, {high!r})."
        )
    return lo, hi


@dataclass(frozen=True)
class ObsMeta:
    """Static metadata attached to every observation field class.

    ``name`` is the stable identifier used in logs, docs, normalizer maps,
    and debugging output. ``unit`` is the physical unit enum emitted by
    :meth:`ObsField.get` or :meth:`PairObsField.get_pair`.

    ``quantity`` is a coarse semantic tag (for example
    :attr:`ObsQuantity.ALTITUDE` or :attr:`ObsQuantity.SPEED`). It is
    descriptive today, but gives wrappers a typed
    alternative to parsing names if field-specific behavior is needed later.

    ``is_pair`` marks fields that require both ownship and intruder indices
    and therefore may only appear in ``EnvConfig.intruder_obs_fields``.
    ``circular`` marks angles where wraparound-aware normalization is usually
    appropriate. ``dynamic_bounds`` means the field's default bounds depend on
    live BlueSky state; constructor ``None`` bounds ask the field to resolve
    those dynamic bounds at runtime.
    """

    name: str
    unit: Unit
    quantity: ObsQuantity
    is_pair: bool = False
    circular: bool = False
    dynamic_bounds: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ObsMeta.name must be a non-empty string.")
        if not isinstance(self.unit, Unit):
            raise TypeError(
                f"ObsMeta({self.name!r}).unit must be Unit, got {self.unit!r}."
            )
        if not isinstance(self.quantity, ObsQuantity):
            raise TypeError(
                f"ObsMeta({self.name!r}).quantity must be ObsQuantity, "
                f"got {self.quantity!r}."
            )


@dataclass(frozen=True)
class ActionMeta:
    """Static metadata attached to every action field class.

    ``name`` is the stable identifier used in logs, docs, debugging output,
    and autopilot dependency metadata. ``unit`` is the physical unit expected
    by :meth:`ActionField.set`.

    ``control_axis`` identifies which aircraft control channel this action
    commands. ``mode`` describes how values should be interpreted. Switch
    actions are 0/1 controls implemented by :class:`SwitchActionMixin` and
    handled separately during action ordering.

    ``requires_on`` lists switch action names that must also be considered
    active when this switch is turned on. ``suppresses_when_on`` lists control
    axes whose same-step non-switch actions should be skipped when this switch
    is ON. ``dynamic_bounds`` means the field's default bounds depend on live
    BlueSky state; constructor ``None`` bounds ask the field to resolve those
    dynamic bounds at runtime.
    """

    name: str
    unit: Unit
    control_axis: ControlAxis | None = None
    mode: ActionMode | None = None
    requires_on: tuple[str, ...] = ()
    suppresses_when_on: tuple[ControlAxis, ...] = ()
    dynamic_bounds: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ActionMeta.name must be a non-empty string.")
        if not isinstance(self.unit, Unit):
            raise TypeError(
                f"ActionMeta({self.name!r}).unit must be Unit, got {self.unit!r}."
            )
        if self.control_axis is not None and not isinstance(
            self.control_axis, ControlAxis
        ):
            raise TypeError(
                f"ActionMeta({self.name!r}).control_axis must be ControlAxis "
                f"or None, got {self.control_axis!r}."
            )
        if self.mode is not None and not isinstance(self.mode, ActionMode):
            raise TypeError(
                f"ActionMeta({self.name!r}).mode must be ActionMode or None, "
                f"got {self.mode!r}."
            )
        for required in self.requires_on:
            if not required:
                raise ValueError(
                    f"ActionMeta({self.name!r}).requires_on entries must be non-empty."
                )
        for axis in self.suppresses_when_on:
            if not isinstance(axis, ControlAxis):
                raise TypeError(
                    f"ActionMeta({self.name!r}).suppresses_when_on entries "
                    f"must be ControlAxis, got {axis!r}."
                )


def _render_metadata(meta: ObsMeta | ActionMeta) -> str:
    """A ``Metadata:`` docstring section for ``meta``: its name and unit, and
    every other field that differs from its default."""
    lines = ["Metadata:"]
    for f in fields(meta):
        value = getattr(meta, f.name)
        if f.name not in ("name", "unit") and value == f.default:
            continue
        if isinstance(value, tuple):
            value = ", ".join(str(v) for v in value)
        lines.append(f"    {f.name}: {value}")
    return "\n".join(lines)


def _without_metadata(doc: str) -> str:
    """``doc`` with any ``Metadata:`` section (the line and its indented body)
    removed."""
    out: list[str] = []
    skipping = False
    for line in doc.splitlines():
        if line.strip() == "Metadata:":
            skipping = True
            continue
        if skipping and (not line.strip() or line.startswith((" ", "\t"))):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out).rstrip()


@dataclass(frozen=True)
class _BoundedField:
    def __init_subclass__(cls, **kwargs: Any) -> None:
        # The docstring's Metadata section is rendered from ``meta``, so the
        # two cannot drift. Only a class with its own docstring and a static
        # ``meta`` gets one; a ``meta`` property depends on the instance.
        super().__init_subclass__(**kwargs)
        doc = cls.__dict__.get("__doc__")
        meta = getattr(cls, "meta", None)
        if doc and isinstance(meta, (ObsMeta, ActionMeta)):
            summary = _without_metadata(inspect.cleandoc(doc))
            cls.__doc__ = f"{summary}\n\n{_render_metadata(meta)}\n"

    # ---- a test case ---------------------------------------------------- #
    def case_value(self, own: int, *, other: int | None = None) -> Any:
        """What this field gives in a test case (:mod:`bluesky_sandbox.checks`)
        for the aircraft at ``own``, about ``other`` (a pair field's other
        aircraft). Each kind of field says it once, here, so a case runner never
        asks which kind it has."""
        raise NotImplementedError(f"{type(self).__name__} cannot be read in a test case yet")

    # ---- optional per-aircraft state ------------------------------------- #
    # A field that reads state the simulator does not keep (a rate, an
    # accumulator, a history) declares here how that state is maintained.
    # Overriding any of these marks the field as stateful, and the environment
    # drives it: ``on_step`` once per env step (after the substep loop),
    # ``on_action_applied`` when an action is dispatched, ``on_aircraft_removed``
    # when an aircraft despawns, ``on_episode_reset`` at reset.
    #
    # The point of putting them on the field rather than in a registry the
    # environment maintains: a store and its "drop this aircraft" call can no
    # longer drift apart. When they lived in separate files, adding a tracker
    # and forgetting its ``forget_x`` leaked one aircraft's state onto the next
    # one to reuse its callsign.
    #
    # Cost note: overriding opts the field INTO per-step bookkeeping. Nothing
    # is recorded for a field no config uses - ``_stateful_fields`` collects
    # only what the config lists - so this is the difference between paying for
    # a tracker and not.

    def on_step(self, ctx: StepContext) -> None:
        """Update this field's state for one env step."""

    def on_action_applied(self, acid: str, action) -> None:
        """Record the action just applied to ``acid`` (once per env step)."""

    def on_aircraft_removed(self, acid: str) -> None:
        """Drop one aircraft's state (it despawned; the callsign may return)."""

    def on_episode_reset(self, seed: int | None = None) -> None:
        """Drop all state (a new episode; callsigns repeat across episodes)."""

    @classmethod
    def is_stateful(cls) -> bool:
        """True when this field overrides any of the state hooks above."""
        return any(
            getattr(cls, name) is not getattr(_BoundedField, name)
            for name in (
                "on_step",
                "on_action_applied",
                "on_aircraft_removed",
                "on_episode_reset",
            )
        )

    low: float | None = None
    high: float | None = None
    normalizer: Any | None = None

    def __post_init__(self) -> None:
        if (self.low is None) != (self.high is None):
            raise ValueError(
                f"{self.__class__.__name__} bounds must provide both low and high."
            )
        if self.low is None or self.high is None:
            return
        _validate_bounds(self.__class__.__name__, self.low, self.high)

    @property
    def bounds_overridden(self) -> bool:
        """True when instance bounds differ from the field's class defaults."""
        default_low, default_high = self._constructor_bounds()
        if default_low is None or default_high is None:
            return self.low is not None or self.high is not None
        return float(self.low) != float(default_low) or float(self.high) != float(
            default_high
        )

    @classmethod
    def _constructor_bounds(cls) -> tuple[float | None, float | None]:
        defaults = {}
        for field in fields(cls):
            if field.name not in {"low", "high"}:
                continue
            if field.default is MISSING:
                raise ValueError(
                    f"{cls.__name__}.{field.name} must define a constructor default."
                )
            defaults[field.name] = field.default
        return defaults.get("low"), defaults.get("high")

    def _configured_bounds(self) -> tuple[float, float]:
        if self.low is None or self.high is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no static constructor bounds."
            )
        return _validate_bounds(self.__class__.__name__, self.low, self.high)

    def _dynamic_or_configured_bounds(
        self,
        resolve_dynamic: Callable[[], tuple[float, float]],
    ) -> tuple[float, float]:
        """Return constructor bounds when set, otherwise runtime dynamic bounds."""
        if self.bounds_overridden:
            return self._configured_bounds()
        low, high = resolve_dynamic()
        return _validate_bounds(self.__class__.__name__, low, high)

    def _validate_bound_policy(self, *, dynamic: bool) -> None:
        """Validate the constructor/default-bound contract for this field."""
        default_low, default_high = self._constructor_bounds()
        if dynamic:
            if (default_low is None) != (default_high is None):
                raise ValueError(
                    f"{self.__class__.__name__} dynamic defaults must define "
                    "both low and high defaults, or neither."
                )
            return
        if self.low is None or self.high is None:
            raise ValueError(
                f"{self.__class__.__name__} has static bounds, so low/high "
                "cannot be None."
            )
        if default_low is None or default_high is None:
            raise ValueError(
                f"{self.__class__.__name__} static fields must define "
                "non-None low/high defaults."
            )
        _validate_bounds(
            f"{self.__class__.__name__} defaults",
            default_low,
            default_high,
        )


def _bounds_rows(bounds: Callable[[int], tuple[Any, Any]], idx: list[int], width: tuple[int, ...]):
    """``bounds`` for each aircraft of ``idx``, as two arrays of rows of
    ``width`` - a scalar bound broadcast over a field's components."""
    rows = [bounds(i) for i in idx]
    low = np.array([np.broadcast_to(np.asarray(b[0], dtype=np.float64), width) for b in rows]).reshape(len(idx), *width)
    high = np.array([np.broadcast_to(np.asarray(b[1], dtype=np.float64), width) for b in rows]).reshape(len(idx), *width)
    return low, high


@dataclass(frozen=True)
class StepContext:
    """What a stateful field needs to update itself for one env step.

    Deliberately thin: fields already read ``bs.traf`` directly in ``get`` /
    ``get_many``, so this carries only what traffic arrays cannot supply - the
    callsigns in index order, the step length (``EnvConfig.dt``, the whole
    multi-substep step) and the sim clock.
    """

    ids: tuple[str, ...]
    dt: float
    sim_time: float
    # Seconds each aircraft has been in the environment. Carried here because
    # only the environment knows spawn times; a field cannot derive it.
    age_s: Mapping[str, float]


@dataclass(frozen=True)
class ObsField(_BoundedField, ABC):
    meta: ClassVar[ObsMeta]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.meta, ObsMeta):
            raise TypeError(f"{self.__class__.__name__}.meta must be ObsMeta.")
        if self.meta.is_pair:
            raise ValueError(
                f"{self.__class__.__name__} uses ObsField but meta.is_pair=True; "
                "use PairObsField instead."
            )
        self._validate_bound_policy(dynamic=self.meta.dynamic_bounds)

    @abstractmethod
    def get(self, idx: Any) -> Any:
        """Return the observation value for one or more BlueSky traffic indices."""

    def case_value(self, own: int, *, other: int | None = None) -> Any:
        """Its value for the aircraft at ``own``: an ownship field reads no
        other aircraft."""
        if other is not None:
            raise ValueError(f"{type(self).__name__} is read for one aircraft, not about another")
        return self.get(own)

    def get_many(self, indices: Any) -> Any:
        """Return observations for multiple traffic indices.

        Subclasses can override this with direct NumPy indexing. The fallback
        preserves scalar-field behavior for task-defined fields.
        """
        return [self.get(int(idx)) for idx in indices]

    def expected(self, idx: int) -> Any:
        """This field's value for the aircraft at ``idx``, stated plainly: the
        reference its bulk code is checked against (:meth:`check_consistency`).

        Optional, and never called at runtime. Built-in fields state it
        independently of their bulk code; a custom field may too. Its older
        name, ``_expected``, still works.
        """
        if type(self)._expected is not ObsField._expected:
            return self._expected(idx)
        raise NotImplementedError

    def _expected(self, idx: int) -> Any:
        """:meth:`expected`'s older name."""
        return self.expected(idx)

    @classmethod
    def states_expected(cls) -> bool:
        """Whether this field states :meth:`expected`, by either name."""
        return cls.expected is not ObsField.expected or cls._expected is not ObsField._expected

    def check_consistency(self, indices: Any) -> list[str]:
        """Where this field's ways of computing disagree for the live aircraft
        at ``indices``: its bulk values against one at a time, and against
        :meth:`expected` where it states it (rules in :mod:`._consistency`).
        None when they agree."""
        idx = [int(i) for i in indices]
        bulk = self.get_many(idx)
        found: list[str] = []
        for k, i in enumerate(idx):
            found += differs(bulk[k], self.get(i), f"aircraft {i}: bulk vs one at a time")
            if self.states_expected():
                found += beyond_rounding(bulk[k], self.expected(i), f"aircraft {i}: bulk vs expected")
        return found

    def values_and_bounds(self, indices: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Its values for the aircraft at ``indices`` - ``(values, low, high,
        defined)``, arrays of one shape: each value, the bounds it is held to,
        and whether it is one (every value is, for an ownship field)."""
        idx = [int(i) for i in indices]
        if not idx:
            empty = np.zeros((0, 1))
            return empty, empty, empty, empty.astype(bool)
        values = np.array(self.get_many(idx), dtype=np.float64).reshape(len(idx), -1)
        low, high = _bounds_rows(self.bounds, idx, values.shape[1:])
        return values, low, high, np.ones(values.shape, dtype=bool)

    def values_about(self, own: int, others: Any) -> Any:
        """The raw values an intruder block holds for ``own``: each of
        ``others``' own value - what it normalizes as a batch. Read as the
        assembler reads it: every aircraft's at once."""
        return np.asarray(self.get_many(list(range(bs.traf.ntraf))))[np.asarray(others, dtype=np.intp)]

    @abstractmethod
    def bounds(self, idx: int) -> tuple[float, float]:
        """Return the low/high bounds for this field at a traffic index."""

    def relative_to_own(
        self,
        *,
        low: float | None = None,
        high: float | None = None,
        normalizer: Any | None = None,
        name: str | None = None,
    ) -> PairObsField:
        """Build an intruder pair field as ``this_field(intruder) - this_field(ownship)``.

        The returned field is a :class:`PairObsField`, so it is valid only in
        ``intruder_obs_fields``. Circular fields use wrapped angle subtraction.
        ``low``/``high`` fix the delta bounds; when omitted they derive from the
        underlying field's (possibly per-aircraft dynamic) bounds.
        """
        # cycle: .observations imports .base
        from .observations.transforms import AngleDifference, Difference  # noqa: PLC0415

        field_name = name or f"relative_{self.meta.name}"
        if self.meta.circular:
            return AngleDifference(
                left=self,
                right=self,
                low=low,
                high=high,
                normalizer=normalizer,
                name=field_name,
            )
        return Difference(
            left=self,
            right=self,
            low=low,
            high=high,
            normalizer=normalizer,
            name=field_name,
        )

    def lagged(self, *, steps: int = 1):
        """This field's value ``steps`` environment steps ago (frame stacking).

        Bounds, normalizer and output size delegate to this field, so a lagged
        channel needs no separate calibration - list it alongside the live one
        to stack. Prefer lagging only quantities with no explicit rate field
        already in the observation: stacking a value whose derivative is a named
        field (``AltFt`` next to ``VsFtMin``) just widens the vector.
        """
        # cycle: .observations imports .base
        from .observations.transforms import LaggedObs  # noqa: PLC0415

        # Inherit the normalizer: the assembler picks it off the OUTER field
        # (``_field_normalizer``/``_field_output_size`` in core.services), so a
        # wrapper with ``normalizer=None`` would emit the lagged channel on the
        # RAW scale next to a normalized live one.
        return LaggedObs(inner=self, steps=int(steps), normalizer=self.normalizer)

    def stacked(self, *, depth: int = 3) -> list:
        """Frame stack: ``[live, lag-1, ..., lag-(depth-1)]`` as a field LIST.

        Replaces the plain entry in an ``obs_fields`` list - ``EnvConfig``
        flattens one level of nesting, so one entry expands to the whole stack.

        A list rather than one wide field on purpose: a normalizer reports its
        width from :meth:`Normalizer.output_size`, which returns 1 regardless of
        the field (only ``CircularNormalizer`` overrides it, to 2). A single
        depth-``N`` field would therefore be declared width 1 while emitting
        ``N``, and the observation space would silently disagree with the data.
        Keeping each lag its own field leaves every normalizer contract intact.
        """
        if int(depth) < 1:
            raise ValueError(f"stacked(depth=) must be >= 1, got {depth}.")
        return [self] + [self.lagged(steps=k) for k in range(1, int(depth))]

    def __call__(self, idx: Any) -> Any:
        return self.get(idx)


@dataclass(frozen=True)
class EnvBound:
    """A field that reads its environment - the design's shapes, its
    queryables, an aircraft's whole context - beyond the traffic arrays every
    field reads. The environment binds every field that is one as it builds,
    whatever its kind: an observation, a pair, a state field, an action.

    :meth:`agent_context` is the context the hooks get, for the aircraft at
    ``idx``. Read it knowing when the field runs:

    - in an observation, ``context.obs`` holds only the fields computed before
      this one this step - read shapes, queries, position and separation;
    - in an action, it is the state the action is applied from, before the
      step it commands.

    It is one aircraft's. A field that computes for many at once reads
    ``bound_env.query_batch(...)`` and ``bound_env.episode_shapes`` instead.
    """

    env: Any | None = dataclass_field(default=None, compare=False, kw_only=True)

    def bind_env(self, env: Any):
        return replace(self, env=env)

    @property
    def bound_env(self) -> Any:
        if self.env is None:
            raise RuntimeError(f"{self.__class__.__name__} is not bound to an env.")
        return self.env

    def agent_context(self, idx: int) -> AgentStepContext:
        """The hooks' context for the aircraft at ``idx``: its shapes, queries,
        position, separation and the task's own ``data``."""
        return self.bound_env.agent_context(idx)


@dataclass(frozen=True)
class EnvObsField(ObsField, EnvBound, ABC):
    """An observation field that reads its environment (:class:`EnvBound`).

    Custom fields subclass it as they would ``ObsField``, without passing
    ``env=self`` into each constructor: the environment binds them.
    """


@dataclass(frozen=True)
class PairObsField(_BoundedField, ABC):
    meta: ClassVar[ObsMeta]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.meta, ObsMeta):
            raise TypeError(f"{self.__class__.__name__}.meta must be ObsMeta.")
        if not self.meta.is_pair:
            raise ValueError(
                f"{self.__class__.__name__} uses PairObsField but meta.is_pair=False; "
                "use ObsField instead."
            )
        self._validate_bound_policy(dynamic=self.meta.dynamic_bounds)

    @abstractmethod
    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        """Return an ownship-relative observation for one or more other indices."""

    def case_value(self, own: int, *, other: int | None = None) -> Any:
        """Its value for the aircraft at ``own`` about the one at ``other``."""
        if other is None:
            raise ValueError(f"{type(self).__name__} is read about another aircraft: name it")
        return self.get_pair(own, other)

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        """Return pair observations for one ownship and multiple intruders."""
        return [self.get_pair(own_idx, int(other_idx)) for other_idx in other_indices]

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        """This field's value for one ownship and intruder, stated plainly: the
        pair-field counterpart of :meth:`ObsField.expected`. Its older name,
        ``_expected_pair``, still works."""
        if type(self)._expected_pair is not PairObsField._expected_pair:
            return self._expected_pair(own_idx, other_idx)
        raise NotImplementedError

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        """:meth:`expected_pair`'s older name."""
        return self.expected_pair(own_idx, other_idx)

    @classmethod
    def states_expected(cls) -> bool:
        """Whether this field states :meth:`expected_pair`, by either name."""
        return (
            cls.expected_pair is not PairObsField.expected_pair
            or cls._expected_pair is not PairObsField._expected_pair
        )

    def check_consistency(self, indices: Any) -> list[str]:
        """Where this field's ways of computing disagree, with ``indices``
        every live aircraft: each ownship's pairs against one at a time, its
        row of the pair matrix against its pairs (a subset of ownships' rows
        too), and against :meth:`expected_pair` where it states it (rules in
        :mod:`._consistency`). None when they agree."""
        owns = [int(i) for i in indices]
        matrix = np.asarray(self.get_pair_matrix(np.array(owns)))
        if matrix.shape[:2] != (len(owns), bs.traf.ntraf):
            return [f"pair matrix shape {matrix.shape[:2]}, not {(len(owns), bs.traf.ntraf)}"]
        found: list[str] = []
        for row, own in enumerate(owns):
            others = [j for j in owns if j != own]
            pairs = self.get_pairs(own, others)
            for k, other in enumerate(others):
                where = f"own {own}, other {other}"
                found += differs(pairs[k], self.get_pair(own, other), f"{where}: bulk vs one at a time")
                if self.states_expected():
                    found += beyond_rounding(
                        matrix[row, other], self.expected_pair(own, other), f"{where}: bulk vs expected"
                    )
            found += differs(matrix[row][others], pairs, f"own {own}: pair matrix vs pairs")
        subset = owns[1::3]
        if subset:
            found += differs(self.get_pair_matrix(np.array(subset)), matrix[1::3], "a subset of ownships' rows")
        return found

    def values_and_bounds(self, indices: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Its values for the aircraft at ``indices``, each with every live
        aircraft - ``(values, low, high, defined)``, arrays of one shape - an
        aircraft with itself undefined (NaN)."""
        idx = [int(i) for i in indices]
        n = bs.traf.ntraf
        if not idx:
            empty = np.zeros((0, n))
            return empty, empty, empty, empty.astype(bool)
        values = np.array(self.get_pair_matrix(np.array(idx)), dtype=np.float64).reshape(len(idx), n, -1)
        defined = np.ones(values.shape, dtype=bool)
        for row, own in enumerate(idx):
            defined[row, own] = False
        values[~defined] = np.nan
        low, high = _bounds_rows(self.bounds, idx, values.shape[2:])
        return values, low[:, None], high[:, None], defined

    def values_about(self, own: int, others: Any) -> Any:
        """The raw values an intruder block holds for ``own``: its pairs with
        ``others`` - what it normalizes as a batch."""
        return self.get_pairs(own, [int(i) for i in others])

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        """Pair observations for several ownships against EVERY live aircraft.

        Row ``r`` holds ownship ``own_indices[r]`` against aircraft
        ``0 .. ntraf-1``; the entry for the ownship itself is undefined and
        never read. The observation assembler calls this once per field per
        step instead of :meth:`get_pairs` once per ownship, so a field that
        computes all pairs at once pays its per-call cost once.

        The default assembles rows from :meth:`get_pairs`. Overrides must
        agree with it value for value - ``tests/test_field_batching.py``
        checks every field.
        """
        n = int(bs.traf.ntraf)
        rows = []
        for own in np.asarray(own_indices, dtype=np.intp).ravel():
            others = np.array([j for j in range(n) if j != own], dtype=np.intp)
            values = np.asarray(self.get_pairs(int(own), others), dtype=np.float64)
            row = np.full((n, *values.shape[1:]), np.nan)
            row[others] = values
            rows.append(row)
        if not rows:
            return np.empty((0, n))
        return np.stack(rows)

    @abstractmethod
    def bounds(self, own_idx: int) -> tuple[float, float]:
        """Return the low/high bounds for this pair field."""

    def lagged(self, *, steps: int = 1):
        """This pair field's value ``steps`` environment steps ago.

        Only sound on fields invariant to ownship ROTATION - conflict-geometry
        scalars (``ConflictTlosS``, ``ConflictTcpaS``, the separation-at-CPA
        fields), not anything in the ownship track frame (``RelPos*``,
        ``RelVel*``, along/cross realized accelerations). See
        :class:`~.observations.transforms.LaggedPair` for why.
        """
        # cycle: .observations imports .base
        from .observations.transforms import LaggedPair  # noqa: PLC0415

        # Inherits the inner normalizer - see :meth:`ObsField.lagged`.
        return LaggedPair(inner=self, steps=int(steps), normalizer=self.normalizer)

    def stacked(self, *, depth: int = 3) -> list:
        """Frame stack ``[live, lag-1, ..., lag-(depth-1)]`` - see
        :meth:`ObsField.stacked`. Same rotation-invariance caveat as
        :meth:`lagged`."""
        if int(depth) < 1:
            raise ValueError(f"stacked(depth=) must be >= 1, got {depth}.")
        return [self] + [self.lagged(steps=k) for k in range(1, int(depth))]

    def __call__(self, own_idx: int, other_idx: Any) -> Any:
        return self.get_pair(own_idx, other_idx)


@dataclass(frozen=True)
class EnvPairObsField(PairObsField, EnvBound, ABC):
    """An intruder observation field that reads its environment
    (:class:`EnvBound`), as :class:`EnvObsField` is for ownship fields."""


@dataclass(frozen=True)
class TaskContextObsField(EnvObsField, ABC, Generic[ContextT]):
    """Ownship field backed by the environment's generic task context."""

    low: float = 0.0
    high: float = 1.0
    normalizer: Any | None = None

    def context(self, idx: int) -> ContextT:
        return cast(ContextT, self.bound_env.agent_context(idx).data)

    def bounds(self, idx: int) -> tuple[float, float]:
        del idx
        return float(self.low), float(self.high)


@dataclass(frozen=True)
class TaskContextPairObsField(EnvPairObsField, ABC, Generic[ContextT]):
    """Intruder field backed by the environment's generic task context."""

    low: float = 0.0
    high: float = 1.0
    normalizer: Any | None = None

    def context(self, idx: int) -> ContextT:
        return cast(ContextT, self.bound_env.agent_context(idx).data)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        del own_idx
        return float(self.low), float(self.high)


@dataclass(frozen=True)
class SwitchActionMixin(ABC):
    """An ON/OFF action: 1 turns it on, 0 turns it off.

    A binary action (:attr:`kind`): it sits in the ``binary`` part of the action
    space, which holds only 0 and 1, so its bounds are fixed at ``(0, 1)`` and
    there is no normalizer and no grid.
    """

    kind: ClassVar[ActionKind] = ActionKind.BINARY
    low: float = dataclass_field(default=0.0, init=False)
    high: float = dataclass_field(default=1.0, init=False)
    normalizer: Any | None = dataclass_field(default=None, init=False)
    grid: Grid | None = dataclass_field(default=None, init=False)

    def _validate_switch_policy(self) -> None:
        if (self.low, self.high, self.normalizer) != (0.0, 1.0, None):
            raise ValueError(
                f"{self.__class__.__name__} is a 0/1 switch: its bounds are (0, 1) "
                f"and it has no normalizer, got low={self.low!r}, "
                f"high={self.high!r}, normalizer={self.normalizer!r}."
            )

    def bounds(self, idx: int) -> tuple[float, float]:
        return 0.0, 1.0

    def switch_command(self, value: float) -> bool:
        """Whether ``value`` turns the switch on: 1 does, 0 does not."""
        if value not in (0.0, 1.0):
            raise ValueError(
                f"{self.__class__.__name__} is a switch: it takes 0 or 1, "
                f"got {value!r}."
            )
        return value == 1.0

    def switch_on_value(self) -> float:
        """The action value that turns this switch on."""
        return 1.0

    @abstractmethod
    def current_switch_state(self, idx: int) -> bool:
        """Whether the switch is on for the aircraft at ``idx``."""


@dataclass(frozen=True)
class ActionField(_BoundedField, ABC):
    meta: ClassVar[ActionMeta]
    #: The part of the action space this action is in.
    kind: ClassVar[ActionKind] = ActionKind.CONTINUOUS

    #: Whole steps the value - or the target it commands - is put on, after
    #: the normalizer: see :class:`~.grid.Grid`.
    grid: Grid | None = None

    def nominal(self, idx: int) -> float:
        """What the action's value counts from for the aircraft at ``idx``, in
        its own unit, so the target it commands is ``nominal + value``: for a
        delta, the present value (or a waypoint's) it adds to; 0 for an
        absolute action."""
        return 0.0

    def reach(self, idx: int) -> tuple[float, float]:
        """The values this action can take for the aircraft at ``idx`` now, in
        its own unit: its bounds, unless they are narrower than what it can
        command - a delta's bounds are symmetric about its nominal so a scaled
        0 stays "no change"; its reach is not. What a normalizer that keeps 0
        fixed by construction (whole steps) clips to."""
        return self.bounds(idx)

    def acting(self, idx: int) -> Any:
        """The action as it acts for the aircraft at ``idx`` now: itself, or -
        an action that changes unit with the regime, as a crossover speed
        action does above its crossover - what its normalizer and grid apply
        to (``bounds``, ``reach``, ``nominal``, ``normalizer``, ``grid``)."""
        return self

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.meta, ActionMeta):
            raise TypeError(f"{self.__class__.__name__}.meta must be ActionMeta.")
        if self.meta.requires_on and self.meta.mode is not ActionMode.SWITCH:
            raise ValueError(
                f"{self.__class__.__name__} requires_on is only valid for "
                "switch actions."
            )
        if self.meta.mode is ActionMode.SWITCH:
            if not isinstance(self, SwitchActionMixin):
                raise TypeError(
                    f"{self.__class__.__name__} switch actions must inherit "
                    "SwitchActionMixin."
                )
            self._validate_switch_policy()
        elif isinstance(self, SwitchActionMixin):
            raise ValueError(
                f"{self.__class__.__name__} inherits SwitchActionMixin but "
                "is not configured as a switch action."
            )
        self._validate_bound_policy(dynamic=self.meta.dynamic_bounds)
        self._validate_grid()

    def _validate_grid(self) -> None:
        """A grid needs one value to put on it - and, on an axis that wraps
        (a heading), one that goes round it evenly."""
        grid = self.grid
        if grid is None:
            return
        name = self.__class__.__name__
        if not isinstance(grid, Grid):
            raise TypeError(f"{name} grid must be a Grid or None, got {grid!r}.")
        output_size = getattr(self, "output_size", None)
        if callable(output_size) and int(output_size()) != 1:
            raise ValueError(f"{name} takes {output_size()} values: no grid fits it.")
        # A delta's change on its grid does not wrap; a value or a target on
        # an axis that does must go round it in whole steps.
        axis = self.meta.control_axis
        period = None if axis is None else axis.period
        snaps_the_axis = grid.on == "target" or self.meta.mode is not ActionMode.DELTA
        if period and snaps_the_axis:
            turns = period / grid.step
            if abs(turns - round(turns)) > 1e-9:
                raise ValueError(
                    f"{name} grid of {grid.step:g} on the {axis} axis, which wraps "
                    f"at {period:g}: it must divide {period:g}, or the values either "
                    "side of the wrap are off it."
                )

    @abstractmethod
    def set(self, idx: int, value: float) -> None:
        """Apply an action value to the aircraft at the BlueSky traffic index."""

    @abstractmethod
    def bounds(self, idx: int) -> tuple[float, float]:
        """Return the low/high action bounds for this field at a traffic index."""

    def __call__(self, idx: int, value: float) -> None:
        self.set(idx, value)

    def case_value(self, own: int, *, other: int | None = None) -> Any:
        """What it holds for the aircraft at ``own`` - the target its clearance
        holds (``held``), ``None`` while it flies its own - so a case can
        apply a value, then read what it commands."""
        if other is not None:
            raise ValueError(f"{type(self).__name__} is read for one aircraft, not about another")
        held = getattr(self, "held", None)
        if held is None:
            raise NotImplementedError(f"{type(self).__name__} cannot say what it holds")
        return held(own)


@dataclass(frozen=True)
class EnvActionField(ActionField, EnvBound, ABC):
    """An action that reads its environment (:class:`EnvBound`): one that
    commands toward a shape or a queryable - direct to the nearest gate - as
    :class:`EnvObsField` reads them."""
