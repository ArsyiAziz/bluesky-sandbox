from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar, cast

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


class ActionMode(StrEnum):
    """How an action value should be interpreted."""

    ABSOLUTE = "absolute"
    DELTA = "delta"
    SWITCH = "switch"


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
    actions are scalar controls implemented by :class:`SwitchActionMixin` and
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


@dataclass(frozen=True)
class _BoundedField:

    # ---- optional per-aircraft state ------------------------------------- #
    # A field that reads state the simulator does not keep (a rate, an
    # accumulator, a history) declares here how that state is maintained.
    # Overriding any of these marks the field as stateful, and the environment
    # drives it: ``on_substep`` each physics substep, ``on_aircraft_removed``
    # when an aircraft despawns, ``on_episode_reset`` at reset.
    #
    # The point of putting them on the field rather than in a registry the
    # environment maintains: a store and its "drop this aircraft" call can no
    # longer drift apart. When they lived in separate files, adding a tracker
    # and forgetting its ``forget_x`` leaked one aircraft's state onto the next
    # one to reuse its callsign.
    #
    # Cost note: overriding also opts the field INTO per-substep work. Nothing
    # is recorded for a field no config uses, and dt/simdt is commonly 100
    # substeps per env step, so this is the difference between paying for a
    # tracker and not.

    def on_substep(self, ctx: SubstepContext) -> None:
        """Update this field's state for one physics substep."""

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
                "on_substep",
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


@dataclass(frozen=True)
class SubstepContext:
    """What a stateful field needs to update itself for one physics substep.

    Deliberately thin: fields already read ``bs.traf`` directly in ``get`` /
    ``get_many``, so this carries only what traffic arrays cannot supply - the
    callsigns in index order, the substep length, and the sim clock.
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

    def get_many(self, indices: Any) -> Any:
        """Return observations for multiple traffic indices.

        Subclasses can override this with direct NumPy indexing. The fallback
        preserves scalar-field behavior for task-defined fields.
        """
        return [self.get(int(idx)) for idx in indices]

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
        from .observations import AngleDifference, Difference

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
        from .observations import LaggedObs

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
class EnvObsField(ObsField, ABC):
    """Observation field that reads task-specific state from its environment.

    Task constructors bind these fields while building ``EnvConfig``. Task
    authors can then write custom observation fields as normal ``ObsField``
    subclasses without passing ``env=self`` into every field constructor.
    """

    env: Any | None = dataclass_field(default=None, compare=False)

    def bind_env(self, env: Any):
        return replace(self, env=env)

    @property
    def bound_env(self) -> Any:
        if self.env is None:
            raise RuntimeError(f"{self.__class__.__name__} is not bound to an env.")
        return self.env


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

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        """Return pair observations for one ownship and multiple intruders."""
        return [self.get_pair(own_idx, int(other_idx)) for other_idx in other_indices]

    @abstractmethod
    def bounds(self, own_idx: int) -> tuple[float, float]:
        """Return the low/high bounds for this pair field."""

    def lagged(self, *, steps: int = 1):
        """This pair field's value ``steps`` environment steps ago.

        Only sound on fields invariant to ownship ROTATION - conflict-geometry
        scalars (``ConflictTlosS``, ``ConflictTcpaS``, the separation-at-CPA
        fields), not anything in the ownship track frame (``RelPos*``,
        ``RelVel*``, along/cross realized accelerations). See
        :class:`~.observations.LaggedPair` for why.
        """
        from .observations import LaggedPair

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
class EnvPairObsField(PairObsField, ABC):
    """Pair observation field that reads task-specific state from its environment.

    Task constructors bind these fields while building ``EnvConfig``, matching
    :class:`EnvObsField` for ownship fields.
    """

    env: Any | None = dataclass_field(default=None, compare=False)

    def bind_env(self, env: Any):
        return replace(self, env=env)

    @property
    def bound_env(self) -> Any:
        if self.env is None:
            raise RuntimeError(f"{self.__class__.__name__} is not bound to an env.")
        return self.env


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


class SwitchActionMixin(ABC):
    """Behavior required by scalar ON/OFF action fields."""

    switch_threshold: ClassVar[float] = 0.5

    def _validate_switch_policy(self) -> None:
        threshold = self.switch_on_value()
        if not math.isfinite(threshold):
            raise ValueError(
                f"{self.__class__.__name__} switch threshold must be finite, "
                f"got {threshold!r}."
            )
        if self.low is not None and self.high is not None:
            lo, hi = _validate_bounds(
                f"{self.__class__.__name__} switch bounds",
                self.low,
                self.high,
            )
            if not lo <= threshold <= hi:
                raise ValueError(
                    f"{self.__class__.__name__} switch threshold must be "
                    f"within bounds ({lo}, {hi}), got {threshold}."
                )

    def is_on(self, value: float) -> bool:
        """Return whether a switch-style action value should be treated as ON."""
        return value >= self.switch_on_value()

    def switch_command(self, value: float) -> bool | None:
        """Return ON/OFF for a switch command, or None to leave state unchanged."""
        return self.is_on(value)

    @abstractmethod
    def current_switch_state(self, idx: int) -> bool:
        """Return current switch state for hold-band semantics."""

    def switch_on_value(self) -> float:
        """Return an action value that forces this switch ON."""
        return float(self.switch_threshold)


@dataclass(frozen=True)
class ActionField(_BoundedField, ABC):
    meta: ClassVar[ActionMeta]

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

    @abstractmethod
    def set(self, idx: int, value: float) -> None:
        """Apply an action value to the aircraft at the BlueSky traffic index."""

    @abstractmethod
    def bounds(self, idx: int) -> tuple[float, float]:
        """Return the low/high action bounds for this field at a traffic index."""

    def __call__(self, idx: int, value: float) -> None:
        self.set(idx, value)
