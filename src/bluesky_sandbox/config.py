from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import bluesky as bs
import numpy as np

from bluesky_sandbox.interface.fields import actions
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import (
    ActionField,
    EnvObsField,
    EnvPairObsField,
    ObsField,
    PairObsField,
)
from bluesky_sandbox.interface.task import TaskInfoProvider
from bluesky_sandbox.sim.performance.bada import bada_install_hint
from bluesky_sandbox.sim.performance.models import available_types
from bluesky_sandbox.sim.sampling.distributions import Categorical
from bluesky_sandbox.sim.spawn import SpawnConfig

DEFAULT_ALLOWED_AIRCRAFT = ("B744",)
DEFAULT_DT = 1.0
DEFAULT_SIMDT = 0.05
DEFAULT_CD_METHOD = "CSTATEBASED"
DEFAULT_RESO_METHOD = None
DEFAULT_PZ_RADIUS_NM = None
DEFAULT_PZ_HEIGHT_FT = None
DEFAULT_LOOKAHEAD_S = None
DEFAULT_PERFORMANCE_MODEL = None


def _flatten_obs_fields(fields):
    """Expand one level of nesting in an observation-field list.

    Lets a single entry stand for several channels, which is what
    :meth:`ObsField.stacked` returns - ``field.stacked(depth=3)`` is one line in
    a config and three fields in the observation. Only one level: a field is
    never itself a sequence, so deeper nesting is a mistake, not a stack.
    """
    if fields is None:
        return None
    out: list = []
    for entry in fields:
        if isinstance(entry, (list, tuple)):
            out.extend(entry)
        else:
            out.append(entry)
    return out


def _bind_env_obs_fields(
    env,
    fields: Sequence[ObsField | PairObsField],
) -> list[ObsField | PairObsField]:
    return [
        field.bind_env(env)
        if isinstance(field, EnvObsField | EnvPairObsField)
        else field
        for field in fields
    ]


#: The model this process asked for. ``bs.settings.performance_model`` is not a
#: reliable record of it: ``bs.init()`` re-reads ``settings.cfg`` and overwrites
#: the value, so a design that selects BADA silently reverts to whatever the
#: user's config file says the moment the runtime initialises. Type-level
#: lookups read this instead.
_REQUESTED_MODEL: str | None = None


def requested_performance_model() -> str:
    """The model this process asked for, falling back to BlueSky's setting."""
    return _REQUESTED_MODEL or str(
        getattr(bs.settings, "performance_model", "openap")
    ).lower()


def apply_performance_model(model: str | None) -> str:
    """Publish ``model`` to ``bs.settings`` and return the resolved name.

    Type-level lookups (envelope ceilings, VMO/MMO, MTOW) ask
    ``bs.settings.performance_model`` which database to read, and they run
    during scenario sampling - which happens before any runtime exists, and in
    the designer's preview happens without an ``EnvConfig`` at all. Anything
    that samples geometry must call this first, or a BADA design silently
    samples its envelopes from OpenAP and fails on the types OpenAP lacks.

    Setting the value does not initialise BlueSky; ``runtime.configure`` still
    owns bs.init and still refuses to switch models after it.
    """
    global _REQUESTED_MODEL

    resolved = (model or _REQUESTED_MODEL or
                getattr(bs.settings, "performance_model", "openap")).lower()
    _REQUESTED_MODEL = resolved
    if getattr(bs.settings, "performance_model", None) != resolved:
        bs.settings.performance_model = resolved
    return resolved


def _available_aircraft(model: str | None = None) -> frozenset[str]:
    """ICAO types the configured performance model carries, lowercased."""
    resolved = (model or requested_performance_model()).lower()
    try:
        return available_types(resolved)
    except Exception as e:
        if resolved == "bada":
            hint = "" if bada_install_hint() in str(e) else f" {bada_install_hint()}"
            raise RuntimeError(
                f"Could not load BADA aircraft database: {e}.{hint}"
            ) from e
        raise


@dataclass
class EnvConfig:
    """Static configuration consumed by ``BlueskyBaseEnvironment``.

    Episode resources such as spawn, queryables, and airspace bounds live on
    :class:`~bluesky_sandbox.sim.sampling.EpisodeSpec`. Config owns the stable API
    shape and simulator settings.

    Parameters
    ----------
    obs_fields:
        Ordered observation field objects that form each agent's ownship
        observation vector.
    intruder_obs_fields:
        Ordered observation field objects emitted per other aircraft
        alongside the ownship block. Use ``PairObsField`` objects such as
        ``obs.DistToOwnNm()`` or ``obs.AltFt().relative_to_own()`` when the
        feature depends on both ownship and intruder. The base env emits a
        variable-length ``Sequence(Box)``; downstream
        :class:`IntruderPaddingWrapper` pads to a fixed
        ``max_intruders x (output_dim + 1)`` block, where ``+1`` is a
        validity flag. ``max_intruders`` derives from
        :meth:`SpawnConfig.max_aircraft` minus one (ownship). Set this
        field to ``None`` to disable intruder observations entirely.
    action_fields:
        Ordered action field objects that form each agent's action vector.
    allowed_aircraft:
        Whitelist of ICAO aircraft-type designators that agents may fly.
    dt:
        Simulation time (seconds) advanced per ``step()`` call.
    simdt:
        BlueSky physics time step in seconds (default 0.05 s = 20 Hz).
    cd_method:
        BlueSky conflict-detection method name passed to ``CDMETHOD``
        (default ``"cstatebased"``).
    performance_model:
        BlueSky performance model name, e.g. ``"openap"`` or ``"bada"``.
        ``None`` uses the current ``bs.settings.performance_model``.
    task_info_providers:
        Optional ordered task info providers. Each provider is invoked every
        step before the reward / termination / truncation hooks. Use these for
        structured goal, constraint, metric, or latch data that should be
        exposed through the agent ``info`` dict.

    Reward / termination / truncation are no longer config functions: they are
    ``@overridable`` hooks on the environment (``reward`` / ``terminated`` /
    ``truncated``), defaulting to ``0.0`` reward and never done.
    """

    obs_fields: list[ObsField] = field(
        default_factory=lambda: [
            obs.LatDeg(),
            obs.LonDeg(),
            obs.AltFt(),
            obs.HdgDeg(),
            obs.CasKts(),
        ]
    )
    intruder_obs_fields: list[ObsField | PairObsField] | None = None
    # Privileged, critic-only observation fields (asymmetric actor-critic / CTDE).
    # These are appended to the *critic's* view of the observation but never
    # reach the actor, so the deployed policy stays a function of the ordinary
    # ``obs_fields`` / ``intruder_obs_fields`` only. Use them for information the
    # value function may exploit at training time but that the policy should not
    # depend on - e.g. other aircraft's route intent (their active-route
    # waypoint), or global-state features. ``critic_obs_fields`` extend the
    # ownship block; ``critic_intruder_obs_fields`` extend each intruder row.
    # Both default to ``None`` (symmetric: critic and actor see the same obs).
    critic_obs_fields: list[ObsField] | None = None
    critic_intruder_obs_fields: list[ObsField | PairObsField] | None = None
    action_fields: list[ActionField] = field(
        default_factory=lambda: [actions.HdgDeg(), actions.SpdKts(), actions.AltFt()]
    )
    allowed_aircraft: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_AIRCRAFT)
    )
    dt: float = DEFAULT_DT
    simdt: float = DEFAULT_SIMDT
    cd_method: str = DEFAULT_CD_METHOD
    # Conflict resolution method applied via BlueSky's ``RESO`` command at each
    # reset. ``None`` (or ``"OFF"``) leaves auto-resolution off - the usual
    # choice for RL, where the agent resolves conflicts itself while conflict
    # *detection* (``cd_method``) still runs for observations/rewards.
    reso_method: str | None = DEFAULT_RESO_METHOD
    # Protected-zone geometry and look-ahead time, applied via ``ZONER`` /
    # ``ZONEDH`` / ``DTLOOK`` at reset. ``None`` keeps BlueSky's default.
    pz_radius_nm: float | None = DEFAULT_PZ_RADIUS_NM
    pz_height_ft: float | None = DEFAULT_PZ_HEIGHT_FT
    lookahead_s: float | None = DEFAULT_LOOKAHEAD_S
    performance_model: str | None = DEFAULT_PERFORMANCE_MODEL
    # Uniform wind field, re-applied every reset (steady mean) and per-step
    # (turbulence). ``wind_dir_deg`` is aviation-standard - the direction the
    # wind blows FROM, degrees true clockwise from north (270 = westerly,
    # pushing aircraft east). ``wind_kts`` is its mean speed; ``turbulence_kts``
    # an Ornstein-Uhlenbeck gust RMS decorrelating over ``gust_tau_s`` seconds.
    # Both speeds 0 = no wind. Consumed by ``BlueSkyRuntime.apply_wind`` and the
    # base env's per-step gust advance.
    wind_dir_deg: float = 270.0
    wind_kts: float = 0.0
    turbulence_kts: float = 0.0
    gust_tau_s: float = 30.0
    task_info_providers: list[TaskInfoProvider] = field(default_factory=list)

    def bind_env(self, env) -> None:
        """Bind environment-aware observation fields to an env instance."""
        self.obs_fields = _bind_env_obs_fields(env, self.obs_fields)
        if self.intruder_obs_fields is not None:
            self.intruder_obs_fields = _bind_env_obs_fields(
                env,
                self.intruder_obs_fields,
            )
        if self.critic_obs_fields is not None:
            self.critic_obs_fields = _bind_env_obs_fields(env, self.critic_obs_fields)
        if self.critic_intruder_obs_fields is not None:
            self.critic_intruder_obs_fields = _bind_env_obs_fields(
                env,
                self.critic_intruder_obs_fields,
            )

    def __post_init__(self) -> None:
        # Before any field validation: ``field.stacked(depth=n)`` puts a LIST in
        # the entry it replaces, so flatten it into real channels first.
        self.obs_fields = _flatten_obs_fields(self.obs_fields)
        self.intruder_obs_fields = _flatten_obs_fields(self.intruder_obs_fields)
        self.critic_obs_fields = _flatten_obs_fields(self.critic_obs_fields)
        self.critic_intruder_obs_fields = _flatten_obs_fields(
            self.critic_intruder_obs_fields
        )
        if not (
            isinstance(self.dt, (int, float)) and self.dt > 0 and np.isfinite(self.dt)
        ):
            raise ValueError(f"dt must be a positive finite number, got {self.dt!r}.")
        if not (
            isinstance(self.simdt, (int, float))
            and self.simdt > 0
            and np.isfinite(self.simdt)
        ):
            raise ValueError(
                f"simdt must be a positive finite number, got {self.simdt!r}."
            )
        if self.simdt > self.dt:
            raise ValueError(
                f"simdt ({self.simdt}) must not exceed dt ({self.dt}); "
                "the physics step cannot be coarser than the agent step."
            )
        ratio = self.dt / self.simdt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"dt ({self.dt}) must be an integer multiple of simdt "
                f"({self.simdt}); got dt/simdt={ratio!r}."
            )
        for provider in self.task_info_providers:
            if not callable(provider):
                raise ValueError("each task info provider must be callable.")

        invalid_actions = [
            f for f in self.action_fields if not isinstance(f, ActionField)
        ]
        if invalid_actions:
            raise TypeError(
                "action_fields must contain ActionField instances, got "
                f"{invalid_actions!r}."
            )

        invalid_obs = [
            f
            for f in self.obs_fields
            if not isinstance(f, (ObsField, PairObsField))
        ]
        if invalid_obs:
            raise TypeError(
                f"obs_fields must contain ObsField instances, got {invalid_obs!r}."
            )

        # Pair-fields (e.g. dist_to_own_nm) need a second aircraft, so
        # they only make sense for intruders - reject them in ownship obs.
        pair_in_own = [f for f in self.obs_fields if isinstance(f, PairObsField)]
        if pair_in_own:
            raise ValueError(
                f"obs_fields contains pair-only fields (intruder-relative; "
                f"not valid for ownship): {[f.meta.name for f in pair_in_own]}"
            )

        if self.intruder_obs_fields is not None:
            invalid_intruders = [
                f
                for f in self.intruder_obs_fields
                if not isinstance(f, (ObsField, PairObsField))
            ]
            if invalid_intruders:
                raise TypeError(
                    "intruder_obs_fields must contain ObsField or PairObsField "
                    f"instances, got {invalid_intruders!r}."
                )

        # Privileged critic-only fields follow the same rules as their actor-side
        # counterparts: ownship-block fields cannot be pair-only (they need a
        # single aircraft); intruder-block fields may be either.
        if self.critic_obs_fields is not None:
            invalid_critic_own = [
                f for f in self.critic_obs_fields if not isinstance(f, ObsField)
            ]
            if invalid_critic_own:
                raise TypeError(
                    "critic_obs_fields must contain ObsField instances, got "
                    f"{invalid_critic_own!r}."
                )
            pair_in_critic_own = [
                f for f in self.critic_obs_fields if isinstance(f, PairObsField)
            ]
            if pair_in_critic_own:
                raise ValueError(
                    "critic_obs_fields contains pair-only fields (intruder-relative; "
                    f"not valid for the ownship block): "
                    f"{[f.meta.name for f in pair_in_critic_own]}"
                )

        if self.critic_intruder_obs_fields is not None:
            invalid_critic_intr = [
                f
                for f in self.critic_intruder_obs_fields
                if not isinstance(f, (ObsField, PairObsField))
            ]
            if invalid_critic_intr:
                raise TypeError(
                    "critic_intruder_obs_fields must contain ObsField or "
                    f"PairObsField instances, got {invalid_critic_intr!r}."
                )

        self.performance_model = apply_performance_model(self.performance_model)
        available = _available_aircraft(self.performance_model)
        assert "b744" in available, (
            "B744 is not present in the performance model database; "
            "check that the OpenAP/BADA data files are installed correctly "
            "(run `python -m bluesky_sandbox.doctor` to see what resolves)."
        )
        invalid = [ac for ac in self.allowed_aircraft if ac.lower() not in available]
        if invalid:
            raise ValueError(
                f"Aircraft type(s) not found in {self.performance_model} database: "
                f"{invalid}."
            )

        self.allowed_aircraft = [ac.upper() for ac in self.allowed_aircraft]

def normalize_spawn_aircraft_types(config: EnvConfig, spawn: SpawnConfig) -> None:
    """Normalize one sampled spawn config against static allowed aircraft."""

    def _normalize_aircraft_type(ac_type, label: str) -> Categorical:
        if isinstance(ac_type, Categorical):
            unknown = [
                t for t in ac_type.weights if t.upper() not in config.allowed_aircraft
            ]
            if unknown:
                raise ValueError(
                    f"{label} references types not in allowed_aircraft: {unknown}"
                )
            return ac_type
        if isinstance(ac_type, str):
            return Categorical({ac_type.upper(): 1.0})
        return Categorical({t: 1.0 for t in config.allowed_aircraft})

    spawn.aircraft_type = _normalize_aircraft_type(
        spawn.aircraft_type,
        "SpawnConfig.aircraft_type",
    )
    for i, region in enumerate(spawn.regions):
        if region.aircraft_type is not None:
            region.aircraft_type = _normalize_aircraft_type(
                region.aircraft_type,
                f"SpawnRegion[{i}].aircraft_type",
            )
