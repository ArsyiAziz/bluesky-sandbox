from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import bluesky as bs
import numpy as np

from bluesky_sandbox.interface.fields import actions
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.actions.mask import check_action_masks
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
DEFAULT_SIMDT = None
DEFAULT_ASAS_DT = None
DEFAULT_CD_METHOD = "CSTATEBASED"
DEFAULT_RESO_METHOD = None
DEFAULT_INTRUDER_OBS_BOUNDS = "ownship"
INTRUDER_OBS_BOUNDS = ("ownship", "intruder")
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
#: user's config file says the moment the runtime initializes. Type-level
#: lookups read this instead.
_REQUESTED_MODEL: str | None = None


def requested_performance_model() -> str:
    """The model this process asked for, falling back to BlueSky's setting."""
    return _REQUESTED_MODEL or str(
        getattr(bs.settings, "performance_model", "openap")
    ).lower()


_SETTINGS_CFG_READ = False


def read_bluesky_settings() -> None:
    """Load settings.cfg into ``bs.settings`` if BlueSky has not yet done so.

    ``bs.init`` reads the file, but configs are built before it runs, and then
    ``bs.settings`` holds only the built-in defaults its modules registered. Read
    the file once, and never after ``bs.init``: re-reading it then would put
    the file's values back over settings changed at runtime.
    """
    global _SETTINGS_CFG_READ
    if _SETTINGS_CFG_READ or bs.sim is not None:
        return
    # Deferred like ``_ensure_navdb_loaded``: pathfinder resolves where
    # settings.cfg lives, and importing simtime registers BlueSky's own simdt
    # default for a file that does not set one.
    from bluesky import pathfinder, settings  # noqa: PLC0415
    from bluesky.core import simtime  # noqa: F401, PLC0415

    pathfinder.init()
    settings.init()
    _SETTINGS_CFG_READ = True


def bluesky_simdt_s() -> float:
    """BlueSky's configured physics step, ``bs.settings.simdt`` (s).

    The sandbox never writes this setting, so it stays BlueSky's value however
    many envs this process has built with other steps.
    """
    read_bluesky_settings()
    return float(bs.settings.simdt)


def apply_performance_model(model: str | None) -> str:
    """Publish ``model`` to ``bs.settings`` and return the resolved name.

    Type-level lookups (envelope ceilings, VMO/MMO, MTOW) ask
    ``bs.settings.performance_model`` which database to read, and they run
    during scenario sampling - which happens before any runtime exists, and in
    the designer's preview happens without an ``EnvConfig`` at all. Anything
    that samples geometry must call this first, or a BADA design silently
    samples its envelopes from OpenAP and fails on the types OpenAP lacks.

    Setting the value does not initialize BlueSky; ``runtime.configure`` still
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
    intruder_obs_bounds:
        Whose bounds normalize a non-pair field in an intruder row, for fields
        whose bounds differ per aircraft (an envelope such as ``CasKts`` or
        ``AltFt``). ``"ownship"`` (default) scales every intruder by the
        observing aircraft's envelope: one shared scale per row block, read as
        "that speed on my scale", but a faster type can fall outside it (and
        saturate under a ``clipped`` normalizer). ``"intruder"`` scales each by
        its own envelope, so a row matches that aircraft's own ownship values;
        the same normalized value then means a different physical value per
        intruder, which only aircraft type in the observation disambiguates.
        No effect on pair fields, fields with fixed bounds (set ``low``/``high``
        for one absolute scale), or fields without a normalizer. Applies to
        ``critic_intruder_obs_fields`` too.
    state_fields:
        Fields computed for each agent every step like its observation, but
        never observed: no agent, actor or critic, sees them. Hooks read them
        by name, in raw values - ``context.state["ownship"]["alt_ft"]``, and
        ``batch.state`` for the batched hooks. For what a reward, a done
        condition or an info needs but the policy must not depend on.
    intruder_state_fields:
        The same per other aircraft, as ``intruder_obs_fields`` are:
        ``context.state["intruders"]["dist_to_own_nm"][i]`` for intruder row ``i``,
        in the observation's intruder order.
    action_fields:
        Ordered action field objects that form each agent's action vector.
    allowed_aircraft:
        Whitelist of ICAO aircraft-type designators that agents may fly.
    dt:
        Simulation time (seconds) advanced per ``step()`` call.
    simdt:
        BlueSky physics time step in seconds. ``None`` takes BlueSky's own
        ``bs.settings.simdt`` - from settings.cfg, else BlueSky's built-in
        default - and resolves it here, so ``config.simdt`` is always a number.
    asas_dt:
        Interval in seconds between BlueSky conflict-detection updates.
        ``None`` keeps BlueSky's own ``bs.settings.asas_dt``. Must be a whole
        number of ``simdt``, and ``dt`` a whole number of it: detection runs on
        a fixed sim-time grid, so any other value gives each step's
        observation separation data of a different age, or none at all.
        Conflict and LoS durations only change at this interval.
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
    # Computed like observations, seen by no agent: read in hooks as context.state.
    state_fields: list[ObsField] | None = None
    intruder_state_fields: list[ObsField | PairObsField] | None = None
    intruder_obs_bounds: str = DEFAULT_INTRUDER_OBS_BOUNDS
    action_fields: list[ActionField] = field(
        default_factory=lambda: [actions.HdgDeg(), actions.SpdKts(), actions.AltFt()]
    )
    allowed_aircraft: list[str] = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_AIRCRAFT)
    )
    dt: float = DEFAULT_DT
    simdt: float | None = DEFAULT_SIMDT
    # Conflict-detection interval, applied to BlueSky's ``asas`` timer at
    # construction and after every reset. ``None`` keeps ``bs.settings.asas_dt``,
    # which only exists once ``bs.init`` has run, so the runtime checks it there.
    asas_dt: float | None = DEFAULT_ASAS_DT
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
        if self.state_fields is not None:
            self.state_fields = _bind_env_obs_fields(env, self.state_fields)
        if self.intruder_state_fields is not None:
            self.intruder_state_fields = _bind_env_obs_fields(
                env, self.intruder_state_fields
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
        self.state_fields = _flatten_obs_fields(self.state_fields)
        self.intruder_state_fields = _flatten_obs_fields(self.intruder_state_fields)
        if not (
            isinstance(self.dt, (int, float)) and self.dt > 0 and np.isfinite(self.dt)
        ):
            raise ValueError(f"dt must be a positive finite number, got {self.dt!r}.")
        if self.intruder_obs_bounds not in INTRUDER_OBS_BOUNDS:
            raise ValueError(
                f"intruder_obs_bounds must be one of {INTRUDER_OBS_BOUNDS}, "
                f"got {self.intruder_obs_bounds!r}."
            )
        if self.simdt is None:
            self.simdt = bluesky_simdt_s()
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
        if self.asas_dt is not None:
            validate_asas_dt(self.asas_dt, dt=self.dt, simdt=self.simdt)
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

        # State fields follow the observation rules: the ownship list holds
        # single-aircraft fields only, the intruder list either kind.
        pair_in_state = [f for f in self.state_fields or () if isinstance(f, PairObsField)]
        if pair_in_state:
            raise ValueError(
                "state_fields contains pair-only fields (intruder-relative; not "
                f"valid for the ownship): {[f.meta.name for f in pair_in_state]}"
            )
        for name, fields, kinds in (
            ("state_fields", self.state_fields, (ObsField,)),
            ("intruder_state_fields", self.intruder_state_fields, (ObsField, PairObsField)),
        ):
            invalid_state = [f for f in fields or () if not isinstance(f, kinds)]
            if invalid_state:
                raise TypeError(
                    f"{name} must contain "
                    f"{' or '.join(k.__name__ for k in kinds)} instances, got "
                    f"{invalid_state!r}."
                )

        check_action_masks(self.action_fields)

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

def validate_asas_dt(
    asas_dt: float,
    *,
    dt: float,
    simdt: float,
    from_bluesky_default: bool = False,
) -> None:
    """Refuse a conflict-detection interval that does not line up with the steps.

    BlueSky knows nothing of the agent step: it runs detection on a fixed
    sim-time grid from the start of the episode, every ``asas_dt // simdt``
    physics ticks. So ``asas_dt`` must be a whole number of ``simdt``, or BlueSky
    silently rounds it down, and ``dt`` a whole number of ``asas_dt``, or each
    step's observation reads separation data of a different age - and an
    interval longer than ``dt`` leaves some steps with no detection at all.

    ``from_bluesky_default`` marks a value read from ``bs.settings.asas_dt``
    because the config left it unset, so the error says where it came from.
    """
    origin = (
        " EnvConfig.asas_dt is unset, so BlueSky's own default "
        "(bs.settings.asas_dt, from settings.cfg) applies; set asas_dt explicitly."
        if from_bluesky_default
        else ""
    )
    if not (
        isinstance(asas_dt, (int, float)) and asas_dt > 0 and np.isfinite(asas_dt)
    ):
        raise ValueError(
            f"asas_dt must be a positive finite number, got {asas_dt!r}.{origin}"
        )
    ticks = asas_dt / simdt
    if round(ticks) < 1 or not np.isclose(ticks, round(ticks), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"asas_dt ({asas_dt}) must be an integer multiple of simdt ({simdt}); "
            f"got asas_dt/simdt={ticks!r}. BlueSky would otherwise round it to "
            f"{max(1, int(ticks + 1e-9)) * simdt!r} s without saying so.{origin}"
        )
    per_step = dt / asas_dt
    if not np.isclose(per_step, round(per_step), rtol=0.0, atol=1e-9) or (
        round(per_step) < 1
    ):
        consequence = (
            "some steps would get no conflict detection at all"
            if asas_dt > dt
            else "the separation data behind each step's observation would be "
            "a different age from step to step"
        )
        raise ValueError(
            f"dt ({dt}) must be an integer multiple of asas_dt ({asas_dt}); got "
            f"dt/asas_dt={per_step!r}. BlueSky runs conflict detection on a fixed "
            f"sim-time grid, so {consequence}. Use a divisor of dt, e.g. "
            f"asas_dt={dt!r} for one detection per step.{origin}"
        )


def resolve_spawn_aircraft_types(config: EnvConfig, spawn: SpawnConfig) -> None:
    """Resolve a sampled spawn config's aircraft types against ``allowed_aircraft``."""

    def _resolve_aircraft_type(ac_type, label: str) -> Categorical:
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

    spawn.aircraft_type = _resolve_aircraft_type(
        spawn.aircraft_type,
        "SpawnConfig.aircraft_type",
    )
    for i, region in enumerate(spawn.regions):
        if region.aircraft_type is not None:
            region.aircraft_type = _resolve_aircraft_type(
                region.aircraft_type,
                f"SpawnRegion[{i}].aircraft_type",
            )
