"""Generate a runnable task package from a :class:`DesignSpec`.

The designer's structured half (geometry/spawn/queryables) is data; the logic
half (fields, task info, reward, termination, truncation, and lifecycle hooks)
is Python. "Generate task structure" emits a small package that captures both:
geometry/spawn/queryables land in ``scenario.py``; static settings, fields, and
hooks land in ``env.py``.

The generated layout is a self-contained package, written wherever the caller
asks (task packages live outside this library - see ``write_task``):

    <package>/
        __init__.py     - exposes Env / Scenario aliases
        design.json     - original DesignSpec for reloading in the designer
        scenario.py     - geometry/spawn/queryables + Scenario subclass
        config.py       - static EnvConfig fields/settings
        setup.py        - module-level helpers/constants the hooks lean on
        env.py          - hooks / BlueskyEnv subclass
        __main__.py     - tiny rollout demo
        README.md

:func:`generate_task` returns ``{relative_path: source}`` (for download/preview);
:func:`write_task` writes that to disk.
"""

from __future__ import annotations

import ast
import copy
import keyword
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bluesky_sandbox.interface.fields.base import ActionKind

from . import setup_code
from .builder import (
    build_design_config,
    build_scenario,
    with_inferred_temporal_tracking,
)
from .catalog import hooks as _hook_catalog
from .emit import emit_env_sources, emit_scenario_sources
from .recording import (
    RecordOptions,
    main_with_wandb,
    record_options,
    recording_block,
    recording_imports,
)
from .setup_code import PRELUDE
from .spec import SCENARIO_HOOKS, DesignSpec, TaskInfoSpec
from .typed_api import MODULE as TYPES_MODULE
from .typed_api import Renderer, TaskTypes


def _valid_package_name(name: str) -> str:
    """Slug an arbitrary title into a dotted-import-safe package name."""
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower()).strip("_")
    if not base:
        raise ValueError(f"package name {name!r} slugs to empty.")
    if base[0].isdigit():
        base = f"task_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_task"
    return base


def _class_stem(pkg: str) -> str:
    """Return a PascalCase class stem from a valid package name."""
    return "".join(part[:1].upper() + part[1:] for part in pkg.split("_") if part)


def generate_task(spec: DesignSpec, package_name: str) -> dict[str, str]:
    """Return ``{relative_path: file_contents}`` for a task package.

    The design's editable code modules (``spec.code``, e.g.
    ``custom_fields.py``) are written into the package, and every code/field
    reference that targets one of them is rewritten to be package-qualified
    (``custom_fields:MyField`` -> ``<pkg>.custom_fields:MyField``) so the
    package is self-contained and importable.
    """
    pkg = _valid_package_name(package_name)
    class_stem = _class_stem(pkg)
    template = template_of(spec)
    processes, watch = run_options(spec)
    record = record_options(spec.metadata) if template != "plain" else None
    if record is not None and watch:
        raise ValueError(
            "watch and record both draw worker 0 - one in a window, one offscreen: pick one."
        )
    title = str(spec.metadata.get("name", pkg))
    meta = dict(spec.metadata)

    # Stems of the design's editable modules (e.g. {"task", "custom_fields"}).
    stems = {f[:-3] for f in spec.code if f.endswith(".py")}

    def rw(ref: Any) -> Any:
        if isinstance(ref, str) and ":" in ref:
            mod = ref.split(":", 1)[0]
            if mod in stems:
                return f"{pkg}.{ref}"
        return ref

    spec_for_pkg = with_inferred_temporal_tracking(
        DesignSpec.from_dict(copy.deepcopy(spec.to_dict()))
    )
    e = spec_for_pkg.env
    # task-info providers are imported by env.py, so package-qualify them.
    e.task_info_providers = [rw(p) for p in e.task_info_providers]

    files: dict[str, str] = {
        f"{pkg}/__init__.py": _init_py(pkg, title, class_stem, meta),
        f"{pkg}/design.json": spec.to_json(),
        f"{pkg}/__main__.py": _main_py(),
        f"{pkg}/README.md": _readme_md(pkg, title, e, meta, template),
    }

    # Write the design's editable code modules verbatim (custom_fields.py, etc.).
    # Reward/termination/truncation are emitted as env hooks, not a task.py file.
    for filename, source in spec.code.items():
        files[f"{pkg}/{filename}"] = source

    # The design itself is emitted as Python (not JSON), split like the
    # hand-written tasks: scenario resources in scenario.py, static config in
    # config.py, hooks in env.py.
    scenario_sources = emit_scenario_sources(spec_for_pkg)
    env_sources = emit_env_sources(spec_for_pkg, package=pkg)
    files[f"{pkg}/scenario.py"] = _scenario_py(
        class_stem,
        scenario_sources,
        spec_for_pkg.scenario_setup,
        spec_for_pkg.scenario_hooks,
    )
    files[f"{pkg}/config.py"] = _config_py(
        env_sources,
    )
    # The design's keys, typed for editors - when the design builds.
    typed = _task_types(spec)
    if typed is not None:
        files[f"{pkg}/{TYPES_MODULE}.py"] = typed.module_source()
    files[f"{pkg}/env.py"], files[f"{pkg}/setup.py"] = _env_py(
        pkg,
        class_stem,
        env_sources,
        e.hook_setup,
        e.hooks,
        e.task_info_setup,
        e.task_info,
        e.task_info_providers,
        has_cost=defines_cost(e),
        typed=typed,
    )
    if template == "sb3":
        files[f"{pkg}/train.py"] = _sb3_train_py(
            pkg,
            class_stem,
            privileged=bool(e.critic_obs_fields or e.critic_intruder_obs_fields),
            notes=sb3_notes(spec),
            processes=processes,
            watch=watch,
            record=record,
        )
    if template == "rl":
        files[f"{pkg}/train.py"] = _train_py(
            pkg,
            class_stem,
            privileged=bool(e.critic_obs_fields or e.critic_intruder_obs_fields),
            has_cost=defines_cost(e),
            processes=processes,
            watch=watch,
            record=record,
        )
    return files


def _task_types(spec: DesignSpec) -> TaskTypes | None:
    """The task's typed classes, or None for a design that does not build -
    it still generates, as it always has, just without them."""
    try:
        config = build_design_config(spec)
        support = build_scenario(spec).support()
    except Exception:
        return None
    return TaskTypes(config, support, _module_aliases(spec))


def _module_aliases(spec: DesignSpec) -> dict[str, str]:
    """The modules the setup code imports whole, by the name it gives them -
    ``{"numpy": "np"}`` - so the annotations read as the code does."""
    aliases: dict[str, str] = {}
    for source in (spec.env.task_info_setup, spec.env.hook_setup):
        try:
            tree = ast.parse(source or "")
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases.setdefault(alias.name, alias.asname or alias.name)
    return aliases


def _typing_imports(render: Renderer | None) -> str:
    """What the annotations need, imported only for type checkers."""
    if render is None:
        return ""
    lines = render.import_lines(indent="    ")
    if render.local_used:
        names = ", ".join(sorted(render.local_used))
        lines.append(f"    from .{TYPES_MODULE} import {names}")
    if not lines:
        return ""
    block = "\n".join(lines)
    return f"\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n{block}\n"


def _ref_import(ref: str, pkg: str, alias: str) -> str:
    """Turn a ``module:attr`` ref into an import line binding it to ``alias``."""
    mod, _, attr = ref.partition(":")
    if mod == pkg or mod.startswith(pkg + "."):
        rel = mod[len(pkg):].lstrip(".") or "__init__"
        return f"from .{rel} import {attr} as {alias}"
    return f"from {mod} import {attr} as {alias}"


def write_task(spec: DesignSpec, package_name: str, dest_dir: str | Path) -> Path:
    """Write a generated task package under ``dest_dir``; return the package dir."""
    files = generate_task(spec, package_name)
    dest = Path(dest_dir)
    for rel, content in files.items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    pkg = next(iter(files)).split("/", 1)[0]
    return dest / pkg


# --------------------------------------------------------------------------- #
# file templates                                                              #
# --------------------------------------------------------------------------- #
def _init_py(pkg: str, title: str, class_stem: str, meta: dict[str, Any] | None = None) -> str:
    version = str((meta or {}).get("version", "") or "0")
    return f'''"""Generated task package: {title}.

Created by the bluesky-sandbox Environment Designer. ``scenario.py`` holds the
geometry/spawn/queryables; ``config.py`` holds static fields/settings;
``setup.py`` holds the module-level helpers; ``env.py`` holds env hooks.
"""

from .scenario import {class_stem}Scenario
from .env import {class_stem}Env

#: Design version, carried from the designer's metadata. Bump it there when
#: the design changes so a checkpoint or a run can be traced to what produced
#: it - the package name alone does not distinguish two edits of one design.
__version__ = "{version}"

Scenario = {class_stem}Scenario
Env = {class_stem}Env

__all__ = [
    "__version__",
    "{class_stem}Scenario",
    "{class_stem}Env",
    "Scenario",
    "Env",
]
'''


def _scenario_setup_block(scenario_setup: str, existing_imports: str) -> str:
    """Module-level scenario setup, with imports the template already has removed."""
    body = setup_code.dedupe_imports(scenario_setup or "", existing_imports).rstrip()
    return f"\n{body}\n" if body else ""


def _emit_scenario_hooks(scenario_hooks: dict[str, str]) -> str:
    """Emit the design's scenario hooks as ``@staticmethod`` blocks.

    Only hooks the design actually customized are emitted - an absent hook is
    the identity, and emitting a pass-through would just be boilerplate that
    obscures which hooks a design really uses.
    """
    blocks: list[str] = []
    for name in sorted(scenario_hooks):
        spec_entry = SCENARIO_HOOKS.get(name)
        if spec_entry is None:
            continue  # validated in DesignSpec; belt-and-braces for hand-built specs
        args, purpose = spec_entry
        body = (scenario_hooks.get(name) or "").rstrip() or "return geometry"
        indented = "\n".join(
            ("        " + ln) if ln.strip() else "" for ln in body.splitlines()
        )
        blocks.append(
            f"    @staticmethod\n"
            f"    def _{name}({', '.join(args)}):\n"
            f'        """{purpose}"""\n'
            f"{indented}\n"
        )
    return ("\n".join(blocks) + "\n") if blocks else ""


def _scenario_py(
    class_stem: str,
    scenario_sources: dict[str, str],
    scenario_setup: str = "",
    scenario_hooks: dict[str, str] | None = None,
) -> str:
    # Sampled region params, waypoint stacks and scenario hooks all need the
    # per-episode geometry hook, so any of them selects the parametric template.
    # ``scenario_setup`` alone does not - module-level helpers are emitted into
    # either template - but a hook body will usually want them, and the
    # parametric form is what gives the hook a geometry dict to post-process.
    scenario_hooks = dict(scenario_hooks or {})
    if (
        scenario_sources.get("region_param_dists") or scenario_hooks
    ):
        return _parametric_scenario_py(
            class_stem, scenario_sources, scenario_setup, scenario_hooks
        )
    setup_block = _scenario_setup_block(scenario_setup, scenario_sources["imports"])
    return f'''"""Scenario sampler for this generated task."""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario

{scenario_sources["imports"]}{setup_block}


class {class_stem}Scenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    def __init__(self) -> None:
        REGIONS = {scenario_sources["regions"]}
        transform = {scenario_sources["transform"]}
        rotation = None
        groups = None
        if transform and transform.get("rotation"):
            rot = transform["rotation"]
            pivot = rot.get("pivot")
            rotation = {{"angle": rot["angle_deg"], "pivot": tuple(pivot) if pivot else None}}
        if transform and transform.get("groups"):
            groups = tuple(transform["groups"])
        super().__init__(
            airspace_bounds={scenario_sources["airspace"]},
            spawn={scenario_sources["spawn"]},
            queryables={scenario_sources["queryables"]},
            rotation=rotation,
            groups=groups,
            sampled_waypoints={scenario_sources["sampled_waypoints"]},
            waypoint_fields={scenario_sources["waypoint_fields"]},
        )

'''


def _parametric_scenario_py(
    class_stem: str,
    scenario_sources: dict[str, str],
    scenario_setup: str = "",
    scenario_hooks: dict[str, str] | None = None,
) -> str:
    """Scenario template for specs with sampled region params.

    Regions become a ``_regions(draw)`` factory and the geometry a
    ``_geometry(REGIONS)`` builder; :class:`RegionParamSampler` wires them into
    the per-episode ``episode_geometry_fn`` hook and the widest-support static
    geometry, so ``support()`` covers every sampled shape.
    """
    region_param_dists = scenario_sources.get("region_param_dists") or "{}"
    scenario_hooks = dict(scenario_hooks or {})
    setup_block = _scenario_setup_block(scenario_setup, scenario_sources["imports"])
    hook_methods = _emit_scenario_hooks(scenario_hooks)
    hooks_doc = (
        """

``scenario_setup`` / ``_episode_geometry`` carry the design's own per-episode
sampling: the structured geometry is rebuilt first, then handed to the hook,
which returns the geometry dict the episode actually runs."""
        if "episode_geometry" in scenario_hooks
        else ""
    )
    # Chain the design's hook after the structured rebuild. Without it the
    # sampler's bound method goes straight in, so hook-free designs regenerate
    # byte-identically.
    if "episode_geometry" in scenario_hooks:
        episode_geometry_wiring = """
        def episode_geometry_fn(rng):
            return self._episode_geometry(dict(sampler.episode_geometry(rng)), rng)

"""
        episode_geometry_expr = "episode_geometry_fn"
    else:
        episode_geometry_wiring = ""
        episode_geometry_expr = "sampler.episode_geometry"
    sampler_args = "\n            self._regions, self._REGION_PARAM_DISTS, self._geometry\n        "
    return f'''"""Scenario sampler for this generated task.

Named regions carry sampled footprint params (see ``_REGION_PARAM_DISTS``):
each episode redraws them and rebuilds the geometry via the
``episode_geometry_fn`` hook; the static/support geometry is the union of the
shapes at every parameter endpoint.{hooks_doc}
"""

from __future__ import annotations

from bluesky_sandbox.sim.scenario import RandomizedScenario, RegionParamSampler

{scenario_sources["imports"]}{setup_block}


class {class_stem}Scenario(RandomizedScenario):
    """Episode sampler for this generated task."""

    # {{'<region>.<param>': (low, high) | scipy dist}} - drawn per episode.
    _REGION_PARAM_DISTS = {region_param_dists}

    @staticmethod
    def _regions(draw):
        return {scenario_sources["regions"]}

{hook_methods}    @staticmethod
    def _geometry(REGIONS):
        return {{
            "airspace_bounds": {scenario_sources["airspace"]},
            "spawn": {scenario_sources["spawn"]},
            "queryables": {scenario_sources["queryables"]},
            "sampled_waypoints": {scenario_sources["sampled_waypoints"]},
        }}

    def __init__(self) -> None:
        transform = {scenario_sources["transform"]}
        rotation = None
        groups = None
        if transform and transform.get("rotation"):
            rot = transform["rotation"]
            pivot = rot.get("pivot")
            rotation = {{"angle": rot["angle_deg"], "pivot": tuple(pivot) if pivot else None}}
        if transform and transform.get("groups"):
            groups = tuple(transform["groups"])
        sampler = RegionParamSampler({sampler_args})
        geometry = self._geometry(sampler.support_regions())
{episode_geometry_wiring}        super().__init__(
            airspace_bounds=geometry["airspace_bounds"],
            spawn=geometry["spawn"],
            queryables=geometry["queryables"],
            rotation=rotation,
            groups=groups,
            sampled_waypoints=geometry["sampled_waypoints"],
            waypoint_fields={scenario_sources["waypoint_fields"]},
            episode_geometry_fn={episode_geometry_expr},
        )

'''


def _config_py(env_sources: dict[str, str]) -> str:
    intruder_obs_fields = env_sources["intruder_obs_fields"]
    intruder_expr = (
        "None"
        if intruder_obs_fields == "None"
        else f"list({intruder_obs_fields})"
    )

    # Privileged critic-only field lists are emitted only when configured, so
    # symmetric designs keep the compact config they had before.
    critic_lines = ""
    for key in ("critic_obs_fields", "critic_intruder_obs_fields"):
        src = env_sources.get(key, "None")
        if src and src != "None":
            critic_lines += f"    {key}=list({src}),\n"
    # Likewise only when changed from the default ("ownship").
    intruder_bounds = env_sources.get("intruder_obs_bounds", "'ownship'")
    if intruder_bounds != "'ownship'":
        critic_lines += f"    intruder_obs_bounds={intruder_bounds},\n"

    return f'''"""Static environment config for this generated task."""

from __future__ import annotations

from bluesky_sandbox.config import EnvConfig

{env_sources["imports"]}

CONFIG = EnvConfig(
    obs_fields=list({env_sources["obs_fields"]}),
    intruder_obs_fields={intruder_expr},
{critic_lines}    action_fields=list({env_sources["action_fields"]}),
    allowed_aircraft=list({env_sources["allowed_aircraft"]}),
    dt={env_sources["dt"]},
    simdt={env_sources["simdt"]},
    asas_dt={env_sources["asas_dt"]},
    cd_method={env_sources["cd_method"]},
    reso_method={env_sources["reso_method"]},
    pz_radius_nm={env_sources["pz_radius_nm"]},
    pz_height_ft={env_sources["pz_height_ft"]},
    lookahead_s={env_sources["lookahead_s"]},
    performance_model={env_sources["performance_model"]},
    wind_dir_deg={env_sources["wind_dir_deg"]},
    wind_kts={env_sources["wind_kts"]},
    turbulence_kts={env_sources["turbulence_kts"]},
    gust_tau_s={env_sources["gust_tau_s"]},
)
'''


# reward/terminated/truncated are always-present hooks (default 0.0 / never done).
_DEFAULT_HOOK_BODIES = {
    "reward": "return 0.0",
    "terminated": "return False",
    "truncated": "return False",
}


def _emit_hooks(
    hooks: dict[str, str], signature: Callable[[str], str | None] | None = None
) -> str:
    """Emit env hook overrides.

    reward/terminated/truncated are always emitted (body from ``hooks`` or their
    default) unless the design defines their batched counterpart instead; other
    hooks are emitted only when the user customized them - uncustomized ones
    inherit the base behavior (no ``super()`` boilerplate).
    """
    sigs = {h["name"]: h["def_signature"] for h in _hook_catalog()}
    # A batched hook replaces its per-agent one, which is then not written.
    batched = {name for name in _DEFAULT_HOOK_BODIES if hooks.get(f"{name}_batch")}
    names = [n for n in _DEFAULT_HOOK_BODIES if n not in batched] + [
        n for n in sorted(hooks) if n not in _DEFAULT_HOOK_BODIES
    ]
    blocks: list[str] = []
    for name in names:
        sig = (signature(name) if signature else None) or sigs.get(name)
        if not sig:
            continue  # unknown hook (e.g. API changed); skip rather than emit invalid code
        body = (hooks.get(name) or _DEFAULT_HOOK_BODIES.get(name) or "").rstrip() or "pass"
        indented = "\n".join(("        " + ln) if ln.strip() else "" for ln in body.splitlines())
        blocks.append(f"    def {name}{sig}:\n{indented}")
    return ("\n\n".join(blocks) + "\n") if blocks else ""


#: The kinds of package the designer generates: ``plain`` is the environment
#: and a smoke rollout; ``rl`` adds ``train.py``, a training-loop scaffold;
#: ``sb3`` adds a ``train.py`` that trains it with Stable-Baselines3 PPO.
TEMPLATES = ("plain", "rl", "sb3")


def template_of(spec: DesignSpec) -> str:
    """The template the design asks for (``metadata["template"]``), else plain."""
    template = str(spec.metadata.get("template") or "plain")
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose one of {TEMPLATES}")
    return template


def run_options(spec: DesignSpec) -> tuple[int, bool]:
    """How a generated training script runs the env: in how many processes
    (``metadata["processes"]``, 1 when unset) and whether one copy is drawn
    (``metadata["watch"]``)."""
    value = spec.metadata.get("processes")
    processes = 1 if value is None else int(value)
    if processes < 1:
        raise ValueError(f"processes must be at least 1, got {processes}")
    return processes, bool(spec.metadata.get("watch"))


def defines_cost(env: Any) -> bool:
    """Whether the design defines a cost hook, per agent or batched."""
    hooks = getattr(env, "hooks", None) or {}
    return any((hooks.get(h) or "").strip() for h in ("cost", "cost_batch"))


def sb3_notes(spec: DesignSpec) -> list[dict[str, str]]:
    """What Stable-Baselines3 cannot do with this design: ``error`` where the
    SB3 package cannot train it at all, ``warning`` where it trains something
    other than the design asks for. Empty when the design does not build."""
    try:
        config = build_design_config(spec)
    except Exception:  # a broken design says so elsewhere
        return []
    notes: list[dict[str, str]] = []
    kinds = {getattr(f, "kind", None) for f in config.action_fields}
    if ActionKind.BINARY in kinds:
        notes.append({
            "level": "error",
            "message": "A switch (0/1) action makes the action space a Dict of a continuous and a "
            "binary part; SB3's algorithms do not train Dict action spaces. Drop the switch "
            "actions, or use the RL template.",
        })
    if defines_cost(spec.env):
        notes.append({
            "level": "warning",
            "message": "SB3 has no constrained algorithm: PPO optimizes the reward alone and "
            "info[\"cost\"] is ignored. Fold the cost into the reward (a Lagrangian penalty), "
            "or use the RL template.",
        })
    if spec.env.critic_obs_fields or spec.env.critic_intruder_obs_fields:
        notes.append({
            "level": "warning",
            "message": "SB3's actor and critic see one observation, so the critic cannot be "
            "privileged: train.py drops the critic-only fields rather than show them to the "
            "actor.",
        })
    notes.append({
        "level": "info",
        "message": "SB3 needs one number per agent per step: a reward hook that returns an "
        "array of components must return their sum here.",
    })
    return notes


def _sb3_train_py(
    pkg: str,
    class_stem: str,
    privileged: bool,
    notes: list[dict[str, str]],
    processes: int = 1,
    watch: bool = False,
    record: RecordOptions | None = None,
) -> str:
    """``train.py`` for Stable-Baselines3: every agent shares one PPO policy,
    through SuperSuit's PettingZoo-to-vector-env conversion. With ``record``,
    every copy records clips, uploaded to wandb as training goes."""
    listed = "".join(
        "\n" + textwrap.fill(f"- {n['level'].upper()}: {n['message']}", width=76, subsequent_indent="  ")
        for n in notes
        if n["level"] != "info"
    )
    blocked = [n for n in notes if n["level"] == "error"]
    guard = (
        "\n    raise NotImplementedError(\n        "
        + repr("SB3 cannot train this design: " + " ".join(n["message"] for n in blocked))
        + "\n    )\n"
        if blocked
        else ""
    )
    actor_view = (
        """

class ActorView(BaseParallelWrapper):
    \"\"\"The observation without its critic-only fields: SB3's critic cannot
    have them apart from the actor, and the actor must not see them.\"\"\"

    def observation_space(self, agent):
        return actor_observation_space(self.env.observation_space(agent))

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return {a: actor_obs(o) for a, o in obs.items()}, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return {a: actor_obs(o) for a, o in obs.items()}, rewards, terminations, truncations, infos
"""
        if privileged
        else ""
    )
    view_imports = (
        "from pettingzoo.utils.wrappers import BaseParallelWrapper\n\n"
        "from bluesky_sandbox import actor_obs, actor_observation_space\n"
        if privileged
        else ""
    )
    wrap_view = "\n    env = ActorView(env)" if privileged else ""
    notes_block = f"\n\nWhat SB3 does differently from this design:{listed}\n" if listed else "\n"
    if record is None:
        stdlib = recorder = sandbox = optional = block = upload_callback = ""
        make = f"    env = {class_stem}Env(render_mode=render_mode){wrap_view}"
        record_arg = ""
        record_doc = ""
        vec_record = ""
        learn = "model.learn(total_timesteps=total_timesteps)"
        drain = ""
        main = "if __name__ == \"__main__\":\n    train()\n"
        install = "pip install stable-baselines3 supersuit"
    else:
        stdlib, sandbox, optional = recording_imports(record, clips=True)
        recorder = "from stable_baselines3.common.callbacks import BaseCallback\n"
        block = recording_block(record, pkg)
        make = (
            "    # A recorded copy draws offscreen, with the recording's driver and views.\n"
            "    drawing = (\n"
            "        {\"frame_driver\": FRAME_DRIVER, \"views\": recorded_views()}\n"
            "        if render_mode == \"rgb_array\"\n"
            "        else {}\n"
            "    )\n"
            f"    env = {class_stem}Env(render_mode=render_mode, **drawing){wrap_view}"
        )
        upload_callback = """

class UploadClips(BaseCallback):
    \"\"\"Uploads each clip the copies finish, as training goes.\"\"\"

    def __init__(self, clips: Clips) -> None:
        super().__init__()
        self.clips = clips

    def _on_step(self) -> bool:
        for clip in self.clips.new():
            upload(clip)
        return True
"""
        record_arg = ",\n    record: bool = True"
        record_doc = "\n    With ``record``, every copy records clips (``RECORDING``)."
        vec_record = ", RECORDING if record else None"
        learn = "model.learn(total_timesteps=total_timesteps, callback=UploadClips(clips))"
        drain = (
            "\n        # The clips the copies finished as they closed."
            "\n        for clip in clips.new():"
            "\n            upload(clip)"
            "\n        clips.close()"
        )
        main = main_with_wandb("train()")
        install = "pip install stable-baselines3 supersuit \"bluesky-sandbox[recording]\" wandb"
    clips_open = "\n    clips = Clips(RECORDING)" if record is not None else ""
    return f'''"""Train this task with Stable-Baselines3: one PPO policy shared by every
agent, the multi-agent env made a vector env by SuperSuit.

    {install}
    python -m {pkg}.train

``train(n_processes=4)`` runs four copies of the env at once, one simulator
per process.
{notes_block}"""

from __future__ import annotations

{stdlib}import gymnasium as gym
{view_imports}from stable_baselines3 import PPO
{recorder}
from bluesky_sandbox.integrations import sb3_vec_env, wrap_parallel_env
{sandbox}
from .env import {class_stem}Env
{optional}{block}{actor_view}

def make_env(render_mode=None):
    """The env as SB3 needs it: a fixed pool of agent IDs, intruders padded to
    a fixed shape."""
{make}
    max_agents = env.unwrapped.scenario.support().max_aircraft
    return wrap_parallel_env(env, max_agents=max_agents)
{upload_callback}

def train(
    total_timesteps: int = 100_000,
    seed: int = 0,
    n_processes: int = {processes},
    watch: bool = {watch}{record_arg},
) -> PPO:
    """Train, the episodes run in ``n_processes`` processes at once - BlueSky
    is one simulator per process - every agent of each an entry of SB3's
    vector env. With ``watch``, one copy is drawn in a pygame window.{record_doc}"""{guard}
    vec = sb3_vec_env(make_env, n_processes, watch{vec_record}){clips_open}
    # From here on the worker processes are running: close them whatever
    # happens, or this process cannot exit.
    try:
        policy = "MultiInputPolicy" if isinstance(vec.observation_space, gym.spaces.Dict) else "MlpPolicy"
        model = PPO(policy, vec, seed=seed, verbose=1)
        {learn}
        model.save("ppo_{pkg}")
        return model
    finally:
        vec.close(){drain}


{main}'''


def _step_unpack(has_cost: bool, indent: str) -> str:
    """The step call, unpacked into what this design's MDP gives."""
    line = f"{indent}next_obs, rewards, terminations, truncations, infos = env.step(actions)\n"
    if has_cost:
        line += (
            f"{indent}# This design defines a cost: each agent's is in its info.\n"
            f'{indent}costs = {{agent: info["cost"] for agent, info in infos.items()}}\n'
        )
    return line


def _train_py(
    pkg: str,
    class_stem: str,
    privileged: bool,
    has_cost: bool,
    processes: int = 1,
    watch: bool = False,
    record: RecordOptions | None = None,
) -> str:
    """``train.py``: how a training loop is arranged around this design's MDP.

    Deliberately not runnable: the networks, buffer and update step are the
    reader's to supply. It fixes what depends on the design - which view each
    network sees (the privileged critic's, when the design has critic-only
    fields), and what a transition holds (a cost, and a critic for it, when the
    design defines a cost) - and leaves the learning to them.
    """
    cost_value_def = (
        """

def build_cost_value(observation_space):
    \"\"\"Your cost critic, built from the CRITIC observation space: it
    estimates the cost-to-go a constraint (a Lagrangian, say) is enforced on.\"\"\"
    raise NotImplementedError("build_cost_value: return a cost-value network.")
"""
        if has_cost
        else ""
    )
    add_args = "obs, action, value, reward, terminated, truncated"
    if has_cost:
        add_args = "obs, action, value, cost_value, reward, cost, terminated, truncated"
    bootstrap_args = "value, cost_value" if has_cost else "value"
    update_args = "policy, value, cost_value, buffer" if has_cost else "policy, value, buffer"
    actor = "actor_obs(ob)" if privileged else "ob"
    critic = "critic_obs(ob)" if privileged else "ob"
    critic_next = "critic_obs(next_obs[agent])" if privileged else "next_obs[agent]"
    actor_space = "actor_observation_space(space)" if privileged else "space"
    critic_space = "critic_observation_space(space)" if privileged else "space"
    imports = (
        """from bluesky_sandbox import (
    actor_obs,
    actor_observation_space,
    critic_obs,
    critic_observation_space,
)

"""
        if privileged
        else ""
    )
    views_note = (
        """
The design declares critic-only observation fields, so every observation
carries ``critic_ownship`` / ``critic_intruders`` alongside the ordinary blocks.
The env never merges them: the actor sees ``actor_obs(ob)``, the critic
``critic_obs(ob)``, and the buffer keeps the raw observation both derive from.
"""
        if privileged
        else ""
    )
    cost_note = (
        """
The design defines a cost, so each step's ``info[agent]["cost"]`` is stored
with the reward, and a cost critic is bootstrapped beside the value.
"""
        if has_cost
        else ""
    )
    cost_value_build = f"\n    cost_value = build_cost_value({critic_space})" if has_cost else ""
    cost_value_call = f"\n            cost_values[agent] = cost_value({critic})" if has_cost else ""
    cost_values_init = ", cost_values" if has_cost else ""
    cost_values_init_rhs = ", {}" if has_cost else ""
    add_call = (
        """            buffer.add(
                agent,
                obs[agent],
                action,
                values[agent],
                cost_values[agent],
                rewards[agent],
                costs[agent],
                terminations[agent],
                truncations[agent],
            )"""
        if has_cost
        else """            buffer.add(
                agent,
                obs[agent],
                action,
                values[agent],
                rewards[agent],
                terminations[agent],
                truncations[agent],
            )"""
    )
    bootstrap_call = (
        f"last = {critic_next}\n                buffer.bootstrap(agent, value(last), cost_value(last))"
        if has_cost
        else f"buffer.bootstrap(agent, value({critic_next}))"
    )
    if processes > 1:
        return _train_vec_py(pkg, class_stem, privileged, has_cost, processes, watch, record)
    render_mode = '"pygame"' if watch else "None"
    draw = "\n            env.render()" if watch else ""
    draw_step = "\n        env.render()" if watch else ""
    loop = f"""    obs, _infos = env.reset(seed=seed)

    # Aircraft arrive on a schedule: wait for the first agent to size the nets.
    while not env.agents and not env.episode_done:
        obs, *_ = env.step({{}})
    space = env.observation_space(env.agents[0])
    policy = build_policy({actor_space})
    value = build_value({critic_space}){cost_value_build}
    buffer = Buffer()

    for _ in range(steps):
        if env.episode_done:
            obs, _infos = env.reset(){draw}
            continue

        actions, values{cost_values_init} = {{}}, {{}}{cost_values_init_rhs}
        for agent, ob in obs.items():
            actions[agent] = policy({actor})
            values[agent] = value({critic}){cost_value_call}

{_step_unpack(has_cost, "        ")}{draw_step}
        for agent, action in actions.items():
{add_call}

        # A truncated trajectory is bootstrapped from its last observation.
        for agent, truncated in truncations.items():
            if truncated and agent in next_obs:
                {bootstrap_call}

        obs = next_obs

    update({update_args})
"""
    if record is None:
        stdlib = sandbox = optional = block = ""
        signature = "steps: int = 200, seed: int = 0"
        body = f"    env = {class_stem}Env(render_mode={render_mode})\n{loop}"
        main = "if __name__ == \"__main__\":\n    training_loop()\n"
    else:
        stdlib, sandbox, optional = recording_imports(record, clips=False)
        imports = imports.rstrip("\n") + "\n" if imports else ""
        sandbox += "\n"
        block = recording_block(record, pkg) + f'''

def make_env(record: bool):
    """The env - recorded offscreen, as ``RECORDING`` says, with ``record``."""
    if not record:
        return {class_stem}Env()
    env = {class_stem}Env(render_mode="rgb_array", frame_driver=FRAME_DRIVER, views=recorded_views())
    return RecordVideo(env, RECORDING, on_clip=upload)
'''
        signature = "steps: int = 200, seed: int = 0, record: bool = True"
        # Closed however the loop ends: the clip it was recording is uploaded.
        body = (
            "    env = make_env(record)\n    try:\n"
            + textwrap.indent(loop, "    ", lambda line: line.strip() != "")
            + "    finally:\n        env.close()\n"
        )
        main = main_with_wandb("training_loop()")
    return f'''"""A training-loop scaffold for this task: what each network sees, and what
a transition holds - the rest is yours.
{views_note}{cost_note}
{"A reward or cost" if has_cost else "A reward"} may be a number or an array of components, per the task's
hooks. Run ``python -m {pkg}.train`` once the stubs are filled in.
"""

from __future__ import annotations

{stdlib}{imports}{sandbox}from .env import {class_stem}Env
{optional}{block}

def build_policy(observation_space):
    """Your actor, built from the ACTOR observation space."""
    raise NotImplementedError("build_policy: return an actor network.")


def build_value(observation_space):
    """Your critic, built from the CRITIC observation space."""
    raise NotImplementedError("build_value: return a value network.")
{cost_value_def}

class Buffer:
    """Your rollout storage. Holds RAW observations."""

    def add(self, agent, {add_args}):
        raise NotImplementedError("Buffer.add: store one transition.")

    def bootstrap(self, agent, {bootstrap_args}):
        raise NotImplementedError("Buffer.bootstrap: record the tail estimate of a truncated trajectory.")


def update({update_args}):
    """Your learning step."""
    raise NotImplementedError("update: fit the networks on the collected rollout.")


def training_loop({signature}) -> None:
{body}

{main}'''


def _train_vec_py(
    pkg: str,
    class_stem: str,
    privileged: bool,
    has_cost: bool,
    processes: int,
    watch: bool,
    record: RecordOptions | None = None,
) -> str:
    """``train.py`` for a training loop over copies of the env in parallel
    processes: the same arrangement as the one-env scaffold, over a vector env
    whose entries are every agent of every copy, one row each."""
    # SuperSuit's vector envs give one entry's space as ``observation_space``.
    single = "vec.observation_space"
    actor = "actor_obs(obs)" if privileged else "obs"
    critic = "critic_obs(obs)" if privileged else "obs"
    actor_space = f"actor_observation_space({single})" if privileged else single
    critic_space = f"critic_observation_space({single})" if privileged else single
    last_view = "critic_obs(last)" if privileged else "last"
    view_imports = (
        "from bluesky_sandbox import (\n    actor_obs,\n    actor_observation_space,\n"
        "    critic_obs,\n    critic_observation_space,\n)\n"
        if privileged
        else ""
    )
    cost_value_def = (
        '\n\ndef build_cost_value(observation_space):\n'
        '    """Your cost critic, built from the CRITIC observation space."""\n'
        '    raise NotImplementedError("build_cost_value: return a cost-value network.")\n'
        if has_cost
        else ""
    )
    if has_cost:
        add_args = "live, obs, actions, values, cost_values, rewards, costs, terminations, truncations"
        add_call = (
            "buffer.add(\n                live, obs, actions, values, cost_values, rewards, costs, "
            "terminations, truncations\n            )"
        )
        bootstrap_args = "row, value, cost_value"
        bootstrap_call = f"buffer.bootstrap(row, value({last_view}), cost_value({last_view}))"
        update_args = "policy, value, cost_value, buffer"
        cost_lines = (
            "\n            # This design defines a cost: each entry's is in its info."
            '\n            costs = np.stack([np.asarray(info.get("cost", 0.0)) for info in infos])'
        )
        cost_value_build = f"\n        cost_value = build_cost_value({critic_space})"
        cost_values_line = f"\n            cost_values = cost_value({critic})"
    else:
        add_args = "live, obs, actions, values, rewards, terminations, truncations"
        add_call = "buffer.add(live, obs, actions, values, rewards, terminations, truncations)"
        bootstrap_args = "row, value"
        bootstrap_call = f"buffer.bootstrap(row, value({last_view}))"
        update_args = "policy, value, buffer"
        cost_lines = cost_value_build = cost_values_line = ""
    watch_note = (
        "One copy - worker 0 - is drawn in a pygame window; the others run headless."
        if watch
        else "All copies run headless; ``watch=True`` draws one."
    )
    if record is None:
        stdlib = sandbox = optional = block = ""
        make = f"    env = {class_stem}Env(render_mode=render_mode)"
        record_arg = vec_record = clips_open = upload_step = drain = ""
        main = "if __name__ == \"__main__\":\n    training_loop()\n"
    else:
        watch_note += (
            "\n\nEvery copy records clips (``RECORDING``), uploaded to wandb as they"
            "\nfinish."
        )
        stdlib, sandbox, optional = recording_imports(record, clips=True)
        block = recording_block(record, pkg)
        make = (
            "    # A recorded copy draws offscreen, with the recording's driver and views.\n"
            "    drawing = (\n"
            "        {\"frame_driver\": FRAME_DRIVER, \"views\": recorded_views()}\n"
            "        if render_mode == \"rgb_array\"\n"
            "        else {}\n"
            "    )\n"
            f"    env = {class_stem}Env(render_mode=render_mode, **drawing)"
        )
        record_arg = ", record: bool = True"
        vec_record = ", RECORDING if record else None"
        clips_open = "\n    clips = Clips(RECORDING)"
        upload_step = "\n            for clip in clips.new():\n                upload(clip)"
        drain = (
            "\n        # The clips the copies finished as they closed."
            "\n        for clip in clips.new():"
            "\n            upload(clip)"
            "\n        clips.close()"
        )
        main = main_with_wandb("training_loop()")
    return f'''"""A training-loop scaffold for this task, its episodes run in {processes}
processes at once - BlueSky is one simulator per process - as one vector env
whose entries are every agent of every copy, one row each. {watch_note}

What each network sees and what a transition holds are fixed here; the
networks, buffer and update are yours. A reward must be one number per agent
here, as the vector env holds it. Run ``python -m {pkg}.train`` once the stubs
are filled in (``pip install bluesky-sandbox[parallel]``).
"""

from __future__ import annotations

{stdlib}import numpy as np

{view_imports}from bluesky_sandbox.integrations import vec_env, wrap_parallel_env
{sandbox}
from .env import {class_stem}Env
{optional}{block}

def make_env(render_mode=None):
    """One copy, as the vector env takes it: a fixed pool of agent IDs, its
    intruders padded to a fixed shape. Each worker process builds its own."""
{make}
    max_agents = env.unwrapped.scenario.support().max_aircraft
    return wrap_parallel_env(env, max_agents=max_agents)


def build_policy(observation_space):
    """Your actor, built from the ACTOR observation space; called on a batch."""
    raise NotImplementedError("build_policy: return an actor network.")


def build_value(observation_space):
    """Your critic, built from the CRITIC observation space; called on a batch."""
    raise NotImplementedError("build_value: return a value network.")
{cost_value_def}

class Buffer:
    """Your rollout storage: one row per entry, RAW observations. ``live``
    marks the rows that are an aircraft this step - the rest pad the fixed
    agent pool and are not transitions."""

    def add(self, {add_args}):
        raise NotImplementedError("Buffer.add: store one step of every live row.")

    def bootstrap(self, {bootstrap_args}):
        raise NotImplementedError("Buffer.bootstrap: record the tail estimate of a truncated trajectory.")


def update({update_args}):
    """Your learning step."""
    raise NotImplementedError("update: fit the networks on the collected rollout.")


def training_loop(
    steps: int = 200, seed: int = 0, n_processes: int = {processes}, watch: bool = {watch}{record_arg}
) -> None:
    vec = vec_env(make_env, n_processes, watch{vec_record}){clips_open}
    # From here on the worker processes are running: close them whatever
    # happens, or this process cannot exit.
    try:
        obs, _infos = vec.reset(seed=seed)
        policy = build_policy({actor_space})
        value = build_value({critic_space}){cost_value_build}
        buffer = Buffer()
        for _ in range(steps):
            actions = policy({actor})
            values = value({critic}){cost_values_line}
            next_obs, rewards, terminations, truncations, infos = vec.step(actions){cost_lines}
            live = np.array([not info.get("_padded", False) for info in infos])
            {add_call}

            # A truncated row is bootstrapped from its final observation: its
            # copy has already moved on.
            for row, info in enumerate(infos):
                last = info.get("final_observation")
                if truncations[row] and last is not None:
                    {bootstrap_call}

            obs = next_obs{upload_step}
        update({update_args})
    finally:
        vec.close(){drain}


{main}'''


def _names_used(source: str) -> set[str]:
    """Every identifier ``source`` reads. Used to import only what is needed."""
    return {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _setup_py(body: str, typing_imports: str = "") -> str:
    """The task's module-level setup code, split out of ``env.py``.

    ``env.py`` is where someone goes to read what the task *does* - the reward,
    the termination rule, the hooks. Before this split those thirty lines sat
    under three hundred lines of geometry helpers and cost machinery. Same
    code, same import-time behavior; only the file boundary moved. The in-process
    builder runs the same ``body`` (:func:`.setup_code.setup_source`).
    """
    return f'''"""Module-level setup for this generated task.

Helpers, constants and task-info providers used by the hooks in ``env.py``.
Split out so ``env.py`` stays readable as the task's behavior; edit either
file, or regenerate both from ``design.json``.

One consequence of the split: ``env.py`` imports these names, so it holds
*references*. A hook that MUTATES shared state behaves exactly as before -
``_CACHE.clear()`` and ``_CACHE[k] = v`` reach the object everything else
reads. A hook that REBINDS one - ``_CACHE = {{}}``, even under ``global`` -
now rebinds only ``env.py``'s name, and the helpers here keep using the
original object. Mutate; never reassign.
"""

{PRELUDE}{typing_imports}
{body}
'''


def _env_py(
    pkg: str,
    class_stem: str,
    env_sources: dict[str, str],
    hook_setup: str,
    hooks: dict[str, str],
    task_info_setup: str,
    task_info: list[TaskInfoSpec],
    task_info_refs: list[str],
    has_cost: bool = False,
    typed: TaskTypes | None = None,
) -> tuple[str, str]:
    """Return ``(env_source, setup_source)`` - the task's behavior, and the
    module-level helpers it leans on, as two files. With ``typed``, the hooks
    and task-info functions are annotated in the task's own types."""
    env_render = typed.renderer() if typed else None
    hook_overrides = _emit_hooks(
        hooks, (lambda hook: typed.hook_signature(hook, env_render)) if typed else None
    )
    provider_aliases = [
        f"_task_info_provider_{i}" for i, _ in enumerate(task_info_refs)
    ]
    provider_names = [
        *setup_code.provider_names(task_info, task_info_setup, hook_setup),
        *provider_aliases,
    ]
    task_info_imports = "\n".join(
        _ref_import(ref, pkg, alias) for ref, alias in zip(task_info_refs, provider_aliases)
    )
    providers = (
        "[" + ", ".join(provider_names) + "]"
        if provider_names
        else "()"
    )
    base_imports = f"""from __future__ import annotations

from bluesky_sandbox.env import BlueskyEnv

from .config import CONFIG
from .scenario import {class_stem}Scenario"""
    setup_render = typed.renderer() if typed else None
    signature = setup_code.PROVIDER_SIGNATURE
    if typed:
        signature = typed.provider_signature(setup_render)
    setup_source = _setup_py(
        setup_code.setup_source(task_info_setup, task_info, hook_setup, signature),
        _typing_imports(setup_render),
    )
    cost_total = "\n    total_cost = 0.0" if has_cost else ""
    cost_or = " or cost" if has_cost else ""
    cost_sum = (
        "\n        total_cost += sum(float(np.sum(c)) for c in costs.values())" if has_cost else ""
    )
    cost_print = ", total cost {total_cost:.3f}" if has_cost else ""
    body = f'''

class {class_stem}Env(BlueskyEnv):
    """Generated environment class with task behavior in Python code."""

    def __init__(
        self,
        *,
        scenario: {class_stem}Scenario | None = None,
        render_mode=None,
        realtime: bool = False,
        views=None,
        frame_driver=None,
    ) -> None:
        scenario = {class_stem}Scenario() if scenario is None else scenario
        super().__init__(
            config=CONFIG,
            scenario=scenario,
            render_mode=render_mode,
            realtime=realtime,
            views=views,
            frame_driver=frame_driver,
        )

    def define_task_info_providers(self):
        return {providers}

    # Env hooks. reward / terminated / truncated are always present; other
    # @overridable hooks appear only when customized (else inherit the base).
{hook_overrides}

def main() -> None:
    """Smoke rollout: print the task spec, then step with random actions.

    This is a multi-agent (PettingZoo parallel) environment - ``step`` takes a
    dict of per-agent actions and returns dicts keyed by agent.
    """
    env = {class_stem}Env(render_mode="pygame")
    env.reset(seed=0)
    if env.render_mode is not None:
        env.render()

    # Task spec: the agent set and per-agent observation / action spaces.
    print(f"agents ({{len(env.agents)}}): {{list(env.agents)}}")
    for agent in env.agents[:1]:
        print(f"observation_space[{{agent}}]: {{env.observation_space(agent)}}")
        print(f"action_space[{{agent}}]:      {{env.action_space(agent)}}")

    import numpy as np

    steps = 0
    total_reward = 0.0{cost_total}
    for _ in range(100):
        if env.episode_done:
            env.reset()
            if env.render_mode is not None:
                env.render()
            continue
        actions = {{
            agent: env.action_space(agent).sample() for agent in env.agents
        }}
{_step_unpack(has_cost, "        ")}        # A reward{cost_or} may be a number or an array of components.
        total_reward += sum(float(np.sum(r)) for r in rewards.values()){cost_sum}
        if env.render_mode is not None:
            env.render()
        steps += 1
    env.close()
    print(f"ran {{steps}} steps OK: total reward {{total_reward:.3f}}{cost_print}")
'''
    # Import only the setup names this module actually reads, so the import
    # line doubles as a summary of what the hooks depend on.
    # Names the header already binds (CONFIG, the Scenario class, provider
    # refs) must not be re-imported from .setup, which re-exports whatever it
    # imported itself.
    already_bound = setup_code.setup_names(
        "\n".join(part for part in (base_imports, task_info_imports) if part)
    )
    exports = setup_code.setup_names(setup_source) - already_bound
    # Annotations are type-only; what they name is imported for type checkers.
    plain_body = body.replace(hook_overrides, _emit_hooks(hooks))
    used = sorted(_names_used(plain_body) & exports)
    setup_import = ""
    if used:
        joined = ",\n    ".join(used)
        setup_import = f"from .setup import (\n    {joined},\n)"
    header = "\n".join(
        part
        for part in (
            base_imports,
            task_info_imports,
            setup_import,
            _typing_imports(env_render).strip("\n"),
        )
        if part
    )
    env_source = f'''"""Environment wrapper for this generated task."""

{header}
{body}'''
    return env_source, setup_source


def _main_py() -> str:
    return '''"""Quick smoke rollout for the generated task."""

from __future__ import annotations

from .env import main


if __name__ == "__main__":
    main()
'''


def _readme_md(
    pkg: str, title: str, env: Any, meta: dict[str, Any] | None = None, template: str = "plain"
) -> str:
    meta = meta or {}
    has_cost = defines_cost(env)
    train_what = {"rl": "training-loop scaffold", "sb3": "Stable-Baselines3 PPO training"}
    train_line = (
        f"\n  train.py      # {train_what[template]} (python -m {pkg}.train)" if template in train_what else ""
    )
    cost_line = (
        '\ncosts = {agent: info["cost"] for agent, info in infos.items()}  # this task defines a cost'
        if has_cost
        else ""
    )
    train_note = {
        "rl": "\n`train.py` lays a training loop out around this MDP: what each network sees,"
        "\nand what a transition holds. Fill in its stubs.\n",
        "sb3": "\n`train.py` trains every agent with one shared Stable-Baselines3 PPO policy"
        "\n(`pip install stable-baselines3 supersuit`); its docstring says what SB3 does"
        "\ndifferently from this design.\n",
    }.get(template, "")
    version = str(meta.get("version", "") or "0")
    note = str(meta.get("note", "") or "").strip()
    note_block = f"\n{note}\n" if note else ""
    return f"""# {title}

Version `{version}`.{note_block}

Generated by the bluesky-sandbox Environment Designer.

```
{pkg}/
  design.json   # original DesignSpec, reloadable in the designer
  scenario.py   # geometry / spawn / queryables / Scenario
  config.py     # static EnvConfig fields/settings
  setup.py      # module-level helpers / constants / task-info providers
  env.py        # hooks / BlueskyEnv subclass
  __main__.py   # python -m {pkg}  (smoke rollout){train_line}
```

## Use

```python
from {pkg} import Env
env = Env(render_mode="pygame")  # "pygame" | "panda3d" | "qtgl" | None (headless)
obs, infos = env.reset(seed=0)
actions = {{agent: env.action_space(agent).sample() for agent in env.agents}}
next_obs, rewards, terminations, truncations, infos = env.step(actions){cost_line}
```

Each is a dict keyed by agent. A reward{" or cost" if has_cost else ""} is a number, or an array of
components if the task's hook returns one.
{train_note}
Reload `design.json` in the designer, or edit `scenario.py` for
geometry/spawn/queryables, `config.py` for fields and simulator settings,
`setup.py` for the helpers the hooks lean on, and `env.py` for hooks, reward,
and episode termination.
{_readme_privileged_section(env)}"""


def _readme_privileged_section(env: Any = None) -> str:
    """README section emitted only when the design declares privileged fields.

    Those fields are the one part of the observation contract that is invisible
    from the outside: the env builds, bounds and normalizes the critic-only
    blocks, then hands them over as extra keys and does nothing further with
    them. Someone who receives this package and feeds the observation straight
    to one network trains with the privileged features silently ignored - no
    error, just a critic that is not actually privileged. So say so, here,
    where they will be reading.
    """
    if env is None:
        return ""
    if not (
        getattr(env, "critic_obs_fields", None)
        or getattr(env, "critic_intruder_obs_fields", None)
    ):
        return ""
    return """
## Privileged (critic-only) observations

This design declares `critic_obs_fields` / `critic_intruder_obs_fields`, so each
observation carries two extra keys the actor must not see:

```
ownship          (own_dim,)               actor + critic
intruders        (n_intruders, intr_dim)  actor + critic
critic_ownship   (c_own_dim,)             critic only
critic_intruders (n_intruders, c_dim)     critic only
```

The env does not merge them - that is the trainer's decision. `bluesky_sandbox`
ships the split; where it goes in a rollout loop:

```python
from bluesky_sandbox import (
    actor_obs, critic_obs,
    actor_observation_space, critic_observation_space,
)

space  = env.observation_space(agent)      # any agent; the layout is shared
policy = build_policy(actor_observation_space(space))
value  = build_value(critic_observation_space(space))

obs, _ = env.reset(seed=0)
for _ in range(n_steps):
    if env.episode_done:
        obs, _ = env.reset()
        continue

    actions, values = {}, {}
    for agent, ob in obs.items():
        actions[agent] = policy(actor_obs(ob))   # privileged keys removed
        values[agent] = value(critic_obs(ob))    # privileged keys merged in

    next_obs, rewards, terminations, truncations, _infos = env.step(actions)

    for agent, action in actions.items():
        # Store the RAW observation, not either view. Both are re-derivable
        # from it, and storing a view pins the rollout to one network's width.
        buffer.add(agent, obs[agent], action, values[agent],
                   rewards[agent], terminations[agent], truncations[agent])

    obs = next_obs

# Bootstrapping a truncated trajectory is a value call, so it takes the
# critic view as well.
last_value = value(critic_obs(obs[agent]))
```

Three things this loop is being careful about:

- **The policy is only ever called on `actor_obs`.** That is what keeps the
  deployed agent a function of information it will actually have.
- **The buffer stores the raw observation.** Re-derive both views at update
  time; a stored view cannot be widened back.
- **`critic_intruders` is row-aligned with `intruders`** - one row per intruder,
  same order. If you subsample intruder rows for the actor, keep the raw
  observation for the critic; `critic_obs` raises on a mismatch rather than
  merging and leaving the critic quietly unprivileged.
"""
