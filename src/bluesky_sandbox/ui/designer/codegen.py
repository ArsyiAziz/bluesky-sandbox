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
from pathlib import Path
from typing import Any

from .builder import with_inferred_temporal_tracking
from .catalog import hooks as _hook_catalog
from .emit import emit_env_sources, emit_scenario_sources
from .spec import SCENARIO_HOOKS, DesignSpec, TaskInfoSpec


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
        f"{pkg}/README.md": _readme_md(pkg, title, e, meta),
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
    files[f"{pkg}/env.py"], files[f"{pkg}/setup.py"] = _env_py(
        pkg,
        class_stem,
        env_sources,
        e.hook_setup,
        e.hooks,
        e.task_info_setup,
        e.task_info,
        e.task_info_providers,
        privileged=bool(e.critic_obs_fields or e.critic_intruder_obs_fields),
    )
    return files


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
    body = _dedupe_setup_imports(scenario_setup or "", existing_imports).rstrip()
    return f"\n{body}\n" if body else ""


def _emit_scenario_hooks(scenario_hooks: dict[str, str]) -> str:
    """Emit the design's scenario hooks as ``@staticmethod`` blocks.

    Only hooks the design actually customised are emitted - an absent hook is
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


def _emit_task_info_providers(task_info: list[TaskInfoSpec]) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    for provider in task_info:
        name = provider.name.strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"invalid task-info provider name {name!r}")
        if name in seen:
            raise ValueError(f"duplicate task-info provider name {name!r}")
        seen.add(name)
        body = provider.body.rstrip() or "pass"
        if _task_info_direct_provider_expr(provider) is not None:
            continue
        blocks.append(
            f"def {name}(obs, action, info, context, rng) -> None:\n"
            f"{textwrap.indent(body, '    ')}\n"
        )
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def _task_info_provider_names(
    task_info: list[TaskInfoSpec],
    task_info_refs: list[str],
) -> tuple[list[str], list[str]]:
    inline_names = [
        _task_info_direct_provider_expr(p) or p.name
        for p in task_info
    ]
    provider_aliases = [f"_task_info_provider_{i}" for i, _ in enumerate(task_info_refs)]
    provider_names = inline_names + provider_aliases
    return provider_aliases, provider_names


def _task_info_direct_provider_expr(provider: TaskInfoSpec) -> str | None:
    body = provider.body.strip()
    if (
        body.isidentifier()
        and not keyword.iskeyword(body)
        and (body.isupper() or body.endswith("_PROVIDER"))
    ):
        return body
    return None


def _dedupe_setup_imports(setup: str, existing_source: str = "") -> str:
    """Drop duplicate exact import lines from generated setup blocks."""
    seen = {
        line.strip()
        for line in existing_source.splitlines()
        if _is_import_line(line)
    }
    out: list[str] = []
    for line in setup.rstrip().splitlines():
        stripped = line.strip()
        if _is_import_line(line):
            if stripped in seen:
                continue
            seen.add(stripped)
        out.append(line)
    return "\n".join(out)


def _is_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("import ", "from "))


# reward/terminated/truncated are always-present hooks (default 0.0 / never done).
_DEFAULT_HOOK_BODIES = {
    "reward": "return 0.0",
    "terminated": "return False",
    "truncated": "return False",
}


def _emit_hooks(hooks: dict[str, str]) -> str:
    """Emit env hook overrides.

    reward/terminated/truncated are always emitted (body from ``hooks`` or their
    default); other hooks are emitted only when the user customised them -
    uncustomised ones inherit the base behaviour (no ``super()`` boilerplate).
    """
    sigs = {h["name"]: h["def_signature"] for h in _hook_catalog()}
    names = list(_DEFAULT_HOOK_BODIES) + [n for n in sorted(hooks) if n not in _DEFAULT_HOOK_BODIES]
    blocks: list[str] = []
    for name in names:
        sig = sigs.get(name)
        if not sig:
            continue  # unknown hook (e.g. API changed); skip rather than emit invalid code
        body = (hooks.get(name) or _DEFAULT_HOOK_BODIES.get(name) or "").rstrip() or "pass"
        indented = "\n".join(("        " + ln) if ln.strip() else "" for ln in body.splitlines())
        blocks.append(f"    def {name}{sig}:\n{indented}")
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def _emit_training_loop(class_stem: str) -> str:
    """A scaffold showing how a training loop is arranged around the
    privileged observations.

    Emitted into ``env.py`` only when the design declares critic-only fields.
    Those fields are the one part of the observation contract that is invisible
    from the outside: the env builds, bounds and normalizes the extra blocks,
    then hands them over and does nothing further with them. Someone who feeds
    the raw observation to a single network trains with the privileged features
    silently ignored - no error, just a critic that is not privileged.

    Deliberately not runnable: the networks, buffer and update step are the
    reader's to supply. Filling them in with stand-ins would make the sketch
    longer and the arrangement harder to see, which is the only thing it is
    here to show.
    """
    return f'''

def build_policy(observation_space):
    """Your actor, built from the ACTOR observation space."""
    raise NotImplementedError(
        "build_policy: return an actor network for the actor observation "
        "space - the ordinary blocks, no privileged keys."
    )


def build_value(observation_space):
    """Your critic, built from the CRITIC observation space."""
    raise NotImplementedError(
        "build_value: return a value network for the critic observation "
        "space - the same blocks, widened by the privileged features."
    )


class Buffer:
    """Your rollout storage. Holds RAW observations, not either view."""

    def add(self, agent, obs, action, value, reward, terminated, truncated):
        raise NotImplementedError("Buffer.add: store one transition.")

    def bootstrap(self, agent, value):
        raise NotImplementedError(
            "Buffer.bootstrap: record V(s_T) for a truncated trajectory."
        )


def update(policy, value, buffer):
    """Your learning step."""
    raise NotImplementedError(
        "update: fit the policy and value networks on the collected rollout."
    )


def training_loop(steps: int = 200, seed: int = 0) -> None:
    """Scaffold: how a training loop is arranged around the privileged obs.

    This design declares critic-only observation fields, so every observation
    carries ``critic_ownship`` / ``critic_intruders`` alongside the ordinary
    blocks. The env never merges them - deciding which network sees what is the
    trainer's job, and the numbered steps below are where that happens.

    Raises ``NotImplementedError`` from the first stub above; the arrangement
    is what this is here to show.
    """
    from bluesky_sandbox import (
        actor_obs,
        actor_observation_space,
        critic_obs,
        critic_observation_space,
    )

    env = {class_stem}Env(render_mode=None)
    obs, _info = env.reset(seed=seed)

    # 1. Build each network from its own SPLIT space, never the raw one.
    #    Aircraft arrive on a schedule, so wait for the first agent.
    while not env.agents and not env.episode_done:
        obs, *_ = env.step({{}})
    space = env.observation_space(env.agents[0])
    policy = build_policy(actor_observation_space(space))
    value = build_value(critic_observation_space(space))
    buffer = Buffer()

    for _ in range(steps):
        if env.episode_done:
            obs, _info = env.reset()
            continue

        # 2. Act on the actor view, score on the critic view - the asymmetry.
        actions, values = {{}}, {{}}
        for agent, ob in obs.items():
            actions[agent] = policy(actor_obs(ob))
            values[agent] = value(critic_obs(ob))

        next_obs, rewards, terminations, truncations, _infos = env.step(actions)

        # 3. Store the RAW observation; a stored view cannot be widened back.
        for agent, action in actions.items():
            buffer.add(
                agent,
                obs[agent],
                action,
                values[agent],
                rewards[agent],
                terminations[agent],
                truncations[agent],
            )

        # 4. Bootstrapping is a value call, so it takes the critic view too.
        for agent, truncated in truncations.items():
            if truncated and agent in next_obs:
                buffer.bootstrap(agent, value(critic_obs(next_obs[agent])))

        obs = next_obs

    # 5. Re-derive both views from the stored raw observations at update time.
    update(policy, value, buffer)

'''


def _setup_exports(source: str) -> set[str]:
    """Top-level names a setup module binds - what ``env.py`` may import.

    Underscore-prefixed names are included deliberately: most of the setup
    block's helpers are private by convention (``_fix_state``, ``_VEL_HIST``),
    so a ``from .setup import *`` would silently miss exactly the names the
    hooks depend on. Hence an explicit import list, computed here.
    """
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _names_used(source: str) -> set[str]:
    """Every identifier ``source`` reads. Used to import only what is needed."""
    return {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _setup_py(task_info_setup: str, inline_blocks: str, hook_setup: str) -> str:
    """The task's module-level setup code, split out of ``env.py``.

    ``env.py`` is where someone goes to read what the task *does* - the reward,
    the termination rule, the hooks. Before this split those thirty lines sat
    under three hundred lines of geometry helpers and cost machinery. Same
    code, same import-time behaviour; only the file boundary moved.
    """
    body = "\n".join(
        part for part in (task_info_setup.rstrip(), inline_blocks, hook_setup.rstrip()) if part.strip()
    )
    return f'''"""Module-level setup for this generated task.

Helpers, constants and task-info providers used by the hooks in ``env.py``.
Split out so ``env.py`` stays readable as the task's behaviour; edit either
file, or regenerate both from ``design.json``.

One consequence of the split: ``env.py`` imports these names, so it holds
*references*. A hook that MUTATES shared state behaves exactly as before -
``_CACHE.clear()`` and ``_CACHE[k] = v`` reach the object everything else
reads. A hook that REBINDS one - ``_CACHE = {{}}``, even under ``global`` -
now rebinds only ``env.py``'s name, and the helpers here keep using the
original object. Mutate; never reassign.
"""

from __future__ import annotations

from .config import CONFIG

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
    privileged: bool = False,
) -> tuple[str, str]:
    """Return ``(env_source, setup_source)`` - the task's behaviour, and the
    module-level helpers it leans on, as two files."""
    hook_overrides = _emit_hooks(hooks)
    training_loop = _emit_training_loop(class_stem) if privileged else ""
    provider_aliases, provider_names = _task_info_provider_names(
        task_info,
        task_info_refs,
    )
    inline_blocks = _emit_task_info_providers(task_info)
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
    existing_imports = "\n".join(
        part for part in (base_imports, task_info_imports) if part
    )
    task_info_setup = _dedupe_setup_imports(task_info_setup, existing_imports)
    hook_setup = _dedupe_setup_imports(
        hook_setup,
        "\n".join(part for part in (existing_imports, task_info_setup) if part),
    )
    setup_source = _setup_py(task_info_setup, inline_blocks, hook_setup)
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
    ) -> None:
        scenario = {class_stem}Scenario() if scenario is None else scenario
        super().__init__(
            config=CONFIG,
            scenario=scenario,
            render_mode=render_mode,
            realtime=realtime,
            views=views,
        )

    def define_task_info_providers(self):
        return {providers}

    # Env hooks. reward / terminated / truncated are always present; other
    # @overridable hooks appear only when customised (else inherit the base).
{hook_overrides}
{training_loop}
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

    steps = 0
    for _ in range(100):
        if env.episode_done:
            env.reset()
            if env.render_mode is not None:
                env.render()
            continue
        actions = {{
            agent: env.action_space(agent).sample() for agent in env.agents
        }}
        env.step(actions)
        if env.render_mode is not None:
            env.render()
        steps += 1
    env.close()
    print(f"ran {{steps}} steps OK")
'''
    # Import only the setup names this module actually reads, so the import
    # line doubles as a summary of what the hooks depend on.
    # Names the header already binds (CONFIG, the Scenario class, provider
    # refs) must not be re-imported from .setup, which re-exports whatever it
    # imported itself.
    already_bound = _setup_exports(
        "\n".join(part for part in (base_imports, task_info_imports) if part)
    )
    exports = _setup_exports(setup_source) - already_bound
    used = sorted(_names_used(body) & exports)
    setup_import = ""
    if used:
        joined = ",\n    ".join(used)
        setup_import = f"from .setup import (\n    {joined},\n)"
    header = "\n".join(
        part for part in (base_imports, task_info_imports, setup_import) if part
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


def _readme_md(pkg: str, title: str, env: Any, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
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
  __main__.py   # python -m {pkg}  (10-step smoke rollout)
```

## Use

```python
from {pkg} import Env
env = Env(render_mode="pygame")  # "pygame" | "panda3d" | "qtgl" | None (headless)
obs, info = env.reset(seed=0)
```

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
