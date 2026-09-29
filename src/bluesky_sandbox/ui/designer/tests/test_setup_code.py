"""A design's code runs the same in-process as in its generated package.

The task-info setup, the inline task-info entries and the hook setup are one
module: ``setup.py`` in a generated package, and the module the builder runs.
A helper one block defines is visible to the others either way - a task-info
entry calling a hook-setup helper used to fail in-process only. A batched hook
replaces its per-agent one in the package, and never sits beside it.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys

import pytest

from bluesky_sandbox.sim.spawn import SpawnConfig
from bluesky_sandbox.ui.designer import codegen, setup_code
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import (
    BuildError,
    build_design_config,
    run_setup_module,
)


def _design(*, task_info_setup="", task_info=(), hook_setup="") -> S.DesignSpec:
    env = S.EnvSpec(obs_fields=[], action_fields=[])
    env.task_info_setup = task_info_setup
    env.task_info = list(task_info)
    env.hook_setup = hook_setup
    return S.DesignSpec(env=env, spawn=S.dump(SpawnConfig(regions=[])))


def _run(provider) -> dict:
    info = {"task": {}}
    provider(None, None, info, None, None)
    return info["task"]


def test_a_task_info_entry_calls_a_helper_from_the_hook_setup():
    design = _design(
        task_info=[S.TaskInfoSpec("doubled", 'info["task"]["doubled"] = _double(21)')],
        hook_setup="def _double(x):\n    return 2 * x",
    )
    (provider,) = build_design_config(design).task_info_providers
    assert _run(provider) == {"doubled": 42}


def test_the_setup_code_reads_config_as_it_loads():
    design = _design(
        task_info=[S.TaskInfoSpec("step", 'info["task"]["dt"] = STEP_S')],
        hook_setup="STEP_S = CONFIG.dt",
    )
    config = build_design_config(design)
    assert _run(config.task_info_providers[0]) == {"dt": config.dt}


def test_a_body_naming_a_setup_object_is_that_provider():
    design = _design(
        task_info_setup=(
            "class _Provider:\n"
            "    def __call__(self, obs, action, info, context, rng):\n"
            '        info["task"]["seen"] = True\n'
            "MY_PROVIDER = _Provider()"
        ),
        task_info=[S.TaskInfoSpec("mine", "MY_PROVIDER")],
    )
    (provider,) = build_design_config(design).task_info_providers
    assert type(provider).__name__ == "_Provider"


def test_the_builder_runs_the_code_the_package_writes():
    design = _design(
        task_info_setup="import math\nLIMIT = math.sqrt(4.0)",
        task_info=[S.TaskInfoSpec("metric", 'info["task"]["m"] = _scale(LIMIT)')],
        hook_setup="import math\ndef _scale(x):\n    return 3 * x",
    )
    written = codegen.generate_task(design, "task_pkg")["task_pkg/setup.py"]
    # The package annotates each entry's parameters for editors; the code is
    # otherwise the same, and so are the parameters.
    signature = re.search(r"^def metric(\(.*\)) -> None:$", written, re.M).group(1)
    source = setup_code.setup_source(
        design.env.task_info_setup,
        design.env.task_info,
        design.env.hook_setup,
        signature,
    )
    assert written.endswith(source + "\n")
    plain = ast.parse(f"def f{setup_code.PROVIDER_SIGNATURE}: ...").body[0].args
    annotated = ast.parse(f"def f{signature}: ...").body[0].args
    assert [a.arg for a in annotated.args] == [a.arg for a in plain.args]
    module = run_setup_module(design.env, build_design_config(design))
    assert setup_code.setup_names(source) <= set(vars(module))


@pytest.mark.parametrize(
    ("design", "where"),
    [
        (_design(hook_setup="OK = 1\nBROKEN = undefined_name"), "hook setup, line 2"),
        (_design(task_info_setup="import no_such_module"), "task-info setup, line 1"),
        (
            _design(
                task_info_setup="A = 1",
                task_info=[
                    S.TaskInfoSpec("bad", 'x = 1\ny = (\ninfo["task"]["y"] = 2')
                ],
            ),
            "task info 'bad', line 2",
        ),
    ],
    ids=["hook setup", "task-info setup", "task-info entry"],
)
def test_an_error_names_its_block_and_line(design, where):
    with pytest.raises(BuildError, match=f"error in {where}"):
        build_design_config(design)


# --- batched hooks ---------------------------------------------------------------
def _batched_design() -> S.DesignSpec:
    design = _design(hook_setup="import numpy as np")
    design.env.hooks["reward_batch"] = (
        'return batch.obs["ownship"]["alt_ft"] * 0.0 + len(batch)'
    )
    return design


def test_a_batched_hook_replaces_its_per_agent_one_in_the_package():
    env_py = codegen.generate_task(_batched_design(), "task_pkg")["task_pkg/env.py"]
    assert "def reward_batch(self, batch: TaskStepBatch) -> np.ndarray:" in env_py
    assert "def reward(self" not in env_py
    assert "def terminated(" in env_py  # still per agent


def test_the_generated_env_uses_the_batched_hook(tmp_path):
    codegen.write_task(_batched_design(), "batch_pkg", tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        env_cls = importlib.import_module("batch_pkg").Env
        assert env_cls._batched_hooks == frozenset({"reward"})
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.split(".")[0] == "batch_pkg"]:
            del sys.modules[name]


def test_a_design_with_a_hook_both_ways_is_refused():
    design = _batched_design()
    design.env.hooks["reward"] = "return 1.0"
    with pytest.raises(BuildError, match="both reward and reward_batch"):
        build_design_config(design)
