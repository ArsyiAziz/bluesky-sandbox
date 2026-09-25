"""A design's setup code runs the same in-process as in its generated package.

The task-info setup, the inline task-info entries and the hook setup are one
module: ``setup.py`` in a generated package, and the module the builder runs.
A helper one block defines is visible to the others either way - a task-info
entry calling a hook-setup helper used to fail in-process only.
"""

from __future__ import annotations

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
    source = setup_code.setup_source(
        design.env.task_info_setup, design.env.task_info, design.env.hook_setup
    )
    assert written.endswith(setup_code.PRELUDE + "\n" + source + "\n")
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
