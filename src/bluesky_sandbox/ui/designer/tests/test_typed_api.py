"""A generated task's types: its design's keys, checked by a type checker.

``task_types.py`` types the context and the step batch for the design, and the
hooks are annotated with them - so an editor completes ``context.obs
["ownship"]["alt_ft"]`` and a type checker flags a key the design lacks.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import bluesky_sandbox
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S

from .test_designer import _example_design_spec


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    design = _example_design_spec()
    design.queryables["wp"] = S.dump(Waypoint(lat=52.0, lon=4.5, alt_ft=3000))
    design.env.hook_setup = "import numpy as np"
    root = tmp_path_factory.mktemp("typed")
    codegen.write_task(design, "typed_pkg", root)
    return root


def test_the_package_has_its_types(package):
    source = (package / "typed_pkg" / "task_types.py").read_text()
    assert "class ObsOwnship(TypedDict):" in source
    assert "    alt_ft: float" in source
    assert "def query(self, name: Literal['wp']) -> WaypointResult: ..." in source
    assert "class TaskAgentStepContext(AgentStepContext):" in source
    assert "class TaskStepBatch(StepBatch):" in source


def test_the_hooks_are_annotated_with_them(package):
    env_py = (package / "typed_pkg" / "env.py").read_text()
    assert "context: TaskAgentStepContext," in env_py
    assert "from .task_types import TaskAgentStepContext" in env_py
    assert "rng: np.random.Generator," in env_py


def test_a_type_checker_reads_the_design_keys(package, monkeypatch):
    api = pytest.importorskip("mypy.api")
    # The library the tests import, wherever it is installed or checked out.
    monkeypatch.setenv("MYPYPATH", str(Path(bluesky_sandbox.__file__).parents[1]))
    check = package / "check.py"
    check.write_text(
        textwrap.dedent(
            """
            from typed_pkg.task_types import TaskAgentStepContext, TaskStepBatch

            def per_agent(context: TaskAgentStepContext) -> None:
                reveal_type(context.obs["ownship"]["alt_ft"])
                reveal_type(context.obs["intruders"]["acid"])
                reveal_type(context.query("wp").current.distance_nm)
                context.obs["ownship"]["alt_fx"]
                context.query("nope")

            def batched(batch: TaskStepBatch) -> None:
                reveal_type(batch.obs["ownship"]["alt_ft"])
                reveal_type(batch.context(0))
            """
        )
    )
    out, _err, _status = api.run(
        [
            "--follow-imports=silent",
            "--ignore-missing-imports",
            "--no-incremental",
            str(check),
            str(package / "typed_pkg" / "task_types.py"),
        ]
    )
    lines = out.splitlines()
    revealed = [
        line.split("Revealed type is ")[1] for line in lines if "Revealed" in line
    ]
    assert revealed == [
        '"builtins.float"',
        '"numpy.ndarray[builtins.tuple[Any, ...], numpy.dtype[Any]]"',
        '"builtins.float"',
        '"numpy.ndarray[builtins.tuple[Any, ...], numpy.dtype[Any]]"',
        '"typed_pkg.task_types.TaskAgentStepContext"',
    ]
    errors = [line for line in lines if ": error:" in line]
    assert len(errors) == 2
    assert 'no key "alt_fx"' in errors[0]
    assert 'No overload variant of "query"' in errors[1]
    assert not [line for line in errors if "task_types.py" in line]
