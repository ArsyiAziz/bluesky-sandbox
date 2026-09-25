"""Problems in a design's code, placed on their block and line.

A block is compiled as it runs, and Python's own compiler says which names it
looks up globally - so what would raise NameError is flagged, and what is local,
a closure, a comprehension variable or a setup helper is not.
"""

from __future__ import annotations

import pytest

from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.diagnostics import diagnostics

from .test_designer import _example_design_spec


def _problems(**env) -> dict:
    design = _example_design_spec()
    for key, value in env.items():
        setattr(design.env, key, value)
    return diagnostics(design)


def _messages(found, block):
    return [(p["line"], p["column"], p["message"]) for p in found.get(block, [])]


def test_a_clean_design_has_no_problems():
    assert _problems() == {}


def test_an_undefined_name_in_a_hook_is_placed_on_its_span():
    found = _problems(hooks={"reward": "x = context.acid\nreturn missing(x)"})
    (problem,) = found["hook:reward"]
    assert (problem["line"], problem["column"], problem["end_column"]) == (2, 8, 15)
    assert problem["message"] == "'missing' is not defined"


@pytest.mark.parametrize(
    "body",
    [
        "y = 1\nreturn y",
        "return [v for v in range(3)]",
        "k = 1\ndef f():\n    return k\nreturn f()",
        "return len(str(context.acid))",
        "return SCALE + _helper()",
    ],
    ids=["local", "comprehension", "closure", "builtin", "setup helper"],
)
def test_what_a_hook_can_reach_is_not_flagged(body):
    found = _problems(
        hooks={"reward": body},
        hook_setup="SCALE = 1.0\ndef _helper():\n    return 0.0",
    )
    assert found.get("hook:reward") is None


def test_a_task_info_entry_sees_the_hook_setup():
    entry = S.TaskInfoSpec("metric", 'info["task"]["m"] = _helper() + nope')
    found = _problems(task_info=[entry], hook_setup="def _helper():\n    return 0.0")
    assert _messages(found, "task_info:metric") == [(1, 33, "'nope' is not defined")]


def test_a_syntax_error_is_placed_in_the_blocks_own_lines():
    found = _problems(hooks={"reward": "x = 1\ny = (\nreturn y"})
    assert _messages(found, "hook:reward") == [(2, 5, "'(' was never closed")]


def test_an_import_that_cannot_be_found_is_flagged():
    found = _problems(
        hook_setup="import math\nfrom bluesky_sandbox.task_types import X"
    )
    assert _messages(found, "hook_setup") == [
        (2, 1, "no module named 'bluesky_sandbox.task_types'")
    ]


def test_an_error_while_the_setup_runs_is_placed_on_its_line():
    found = _problems(hook_setup="A = 1\nB = 1 / 0")
    (problem,) = found["hook_setup"]
    assert problem["line"] == 2
    assert "division by zero" in problem["message"]


def test_class_bodies_and_globals_resolve_as_python_does():
    found = _problems(
        hook_setup=(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class P:\n"
            "    x: int = 1\n"
            "    y: int = x + 1\n"
            "def set_g():\n"
            "    global G\n"
            "    G = 1\n"
            "def read_g():\n"
            "    return G"
        )
    )
    assert found.get("hook_setup") is None
