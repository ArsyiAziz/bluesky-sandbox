"""A design as a folder: ``design.json`` for its structure, its code as real
Python modules - read back as the design it was written from, by the store,
the command line and any JSON editor (its schema)."""

from __future__ import annotations

import ast
import json

import pytest

from bluesky_sandbox.cli import main as cli
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.folder import (
    FolderError,
    load_design,
    read_folder,
    save_design,
    split_module,
    write_folder,
)
from bluesky_sandbox.ui.designer.store import SpecStore

from .test_designer import _example_design_spec


def _design() -> S.DesignSpec:
    spec = _example_design_spec()
    spec.env.hook_setup = "import math\n\nSCALE = 2.0\n\n\ndef _half(x):\n    return x / 2\n"
    spec.env.hooks = {**spec.env.hooks, "reward": "# shaped\nreturn _half(SCALE) * math.pi\n"}
    spec.env.task_info_setup = "LIMIT = 3\n"
    spec.env.task_info = [S.TaskInfoSpec("near", "info['near'] = LIMIT\n")]
    spec.scenario_setup = "TIMES = (0.0, 30.0)\n"
    spec.scenario_hooks = {"episode_geometry": "return geometry\n"}
    spec.spawn = {
        **spec.spawn,
        "sources": [{"name": "adsb", "plan": "return []\n", "when_blocked": "skip"}],
    }
    spec.code = {"custom_fields.py": "X = 1\n"}
    return S.DesignSpec.from_json(spec.to_json())


def _same(a: S.DesignSpec, b: S.DesignSpec) -> None:
    def norm(value):
        if isinstance(value, dict):
            return {k: norm(v) for k, v in value.items()}
        if isinstance(value, list):
            return [norm(v) for v in value]
        return value.strip("\n") if isinstance(value, str) else value

    assert norm(a.to_dict()) == norm(b.to_dict())


def test_it_reads_back_as_the_design_it_was_written_from(tmp_path):
    spec = _design()
    write_folder(spec, tmp_path / "d")
    _same(read_folder(tmp_path / "d"), spec)
    assert sorted(p.name for p in (tmp_path / "d" / "code").iterdir()) == [
        "custom_fields.py", "hooks.py", "scenario.py", "task_info.py",
    ]


def test_its_code_is_real_python_and_its_structure_has_none(tmp_path):
    write_folder(_design(), tmp_path / "d")
    hooks = (tmp_path / "d" / "code" / "hooks.py").read_text()
    tree = ast.parse(hooks)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in functions["reward"].args.args][:2] == ["self", "obs"]
    assert "_half" in functions  # the setup's own helper, at module scope
    scenario = ast.parse((tmp_path / "d" / "code" / "scenario.py").read_text())
    assert {"episode_geometry", "plan_adsb"} <= {n.name for n in scenario.body if isinstance(n, ast.FunctionDef)}
    design = json.loads((tmp_path / "d" / "design.json").read_text())
    assert design["$schema"] == "./design.schema.json"
    assert "reward" in design["env"]["hooks"] and "hook_setup" not in design["env"]
    assert design["spawn"]["sources"] == [{"name": "adsb", "when_blocked": "skip"}]
    assert design["code"] == ["custom_fields.py"]


def test_edits_to_its_code_files_are_the_designs(tmp_path):
    write_folder(_design(), tmp_path / "d")
    hooks = tmp_path / "d" / "code" / "hooks.py"
    text = hooks.read_text()
    text = text.replace("return _half(SCALE) * math.pi", "return 1.0")
    text = text.replace("from __future__ import annotations", "import os  # in scope: not kept")
    text += "\n\ndef terminated(self, obs, action, context, info, rng):\n    return False\n\n\nHELPER = 1\n"
    hooks.write_text(text)
    back = read_folder(tmp_path / "d")
    assert back.env.hooks["reward"] == "# shaped\nreturn 1.0\n"
    assert back.env.hooks["terminated"] == "return False\n"  # a hook by its name
    assert "HELPER = 1" in back.env.hook_setup and "import os" not in back.env.hook_setup
    assert "def _half(x):" in back.env.hook_setup


def test_comments_under_a_function_are_its_own():
    setup, bodies = split_module(
        "A = 1\n\n\ndef reward(self):\n    x = 1\n    return x\n    # why\n\n\nB = 2\n", {"reward"}
    )
    assert bodies["reward"] == "x = 1\nreturn x\n# why\n"
    assert setup == "A = 1\n\n\nB = 2\n"


def test_a_module_named_as_its_own_code_files_is_refused(tmp_path):
    spec = _design()
    spec.code = {"hooks.py": "X = 1\n"}
    with pytest.raises(FolderError, match="rename it"):
        write_folder(spec, tmp_path / "d")


def test_a_module_the_design_drops_is_removed(tmp_path):
    spec = _design()
    write_folder(spec, tmp_path / "d")
    spec.code = {}
    write_folder(spec, tmp_path / "d")
    assert not (tmp_path / "d" / "code" / "custom_fields.py").exists()


def test_either_form_loads_and_saves(tmp_path):
    spec = _design()
    save_design(spec, tmp_path / "one.json")
    save_design(spec, tmp_path / "folder")
    _same(load_design(tmp_path / "one.json"), spec)
    _same(load_design(tmp_path / "folder"), spec)
    _same(load_design(tmp_path / "folder" / "design.json"), spec)


def test_the_store_saves_a_new_design_as_a_folder_and_a_file_as_a_file(tmp_path):
    store = SpecStore(tmp_path)
    spec = _design()
    (tmp_path / "old.json").write_text(spec.to_json())
    assert store.save("old", spec) == "old" and not (tmp_path / "old").exists()
    assert store.save("new", spec) == "new" and (tmp_path / "new" / "design.json").is_file()
    assert [d["name"] for d in store.list()] == ["new", "old"]
    _same(store.load("new"), spec)
    _same(store.load("old"), spec)
    store.delete("new")
    assert not (tmp_path / "new").exists() and [d["name"] for d in store.list()] == ["old"]


def test_its_schema_checks_both_forms(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from bluesky_sandbox.ui.designer.schema import design_schema

    validator = jsonschema.Draft202012Validator(design_schema())
    spec = _design()
    assert not list(validator.iter_errors(spec.to_dict()))
    write_folder(spec, tmp_path / "d")
    assert not list(validator.iter_errors(json.loads((tmp_path / "d" / "design.json").read_text())))
    wrong = spec.to_dict()
    wrong["env"]["obs_fields"][0]["field"] = "NoSuchField"
    wrong["env"]["dtt"] = 1.0
    messages = " ".join(e.message for e in validator.iter_errors(wrong))
    assert "dtt" in messages
    assert any("NoSuchField" in str(e.instance) for e in validator.iter_errors(wrong))


def test_the_command_line_checks_previews_builds_and_converts(tmp_path, capsys):
    spec = _design()
    write_folder(spec, tmp_path / "d")
    assert cli(["design", "check", str(tmp_path / "d")]) == 0
    assert "builds" in capsys.readouterr().out
    assert cli(["design", "preview", str(tmp_path / "d"), "--seed", "3"]) == 0
    assert "seed 3:" in capsys.readouterr().out
    assert cli(["design", "convert", str(tmp_path / "d"), str(tmp_path / "one.json")]) == 0
    _same(load_design(tmp_path / "one.json"), spec)
    assert cli(["design", "build", str(tmp_path / "one.json"), "--out", str(tmp_path / "pkg"), "--name", "cli_task"]) == 0
    assert (tmp_path / "pkg" / "cli_task" / "scenario.py").is_file()
    hooks = tmp_path / "d" / "code" / "hooks.py"
    hooks.write_text(hooks.read_text().replace("return _half(SCALE) * math.pi", "return nowhere"))
    assert cli(["design", "check", str(tmp_path / "d")]) == 1
    assert "code/hooks.py: reward(), line 2" in capsys.readouterr().out
