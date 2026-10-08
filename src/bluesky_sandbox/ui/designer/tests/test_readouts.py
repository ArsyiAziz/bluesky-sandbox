"""An aircraft's readout rows from a hook: plain ``{label: value}``, pairs, or
items - and the types the hooks name in scope in every hook body, in the
designer and in the generated package alike."""

from __future__ import annotations

import importlib
import sys

from bluesky_sandbox.interface.task import AircraftReadoutItem, aircraft_readout_items
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.diagnostics import diagnostics

from .test_designer import _example_design_spec


def test_a_readout_hook_may_return_plain_rows():
    assert aircraft_readout_items({"SEP": "3.2 nm", "ETA": "+12 s"}) == (
        AircraftReadoutItem("SEP", "3.2 nm"),
        AircraftReadoutItem("ETA", "+12 s"),
    )
    assert aircraft_readout_items([("A", 1), AircraftReadoutItem("B", 2)]) == (
        AircraftReadoutItem("A", 1),
        AircraftReadoutItem("B", 2),
    )
    assert aircraft_readout_items(None) == ()


def _with_readouts(body: str):
    spec = _example_design_spec()
    spec.env.hooks["define_aircraft_readouts"] = body
    return spec


def test_a_hook_body_uses_the_hook_types_without_importing_them():
    spec = _with_readouts(
        'return [AircraftReadoutItem("ACID", acid)] if AircraftControlState else []'
    )
    assert "hook:define_aircraft_readouts" not in diagnostics(spec)


def test_the_generated_package_imports_the_types_its_hooks_use(tmp_path):
    spec = _with_readouts('return [AircraftReadoutItem("ACID", acid)]')
    files = codegen.generate_task(spec, "Readouts")
    env_py = next(t for p, t in files.items() if p.endswith("/env.py"))
    assert "from bluesky_sandbox import (\n    AircraftReadoutItem,\n)" in env_py
    assert "AircraftControlState" not in env_py  # only what its hooks use
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(next(p for p in files if p.endswith("/env.py"))[:-3].replace("/", "."))
    finally:
        sys.path.remove(str(tmp_path))
    cls = next(v for k, v in vars(module).items() if k.endswith("Env") and k != "BlueskyEnv")
    rows = cls.define_aircraft_readouts(object.__new__(cls), "KLM1")
    assert aircraft_readout_items(rows) == (AircraftReadoutItem("ACID", "KLM1"),)


def test_a_dict_readout_needs_no_type_at_all():
    spec = _with_readouts('return {"ACID": acid, "FL": 120}')
    env_py = next(t for p, t in codegen.generate_task(spec, "Plain").items() if p.endswith("/env.py"))
    assert "from bluesky_sandbox import (" not in env_py
