"""A crossover speed action's Mach regime and the crossover flag in a design:
built, generated and listed as the code API has them."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.ui.designer import catalog, codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import BuildError, build_design_config

from .test_designer import _example_design_spec

XO = {"type": "crossover", "cas_kts": 290.0, "mach": 0.8}
SYM = {"type": "normalizer", "name": "SymmetricNormalizer", "kwargs": {}}


def _design(**regime) -> S.DesignSpec:
    spec = _example_design_spec()
    above = {"type": "mach_regime", "crossover": XO, "low": -0.03, "high": 0.03, **regime}
    spec.env.action_fields = [
        S.FieldRef("ApSpdDeltaCrossover", {"normalizer": SYM, "above_crossover": above}),
    ]
    spec.env.obs_fields = [*spec.env.obs_fields, S.FieldRef("AboveCrossover", {"crossover": XO})]
    return S.DesignSpec.from_json(spec.to_json())


def test_it_is_built_as_the_code_api_has_it():
    config = build_design_config(_design(grid={"type": "grid", "step": 0.01, "on": "target"}))
    (speed,) = config.action_fields
    xo = act.Crossover(290.0, 0.8)
    assert speed.above_crossover == act.MachRegime(xo, low=-0.03, high=0.03, grid=act.Grid(0.01, on="target"))
    assert config.obs_fields[-1].crossover == xo


def test_the_generated_package_states_it():
    files = codegen.generate_task(_design(normalizer=SYM), "Cruise")
    config = next(t for p, t in files.items() if p.endswith("config.py"))
    assert (
        "above_crossover=act.MachRegime(act.Crossover(cas_kts=290.0, mach=0.8), "
        "low=-0.03, high=0.03, normalizer=SymmetricNormalizer())"
    ) in config
    assert "obs.AboveCrossover(crossover=act.Crossover(cas_kts=290.0, mach=0.8))" in config


def test_a_regime_that_changes_the_action_space_is_a_build_error():
    step = {"type": "normalizer", "name": "StepNormalizer", "kwargs": {"step": 0.01}}
    with pytest.raises(BuildError, match="another part of the action space"):
        build_design_config(_design(normalizer=step))


def test_the_catalog_says_which_take_it():
    actions = {a["name"]: a for a in catalog.action_fields()}
    assert actions["ApSpdDeltaCrossover"]["mach_regime"] and not actions["SpdDeltaKts"]["mach_regime"]
    observations = {o["name"]: o for o in catalog.obs_fields()}
    assert observations["AboveCrossover"]["crossover"] and observations["CrossoverAltMarginFt"]["crossover"]
    assert not observations["CasKts"]["crossover"]


def _crossover_curve(spec):
    from bluesky_sandbox.ui.designer.mdp import mdp_summary

    actions = mdp_summary(spec)["action"]
    (field,) = [f for part in actions["parts"] for f in part["fields"] if f["class"] == "ApSpdDeltaCrossover"]
    return field["crossover"]


def _with_bounds(**regime) -> S.DesignSpec:
    spec = _example_design_spec()
    above = {"type": "mach_regime", "crossover": XO, **regime}
    spec.env.action_fields = [
        S.FieldRef("ApSpdDeltaCrossover", {"normalizer": SYM, "low": -40.0, "high": 40.0, "above_crossover": above}),
    ]
    return S.DesignSpec.from_json(spec.to_json())


def test_spaces_shows_the_mach_regime_on_its_own_scale():
    curve = _crossover_curve(_with_bounds(low=-0.02, high=0.02))
    assert curve["title"].startswith("above the crossover (FL321)") and curve["y_label"] == "Mach change"
    (ys,) = curve["series"]
    at = lambda v: float(np.interp(v, curve["x"], ys))  # noqa: E731 - the plot runs past +-1
    assert at(1.0) == pytest.approx(0.02, abs=1e-4) and at(-1.0) == pytest.approx(-0.02, abs=1e-4)


def test_spaces_shows_a_mach_range_that_is_each_aircrafts_own_by_position():
    curve = _crossover_curve(_with_bounds())  # no Mach bounds: the envelope
    assert curve["y_label"] == "position in range"


def _stepping(mach_step: float) -> S.DesignSpec:
    spec = _example_design_spec()
    step = lambda s: {"type": "normalizer", "name": "StepNormalizer", "kwargs": {"steps_each_way": 4, "step": s}}  # noqa: E731
    spec.env.action_fields = [
        S.FieldRef(
            "ApSpdDeltaCrossover",
            {"normalizer": step(10.0), "above_crossover": {"type": "mach_regime", "crossover": XO, "normalizer": step(mach_step)}},
        )
    ]
    return S.DesignSpec.from_json(spec.to_json())


def test_spaces_shows_a_stepping_action_in_mach_steps_above():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 10 kt below, 0.01 Mach above: as meant
        curve = _crossover_curve(_stepping(0.01))
    assert curve["discrete"] and curve["x"] == [float(k) for k in range(-4, 5)]
    (ys,) = curve["series"]
    assert ys == pytest.approx([k * 0.01 for k in range(-4, 5)], abs=1e-9)
