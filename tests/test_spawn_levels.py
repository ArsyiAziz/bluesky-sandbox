"""Altitudes drawn from the flight envelope on a level grid: with
``alt_step_ft`` an aircraft spawns, and a fix is set, at a flight level - drawn
uniformly among the levels the envelope (and any band) allows."""

from __future__ import annotations

import numpy as np
import pytest
from bluesky.tools.aero import ft

from bluesky_sandbox.sim.performance.envelope import (
    EnvelopeSample,
    _sim_limits,
    feasible_alt_for_type,
)


def test_a_spawn_on_levels_is_a_whole_flight_level_within_the_envelope():
    hmax_ft = float(_sim_limits("B744")["hmax"]) / ft
    rng = np.random.default_rng(0)
    drawn = np.array(
        [
            feasible_alt_for_type("B744", rng, 1000.0, alt_step_ft=1000.0)
            for _ in range(2000)
        ]
    )
    assert np.all(drawn % 1000.0 == 0.0)
    assert drawn.min() >= 1000.0 and drawn.max() <= hmax_ft
    # Uniform among the levels: every level in the window comes up.
    levels = np.arange(1000.0, np.floor(hmax_ft / 1000.0) * 1000.0 + 1, 1000.0)
    assert set(drawn) == set(levels)


def test_a_band_narrower_than_a_level_is_drawn_continuously():
    rng = np.random.default_rng(0)
    drawn = [
        feasible_alt_for_type(
            "B744", rng, alt_min_ft=21_200.0, alt_max_ft=21_800.0, alt_step_ft=1000.0
        )
        for _ in range(50)
    ]
    assert min(drawn) >= 21_200.0 and max(drawn) <= 21_800.0
    assert len(set(drawn)) > 1


def test_without_a_step_nothing_changes():
    a = feasible_alt_for_type("B744", np.random.default_rng(3))
    b = feasible_alt_for_type("B744", np.random.default_rng(3), alt_step_ft=None)
    assert a == b and a % 1000.0 != 0.0


@pytest.mark.parametrize("step", [0.0, -1000.0])
def test_a_level_grid_must_be_positive(step):
    with pytest.raises(ValueError, match="alt_step_ft"):
        EnvelopeSample(alt_step_ft=step)
