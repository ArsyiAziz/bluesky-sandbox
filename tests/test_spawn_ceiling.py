"""A spawn altitude drawn from a type's envelope is one it can be flown at.

BlueSky clamps the selected altitude to ``perf.hmax``, thousands of feet below
the certified ceiling for much of a fleet: an aircraft spawned between the two
can never hold its level, and a route-relative altitude action loses its
authority.
"""

from __future__ import annotations

import numpy as np
import pytest
from bluesky.tools.aero import ft

from bluesky_sandbox.sim.performance.envelope import (
    _ceiling_ft_for_type,
    _sim_limits,
    feasible_alt_for_type,
)


@pytest.mark.parametrize("actype", ["A320", "B744", "C550", "E145"])
def test_an_envelope_spawn_stays_under_what_the_simulator_flies(actype):
    hmax_ft = float(_sim_limits(actype)["hmax"]) / ft
    assert hmax_ft < _ceiling_ft_for_type(actype)  # the gap this guards
    rng = np.random.default_rng(0)
    drawn = [feasible_alt_for_type(actype, rng) for _ in range(500)]
    assert max(drawn) <= hmax_ft
