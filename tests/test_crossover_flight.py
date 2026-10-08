"""A crossover speed action, flown: the aircraft reaches and holds what it is
commanded - a CAS below the crossover, a Mach above - never outside its
envelope; its speed hold is handed over at the crossover as an FMS does, with
no jump in true airspeed and no flipping at a level-off near it; the flag
says the regime the action is in at every step; and each regime commands on its own scale - knots below, Mach above."""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts, vcas2mach, vmach2cas

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.services import _denormalize_action_value
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import StepContext
from bluesky_sandbox.interface.wrappers.observations.normalizer import SymmetricNormalizer
from bluesky_sandbox.sim.performance.speeds import is_mach, selected_cas_ms
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

XO = act.Crossover(cas_kts=300.0, mach=0.78)  # about FL293
XO_FT = XO.altitude_m / ft


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=SpawnConfig(regions=[]), queryables={}, max_aircraft=0)


@pytest.fixture(scope="module")
def env():
    env = BlueskyEnv(scenario=_Empty(), config=EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], action_fields=[]))
    yield env
    env.close()


def _action(**regime):
    return act.ApSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(), low=-40.0, high=40.0, above_crossover=act.MachRegime(XO, **regime)
    )


def _fly(env, acid: str, alt_ft: float, cas_kts: float = 300.0) -> int:
    env.reset(seed=0)
    assert bs.traf.cre(acid, "B744", 52.0, 4.0, 90.0, alt_ft * ft, cas_kts * kts)
    env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
    bs.stack.process()
    return bs.traf.id.index(acid)


def _command(action, idx: int, policy_value: float) -> None:
    action.set(idx, _denormalize_action_value(action, [policy_value], idx))
    bs.stack.process()


def _fly_for(seconds: float, action=None, each=None) -> None:
    """Advance the sim ``seconds`` in env steps of 5 s, the action handing
    over holds after each - as the environment drives it."""
    for _ in range(int(seconds / 5.0)):
        for _sub in range(int(5.0 / bs.sim.simdt)):
            bs.sim.step()
        if action is not None:
            action.on_step(StepContext(ids=tuple(bs.traf.id), dt=5.0, sim_time=float(bs.sim.simt), age_s={}))
            bs.stack.process()
        if each is not None:
            each()


# ---- commanded, and flown ------------------------------------------------- #


def test_below_the_crossover_a_cas_is_reached_and_held(env):
    idx = _fly(env, "LOW1", 20_000, 280.0)
    action = _action()
    _command(action, idx, 0.5)  # +20 kt
    assert not is_mach(bs.traf.selspd[idx])
    _fly_for(120)
    assert bs.traf.cas[idx] / kts == pytest.approx(300.0, abs=2.0)


def test_above_the_crossover_a_mach_is_reached_and_held(env):
    idx = _fly(env, "HIGH1", 35_000, 260.0)
    action = _action()
    start = float(bs.traf.M[idx])
    _command(action, idx, 0.5)
    assert is_mach(bs.traf.selspd[idx])
    target = float(bs.traf.selspd[idx])
    assert target > start
    _fly_for(180)
    assert float(bs.traf.M[idx]) == pytest.approx(target, abs=0.005)


@pytest.mark.parametrize("alt_ft", [10_000, 25_000, 29_000, 30_000, 35_000, 41_000])
def test_no_command_leaves_the_envelope(env, alt_ft):
    rng = np.random.default_rng(int(alt_ft))
    idx = _fly(env, "ENV1", alt_ft, 270.0)
    action = _action()
    for value in rng.uniform(-1.0, 1.0, 12):
        _command(action, idx, float(value))
        cas = float(selected_cas_ms(idx)[0])
        mach = float(vcas2mach(cas, bs.traf.alt[idx]))
        assert cas >= float(bs.traf.perf.vmin[idx]) - 0.5
        assert cas <= float(bs.traf.perf.vmax[idx]) + 0.5
        assert mach <= float(bs.traf.perf.mmo[idx]) + 1e-3


# ---- the hand-over at the crossover ---------------------------------------- #


def _climb_through(env, acid, action, flag, regimes):
    def record():
        idx = bs.traf.id.index(acid)
        regimes.append((float(bs.traf.alt[idx]) / ft, float(bs.traf.tas[idx]), float(bs.traf.selspd[idx]),
                        flag.get(idx), action.acting(idx) is not action))

    return record


def test_climbing_through_the_crossover_the_cas_hold_becomes_its_mach(env):
    idx = _fly(env, "UP1", XO_FT - 3_000, 300.0)
    action = _action()
    action.on_action_applied("UP1", None)  # the action commands it
    bs.stack.stack(f"SPD UP1 300")
    bs.stack.stack(f"ALT UP1 {XO_FT + 4_000:.0f}")
    bs.stack.process()
    flag = obs.AboveCrossover(crossover=XO)
    trace: list = []
    _fly_for(600, action, _climb_through(env, "UP1", action, flag, trace))
    alts = [t[0] for t in trace]
    assert alts[-1] > XO_FT + XO.margin_ft  # it climbed through
    idx = bs.traf.id.index("UP1")
    assert is_mach(bs.traf.selspd[idx])
    # Handed over at the Mach the CAS was there: the schedule's, near enough.
    assert float(bs.traf.selspd[idx]) == pytest.approx(XO.mach, abs=0.01)
    # No jump in true airspeed at the hand-over.
    tas = [t[1] for t in trace]
    assert max(abs(b - a) for a, b in zip(tas, tas[1:])) < 5.0  # m/s in 5 s
    # The flag is the regime the action acts in, at every step.
    assert all(bool(t[3]) == t[4] for t in trace)


def test_descending_through_the_crossover_the_mach_hold_becomes_its_cas(env):
    idx = _fly(env, "DN1", XO_FT + 3_000, 290.0)
    action = _action()
    action.on_action_applied("DN1", None)
    bs.stack.stack("SPD DN1 0.78")
    bs.stack.stack(f"ALT DN1 {XO_FT - 4_000:.0f}")
    bs.stack.process()
    _fly_for(600, action)
    idx = bs.traf.id.index("DN1")
    assert float(bs.traf.alt[idx]) / ft < XO_FT - XO.margin_ft
    assert not is_mach(bs.traf.selspd[idx])
    assert float(bs.traf.selspd[idx]) / kts == pytest.approx(300.0, abs=6.0)


def test_a_level_off_near_the_crossover_does_not_flip_the_hold(env):
    idx = _fly(env, "LVL1", XO_FT - 2_000, 300.0)
    action = _action()
    action.on_action_applied("LVL1", None)
    bs.stack.stack("SPD LVL1 300")
    bs.stack.stack(f"ALT LVL1 {XO_FT + XO.margin_ft * 0.5:.0f}")  # inside the margin
    bs.stack.process()
    holds = []
    _fly_for(400, action, lambda: holds.append(bool(is_mach(bs.traf.selspd[bs.traf.id.index("LVL1")]))))
    assert not any(holds)  # held its CAS throughout
    assert not obs.AboveCrossover(crossover=XO).get(bs.traf.id.index("LVL1"))


def test_an_aircraft_the_action_does_not_command_is_left_alone(env):
    idx = _fly(env, "BG1", XO_FT + 2_000, 300.0)
    action = _action()  # never applied to BG1
    bs.stack.stack("SPD BG1 300")
    bs.stack.process()
    _fly_for(30, action)
    assert not is_mach(bs.traf.selspd[bs.traf.id.index("BG1")])


def test_no_hand_over_when_turned_off(env):
    idx = _fly(env, "OFF1", XO_FT + 2_000, 300.0)
    action = _action(handover=False)
    action.on_action_applied("OFF1", None)
    bs.stack.stack("SPD OFF1 300")
    bs.stack.process()
    _fly_for(30, action)
    assert not is_mach(bs.traf.selspd[bs.traf.id.index("OFF1")])


# ---- each regime on its own scale ------------------------------------------- #


def test_each_regime_commands_on_its_own_scale(env):
    # +-40 kt below, +-0.02 Mach above: the same output, half of each.
    action = _action(low=-0.02, high=0.02)
    idx = _fly(env, "SC1", XO_FT - XO.margin_ft - 500, 280.0)
    _command(action, idx, 0.5)
    assert float(bs.traf.selspd[idx]) / kts == pytest.approx(300.0, abs=0.01)
    idx = _fly(env, "SC2", XO_FT + XO.margin_ft + 3_000, 270.0)
    mach = float(bs.traf.M[idx])
    _command(action, idx, 0.5)
    assert float(bs.traf.selspd[idx]) == pytest.approx(mach + 0.01, abs=1e-4)


def test_unset_mach_bounds_are_the_aircrafts_own_envelope(env):
    idx = _fly(env, "SC3", 35_000, 260.0)
    view = _action().acting(idx)
    low, high = view.bounds(idx)
    assert low == pytest.approx(-high) and 0.0 < high < 0.5  # not the knots' +-40
