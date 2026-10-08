"""Speed above the CAS/Mach crossover: BlueSky holds a Mach as the selected
speed there - read back as the CAS it is - and the crossover flag says which
regime the aircraft's speed is commanded in."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.actions.mask import _resume_axis
from bluesky_sandbox.interface.fields.base import ControlAxis
from bluesky_sandbox.sim.performance.speeds import above_crossover, crossover_speed_state, selected_cas_ms
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


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


def _fly(env, acid: str, alt_ft: float, cas_kts: float) -> int:
    assert bs.traf.cre(acid, "B744", 52.0, 4.0, 90.0, alt_ft * ft, cas_kts * kts)
    env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
    return bs.traf.id.index(acid)


def test_a_mach_selection_reads_back_as_the_cas_it_is(env):
    env.reset(seed=0)
    idx = _fly(env, "HIGH1", 39_000.0, 250.0)
    bs.stack.stack("SPD HIGH1 0.80")
    bs.stack.process()
    assert float(bs.traf.selspd[idx]) == pytest.approx(0.80)  # BlueSky's own: a Mach
    cas_kts = float(selected_cas_ms(idx)[0]) / kts
    assert 220 < cas_kts < 270  # M0.80 at FL390 is ~250 kt CAS
    assert obs.ApCasKts().get(idx) == pytest.approx(cas_kts, rel=1e-6)
    assert abs(obs.ApCasErrorKts().get(idx)) < 20  # not 0.80 - 250


def test_the_flag_is_the_regime_the_speed_is_commanded_in(env):
    env.reset(seed=0)
    high = _fly(env, "HIGH1", 39_000.0, 250.0)
    low = _fly(env, "LOW1", 10_000.0, 250.0)
    flags = obs.AboveCrossover()
    # Held at 250 kt, both fly CAS: the crossover is where CAS meets Mmo.
    assert flags.get(high) == 0.0 and flags.get(low) == 0.0
    bs.stack.stack("SPD HIGH1 300")  # past the Mmo ceiling at FL390: Mach-limited
    bs.stack.stack("SPD LOW1 300")
    bs.stack.process()
    assert flags.get(high) == 1.0 and flags.get(low) == 0.0
    assert list(above_crossover([high, low])) == [True, False]


@pytest.mark.parametrize("fl", [350, 390, 410])
def test_a_target_past_the_mmo_ceiling_is_mach_at_every_level(env, fl):
    env.reset(seed=0)
    idx = _fly(env, "CEIL1", fl * 100.0, 250.0)
    assert crossover_speed_state(idx, 340 * kts).in_mach  # not left to rounding


def test_resuming_the_route_speed_above_crossover_commands_mach(env, monkeypatch):
    env.reset(seed=0)
    idx = _fly(env, "HIGH1", 39_000.0, 250.0)
    from bluesky_sandbox.interface.fields.actions import mask

    monkeypatch.setattr(mask, "_active_route_waypoint", lambda i: (52.0, 5.0, 39_000 * ft, 300 * kts))
    _resume_axis(idx, "HIGH1", ControlAxis.SPEED)
    bs.stack.process()
    assert float(bs.traf.selspd[idx]) < 1.0  # held as Mach, not a CAS past Mmo
