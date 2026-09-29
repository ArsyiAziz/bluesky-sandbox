"""Raw observation and action values, read by name, computed once per step.

``context.obs`` / ``env.raw_observation(agent)`` give an aircraft's
observation in each field's own unit - ``raw["ownship"]["alt_ft"]``,
``raw["intruders"]["dist_to_own_nm"][i]`` for intruder row ``i`` - and
``raw_action`` the value each action field was set to. They read the arrays the
observation is built from, so reading them costs no recomputation.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.wrappers.observations.normalizer import (
    CircularNormalizer,
    SymmetricNormalizer,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

_ALTITUDES_FT = (9_000.0, 11_000.0, 13_000.0, 15_000.0)


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


class _RecordingEnv(BlueskyEnv):
    """Keeps what its hooks saw of the raw values."""

    seen: dict

    def reward(self, _obs, _action, _terminated, _truncated, context, _info, _rng):
        self.seen[context.acid] = (
            context.obs["ownship"]["alt_ft"],
            dict(context.action),
        )
        return 0.0


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(
        dt=12.0,
        obs_fields=[
            obs.AltFt(),
            obs.TrkDeg(normalizer=CircularNormalizer()),
            obs.FlightPhaseOneHot(),
        ],
        intruder_obs_fields=[obs.DistToOwnNm(), obs.AltFt()],
        critic_intruder_obs_fields=[obs.TcpaS()],
        action_fields=[
            act.HdgDeltaDeg(normalizer=SymmetricNormalizer()),
            act.AltDeltaFt(
                low=-2_000.0, high=2_000.0, normalizer=SymmetricNormalizer()
            ),
            act.AutopilotLnav(),
        ],
    )
    env = _RecordingEnv(scenario=_Empty(), config=config)
    yield env
    env.close()


@pytest.fixture
def agents(env):
    """Four controlled aircraft at different altitudes; their callsigns."""
    env.reset(seed=0)
    env.seen = {}
    callsigns = []
    for i, alt in enumerate(_ALTITUDES_FT):
        acid = f"RAW{i:03d}"
        assert bs.traf.cre(acid, "B744", 52.0 + 0.1 * i, 4.5, 90, alt * ft, 250 * kts)
        env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
        callsigns.append(acid)
    return callsigns


def _action(env, agent, hdg=0.5, alt=-0.25, lnav=0):
    return {"continuous": np.array([hdg, alt], np.float32), "binary": np.array([lnav])}


# --- observations ----------------------------------------------------------------
def test_ownship_values_are_each_fields_own_value(env, agents):
    raw = env.raw_observation(agents[1])["ownship"]
    idx = bs.traf.id.index(agents[1])
    assert raw["alt_ft"] == pytest.approx(_ALTITUDES_FT[1])
    # A circular field observed as (sin, cos) is its angle raw; a one-hot its vector.
    assert raw["trk_deg"] == pytest.approx(obs.TrkDeg().get(idx))
    assert np.asarray(raw["flight_phase_one_hot"]).shape == (
        obs.FlightPhaseOneHot().output_size(),
    )


def test_intruder_row_i_is_the_observations_intruder_i(env, agents):
    own = agents[0]
    observed, *_ = env.step({a: _action(env, a) for a in agents})
    raw = env.raw_observation(own)["intruders"]
    assert list(raw["acid"]) == [a for a in bs.traf.id if a != own]
    distance = env.observation_layout()["intruders"][0].columns
    assert np.allclose(
        observed[own]["intruders"][:, distance].ravel(),
        raw["dist_to_own_nm"].astype(np.float32),
    )
    for i, other in enumerate(raw["acid"]):
        other_idx = bs.traf.id.index(other)
        assert raw["alt_ft"][i] == pytest.approx(float(bs.traf.alt[other_idx]) / ft)


def test_every_configured_part_is_there(env, agents):
    raw = env.raw_observation(agents[0])
    assert list(raw) == ["ownship", "intruders", "critic_intruders"]
    assert len(raw["critic_intruders"]["tcpa_s"]) == len(agents) - 1
    with pytest.raises(KeyError, match="critic_ownship"):
        raw["critic_ownship"]


# --- actions ---------------------------------------------------------------------
def test_the_raw_action_is_what_each_field_was_set_to(env, agents):
    env.step({a: _action(env, a, hdg=0.5, alt=-0.25, lnav=1) for a in agents})
    raw = env.raw_action(agents[0])
    assert raw["hdg_delta_deg"] == pytest.approx(90.0)  # 0.5 of +/-180
    assert raw["alt_delta_ft"] == pytest.approx(-500.0)  # -0.25 of +/-2000
    assert raw["autopilot_lnav"] == 1.0


def test_an_agent_without_an_action_has_an_empty_raw_action(env, agents):
    env.step({agents[0]: _action(env, agents[0])})
    assert env.raw_action(agents[1]) == {}


def test_hooks_read_the_same_raw_values(env, agents):
    env.step({a: _action(env, a, lnav=1) for a in agents})
    alt, action = env.seen[agents[2]]
    assert alt == env.raw_observation(agents[2])["ownship"]["alt_ft"]
    assert action == env.raw_action(agents[2])


# --- the cache -------------------------------------------------------------------
@pytest.fixture
def calls(monkeypatch):
    """How many times each field class computed its batch this test."""
    counts: dict[str, int] = {}

    def counting(cls, method):
        original = getattr(cls, method)

        def wrapper(self, *args, **kwargs):
            counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method, wrapper)

    counting(obs.AltFt, "get_many")
    counting(obs.DistToOwnNm, "get_pair_matrix")
    return counts


def test_a_step_computes_each_field_once_however_often_it_is_read(env, agents, calls):
    env.step({a: _action(env, a) for a in agents})
    # AltFt is configured twice - ownship and intruder - which is two fields.
    assert calls == {"AltFt": 2, "DistToOwnNm": 1}
    for agent in agents:
        raw = env.raw_observation(agent)
        raw["ownship"]["alt_ft"], raw["intruders"]["dist_to_own_nm"]
    assert calls == {"AltFt": 2, "DistToOwnNm": 1}


def test_values_read_before_the_observation_are_reused_by_it(env, agents, calls):
    env.raw_observation(agents[0])["intruders"]["dist_to_own_nm"]
    assert calls == {"AltFt": 1, "DistToOwnNm": 1}
    env._assemble_observations(agents)
    assert calls == {"AltFt": 2, "DistToOwnNm": 1}


def test_values_follow_the_traffic(env, agents):
    before = env.raw_observation(agents[0])["ownship"]["alt_ft"]
    bs.traf.alt[bs.traf.id.index(agents[0])] += 1_000 * ft
    env.step({})  # the sim moves on; the values are recomputed
    after = env.raw_observation(agents[0])["ownship"]["alt_ft"]
    assert after != before
    assert after == pytest.approx(float(bs.traf.alt[bs.traf.id.index(agents[0])]) / ft)
