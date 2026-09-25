"""A never-reused uid per aircraft, kept aligned by BlueSky.

BlueSky names aircraft only by callsign and reuses a callsign once its aircraft
is deleted. The runtime registers a BlueSky traffic array holding a serial
number per aircraft; the monitors key their rows by it, and the callsign issuer
keeps clear of every callsign the runtime saw created. Here aircraft come and go
straight through ``bs.traf`` - the way task hooks, qtgl commands and plugins do,
outside the spawner.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.spawning import _CallsignIssuer
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

_SECTOR = RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.7))


class _Scenario:
    """No spawns: every aircraft here is created by hand, like user code."""

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={
                "sector": QueryRegion(bounds=_SECTOR, track_temporal_state=True)
            },
            max_aircraft=0,
        )


@pytest.fixture
def env():
    env = BlueskyEnv(
        scenario=_Scenario(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    env.reset(seed=0)
    yield env
    env.close()


def _cre(callsign: str, lat: float = 52.0, lon: float = 4.55) -> None:
    assert bs.traf.cre(callsign, "B744", lat, lon, 90, 8_000 * 0.3048, 125) is True


def _index(callsign: str) -> int:
    return bs.traf.id.index(callsign)


def test_uids_follow_create_and_delete_and_never_repeat(env):
    for callsign in ("AAA001", "AAA002", "AAA003"):
        _cre(callsign)
    first = env._runtime.aircraft_uids.tolist()
    assert len(set(first)) == 3

    bs.traf.delete(_index("AAA002"))
    assert env._runtime.aircraft_uids.tolist() == [first[0], first[2]]

    _cre("AAA002")  # the same callsign, a different aircraft
    now = env._runtime.aircraft_uids.tolist()
    assert now[:2] == [first[0], first[2]]
    assert now[2] not in first


def test_every_creation_is_logged_until_bluesky_resets(env):
    _cre("AAA001")
    bs.traf.delete(_index("AAA001"))
    _cre("AAA001")
    assert env._runtime.created_callsigns[-2:] == ["AAA001", "AAA001"]
    env.reset(seed=1)
    assert env._runtime.created_callsigns == []
    assert len(env._runtime.aircraft_uids) == 0


def test_closing_an_env_detaches_its_tracker():
    # Make sure BlueSky is initialised before taking stock of its children.
    BlueskyEnv(
        scenario=_Scenario(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    ).close()
    children = list(bs.traf._children)
    for _ in range(2):  # a second env in the same process, the routine case
        env = BlueskyEnv(
            scenario=_Scenario(),
            config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[]),
        )
        tracker = env._runtime._uids
        assert tracker in bs.traf._children
        env.close()
        assert tracker not in bs.traf._children
    assert [c for c in bs.traf._children if c not in children] == []


def test_a_reused_callsign_starts_with_no_conflict_history(env):
    # Two aircraft nose to nose: in conflict from the first step.
    _cre("AAA001", lon=4.50)
    bs.traf.cre("AAA002", "B744", 52.0, 4.60, 270, 8_000 * 0.3048, 125)
    for _ in range(3):
        env.step({})
    monitor = env._traffic_monitor
    before = monitor.build_separation_context("AAA001", _index("AAA001"))
    assert before.conflict.time.total_s > 0

    # Delete it and create a new AAA001 alone over the Atlantic.
    bs.traf.delete(_index("AAA001"))
    _cre("AAA001", lat=40.0, lon=-30.0)
    env.step({})
    after = monitor.build_separation_context("AAA001", _index("AAA001"))
    assert after.conflict.time.total_s == 0.0


def test_a_reused_callsign_starts_with_no_dwell_history(env):
    _cre("AAA001")  # inside the tracked sector
    for _ in range(2):
        env.step({})
    monitor = env._query_state_monitor
    sector = env.episode_queryables["sector"]
    inside = monitor.query("AAA001", _index("AAA001"), "sector", sector)
    assert inside.time.total_s > 0

    bs.traf.delete(_index("AAA001"))
    _cre("AAA001", lat=40.0, lon=-30.0)  # far outside it
    env.step({})
    fresh = monitor.query("AAA001", _index("AAA001"), "sector", sector)
    assert fresh.time.total_s == 0.0


def test_the_issuer_keeps_clear_of_every_logged_creation():
    # Aircraft user code created and deleted between two spawns are never
    # live when the spawner looks; only the creation log knows them.
    created: list[str] = []
    issuer = _CallsignIssuer()
    issuer.start_episode(np.random.default_rng(0), created)
    created.extend(f"KL{n:03d}" for n in range(1, 999))  # all but KL999
    assert issuer.issue(set(), "KL") == "KL999"


def test_the_issuer_rereads_a_log_bluesky_cleared():
    created: list[str] = []
    issuer = _CallsignIssuer()
    issuer.start_episode(np.random.default_rng(0), created)
    created.extend(f"ZZZ{n:03d}" for n in range(1, 11))
    issuer.issue(set(), "BA")  # has read all 10 entries
    created.clear()  # bs.sim.reset() outside an env reset (e.g. qtgl RESET)
    created.append("KL005")  # a new log, shorter than what was read before
    issuer.issue(set(), "BA")
    assert "KL005" in issuer._taken


def test_the_spawner_hands_the_runtime_log_to_the_issuer(env):
    assert env._spawn_generator._callsigns._created is env._runtime.created_callsigns


def test_a_reused_callsign_starts_with_no_lag_history():
    # Deleted and re-created between two observations, so the env never sees
    # the callsign leave: only uid-keyed history can tell the two apart.
    lagged = obs.CasKts().lagged(steps=1)
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(dt=12.0, obs_fields=[obs.CasKts(), lagged], action_fields=[]),
    )
    try:
        env.reset(seed=0)
        assert bs.traf.cre("AAA001", "B744", 52.0, 4.55, 90, 8_000 * 0.3048, 125)
        assembler = env._observation_assembler  # hand-made aircraft aren't agents
        for _ in range(2):
            env.step({})
            old_cas = float(assembler.get_obs(["AAA001"])["AAA001"][0])

        bs.traf.delete(_index("AAA001"))
        assert bs.traf.cre("AAA001", "B744", 40.0, -30.0, 90, 8_000 * 0.3048, 160)
        env.step({})
        live, lag = (float(v) for v in assembler.get_obs(["AAA001"])["AAA001"])
        assert abs(live - old_cas) > 20.0, "the two aircraft must fly different speeds"
        # One observation of its own: the lag holds that, not the old aircraft's.
        assert lag == live
    finally:
        env.close()


def test_a_reused_callsign_starts_with_no_memory(env):
    # Last action, age and broadcast all live in AircraftMemory: a new aircraft
    # given a deleted one's callsign must not read what the old one left.
    from bluesky_sandbox.interface.fields import _state  # noqa: PLC0415

    _cre("AAA001")
    idx = _index("AAA001")
    _state.record_comm_message(idx, 0, 0.7)
    _state._LAST_NORM_ACTION.write(idx, np.array([0.5], dtype=np.float32))
    assert _state.comm_messages(0)[idx] == 0.7

    bs.traf.delete(idx)
    _cre("AAA001")  # nothing told the memory: no step ran in between
    idx = _index("AAA001")
    assert _state.comm_messages(0)[idx] == 0.0
    assert _state._LAST_NORM_ACTION.read_one(idx) is None


def test_memory_follows_its_aircraft_when_others_leave(env):
    from bluesky_sandbox.interface.fields import _common, _state  # noqa: PLC0415

    for callsign in ("AAA001", "AAA002", "AAA003"):
        _cre(callsign)
    memory = _state.AircraftMemory(default=-1.0)
    for callsign, value in (("AAA001", 1.0), ("AAA003", 3.0)):
        memory.write(_index(callsign), value)
    bs.traf.delete(_index("AAA002"))
    assert memory.read([_index("AAA001"), _index("AAA003")]) == [1.0, 3.0]
    _common.reset_field_state()
    assert memory.read([_index("AAA001")]) == [-1.0]


def test_without_a_runtime_memory_falls_back_to_callsigns(monkeypatch):
    from types import SimpleNamespace  # noqa: PLC0415

    from bluesky_sandbox.interface.fields import _state  # noqa: PLC0415
    from bluesky_sandbox.sim import aircraft_uids  # noqa: PLC0415

    monkeypatch.setattr(aircraft_uids, "_ATTACHED", [])
    monkeypatch.setattr(
        _state, "bs", SimpleNamespace(traf=SimpleNamespace(id=["A", "B"]))
    )
    memory = _state.AircraftMemory(default=0.0)
    memory.write(1, 2.0)
    assert memory.read([0, 1]) == [0.0, 2.0]
    memory.forget("B")  # the despawn hook: callsign keys must be dropped
    assert memory.read([1]) == [0.0]


def test_an_env_reset_clears_every_field_store(env):
    from bluesky_sandbox.interface.fields import _lag, _pairs, _state  # noqa: PLC0415

    _cre("AAA001")
    _state._TIME_IN_ENV.write(_index("AAA001"), 5.0)
    _lag._LAG_HISTORY[("obs", "stale")] = object()
    _pairs._CD_PAIR_CACHE["tcpa"] = ((), (None, None))
    _state._COMM_NOISE_RNG.normal(size=3)  # advance the stream
    env.reset(seed=4)
    assert not _lag._LAG_HISTORY and not _pairs._CD_PAIR_CACHE
    _cre("AAA001")
    assert _state._TIME_IN_ENV.read_one(_index("AAA001")) == 0.0
    fresh = np.random.default_rng(4).normal(size=3)
    np.testing.assert_array_equal(_state._COMM_NOISE_RNG.normal(size=3), fresh)


def test_a_despawn_reaches_every_stateful_fields_hook():
    # The env tells each configured stateful field when one of its aircraft
    # leaves; a hook that fails only then goes unnoticed in runs where none do.
    from bluesky_sandbox.sim.spawn import SpawnRegion  # noqa: PLC0415

    class _Spawning(_Scenario):
        def support(self):
            region = SpawnRegion(
                bounds=_SECTOR,
                n_aircraft=3,
                params={"alt_ft": (8_000, 9_000), "spd_kts": (230, 270)},
            )
            return EpisodeSpec(
                airspace_bounds=None,
                spawn=SpawnConfig(
                    regions=[region], aircraft_type="B744", conflict_free_spawn=False
                ),
                queryables={},
                max_aircraft=3,
            )

    env = BlueskyEnv(
        scenario=_Spawning(),
        config=EnvConfig(
            dt=12.0,
            obs_fields=[
                obs.CasKts().lagged(steps=1),
                obs.PrevActionNorm(),
                obs.TimeInEnvS(),
            ],
            intruder_obs_fields=[
                obs.DistToOwnNm().lagged(steps=1),
                obs.IntruderCommMessage(),
            ],
            action_fields=[],
        ),
    )
    try:
        env.reset(seed=0)
        assert bs.traf.ntraf == 3
        env.step({})
        gone = bs.traf.id[0]
        bs.traf.delete(0)
        env.step({})  # the env notices the aircraft left and calls every hook
        assert gone not in bs.traf.id
    finally:
        env.close()
