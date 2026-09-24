"""``SpawnConfig.spawn_max_tries`` / ``spawn_warn_after``: how hard a spawn tries.

A spawn with a clearance check resamples its state until it clears live
traffic. These pin that the number of tries and the failures before a warning
come from the config, on both paths that retry: a queued conflict-free spawn
and a ``maintain`` top-up.
"""

from __future__ import annotations

import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.scenario.transforms import rotate_spawn
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


def _spawn(*, maintain: bool, **kwargs) -> SpawnConfig:
    return SpawnConfig(
        regions=[
            SpawnRegion(
                bounds=RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.7)),
                n_aircraft=2,
                params={"alt_ft": (8_000, 9_000), "spd_kts": (230, 270)},
                maintain=maintain,
            )
        ],
        aircraft_type="B744",
        conflict_free_spawn=True,
        **kwargs,
    )


class _Scenario:
    def __init__(self, spawn: SpawnConfig) -> None:
        self.spawn = spawn

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None, spawn=self.spawn, queryables={}, max_aircraft=2
        )


def _env_that_never_clears(spawn: SpawnConfig):
    """An env whose every candidate spawn state fails its clearance check."""
    env = BlueskyEnv(
        scenario=_Scenario(spawn),
        config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[]),
    )
    checks = []

    def never_clear(*_args, **_kwargs):
        checks.append(1)
        return False

    env._spawn_generator._spawn_position_clear = never_clear
    return env, checks


def test_the_defaults_live_on_spawn_config():
    spawn = SpawnConfig(regions=[])
    assert (spawn.spawn_max_tries, spawn.spawn_warn_after) == (20, 5)


@pytest.mark.parametrize("name", ["spawn_max_tries", "spawn_warn_after"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True, "3"])
def test_a_non_positive_or_non_integer_count_is_refused(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        SpawnConfig(regions=[], **{name: value})


def test_rotation_keeps_the_counts():
    spawn = _spawn(maintain=True, spawn_max_tries=7, spawn_warn_after=2)
    rotated = rotate_spawn(spawn, (52.0, 4.5), 30.0)
    assert (rotated.spawn_max_tries, rotated.spawn_warn_after) == (7, 2)


@pytest.mark.parametrize(
    ("maintain", "expected"),
    [
        # Both queued aircraft are due at reset; each tries 7 states, then is
        # deferred to the next step.
        (False, 2 * 7),
        # A top-up stops at the first aircraft it cannot place and retries
        # next step, so only one aircraft's 7 tries run.
        (True, 7),
    ],
    ids=["queued", "maintain"],
)
def test_a_spawn_tries_exactly_spawn_max_tries_states(maintain, expected):
    env, checks = _env_that_never_clears(_spawn(maintain=maintain, spawn_max_tries=7))
    try:
        env.reset(seed=0)
        assert len(checks) == expected
    finally:
        env.close()


def test_a_maintain_region_warns_after_spawn_warn_after_failed_top_ups():
    env, _checks = _env_that_never_clears(
        _spawn(maintain=True, spawn_max_tries=2, spawn_warn_after=3)
    )
    try:
        with pytest.warns(RuntimeWarning, match=r"3 consecutive top-ups \(2 tries each\)"):
            env.reset(seed=0)          # failed top-up 1
            env.step({})               # 2
            env.step({})               # 3 -> warns
    finally:
        env.close()
