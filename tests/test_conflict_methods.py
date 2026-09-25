"""``cd_method`` / ``reso_method`` select a method BlueSky has, or fail.

BlueSky's ``CDMETHOD`` and ``RESO`` report an unknown name as a failed command
result rather than by raising, and the stack drops that result. A typo - or
``CSTATEBASED`` on a machine without the compiled detector - then ran the
episode with conflict detection or resolution silently off.
"""

from __future__ import annotations

import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


def _reset(**config):
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[], **config),
    )
    try:
        env.reset(seed=0)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            {"cd_method": "CSTATEBASD"},
            r"EnvConfig\.cd_method: CSTATEBASD doesn't exist",
        ),
        ({"reso_method": "MPV"}, r"EnvConfig\.reso_method: MPV doesn't exist"),
    ],
)
def test_an_unknown_method_is_an_error(config, match):
    with pytest.raises(ValueError, match=match):
        _reset(**config)


def test_the_error_lists_the_methods_available():
    with pytest.raises(ValueError, match=r"Available CD methods: OFF, .*CSTATEBASED"):
        _reset(cd_method="NOPE")


def test_the_configured_methods_are_selected_whatever_their_case():
    # Not at module level: importing BlueSky's traffic package before
    # bs.init() breaks its performance-model selection for later tests.
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution  # noqa: PLC0415

    _reset(cd_method="statebased", reso_method="mvp")
    assert ConflictDetection.selected().__name__ == "StateBased"
    assert ConflictResolution.selected().__name__ == "MVP"
    _reset()  # the defaults
    assert ConflictDetection.selected().__name__ == "CStateBased"
    assert ConflictResolution.selected() is ConflictResolution  # OFF
