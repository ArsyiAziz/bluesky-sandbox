"""An unset ``EnvConfig.simdt`` takes BlueSky's own physics step.

The config resolves it from ``bs.settings.simdt`` when built, so everything that
needs it then - the ``dt`` checks, the substep count - still has a number. The
runtime applies the config's step to BlueSky's clock without writing the
setting, so the setting stays BlueSky's value however many envs have run.
"""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.core import simtime

from bluesky_sandbox import config as config_module
from bluesky_sandbox.config import EnvConfig, bluesky_simdt_s, read_bluesky_settings
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig


class _EmptyScenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


def _config(**kwargs) -> EnvConfig:
    return EnvConfig(obs_fields=[], intruder_obs_fields=None, action_fields=[], **kwargs)


@pytest.fixture(scope="module", autouse=True)
def _bluesky_initialized():
    # ``bs.init`` re-reads settings.cfg over anything a test patched into
    # ``bs.settings``, so have it run before any test patches.
    BlueskyEnv(scenario=_EmptyScenario(), config=_config(dt=1.0)).close()


def test_an_unset_step_is_blueskys_setting():
    assert _config(dt=1.0).simdt == bluesky_simdt_s() == float(bs.settings.simdt)


def test_an_unset_step_follows_the_setting_not_a_written_in_default(monkeypatch):
    monkeypatch.setattr(bs.settings, "simdt", 0.1)
    assert _config(dt=1.0).simdt == 0.1


def test_an_explicit_step_wins():
    assert _config(dt=1.0, simdt=0.25).simdt == 0.25


def test_an_unset_step_is_resolved_before_the_dt_checks(monkeypatch):
    monkeypatch.setattr(bs.settings, "simdt", 0.3)
    with pytest.raises(ValueError, match=r"integer multiple of simdt \(0\.3\)"):
        _config(dt=1.0)


def test_running_an_env_leaves_blueskys_setting_alone():
    before = bs.settings.simdt
    env = BlueskyEnv(scenario=_EmptyScenario(), config=_config(dt=1.0, simdt=0.1))
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            assert simtime._clock.fdt == 0.1
            assert bs.sim.simdt == 0.1
            start = bs.sim.simt
            env.step({})
            assert bs.sim.simt == pytest.approx(start + 1.0)
    finally:
        env.close()
    assert bs.settings.simdt == before
    assert _config(dt=1.0).simdt == float(before)


def test_settings_cfg_is_never_reread_after_bluesky_is_initialized(monkeypatch):
    # Re-reading the file after bs.init would put its values back over settings
    # changed at runtime. Force the "not yet read" state to reach the guard.
    monkeypatch.setattr(config_module, "_SETTINGS_CFG_READ", False)
    monkeypatch.setattr(bs.settings, "asas_pzr", 123.0)
    read_bluesky_settings()
    assert bs.settings.asas_pzr == 123.0
