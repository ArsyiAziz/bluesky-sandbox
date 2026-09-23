"""``EnvConfig.asas_dt``: BlueSky's conflict-detection interval.

BlueSky runs detection on its ``asas`` timer, a fixed sim-time grid that knows
nothing of the agent step and that ``bs.sim.reset()`` returns to
``bs.settings.asas_dt``. These pin that the config refuses intervals which would
not line up with the steps, that an explicit interval is re-applied after every
reset, and that an unset one follows BlueSky's own setting rather than a value
written into the sandbox.
"""

from __future__ import annotations

import math

import bluesky as bs
import pytest
from bluesky.core.simtime import Timer

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


class _Scenario:
    """Two aircraft: BlueSky skips traffic updates, and so detection, without any."""

    def __init__(self) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.9, 52.1, 4.4, 4.7)),
                    n_aircraft=2,
                    params={"alt_ft": (8_000, 9_000), "spd_kts": (220, 280)},
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None, spawn=self.spawn, queryables={}, max_aircraft=2
        )


def _config(**kwargs) -> EnvConfig:
    return EnvConfig(obs_fields=[], intruder_obs_fields=None, action_fields=[], **kwargs)


def _asas_timer_dt() -> float:
    return float(Timer.gettimer("asas").dt_act)


@pytest.fixture(scope="module", autouse=True)
def _bluesky_initialised():
    # The first env in a process runs ``bs.init``, which re-reads settings.cfg
    # over anything a test patched into ``bs.settings``. Get that done first.
    BlueskyEnv(scenario=_Scenario(), config=_config(dt=1.0)).close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(dt=12.0, asas_dt=5.0), "different age from step to step"),
        (dict(dt=12.0, asas_dt=20.0), "no conflict detection at all"),
        (dict(dt=12.0, asas_dt=1.03), "integer multiple of simdt"),
        (dict(dt=12.0, asas_dt=0.01), "integer multiple of simdt"),
        (dict(dt=12.0, asas_dt=0.0), "positive finite"),
        (dict(dt=12.0, asas_dt=-1.0), "positive finite"),
        (dict(dt=12.0, asas_dt=math.nan), "positive finite"),
    ],
    ids=["not-dividing-dt", "longer-than-dt", "off-simdt-grid", "below-simdt",
         "zero", "negative", "nan"],
)
def test_config_refuses_an_interval_that_does_not_line_up(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


@pytest.mark.parametrize("asas_dt", [0.05, 1.0, 4.0, 12.0])
def test_config_accepts_a_divisor_of_dt_on_the_simdt_grid(asas_dt):
    assert _config(dt=12.0, asas_dt=asas_dt).asas_dt == asas_dt


def test_an_unset_interval_is_not_judged_before_bluesky_can_supply_it():
    # BlueSky's default exists only after bs.init, so EnvConfig must not guess
    # at it; the runtime checks it when the env is built.
    assert _config(dt=0.2, simdt=0.1).asas_dt is None


def test_an_explicit_interval_survives_every_reset_and_sets_the_cadence():
    env = BlueskyEnv(scenario=_Scenario(), config=_config(dt=12.0, asas_dt=4.0))
    detections = []
    try:
        assert _asas_timer_dt() == 4.0
        for seed in (0, 1):
            env.reset(seed=seed)
            assert _asas_timer_dt() == 4.0

            detector = bs.traf.cd._refobj
            original = detector.detect

            def counting(*args, _original=original):
                detections.append(bs.sim.simt)
                return _original(*args)

            # CStateBased binds its compiled kernel on the instance: put that
            # back, rather than deleting the patch and exposing the slow
            # pure-Python StateBased.detect to every later test.
            detector.__dict__["detect"] = counting
            try:
                env.step({})
            finally:
                detector.__dict__["detect"] = original
    finally:
        env.close()
    # dt / asas_dt = 3 detections per step, on the 4 s grid from each reset.
    assert [round(t, 6) for t in detections] == [4.0, 8.0, 12.0] * 2


def test_an_unset_interval_follows_blueskys_setting(monkeypatch):
    monkeypatch.setattr(bs.settings, "asas_dt", 3.0)
    env = BlueskyEnv(scenario=_Scenario(), config=_config(dt=12.0))
    try:
        env.reset(seed=0)
        assert _asas_timer_dt() == 3.0
    finally:
        env.close()


def test_an_unset_interval_that_blueskys_setting_breaks_fails_at_construction(
    monkeypatch,
):
    monkeypatch.setattr(bs.settings, "asas_dt", 5.0)
    with pytest.raises(ValueError, match=r"bs\.settings\.asas_dt") as excinfo:
        BlueskyEnv(scenario=_Scenario(), config=_config(dt=12.0))
    assert "set asas_dt explicitly" in str(excinfo.value)
