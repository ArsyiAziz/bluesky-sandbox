from __future__ import annotations

import time
import warnings

import bluesky as bs
import pytest

from bluesky_sandbox import config as config_module
from bluesky_sandbox.config import apply_performance_model, requested_performance_model
import bluesky_sandbox.core.base_environment as base_env
from bluesky_sandbox.core.base_environment import (
    BlueskyBaseEnvironment as BlueSkyBaseEnvironmentForTest,
)
from bluesky_sandbox.core.runtime import BlueSkyRuntime
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.performance.envelope import _aircraft_limits_cached
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig
from bluesky_sandbox.ui.drivers import sim_driver
from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver
from bluesky_sandbox.ui.drivers.sim_driver import SimDriver


def test_env_config_rejects_non_integer_substep_ratio(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_available_aircraft",
        lambda _model: frozenset({"b744"}),
    )

    with pytest.raises(ValueError, match="integer multiple"):
        config_module.EnvConfig(dt=1.0, simdt=0.06)


def test_env_step_advances_exact_config_dt(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_available_aircraft",
        lambda _model: frozenset({"b744"}),
    )

    class EmptyScenario:
        def sample(self, _rng):
            return self.support()

        def support(self):
            return EpisodeSpec(
                airspace_bounds=None,
                spawn=SpawnConfig(regions=[]),
                queryables={},
                max_aircraft=0,
            )

    env = BlueskyEnv(
        scenario=EmptyScenario(),
        config=config_module.EnvConfig(
            obs_fields=[],
            intruder_obs_fields=None,
            action_fields=[],
            dt=0.2,
            simdt=0.1,
        ),
    )
    try:
        env.reset(seed=123)
        start = env.sim_time

        env.step({})

        assert env._n_substeps == 2
        assert env.sim_time == pytest.approx(start + env.config.dt)
    finally:
        env.close()


def test_operate_does_not_advance_clock_without_step():
    class _Env:
        class config:
            simdt = 0.1
            performance_model = "openap"
            cd_method = "CSTATEBASED"
            reso_method = None
            pz_radius_nm = None
            pz_height_ft = None
            lookahead_s = None
            wind_dir_deg = 270.0
            wind_kts = 0.0
            turbulence_kts = 0.0

    runtime = BlueSkyRuntime(_Env())
    runtime.configure()
    runtime.reset(seed=123)

    start = runtime.sim_time
    runtime.operate()
    time.sleep(0.05)

    assert runtime.sim_time == pytest.approx(start)
    bs.sim.step()
    assert runtime.sim_time == pytest.approx(start + 0.1)


def test_realtime_driver_step_uses_fixed_simdt_despite_wall_clock_lag():
    class _Env:
        class config:
            simdt = 0.01
            performance_model = "openap"
            cd_method = "CSTATEBASED"
            reso_method = None
            pz_radius_nm = None
            pz_height_ft = None
            lookahead_s = None
            wind_dir_deg = 270.0
            wind_kts = 0.0
            turbulence_kts = 0.0

    class LaggyDriver(SimDriver):
        def update(self) -> None:
            time.sleep(0.02)

    runtime = BlueSkyRuntime(_Env())
    runtime.configure()
    runtime.reset(seed=123)
    runtime.operate()

    driver = LaggyDriver(realtime=True)
    start = runtime.sim_time

    driver.step()

    assert runtime.sim_time == pytest.approx(start + 0.01)


def _record_sleeps(monkeypatch) -> list[float]:
    """Capture the driver's pacing sleeps without touching the real ``time``.

    ``monkeypatch.setattr(sim_driver.time, "sleep", ...)`` would mutate the
    shared stdlib module, and BlueSky's ``network.receiver`` threads poll with
    ``while True: ... time.sleep(1.0)``. Replacing that sleep with a no-op turns
    every receiver thread into a busy-loop that starves the GIL and grows this
    list without bound - the whole suite crawls once any earlier test has run
    ``bs.init``. Swapping the module *reference* inside ``sim_driver`` instead
    leaves those threads on the real clock.
    """
    sleeps: list[float] = []

    class _Clock:
        monotonic = staticmethod(time.monotonic)

        @staticmethod
        def sleep(seconds: float) -> None:
            sleeps.append(seconds)

    monkeypatch.setattr(sim_driver, "time", _Clock)
    return sleeps


def test_realtime_driver_fastforward_skips_pacing_sleep(monkeypatch):
    class _Env:
        class config:
            simdt = 0.01
            performance_model = "openap"
            cd_method = "CSTATEBASED"
            reso_method = None
            pz_radius_nm = None
            pz_height_ft = None
            lookahead_s = None
            wind_dir_deg = 270.0
            wind_kts = 0.0
            turbulence_kts = 0.0

    sleeps = _record_sleeps(monkeypatch)

    runtime = BlueSkyRuntime(_Env())
    runtime.configure()
    runtime.reset(seed=123)
    runtime.operate()
    runtime.configure_timestep()

    driver = SimDriver(realtime=True)
    start = runtime.sim_time
    sim_driver.bs.sim.fastforward()

    driver.step()

    assert runtime.sim_time == pytest.approx(start + 0.01)
    assert sim_driver.bs.sim.ffmode is True
    assert sleeps == []


def test_realtime_driver_finite_fastforward_stops_at_ffstop(monkeypatch):
    class _Env:
        class config:
            simdt = 0.01
            performance_model = "openap"
            cd_method = "CSTATEBASED"
            reso_method = None
            pz_radius_nm = None
            pz_height_ft = None
            lookahead_s = None
            wind_dir_deg = 270.0
            wind_kts = 0.0
            turbulence_kts = 0.0

    sleeps = _record_sleeps(monkeypatch)

    runtime = BlueSkyRuntime(_Env())
    runtime.configure()
    runtime.reset(seed=123)
    runtime.operate()
    runtime.configure_timestep()

    driver = SimDriver(realtime=True)
    sim_driver.bs.sim.fastforward(0.01)

    driver.step()

    assert sim_driver.bs.sim.ffmode is False
    assert sim_driver.bs.sim.ffstop is None
    assert sleeps == []


def test_panda_update_refreshes_scene_before_render():
    driver = Panda3DSimDriver.__new__(Panda3DSimDriver)
    calls: list[str] = []

    class _TaskMgr:
        def step(self) -> None:
            calls.append("render")

    class _Show:
        taskMgr = _TaskMgr()

    driver._show = _Show()
    driver._apply_held_motion = lambda: calls.append("motion")
    driver._dispatch_step = lambda: calls.append("dispatch")
    driver._refresh_hud = lambda: calls.append("hud")

    Panda3DSimDriver.update(driver)

    assert calls == ["motion", "dispatch", "hud", "render"]


def test_type_limit_caches_are_keyed_by_performance_model():
    """Regression: the caches were keyed by aircraft type alone.

    The performance model is chosen when an ``EnvConfig`` is built, which can
    happen after something has already asked about a type. A ``None`` cached
    during the default-OpenAP phase then survived the switch to BADA, and every
    later lookup returned it - reported as "openap ceiling data is unavailable"
    for a type the configured model knows.
    """
    # Same type, two models -> two cache entries, not one shared answer.
    a = _aircraft_limits_cached("openap", "A320")
    b = _aircraft_limits_cached("bada", "A320")
    assert a is not b or a is None, "model must be part of the cache key"
    info = _aircraft_limits_cached.cache_info()
    assert info.currsize >= 2
    print("  type-limit cache keyed by model: OK")


def test_requested_performance_model_survives_bs_init():
    """Regression: ``bs.init()`` re-reads ``settings.cfg`` and overwrites
    ``bs.settings.performance_model``.

    A design selecting BADA therefore reverted to whatever the user's config
    file pinned (commonly ``openap``) the instant the runtime initialised - and
    every later envelope lookup failed on types OpenAP does not carry, while
    reporting the model as ``openap`` and looking like the design had been
    ignored.
    """
    before = requested_performance_model()
    try:
        apply_performance_model("bada")
        bs.settings.performance_model = "openap"  # what bs.init does
        assert requested_performance_model() == "bada"
    finally:
        apply_performance_model(before)
    print("  requested performance model is sticky: OK")


# ---- the task-hook contract --------------------------------------------- #


def _empty_config():
    return config_module.EnvConfig(
        obs_fields=[], intruder_obs_fields=None, action_fields=[], dt=0.2, simdt=0.1
    )


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


def test_base_env_rejects_a_class_with_no_task_hooks(monkeypatch):
    """``self._hooks = cast(TaskHooks, self)`` is an unverified claim.

    ``BlueskyBaseEnvironment`` deliberately defines none of the hooks - they
    live on ``BlueskyEnv`` - so constructing the base directly used to succeed
    and then die part-way through ``reset`` with a bare AttributeError. The
    cast suppresses exactly the warning that would have caught it, and the
    project runs no type checker, so the check has to be at runtime.
    """
    monkeypatch.setattr(
        config_module, "_available_aircraft", lambda _model: frozenset({"b744"})
    )
    with pytest.raises(TypeError, match="does not implement the task hooks"):
        BlueSkyBaseEnvironmentForTest(config=_empty_config(), scenario=_EmptyScenario())


def test_blueskyenv_satisfies_every_declared_hook(monkeypatch):
    """The public authoring base must supply a default for all of them."""
    monkeypatch.setattr(
        config_module, "_available_aircraft", lambda _model: frozenset({"b744"})
    )
    env = BlueskyEnv(scenario=_EmptyScenario(), config=_empty_config())
    try:
        assert env._hooks is env  # the cast is an alias, not an indirection
        assert not [n for n in base_env._TASK_HOOK_NAMES if not hasattr(env, n)]
    finally:
        env.close()


def test_task_shadowing_the_runtime_api_warns():
    """``def step`` in a task reads like a hook and is not one.

    The runtime and the hook surface share one MRO, so a task that defines
    ``step`` / ``reset`` / ``render`` replaces the environment's own machinery
    with nothing to say so. Warn at class definition, before anything runs.
    """
    with pytest.warns(RuntimeWarning, match=r"runtime API \['step'\]"):
        type("ShadowingTask", (BlueskyEnv,), {"step": lambda self, actions: None})


def test_shadowing_a_runtime_property_warns_too():
    """Properties are the easier accident: ``rng`` looks like task state."""
    with pytest.warns(RuntimeWarning, match=r"runtime API \['rng'\]"):
        type("ShadowingRng", (BlueskyEnv,), {"rng": property(lambda self: None)})


def test_hooks_and_new_helpers_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        type(
            "PlainTask",
            (BlueskyEnv,),
            {
                "reward": lambda self, *a: 0.0,
                "on_before_step": lambda self: None,
                "my_own_helper": lambda self: 1,
                # PettingZoo subclasses are expected to set this.
                "metadata": {"name": "plain-v0"},
            },
        )


def test_a_deliberate_override_can_be_declared():
    """Extending ``close`` and calling ``super()`` is legitimate; let it say so."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class DeliberateTask(BlueskyEnv, allow_runtime_override={"close"}):
            def close(self):
                return super().close()

    assert DeliberateTask.close is not BlueskyEnv.close
