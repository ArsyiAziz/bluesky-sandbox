"""An env step timed phase by phase (bluesky_sandbox.core.step_timing): off
unless switched on, each phase its own time, together the step's total."""

from __future__ import annotations

import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core import step_timing
from bluesky_sandbox.core.step_timing import OTHER, StepTimer
from bluesky_sandbox.core.layout import zero_action
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs

from test_env_bound import _Scenario


class _Clock:
    """``time`` as the timer reads it: a clock moved by hand, in seconds."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def test_a_phase_within_another_counts_to_the_inner_one_only(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(step_timing, "time", clock)
    timer = StepTimer()
    timer.enabled = True
    timer.begin_step()
    clock.now += 0.001  # outside every phase
    with timer.phase("outer"):
        clock.now += 0.002
        with timer.phase("inner"):
            clock.now += 0.004
    timer.end_step()
    step = timer.last
    assert step.phases["inner"] == pytest.approx(4.0)
    assert step.phases["outer"] == pytest.approx(2.0)
    assert step.phases[OTHER] == pytest.approx(1.0)
    assert step.total_ms == pytest.approx(7.0) == pytest.approx(sum(step.phases.values()))


def test_off_it_times_nothing():
    timer = StepTimer()
    timer.begin_step()
    with timer.phase("anything"):
        pass
    timer.end_step()
    assert timer.last is None


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(
        dt=5.0, obs_fields=[obs.AltFt()], intruder_obs_fields=[obs.DistToOwnNm()], action_fields=[act.HdgDeltaDeg()]
    )
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    yield env
    env.close()


def test_a_step_is_timed_by_phase_and_by_field(env):
    env.reset(seed=0)
    env.step_timer.enabled = True
    try:
        env.step({a: zero_action(env.action_space(a)) for a in env.agents})
    finally:
        env.step_timer.enabled = False
    step = env.step_timer.last
    assert {"actions", "simulation", "bookkeeping", "lifecycle", "fields", "packing", "hooks", OTHER} <= set(step.phases)
    assert sum(step.phases.values()) == pytest.approx(step.total_ms)
    assert step.phases["simulation"] > 0
    # Each field's raw computation, by the field.
    timed = {id(f) for f in [*env.config.obs_fields, *env.config.intruder_obs_fields]}
    assert timed <= set(step.fields) and all(v > 0 for v in step.fields.values())


def test_a_step_untimed_keeps_no_timing(env):
    env.reset(seed=0)
    env.step_timer.last = None
    env.step({a: zero_action(env.action_space(a)) for a in env.agents})
    assert env.step_timer.last is None
