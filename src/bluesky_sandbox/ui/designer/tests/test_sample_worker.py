"""Designs are sampled in one warm process, reused from sample to sample: a
fresh one only for another performance model, and after an error it keeps
running; a stream closed early is let finish, the process kept."""

from __future__ import annotations

import time

import pytest

from bluesky_sandbox.sim.performance import bada_available
from bluesky_sandbox.ui.designer import runner
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.builder import BuildError

from .test_designer import _example_design_spec


@pytest.fixture(scope="module")
def design():
    spec = _example_design_spec()
    yield S.DesignSpec.from_json(spec.to_json())
    runner._WORKER.stop()


def _pid():
    return runner._WORKER._proc.pid


def test_samples_reuse_one_warm_process(design):
    runner.sample_design(design, seed=0)  # starts it
    pid = _pid()
    start = time.monotonic()
    out = runner.sample_design(design, seed=1)
    assert out["agents"] and _pid() == pid
    assert time.monotonic() - start < 2.0  # no imports paid again


def test_an_error_is_reported_and_the_process_kept(design):
    runner.sample_design(design, seed=0)
    pid = _pid()
    bad = S.DesignSpec.from_json(design.to_json())
    bad.env.hooks["on_episode_reset"] = "raise RuntimeError('boom in reset')"
    with pytest.raises(BuildError, match="boom in reset"):
        runner.sample_design(bad, seed=0)
    assert runner.sample_design(design, seed=2)["agents"] and _pid() == pid


def test_a_stream_closed_early_finishes_and_the_process_is_kept(design):
    runner.sample_design(design, seed=0)
    pid = _pid()
    stream = runner.iter_episode_spawns(design, seed=0)
    next(stream)
    stream.close()
    assert runner.sample_design(design, seed=3)["agents"] and _pid() == pid


def test_another_performance_model_gets_a_process_of_its_own(design):
    if not bada_available():  # licensed: CI has none
        pytest.skip("no BADA database installed")
    runner.sample_design(design, seed=0)
    pid = _pid()
    other = S.DesignSpec.from_json(design.to_json())
    other.env.performance_model = "bada"
    runner.sample_design(other, seed=0)
    assert _pid() != pid and runner._WORKER._model == "bada"


def test_a_busy_process_leaves_the_sample_to_one_of_its_own(design):
    runner.sample_design(design, seed=0)
    assert runner._WORKER.lock.acquire(blocking=False)
    try:
        assert runner.sample_design(design, seed=4)["agents"]  # not waiting on it
    finally:
        runner._WORKER.lock.release()
