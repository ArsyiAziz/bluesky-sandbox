"""Recording an env as clips: drawn offscreen, only while a clip records."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.interface.wrappers import Clips, Recording, RecordVideo

pytest.importorskip("pygame")
imageio = pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

from test_vector_env import make_env  # noqa: E402 - after the skips


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def _frames(path):
    with imageio.get_reader(path) as reader:
        return sum(1 for _ in reader)


def test_rgb_array_renders_a_frame_and_opens_no_window():
    env = make_env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        first = env.render()
        env.step({a: env.action_space(a).sample() for a in env.agents})
        second = env.render()
        assert first.dtype == np.uint8 and first.ndim == 3 and first.shape[2] == 3
        assert first.shape == second.shape and (first != second).any()
    finally:
        env.close()


def test_clips_are_recorded_in_bursts_and_nothing_is_drawn_between(
    tmp_path, monkeypatch
):
    env = RecordVideo(
        make_env(render_mode="rgb_array"), Recording(tmp_path, every=4, length=2)
    )
    drawn = []
    driver = env.unwrapped._driver
    frame = driver.frame
    monkeypatch.setattr(driver, "frame", lambda: drawn.append(1) or frame())
    clips = Clips(tmp_path)
    try:
        env.reset(seed=0)
        for _ in range(8):
            if not env.agents:
                env.reset()
            env.step({a: env.action_space(a).sample() for a in env.agents})
        # Steps 0-1 and 4-5 are whole; the burst at step 8 has one frame so far.
        assert [(c.name, c.step) for c in clips.new()] == [("clip", 0), ("clip", 4)]
        assert clips.new() == []  # each clip is handed out once
    finally:
        env.close()
    (last,) = clips.new()  # what there was of it, written on close
    assert last.step == 8 and _frames(last.path) == 1
    assert len(drawn) == 5
    assert all(_frames(tmp_path / f"clip-step{s:09d}.mp4") == 2 for s in (0, 4))


def test_an_env_that_does_not_draw_offscreen_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rgb_array"):
        RecordVideo(make_env(), Recording(tmp_path, every=10, length=5))


def test_overlapping_clips_are_refused(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        Recording(tmp_path, every=5, length=10)


def test_every_worker_of_a_parallel_run_records_its_own_clips(tmp_path):
    from bluesky_sandbox.integrations import vec_env  # noqa: PLC0415

    v = vec_env(make_env, n_processes=2, record=Recording(tmp_path, every=10, length=3))
    try:
        v.reset(seed=0)
        for _ in range(3):
            v.step(np.stack([v.action_space.sample() for _ in range(v.num_envs)]))
        clips = Clips(tmp_path).new()
        assert [(c.name, c.step) for c in clips] == [("worker0", 0), ("worker1", 0)]
        assert all(_frames(c.path) == 3 for c in clips)
    finally:
        v.close()


def test_watch_and_record_cannot_share_worker_zero(tmp_path):
    from bluesky_sandbox.integrations import vec_env  # noqa: PLC0415

    with pytest.raises(ValueError, match="pick one"):
        vec_env(make_env, watch=True, record=Recording(tmp_path, every=10, length=3))
