"""Recording an env as clips: drawn offscreen, only while a clip records, and
handed out in memory - kept on disk only when asked."""

from __future__ import annotations

import io

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


def _frames(data: bytes) -> int:
    with imageio.get_reader(io.BytesIO(data), format="mp4") as reader:
        return sum(1 for _ in reader)


def _play(env, steps):
    env.reset(seed=0)
    for _ in range(steps):
        if not env.agents:
            env.reset()
        env.step({a: env.action_space(a).sample() for a in env.agents})


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


def test_panda3d_draws_frames_offscreen_too():
    pytest.importorskip("panda3d")
    env = make_env(render_mode="rgb_array", frame_driver="panda3d")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert frame.dtype == np.uint8 and frame.shape[2] == 3
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1  # not a blank buffer
    finally:
        env.close()


def test_a_frame_driver_is_only_for_rgb_array():
    with pytest.raises(ValueError, match="frame_driver"):
        make_env(render_mode="pygame", frame_driver="panda3d")
    with pytest.raises(ValueError, match="frame_driver must be one of"):
        make_env(render_mode="rgb_array", frame_driver="qtgl")


def test_clips_are_recorded_in_bursts_and_nothing_is_drawn_between(monkeypatch):
    clips = []
    env = RecordVideo(
        make_env(render_mode="rgb_array"),
        Recording(every=4, length=2),
        on_clip=clips.append,
    )
    drawn = []
    driver = env.unwrapped._driver
    frame = driver.frame
    monkeypatch.setattr(driver, "frame", lambda: drawn.append(1) or frame())
    try:
        _play(env, 8)
        # Steps 0-1 and 4-5 are whole; the burst at step 8 has one frame so far.
        assert [(c.name, c.step) for c in clips] == [("clip", 0), ("clip", 4)]
    finally:
        env.close()
    assert [c.step for c in clips] == [0, 4, 8]  # what there was of the last, on close
    assert [_frames(c.data) for c in clips] == [2, 2, 1]
    assert len(drawn) == 5


def test_a_clip_not_kept_leaves_nothing_on_disk():
    recording = Recording(every=4, length=2)
    clips = []
    env = RecordVideo(
        make_env(render_mode="rgb_array"), recording, on_clip=clips.append
    )
    try:
        _play(env, 3)
    finally:
        env.close()
    (clip,) = clips
    assert clip.path is None and clip.data[4:8] == b"ftyp"  # an mp4, in memory
    assert not recording.path.exists()  # nothing left behind


def test_a_folder_keeps_every_clip(tmp_path):
    clips = []
    env = RecordVideo(
        make_env(render_mode="rgb_array"),
        Recording(every=4, length=2, folder=tmp_path),
        on_clip=clips.append,
    )
    try:
        _play(env, 3)
    finally:
        env.close()
    (clip,) = clips
    assert clip.path == tmp_path / "clip-step000000000.mp4"
    assert clip.path.read_bytes() == clip.data


def test_an_env_that_does_not_draw_offscreen_is_refused():
    with pytest.raises(ValueError, match="rgb_array"):
        RecordVideo(make_env(), Recording(every=10, length=5))


def test_overlapping_clips_are_refused():
    with pytest.raises(ValueError, match="overlap"):
        Recording(every=5, length=10)


def test_every_worker_of_a_parallel_run_records_its_own_clips():
    from bluesky_sandbox.integrations import vec_env  # noqa: PLC0415

    recording = Recording(every=10, length=3)
    clips = Clips(recording)
    v = vec_env(make_env, n_processes=2, record=recording)
    try:
        v.reset(seed=0)
        for _ in range(3):
            v.step(np.stack([v.action_space.sample() for _ in range(v.num_envs)]))
        found = clips.new()
        assert [(c.name, c.step) for c in found] == [("worker0", 0), ("worker1", 0)]
        assert all(_frames(c.data) == 3 and c.path is None for c in found)
        assert clips.new() == []  # each clip is handed out once
    finally:
        v.close()
        clips.close()
    assert not recording.path.exists()


def test_watch_and_record_cannot_share_worker_zero():
    from bluesky_sandbox.integrations import vec_env  # noqa: PLC0415

    with pytest.raises(ValueError, match="pick one"):
        vec_env(make_env, watch=True, record=Recording(every=10, length=3))
