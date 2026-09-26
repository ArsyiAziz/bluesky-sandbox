"""Video recording in a generated training script: the options a design keeps,
checked, and the code they become."""

from __future__ import annotations

import ast

import pytest

from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer.recording import (
    RECORD_DEFAULTS,
    record_catalog,
    record_options,
)
from bluesky_sandbox.ui.drivers import FRAME_DRIVERS

from .test_designer import _example_design_spec


def _train(template: str, processes: int = 1, **record) -> str:
    spec = _example_design_spec()
    spec.metadata.update(template=template, processes=processes, record=record or None)
    files = codegen.generate_task(spec, "Rec")
    return next(t for p, t in files.items() if p.endswith("train.py"))


def test_no_record_asks_for_none():
    assert record_options({}) is None and record_options({"record": None}) is None


def test_a_recording_takes_the_defaults_it_is_not_given():
    options = record_options({"record": {"length": 50}})
    assert (options.every, options.length, options.fps) == (
        RECORD_DEFAULTS["every"],
        50,
        10,
    )
    first = next(iter(FRAME_DRIVERS))
    assert options.driver == first and options.views == FRAME_DRIVERS[first].default
    assert options.folder is None


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"driver": "qtgl"}, "record driver must be one of"),
        ({"driver": "pygame", "views": ["world"]}, "pygame has no view world"),
        ({"every": 10, "length": 20}, "at most every"),
        ({"fps": 0}, "fps must be at least 1"),
    ],
)
def test_a_recording_that_cannot_be_made_is_refused(record, match):
    with pytest.raises(ValueError, match=match):
        record_options({"record": record})


def test_the_catalog_offers_every_frame_drivers_views():
    catalog = record_catalog()
    assert set(catalog["drivers"]) == set(FRAME_DRIVERS)
    for name, driver in FRAME_DRIVERS.items():
        assert catalog["drivers"][name]["views"] == list(driver.views)
    assert catalog["defaults"]["driver"] == next(iter(FRAME_DRIVERS))


@pytest.mark.parametrize(
    ("template", "processes"), [("sb3", 1), ("sb3", 3), ("rl", 1), ("rl", 3)]
)
def test_every_training_template_records_and_uploads(template, processes):
    train = _train(
        template, processes, every=500, length=20, views=["horizontal", "tsas"]
    )
    ast.parse(train)
    assert "RECORDING = Recording(every=500, length=20, fps=10)" in train
    assert 'FRAME_DRIVER = "pygame"' in train
    assert "return (HorizontalView, TSASView)" in train
    assert (
        "from bluesky_sandbox.ui.drivers.pygame.views import HorizontalView, TSASView"
        in train
    )
    # Uploaded from memory; saved only with no wandb run to take it.
    assert "wandb.Video(io.BytesIO(clip.data)" in train
    assert "wandb.init(project=WANDB_PROJECT)" in train
    assert "record: bool = True" in train


def test_panda3d_views_are_built_as_instances_and_a_folder_keeps_the_clips():
    train = _train("sb3", 2, driver="panda3d", views=["world"], folder="clips")
    assert 'FRAME_DRIVER = "panda3d"' in train and "return [WorldView()]" in train
    assert 'Recording(every=10000, length=200, fps=10, folder="clips")' in train


def test_parallel_copies_upload_as_they_train_and_as_they_close():
    sb3 = _train("sb3", 2, length=5)
    assert (
        "sb3_vec_env(make_env, n_processes, watch, RECORDING if record else None)"
        in sb3
    )
    assert "callback=UploadClips(clips)" in sb3
    closing = sb3.split("finally:\n        vec.close()", 1)[1]
    assert "for clip in clips.new():" in closing and "clips.close()" in closing

    rl = _train("rl", 2, length=5)
    assert "vec_env(make_env, n_processes, watch, RECORDING if record else None)" in rl
    assert "for clip in clips.new():\n                upload(clip)" in rl


def test_one_rl_copy_records_itself_and_closes_however_the_loop_ends():
    rl = _train("rl", 1, length=5)
    assert "RecordVideo(env, RECORDING, on_clip=upload)" in rl
    loop = rl.split("def training_loop(", 1)[1]
    assert (
        loop.index("try:")
        < loop.index("env.reset(seed=seed)")
        < loop.index("env.close()")
    )


def test_a_package_without_a_recording_is_as_before():
    sb3 = _train("sb3", 2, length=5)
    spec = _example_design_spec()
    spec.metadata.update(template="sb3", processes=2)
    plain = next(
        t
        for p, t in codegen.generate_task(spec, "Rec").items()
        if p.endswith("train.py")
    )
    assert "Recording" in sb3 and "Recording" not in plain and "wandb" not in plain


def test_watch_and_record_are_refused_together():
    spec = _example_design_spec()
    spec.metadata.update(template="rl", watch=True, record={"length": 5})
    with pytest.raises(ValueError, match="pick one"):
        codegen.generate_task(spec, "Rec")


def test_a_plain_package_ignores_a_recording():
    spec = _example_design_spec()
    spec.metadata.update(template="plain", record={"length": 5})
    files = codegen.generate_task(spec, "Rec")
    assert not any(p.endswith("train.py") for p in files)
