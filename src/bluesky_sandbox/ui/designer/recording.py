"""Video recording in a generated training script: the options a design keeps,
and the code they become.

A design asks for clips with ``metadata["record"]``: ``every`` and ``length``
in steps of each copy of the env, ``fps``, the ``driver`` that draws them and
its ``views``, and optionally a ``folder`` to keep them in. The generated
``train.py`` records them offscreen (:class:`~bluesky_sandbox.interface.
wrappers.RecordVideo`) and uploads each, from memory, to the running wandb run.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any

from bluesky_sandbox.ui.drivers import FRAME_DRIVERS

__all__ = ["RECORD_DEFAULTS", "RecordOptions", "record_catalog", "record_options"]

#: A new recording's settings, before the design changes them.
RECORD_DEFAULTS: dict[str, Any] = {"every": 10_000, "length": 200, "fps": 10}


@dataclass(frozen=True)
class RecordOptions:
    """What a design asks of its clips, checked."""

    every: int
    length: int
    fps: int
    driver: str
    views: tuple[str, ...]
    folder: str | None

    def recording_source(self) -> str:
        """The ``Recording(...)`` it becomes, as source."""
        folder = f", folder={json.dumps(self.folder)}" if self.folder else ""
        return f"Recording(every={self.every}, length={self.length}, fps={self.fps}{folder})"


def record_options(metadata: dict[str, Any]) -> RecordOptions | None:
    """The recording a design's metadata asks for, or ``None`` for none."""
    raw = metadata.get("record")
    if not raw:
        return None
    options = {**RECORD_DEFAULTS, **raw}
    driver = str(options.get("driver") or next(iter(FRAME_DRIVERS)))
    if driver not in FRAME_DRIVERS:
        raise ValueError(
            f"record driver must be one of {list(FRAME_DRIVERS)}, got {driver!r}"
        )
    offered = FRAME_DRIVERS[driver]
    views = tuple(options.get("views") or offered.default)
    unknown = [v for v in views if v not in offered.views]
    if unknown:
        raise ValueError(
            f"{driver} has no view {', '.join(unknown)}; it has {', '.join(offered.views)}"
        )
    folder = str(options.get("folder") or "").strip() or None
    every, length, fps = (int(options[k]) for k in ("every", "length", "fps"))
    if fps < 1:
        raise ValueError(f"record fps must be at least 1, got {fps}")
    if length < 1 or every < length:
        raise ValueError(
            f"record length must be at least 1 and at most every ({every}), got {length}"
        )
    return RecordOptions(every, length, fps, driver, views, folder)


def record_catalog() -> dict[str, Any]:
    """What the designer offers for recording: each driver's views, and the
    defaults."""
    return {
        "drivers": {
            name: {"views": list(driver.views), "default": list(driver.default)}
            for name, driver in FRAME_DRIVERS.items()
        },
        "defaults": {**RECORD_DEFAULTS, "driver": next(iter(FRAME_DRIVERS))},
    }


def view_imports(options: RecordOptions) -> str:
    """The import of the views the clips show."""
    by_module: dict[str, list[str]] = {}
    for name in options.views:
        module, cls = FRAME_DRIVERS[options.driver].views[name].split(":")
        by_module.setdefault(module, [])
        if cls not in by_module[module]:
            by_module[module].append(cls)
    return "".join(
        f"from {module} import {', '.join(sorted(classes))}\n"
        for module, classes in sorted(by_module.items())
    )


def views_source(options: RecordOptions) -> str:
    """What ``recorded_views()`` returns, as source."""
    driver = FRAME_DRIVERS[options.driver]
    classes = [driver.views[name].split(":")[1] for name in options.views]
    if driver.instances:
        return "[" + ", ".join(f"{cls}()" for cls in classes) + "]"
    return "(" + ", ".join(classes) + ("," if len(classes) == 1 else "") + ")"


def recording_block(options: RecordOptions, project: str) -> str:
    """The module-level recording settings and ``upload``, as source."""
    driver = FRAME_DRIVERS[options.driver]
    order = "" if driver.instances else ", top to bottom"
    kept = (
        f"each is also kept in {options.folder}/."
        if options.folder
        else "none is kept on disk."
    )
    about = textwrap.fill(
        f"Video clips of training: {options.length} steps of each copy every "
        f"{options.every}, drawn offscreen - nothing is drawn between clips - and "
        f"uploaded as they finish; {kept}",
        width=78,
        initial_indent="#: ",
        subsequent_indent="#: ",
    )
    return f'''

{about}
RECORDING = {options.recording_source()}
#: Who draws the clips.
FRAME_DRIVER = {json.dumps(options.driver)}
#: The wandb project the clips are uploaded to.
WANDB_PROJECT = {json.dumps(project)}


def recorded_views():
    """The views a clip shows{order}."""
    return {views_source(options)}


def upload(clip: Clip) -> None:
    """Where a finished clip goes: to the running wandb run, straight from
    memory. With no run it is saved to videos/ instead, so none is lost."""
    if wandb is not None and wandb.run is not None:
        video = wandb.Video(io.BytesIO(clip.data), fps=RECORDING.fps, format="mp4")
        wandb.log({{f"video/{{clip.name}}": video}})
    elif clip.path is None:
        path = Path("videos") / f"{{clip.name}}-step{{clip.step:09d}}.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(clip.data)
'''


def recording_imports(options: RecordOptions, *, clips: bool) -> tuple[str, str, str]:
    """The imports the recording code needs: the standard library's, the
    sandbox's, and wandb's - optional, so imported after the rest."""
    names = "Clip, Clips, Recording" if clips else "Clip, Recording, RecordVideo"
    return (
        "import io\nfrom pathlib import Path\n\n",
        f"from bluesky_sandbox.interface.wrappers import {names}\n"
        + view_imports(options),
        "\ntry:\n    import wandb\nexcept ImportError:  # clips go to videos/ instead\n"
        "    wandb = None\n",
    )


def main_with_wandb(call: str) -> str:
    """The ``__main__`` block: the call, inside a wandb run the clips go to."""
    return f"""if __name__ == "__main__":
    run = wandb.init(project=WANDB_PROJECT) if wandb is not None else None
    try:
        {call}
    finally:
        if run is not None:
            run.finish()
"""
