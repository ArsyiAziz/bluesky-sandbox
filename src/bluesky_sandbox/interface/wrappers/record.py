"""Record a running env as video clips, drawn only while a clip records.

:class:`RecordVideo` records ``length`` steps of the env it wraps every
``every`` steps, each clip an mp4 of its own. Between clips it draws nothing,
so an env recorded in short bursts trains at nearly full speed. The env must
be built with ``render_mode="rgb_array"``, which draws offscreen.

A clip is named ``{name}-step{start}.mp4``, after the step it starts at, and
is written under a temporary name until it is whole, so a clip found in the
folder is always complete. :class:`Clips` finds the clips a run has finished
since it last looked - in parallel runs each worker records its own, and the
main process is the one that can log them::

    clips = Clips("videos")
    ...
    for clip in clips.new():
        wandb.log({f"video/{clip.name}": wandb.Video(str(clip.path))})

Writing mp4 needs the ``recording`` extra (``imageio``, ``imageio-ffmpeg``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

__all__ = ["Clip", "Clips", "RecordVideo", "Recording"]

_CLIP = re.compile(r"^(?P<name>.+)-step(?P<step>\d+)\.mp4$")
_PART = ".part.mp4"


@dataclass(frozen=True)
class Recording:
    """How to record: ``length`` steps every ``every`` steps, into ``folder``,
    played back at ``fps``."""

    folder: str | Path
    every: int
    length: int
    fps: int = 10

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(
                f"a clip is at least one step long, got length={self.length}"
            )
        if self.every < self.length:
            raise ValueError(
                f"clips would overlap: every={self.every} is shorter than length={self.length}"
            )


class RecordVideo(BaseParallelWrapper):
    """Record the wrapped env as :class:`Recording` says, a clip per burst.

    ``name`` starts each clip's file name - a worker's, say, among the copies
    of a parallel run. ``on_clip`` is called with each clip's path once it is
    written.
    """

    def __init__(
        self,
        env: Any,
        recording: Recording,
        *,
        name: str = "clip",
        on_clip: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(env)
        if self.env.unwrapped.render_mode != "rgb_array":
            raise ValueError(
                "RecordVideo draws frames offscreen: build the env with "
                f'render_mode="rgb_array", not {self.env.unwrapped.render_mode!r}'
            )
        self.recording = recording
        self.name = name
        self.on_clip = on_clip
        self.folder = Path(recording.folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._steps = 0
        self._frames: list[np.ndarray] | None = None
        self._start = 0

    def reset(self, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        self._record()
        return out

    def step(self, actions):
        out = self.env.step(actions)
        self._steps += 1
        self._record()
        return out

    def close(self):
        try:
            if self._frames:
                self._write()  # what there is of the clip
        finally:
            self.env.close()

    def _record(self) -> None:
        if self._frames is None and self._steps % self.recording.every == 0:
            self._frames, self._start = [], self._steps
        if self._frames is None:
            return
        self._frames.append(self.env.render())
        if len(self._frames) >= self.recording.length:
            self._write()

    def _write(self) -> None:
        import imageio  # noqa: PLC0415 - optional extra: [recording]

        frames, self._frames = self._frames or [], None
        path = self.folder / f"{self.name}-step{self._start:09d}.mp4"
        part = path.with_name(path.stem + _PART)
        with imageio.get_writer(
            part,
            fps=self.recording.fps,
            codec="libx264",
            macro_block_size=1,
            output_params=["-r", str(self.recording.fps)],
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        part.replace(path)
        if self.on_clip is not None:
            self.on_clip(path)


@dataclass(frozen=True)
class Clip:
    """A finished clip: its file, whose it is, and the step it starts at."""

    path: Path
    name: str
    step: int


class Clips:
    """The clips in a folder, handed out once each as they are finished."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self._seen: set[Path] = set()

    def new(self) -> list[Clip]:
        """The clips finished since the last call, by step, then name."""
        found = []
        for path in self.folder.glob("*.mp4") if self.folder.is_dir() else ():
            match = _CLIP.match(path.name)
            if path in self._seen or path.name.endswith(_PART) or match is None:
                continue
            self._seen.add(path)
            found.append(Clip(path, match["name"], int(match["step"])))
        return sorted(found, key=lambda c: (c.step, c.name))
