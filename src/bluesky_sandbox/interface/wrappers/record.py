"""Record a running env as video clips, drawn only while a clip records.

:class:`RecordVideo` records ``length`` steps of the env it wraps every
``every`` steps, each clip an mp4 of its own. Between clips it draws nothing,
so an env recorded in short bursts trains at nearly full speed. The env must
be built with ``render_mode="rgb_array"``, which draws offscreen.

A finished clip is a :class:`Clip`: its mp4 in memory (:attr:`Clip.data`),
ready to upload. With no ``folder`` the clips are not kept - each is encoded in
a temporary file, handed out and deleted - so a run that uploads its clips,
to wandb, say, leaves none on disk::

    recording = Recording(every=10_000, length=200)
    env = RecordVideo(Env(render_mode="rgb_array"), recording, on_clip=upload)

    def upload(clip):
        wandb.log({f"video/{clip.name}": wandb.Video(io.BytesIO(clip.data), format="mp4")})

With a ``folder`` each clip is kept there too, as ``{name}-step{start}.mp4``.

In a parallel run each worker records its own clips, and the main process is
the one that can upload them: :class:`Clips` hands out the clips the workers
have finished since it last looked, once each.

Writing mp4 needs the ``recording`` extra (``imageio``, ``imageio-ffmpeg``).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pettingzoo.utils.wrappers import BaseParallelWrapper

__all__ = ["Clip", "Clips", "RecordVideo", "Recording"]

_CLIP = re.compile(r"^(?P<name>.+)-step(?P<step>\d+)\.mp4$")
_PART = ".part.mp4"


@dataclass(frozen=True)
class Recording:
    """How to record: ``length`` steps every ``every`` steps, played back at
    ``fps``.

    ``folder`` keeps every clip there. Left out, clips are not kept: they are
    written to a temporary folder only until they are handed out. That folder
    is made when first used (:attr:`path`) and travels with the recording, so
    every worker it is sent to writes where one :class:`Clips` finds them.
    """

    every: int
    length: int
    fps: int = 10
    folder: str | Path | None = None

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(
                f"a clip is at least one step long, got length={self.length}"
            )
        if self.every < self.length:
            raise ValueError(
                f"clips would overlap: every={self.every} is shorter than length={self.length}"
            )

    @property
    def keep(self) -> bool:
        """Whether clips stay in :attr:`path` once handed out."""
        return self.folder is not None

    @property
    def path(self) -> Path:
        """Where clips are written: :attr:`folder`, or the temporary one."""
        if self.folder is not None:
            return Path(self.folder)
        made = self.__dict__.get("_temporary")
        if made is None:
            made = Path(tempfile.mkdtemp(prefix="bluesky-clips-"))
            object.__setattr__(self, "_temporary", made)
        return made


@dataclass(frozen=True)
class Clip:
    """A finished clip: whose it is, the step it starts at, and its mp4.
    ``path`` is where it is kept, or ``None`` when it is not."""

    name: str
    step: int
    data: bytes = field(repr=False)
    path: Path | None = None


def _hand_out(path: Path, name: str, step: int, keep: bool) -> Clip:
    data = path.read_bytes()
    if not keep:
        path.unlink(missing_ok=True)
    return Clip(name, step, data, path if keep else None)


class RecordVideo(BaseParallelWrapper):
    """Record the wrapped env as :class:`Recording` says, a clip per burst.

    ``name`` starts each clip's name - a worker's, say, among the copies of a
    parallel run. ``on_clip`` is given each clip once it is finished; without
    one, a clip waits in the recording's folder for :class:`Clips`.
    """

    def __init__(
        self,
        env: Any,
        recording: Recording,
        *,
        name: str = "clip",
        on_clip: Callable[[Clip], None] | None = None,
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
        self.folder = recording.path
        self.folder.mkdir(parents=True, exist_ok=True)
        self._steps = 0
        self._writer: Any = None
        self._frames = 0
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
            if self._writer is not None:
                self._finish()  # what there is of the clip
        finally:
            self.env.close()
        # Every clip was handed to on_clip and none kept: the temporary folder
        # is empty now. Another recorder may still be using it; then it stays.
        if self.on_clip is not None and not self.recording.keep:
            try:
                self.folder.rmdir()
            except OSError:
                pass

    def _path(self) -> Path:
        return self.folder / f"{self.name}-step{self._start:09d}.mp4"

    def _record(self) -> None:
        if self._writer is None and self._steps % self.recording.every == 0:
            self._begin()
        if self._writer is None:
            return
        # Each frame goes straight to the encoder: a clip is never held whole.
        self._writer.append_data(self.env.render())
        self._frames += 1
        if self._frames >= self.recording.length:
            self._finish()

    def _begin(self) -> None:
        import imageio  # noqa: PLC0415 - optional extra: [recording]

        self._start, self._frames = self._steps, 0
        fps = self.recording.fps
        self._writer = imageio.get_writer(
            self._path().with_name(self._path().stem + _PART),
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            output_params=["-r", str(fps)],
        )

    def _finish(self) -> None:
        writer, self._writer = self._writer, None
        writer.close()
        path = self._path()
        # Complete under its own name only now, so Clips never sees half a clip.
        path.with_name(path.stem + _PART).replace(path)
        if self.on_clip is not None:
            self.on_clip(_hand_out(path, self.name, self._start, self.recording.keep))


class Clips:
    """The clips a recording's workers finish, handed out once each - in the
    main process of a parallel run, to upload. Clips not kept are deleted as
    they are handed out; :meth:`close` removes the temporary folder."""

    def __init__(self, recording: Recording | str | Path) -> None:
        if isinstance(recording, Recording):
            self.folder, self.keep = recording.path, recording.keep
        else:
            self.folder, self.keep = Path(recording), True
        self._seen: set[Path] = set()

    def new(self) -> list[Clip]:
        """The clips finished since the last call, by step, then name."""
        found = []
        for path in self.folder.glob("*.mp4") if self.folder.is_dir() else ():
            match = _CLIP.match(path.name)
            if path in self._seen or path.name.endswith(_PART) or match is None:
                continue
            self._seen.add(path)
            found.append((int(match["step"]), match["name"], path))
        return [
            _hand_out(path, name, step, self.keep) for step, name, path in sorted(found)
        ]

    def close(self) -> None:
        """Remove the temporary folder of a recording that keeps no clips."""
        if not self.keep:
            shutil.rmtree(self.folder, ignore_errors=True)
