# Rendering

You can switch renderers with a single argument. By default `render_mode=None` runs headless, which is what you
want during training.

| Mode | Visual | Best for |
| :--- | :--- | :--- |
| `pygame` | ![Pygame](media/screenshots/pygame.png) | High-speed 2D prototyping and quick debugging |
| `panda3d` | ![Panda3D](media/screenshots/panda3d.png) | 3D altitude visualization and spatial analysis |
| `qtgl` | ![QtGL](media/screenshots/qtgl.png) | High-fidelity BlueSky native radar display |

```python
env = Env(render_mode="pygame")
```

Each renderer contains optional extras — see [Installation](installation.md#optional-extras).

## Recording video

`render_mode="rgb_array"` opens no window: `env.render()` draws the pygame views
offscreen and returns the frame, a `(height, width, 3)` RGB array. Nothing is
drawn between `render()` calls. `views` sets its layout as it does the window's.

`RecordVideo` records clips from it: `length` steps every `every` steps, each
clip an mp4, drawing only while a clip records. It needs the `recording` extra.

```python
from bluesky_sandbox.interface.wrappers import Clips, Recording, RecordVideo

env = RecordVideo(Env(render_mode="rgb_array"), Recording("videos", every=10_000, length=200))
```

`Clips("videos").new()` returns the clips finished since it last looked, once
each - for logging them, to wandb, say:

```python
clips = Clips("videos")
...
for clip in clips.new():
    wandb.log({f"video/{clip.name}": wandb.Video(str(clip.path))})
```

## Real time and views

`realtime=True` runs the simulation following wall-clock time.

The `views` argument controls the panel layout. See
[Rendering and drivers](api/rendering.md) for the driver and view types.
